# ABOUTME: Owns one bounded RunPod rental and standard matched ODCV or MASK evaluation.
# ABOUTME: Records pod/process ownership immediately, preserves logs, audits output and verifies teardown.
"""PLAN.json fields: eval,target,target_revision,base_revision,pod_name,port,output_dir,
max_hours,hourly_ceiling_usd; optional cloud=SECURE,boot_timeout_s=2400,idle_timeout_s=3600,
resume_from,ssh_key. Invocation: python scratch/plain_dose/eval_owner.py PLAN.json.
No resumption of owner state: a recovery plan gets a new output_dir and explicit resume_from.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra import runpod
from src.infra.endpoints.vllm import SshExec, resolve_target
from scratch.plain_dose.audit_evals import BASE_REVISION, audit_entry


def command(plan, host, output):
    argv = [sys.executable, str(ROOT / 'scripts/run_eval.py'), '--name', plan['eval'],
            '--target', plan['target'], '--server', host, '--port', str(plan['port']),
            '--terminate-pod']
    if plan.get('ssh_key'):
        argv += ['--ssh-key', plan['ssh_key']]
    if plan['eval'] == 'odcv' and os.name == 'nt':
        argv += ['--server-bind', '0.0.0.0']
    argv += [f"target_revision={plan['target_revision']}", f'output_root={output.as_posix()}']
    if plan['eval'] == 'odcv':
        argv += ['concurrency=16', 'passes=3', 'smoke=false']
    else:
        argv += ['max_generation_error_rate=0.05', 'passes=1', 'subsample=null', 'mode=think']
    if plan.get('resume_from'):
        argv += [f"resume_from={Path(plan['resume_from']).absolute().as_posix()}"]
    return argv


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    assert plan['eval'] in ('odcv', 'mask')
    assert re.fullmatch('[a-f0-9]{40}', plan['target_revision'])
    assert plan['base_revision'] == BASE_REVISION
    assert plan['pod_name'].startswith('nika-')
    assert 1024 <= int(plan['port']) <= 65535 and int(plan['port']) != 8080
    assert 0 < float(plan['max_hours']) <= 24 and float(plan['hourly_ceiling_usd']) > 0
    assert Path.cwd().resolve() == ROOT, f'Run from {ROOT}'
    spec = resolve_target(plan['target'], revision=plan['target_revision'])
    assert spec.revision == plan['target_revision'] and spec.base_revision == BASE_REVISION
    assert spec.mode == 'think' and spec.base_model == 'Qwen/Qwen3.6-27B'
    if plan['eval'] == 'odcv':
        from src.eval.docker import docker_preflight, require_lf_shell_scripts
        docker_preflight()
        require_lf_shell_scripts(ROOT / 'src/eval/misalignment/odcv/third_party/odcv-bench')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', int(plan['port'])))
    out = Path(plan['output_dir']).absolute()
    canonical_out = out.resolve()
    assert canonical_out.is_relative_to(ROOT / 'output'), 'Output must remain under this worktree/output'
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents concurrent or accidental duplicate rentals for this owner.
    status = out / 'status.json'
    with status.open('x', encoding='utf-8') as stream:
        json.dump({'phase': 'claimed', 'pid': os.getpid()}, stream)
    state = {'phase': 'preflight', 'pid': os.getpid(), 'plan': plan,
             'owned_pod': None, 'started_epoch': time.time(), 'source_root': str(ROOT),
             'canonical_output_dir': str(canonical_out), 'lexical_output_dir': str(out)}
    child = guard = None
    pod = None

    def save(**updates):
        state.update(updates, updated_epoch=time.time())
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        tmp.replace(status)

    def registered(pod_id):
        nonlocal pod, guard
        pod = pod_id
        save(phase='booting', owned_pod=pod, created_epoch=time.time())
        guard = runpod.start_watchdog(pod, int(float(plan['max_hours']) * 3600), out / 'watchdog.log')
        save(watchdog_pid=guard.pid)
        info = runpod.call('GET', f'/pods/{pod}')
        price = float(info['costPerHr'])
        save(hourly_usd=price)
        assert price <= float(plan['hourly_ceiling_usd']), f'Hourly price {price} exceeds ceiling'

    failed = False
    try:
        save()
        provision = runpod.up(name=plan['pod_name'], eval=plan['eval'], target=plan['target'],
                             count=1, cloud=plan.get('cloud', 'SECURE'), push_env=True,
                             max_hours=float(plan['max_hours']),
                             boot_timeout_s=int(plan.get('boot_timeout_s', 2400)),
                             on_provisioned=registered)
        (out / 'provision.txt').write_text(provision, encoding='utf-8')
        host = next(line.split(None, 1)[1] for line in provision.splitlines() if line.startswith('host:'))
        save(host=host)
        remote = SshExec(host, port=int(plan['port']), identity=plan.get('ssh_key', ''))
        deadline = state['created_epoch'] + float(plan['max_hours']) * 3600
        boot_end = min(deadline, state['created_epoch'] + int(plan.get('boot_timeout_s', 2400)))
        while time.time() < boot_end:
            tail = remote._ssh('tail -c 3500 /workspace/boot.log', timeout=30)
            save(boot_tail=tail)
            if any(line.startswith('READY') for line in tail.splitlines()):
                break
            time.sleep(15)
        else:
            raise TimeoutError('Pod bootstrap exceeded its time bound')
        remote.check_ready()
        argv = command(plan, host, out / 'eval')
        env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'}
        with (out / 'eval.log').open('wb') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        save(phase='evaluating', eval_pid=child.pid, command=argv)
        last_progress = time.time()
        previous = None
        while child.poll() is None:
            files = [p for p in (out / 'eval').rglob('*') if p.is_file()]
            sizes = []
            for path in files:
                try:
                    stat = path.stat()
                    sizes.append((stat.st_size, stat.st_mtime_ns))
                except FileNotFoundError:
                    pass  # Packaging moves files while the process is alive.
            marker = (len(files), sum(x[0] for x in sizes), max((x[1] for x in sizes), default=0),
                      (out / 'eval.log').stat().st_size)
            if marker != previous:
                previous, last_progress = marker, time.time()
            save(last_progress_epoch=last_progress, artifact_files=len(files))
            if time.time() >= deadline:
                raise TimeoutError('Evaluation owner lifetime reached')
            if time.time() - last_progress > int(plan.get('idle_timeout_s', 3600)):
                raise TimeoutError('No evaluation artifact/log progress within idle bound')
            time.sleep(20)
        save(eval_exit=child.returncode, phase='auditing')
        assert child.returncode == 0, f'Eval exited {child.returncode}; retain local evidence'
        candidates = [p.parent.parent for p in (out / 'eval').rglob('metadata/run_meta.json')
                      if (p.parent.parent / 'results/results.json').is_file()]
        candidates = [p for p in candidates if json.loads((p / 'metadata/run_meta.json').read_text())
                      .get('target_revision') == plan['target_revision']]
        assert len(candidates) == 1, f'Expected one packaged run, found {candidates}'
        receipt = audit_entry({'eval': plan['eval'], 'run_dir': str(candidates[0]),
                               'target': plan['target'], 'target_revision': plan['target_revision']})
        (out / 'local_audit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        save(phase='local_verified', packaged_run=str(candidates[0]),
             publication_verification='pending pinned-HF hash audit')
    except BaseException as exc:
        failed = True
        save(phase='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
        except BaseException as exc:
            failed = True
            save(child_cleanup_error=f'{type(exc).__name__}: {exc}')
        if pod:
            try:
                runpod.teardown(pod)
                save(terminated=True, termination_verified_epoch=time.time())
                if guard is not None:
                    guard.terminate()
            except BaseException as exc:
                failed = True
                save(teardown_error=f'{type(exc).__name__}: {exc}')
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    raise SystemExit(run(parser.parse_args().plan))
