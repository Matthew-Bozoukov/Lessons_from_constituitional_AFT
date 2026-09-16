# ABOUTME: Bounded owner for one overnight baseline pod; dispatches the existing eval runner.
# ABOUTME: Arms independent teardown before bootstrap and records actual rate and estimated accrual.
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid

from omegaconf import OmegaConf
from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target, SshExec, ssh_argv, _SERVER_PATTERN
from src.eval.run_eval import main as evaluate
from src.eval.docker import docker_preflight, require_network_capacity, require_lf_shell_scripts
from src.infra.huggingface import hf_org

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/nonmoral_overnight/20260909'
TARGET = 'dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch'
REVISION = '2225547cec8bd312a1e025f02fb6b3321c047e4a'
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
LIFETIME = 8 * 3600
CHECKPOINTS = {
    'nonmoral': (TARGET, REVISION, 'baseline'),
    'nonmoral_lf': (TARGET, REVISION, 'baseline_lf'),
    'math': ('matboz/qwen3.6-27b-lora-9284-numina-control-716-r64',
             'edfb4287c10f553c541ba28216f202d0c0f47055', 'math_baseline'),
    'table2': ('dougalldeepmind/2026-08-04-qwen36-lora-table2-only-9284-rank-64',
               '2c513ea7513baf792bd2becf0900b5c9d858c92d', 'table2_baseline'),
}


def load_plan(path):
    """Validate the one-shot broader evaluation contract before any network/rental."""
    plan = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    required = {'target', 'target_revision', 'base_model', 'base_revision', 'output_dir',
                'run_name', 'eval_output_root', 'eval_config', 'eval_config_sha256', 'expected_cells', 'passes',
                'gpu_cap_usd', 'backup_reserve_usd', 'judge_cap_usd',
                'max_gpu_hourly_usd', 'storage_hourly_reserve_usd'}
    if set(plan) != required:
        raise ValueError(f'Plan fields differ: missing={required-set(plan)}, extra={set(plan)-required}')
    for field in ('target_revision', 'base_revision'):
        if not re.fullmatch('[0-9a-f]{40}', plan[field]):
            raise ValueError(f'{field} must be an exact commit SHA')
    if plan['base_model'] != 'Qwen/Qwen3.6-27B' or plan['base_revision'] != BASE_REVISION:
        raise ValueError('Broader evaluation must use the frozen Qwen baseline revision')
    if not re.fullmatch(r'dougalldeepmind/[a-zA-Z0-9_.-]+', plan['target']):
        raise ValueError('Expected one public dougalldeepmind adapter')
    if not re.fullmatch(r'odcv-(?:broader|stakes)-[a-z0-9-]+', plan['run_name']):
        raise ValueError('Use a distinct odcv-broader-* or odcv-stakes-* run_name')
    eval_root = PureWindowsPath(plan['eval_output_root'])
    if not eval_root.is_absolute() or len(str(eval_root)) > 24 or len(plan['run_name']) > 40:
        raise ValueError('Use a short absolute Windows eval_output_root (e.g. C:/nm-eval) and run_name <=40 chars')
    if eval_root == PureWindowsPath(plan['output_dir']):
        raise ValueError('eval_output_root must be separate from metadata output_dir')
    # Timestamped directories use underscore-normalized run names. Do not launch this arm twice.
    if Path(str(eval_root)).exists() and any(Path(str(eval_root)).glob(
            '*_'+plan['run_name'].replace('-', '_')+'_*')):
        raise ValueError('An evaluation with this run_name already exists; no automatic rerun')
    if (plan['expected_cells'], plan['passes']) != (80, 3):
        raise ValueError('Exactly one 3x80 evaluation is authorized')
    gpu, judge, reserve = (float(plan[k]) for k in
                          ('gpu_cap_usd', 'judge_cap_usd', 'backup_reserve_usd'))
    rate, storage = (float(plan[k]) for k in
                     ('max_gpu_hourly_usd', 'storage_hourly_reserve_usd'))
    if not (0 < judge <= 5 and 2 <= reserve < gpu <= 15 and gpu+judge <= 20
            and 0 < rate <= 4.5 and storage >= 0.1):
        raise ValueError('Plan exceeds the $20 allocation or omits the $2 backup reserve')
    config = Path(plan['eval_config']).resolve()
    if hashlib.sha256(config.read_bytes()).hexdigest() != plan['eval_config_sha256']:
        raise ValueError('Frozen eval config SHA256 mismatch')
    cfg = OmegaConf.load(config)
    expected = {'temperature': 0.7, 'passes': 3, 'concurrency': 8,
                'scenario_timeout_s': 2400, 'progress_judge': True, 'smoke': False,
                'judge_workers': 4}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen config differs from the common baseline protocol')
    judge_spec = {'gemini-3-flash-preview': 'google/gemini-3-flash-preview'}
    if cfg.judges != judge_spec or cfg.progress_judges != judge_spec:
        raise ValueError('Both judges must match the common baseline protocol')
    if cfg.serving != dict(context_window=28000, needs_tool_calls=True, reuses_long_prefixes=True):
        raise ValueError('Serving protocol differs from the common baseline')
    if cfg.get('exclude_scenarios') or cfg.get('mode') or cfg.get('system_preamble_file'):
        raise ValueError('No exclusions or intervention overrides in this evaluation')
    out = Path(plan['output_dir']).resolve()
    if out == OUT.resolve() or out.is_relative_to(OUT.resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Plan output_dir must be fresh and distinct from the baseline lane')
    return plan


def checked_spec(plan):
    spec = resolve_target(plan['target'])
    if (spec.revision, spec.base_model, spec.base_revision) != (
            plan['target_revision'], plan['base_model'], plan['base_revision']):
        raise ValueError('Target/base revision differs from the frozen plan')
    if not spec.adapter or spec.mode != 'think':
        raise ValueError('Expected a thinking-mode LoRA')
    return spec


LOG_FILES = ('boot.log', 'output/serve/vllm.log')


def fetch_eval_logs(host, identity, out, require_server, timeout=300):
    """Snapshot only the two owned logs, then verify transfer bytes before teardown."""
    remote = SshExec(host, port=8000, identity=identity)
    deadline = time.monotonic()+timeout
    def remaining():
        seconds = deadline-time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Remote-log fetch exhausted its remaining recovery time')
        return seconds
    if require_server:
        # A budget-killed local driver may not have reached run_eval's server cleanup.
        remote._ssh(f'pkill -f {shlex.quote(_SERVER_PATTERN)} || true', timeout=remaining())
        alive = remote._ssh(f'pgrep -f {shlex.quote(_SERVER_PATTERN)} >/dev/null && echo up || echo down',
                            timeout=remaining()).strip()
        if alive != 'down':
            raise RuntimeError('Owned vLLM server still active; log snapshot is not final')
    script = f'''import hashlib,json,tarfile
from pathlib import Path
root=Path('/workspace')
names={LOG_FILES!r}
required=names if {require_server!r} else names[:1]
for name in required:
 if not (root/name).is_file(): raise RuntimeError('Missing required log: '+name)
archive=root/'eval-log-backup.tar'
with tarfile.open(archive,'w') as tar:
 for name in names:
  p=root/name
  if p.is_symlink(): raise RuntimeError('Symlink log refused')
  if p.is_file():
   before=p.stat()
   tar.add(p,arcname=name,recursive=False)
   after=p.stat()
   if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
    raise RuntimeError('Log changed during archive: '+name)
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''
    manifest = json.loads(remote._ssh('python3 -c '+shlex.quote(script), timeout=min(120, remaining())))
    path = Path(out)/f'remote_logs_{uuid.uuid4().hex}.tar.partial'
    argv, target = ssh_argv(host, identity)
    with path.open('xb') as stream:
        result = subprocess.run([*argv, target, 'cat /workspace/eval-log-backup.tar'],
                                stdout=stream, stderr=subprocess.PIPE, timeout=min(180, remaining()))
    if result.returncode or path.stat().st_size != manifest['bytes'] or (
            hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']):
        raise RuntimeError('Remote log transfer hash/size verification failed')
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or any(not m.isfile() or m.name not in LOG_FILES for m in members):
            raise RuntimeError('Unexpected remote backup member')
        if not set(LOG_FILES if require_server else LOG_FILES[:1]) <= set(names):
            raise RuntimeError('Required remote log absent from archive')
        files = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    final = Path(out)/'remote_logs.tar'
    path.replace(final)
    return dict(verified=True, path=str(final), **manifest, file_sha256=files,
                scope='Only boot and vLLM logs; Docker rollouts and eval outputs are already local')


def recover_eval_logs(state, identity, out, save, *, now=None, sleep=None):
    """Use the remaining fixed lifetime for recovery; owner death must not cut it short."""
    now = now or time.time
    sleep = sleep or time.sleep
    deadline = state['rented_at_unix']+state['max_lifetime_s']
    attempts = 0
    save(phase='recovering_remote_logs', recovery_deadline_unix=deadline)
    while now() < deadline-15:
        attempts += 1
        try:
            host = state.get('host')
            if not host:
                ip, port = runpod._ssh_endpoint(state['pod_id'], timeout_s=min(60, int(deadline-now()-10)))
                host = f'root@{ip}:{port}'
                save(host=host)
            remaining = deadline-now()-10
            if remaining <= 0:
                raise TimeoutError('No recovery time remains before the fixed watchdog')
            receipt = fetch_eval_logs(host,identity,out,bool(state.get('evaluation_started')),
                                      timeout=min(300, remaining))
            if receipt.get('verified') is not True:
                raise RuntimeError('Remote-log receipt is not verified')
            save(local_log_backup=receipt, recovery_attempts=attempts,
                 ordinary_teardown_blocked=False, phase='remote_logs_verified')
            return True
        except Exception as exc:
            save(backup_error=f'{type(exc).__name__}: {exc}', recovery_attempts=attempts,
                 ordinary_teardown_blocked=True,
                 partial_log_archives=sorted(str(p) for p in Path(out).glob('remote_logs_*.tar.partial')))
            sleep(max(0, min(15, deadline-now())))
    save(phase='artifact_recovery_required', ordinary_teardown_blocked=True,
         recovery_attempts=attempts)
    print('URGENT: remote logs unverified; retaining the owner until the fixed watchdog deadline.',flush=True)
    while now() < deadline:
        sleep(min(15, deadline-now()))
    save(phase='recovery_deadline_reached', emergency_watchdog_handoff=True)
    return False


def evaluate_frozen(plan_path, config, host, identity):
    """Child mode: freeze the single resolved spec while the shared runner owns serving."""
    from unittest.mock import patch
    plan = OmegaConf.to_container(OmegaConf.load(plan_path), resolve=True)
    status = json.loads((Path(plan_path).parent/'broader_eval_status.json').read_text())
    if hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() != status['plan_sha256'] or (
            hashlib.sha256(Path(config).read_bytes()).hexdigest() != status['config_sha256']):
        raise ValueError('Frozen plan or runtime configuration changed before eval dispatch')
    spec = checked_spec(plan)
    def one_target(target):
        if target != spec.hf_path:
            raise ValueError('Unexpected extra eval target')
        return spec
    with patch('src.eval.run_eval.resolve_target', one_target):
        evaluate(['--name', 'odcv', '--config', str(config), '--target', spec.hf_path,
                  '--server', host, '--ssh-key', identity])


def dispatch_frozen(plan_path, config, host, identity, timeout):
    """Bound the owned eval process tree so remote-log backup has reserved time."""
    argv = [sys.executable, str(Path(__file__).resolve()), '--evaluate-plan', str(plan_path),
            '--eval-config', str(config), '--server', host, '--ssh-key', identity]
    proc = subprocess.Popen(argv, start_new_session=os.name != 'nt')
    try:
        code = proc.wait(timeout=max(1, timeout))
        if code:
            raise RuntimeError(f'Owned evaluation exited with status {code}; no automatic rerun')
    finally:
        if proc.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True, timeout=30)
            else:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def prior_lane_spend(out: Path, prefix: str) -> float:
    """Refuse overlap/replacement; charge closed rentals by full observed lifetime."""
    total = 0.0
    for _, _, known_prefix in CHECKPOINTS.values():
        path = out / f'{known_prefix}_status.json'
        if not path.exists():
            continue
        prior = json.loads(path.read_text())
        if not prior.get('pod_id'):
            continue
        if known_prefix == prefix or not prior.get('termination_verified'):
            raise RuntimeError(f'Existing owned rental must be preserved/closed: {path}')
        elapsed = prior['terminated_at_unix'] - prior['rented_at_unix']
        total += elapsed / 3600 * (prior.get('actual_gpu_hourly_usd', 4.5) + 0.1)
    return total


def main(checkpoint='nonmoral', plan_path=None):
    os.chdir(ROOT)
    plan = load_plan(plan_path) if plan_path else None
    out = Path(plan['output_dir']).resolve() if plan else OUT
    out.mkdir(parents=True, exist_ok=not bool(plan))
    if plan:
        target, revision, prefix = plan['target'], plan['target_revision'], 'broader_eval'
        prior_spend = 0
        gpu_cap, judge_cap = float(plan['gpu_cap_usd']), float(plan['judge_cap_usd'])
        max_rate, storage = float(plan['max_gpu_hourly_usd']), float(plan['storage_hourly_reserve_usd'])
        reserve = float(plan['backup_reserve_usd'])
        lifetime = int(gpu_cap/(max_rate+storage)*3600) - 60  # watchdog poll/teardown margin
        work_lifetime = int((gpu_cap-reserve)/(max_rate+storage)*3600) - 60
        frozen_plan = out/'plan.yaml'
        OmegaConf.save(OmegaConf.create(plan), frozen_plan)
    else:
        target, revision, prefix = CHECKPOINTS[checkpoint]
        prior_spend = prior_lane_spend(out, prefix)
        gpu_cap, judge_cap, max_rate, storage, reserve = 60, 10, 4.5, 0.1, 5
        lifetime = min(LIFETIME, int((55 - prior_spend) / 4.6 * 3600))
    assert lifetime >= 3600, 'Less than one hour remains in the GPU lane; no rental'
    state = dict(phase='preflight', target=target, target_revision=revision,
                 base_revision=BASE_REVISION, pid=os.getpid(), pod_id=None,
                 gpu_cap_usd=gpu_cap, judge_cap_usd=judge_cap, max_lifetime_s=lifetime,
                 prior_lane_gpu_storage_estimate_usd=prior_spend,
                 max_gpu_hourly_usd=max_rate, storage_hourly_reserve_usd=storage,
                 cost_note=f'Elapsed-rate estimates, not provider invoice; ${reserve} teardown reserve.',
                 passes=3, expected_rollouts=240)
    if plan:
        state.update(plan_sha256=hashlib.sha256(frozen_plan.read_bytes()).hexdigest(),
                     total_allocation_usd=gpu_cap+judge_cap, work_lifetime_s=work_lifetime)
    shared_ledger = out/('judge_ledger.json' if plan else 'baseline_judge_ledger.json')
    state['judge_ledger_start_index'] = len(json.loads(shared_ledger.read_text())) if shared_ledger.exists() else 0
    lock = threading.Lock()
    def save(**updates):
        with lock:
            state.update(updates)
            state['updated_at_unix'] = time.time()
            if state.get('rented_at_unix'):
                elapsed = (state.get('terminated_at_unix') or time.time())-state['rented_at_unix']
                state['elapsed_seconds'] = elapsed
                state['estimated_gpu_and_storage_usd'] = elapsed/3600*(state.get('actual_gpu_hourly_usd',max_rate)+storage)
            ledger = shared_ledger
            if ledger.exists():
                try:
                    entries = json.loads(ledger.read_text())
                    state['judge_charged_or_reserved_usd'] = sum(e['charged_or_reserved_usd'] for e in entries)
                except (ValueError, OSError):
                    state['judge_ledger_read_pending'] = True
            tmp=out/f'{prefix}_status.tmp'
            tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
            tmp.replace(out/f'{prefix}_status.json')
    save()
    docker_preflight()
    require_network_capacity(16,because='ODCV concurrency 8')
    keypair=runpod.default_keypair()
    assert keypair, 'SSH keypair missing; no rental'
    assert hf_org() == 'dougalldeepmind'
    spec=checked_spec(plan) if plan else resolve_target(target)
    assert (spec.revision,spec.base_revision)==(revision,BASE_REVISION)
    cfg=OmegaConf.load(plan['eval_config'] if plan else ROOT/'scratch/nonmoral/odcv-paired.yaml')
    cfg.passes=3
    cfg.output_root=plan['eval_output_root'] if plan else 'C:/nm-eval'
    cfg.run_name=plan['run_name'] if plan else f'odcv-{checkpoint}-common-3x'
    cfg.bench_dir=str((ROOT/str(cfg.bench_dir)).resolve())
    require_lf_shell_scripts(Path(cfg.bench_dir))
    if plan:
        from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names
        cells = {v: scenario_names(Path(cfg.bench_dir), v) for v in VARIANTS}
        if any(len(names) != 40 for names in cells.values()):
            raise ValueError('Expected exactly 40 scenarios in each of two variants')
        save(expected_cells=cells)
    cfg.judge_budget=dict(ledger=str(shared_ledger),cap_usd=judge_cap,max_tokens=8192)
    config=out/f'{prefix}_config.yaml'
    OmegaConf.save(cfg,config)
    sources=['src/eval/run_eval.py','src/eval/misalignment/odcv/runner.py',
             'src/eval/docker.py','src/eval/misalignment/odcv/odcv_rollout.py',
             'src/eval/misalignment/odcv/recover.py',
             'src/eval/misalignment/odcv/odcv_judge.py','src/eval/misalignment/odcv/progress_judge.py',
             'src/infra/runpod.py','scratch/nonmoral/overnight_baseline.py']
    save(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
         git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
         shell_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT/str(cfg.bench_dir)).rglob('*.sh'))})
    watchdog=None
    finished=threading.Event()
    def monitor():
        while not finished.wait(30):
            save()
    def arm(pod_id):
        nonlocal watchdog
        save(pod_id=pod_id,rented_at_unix=time.time(),phase='provisioned')
        watchdog=runpod.start_watchdog(pod_id,lifetime,out/f'watchdog_{pod_id}.log')
        save(watchdog_pid=watchdog.pid)
        info=runpod.call('GET',f'/pods/{pod_id}')
        rate=float(info['costPerHr'])
        save(actual_gpu_hourly_usd=rate,provider_created_at=info.get('createdAt'),
             worst_case_gpu_storage_and_teardown_usd=prior_spend+lifetime/3600*(rate+storage)+(0 if plan else reserve))
        if plan and not 0 < rate <= max_rate:
            # This is a budget emergency before any evaluation, not normal teardown.
            save(emergency_budget_shutdown=True)
            gone = runpod.terminate(pod_id)
            still = any(p.get('id') == pod_id for p in runpod.active_pods())
            if gone and not still:
                save(termination_verified=True, terminated_at_unix=time.time())
                watchdog.terminate()
            raise RuntimeError(f'Provider rate ${rate}/h violates the frozen budget; emergency shutdown requested')
        assert 0 < rate <= max_rate, f'Unexpected GPU rate {rate}; teardown'
        assert prior_spend+lifetime/3600*(rate+storage)+(0 if plan else reserve) <= gpu_cap
        threading.Thread(target=monitor,daemon=True).start()
        print(f'OWNED POD {pod_id}: ${rate}/h; watchdog {watchdog.pid}; cap {lifetime}s',flush=True)
    try:
        pod=runpod.provision_eval_pod([target],name=plan['run_name'] if plan else f'nikak-{checkpoint}-baseline-20260909',
                                    pubkey_path=keypair[0],identity=keypair[1],on_provisioned=arm)
        save(phase='bootstrapping',host=pod.host)
        bootstrap_timeout = min(3600, max(1, int(state['rented_at_unix']+work_lifetime-time.time()))) if plan else 3600
        if not runpod.wait_bootstrapped(pod.id,timeout_s=bootstrap_timeout):
            raise TimeoutError('Baseline pod bootstrap exceeded one hour')
        save(phase='evaluating',evaluation_started=True)
        if plan:
            remaining = state['rented_at_unix']+work_lifetime-time.time()
            if remaining <= 0:
                raise TimeoutError('Work budget exhausted before eval; preserving logs')
            dispatch_frozen(frozen_plan,config,pod.host,keypair[1],remaining)
        else:
            evaluate(['--name','odcv','--config',str(config),'--target',target,
                      '--server',pod.host,'--ssh-key',keypair[1]])
        save(phase='evaluation_completed', evaluation_driver_completed=True)
    except BaseException as exc:
        save(phase='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
    finally:
        if state['pod_id'] and not state.get('termination_verified'):
            backup_ok = not plan
            if plan:
                backup_ok = recover_eval_logs(state,keypair[1],out,save)
            if backup_ok:
                gone=runpod.terminate(state['pod_id'])
                still=any(p.get('id')==state['pod_id'] for p in runpod.active_pods())
                if gone and not still:
                    save(terminated_at_unix=time.time(),termination_verified=True)
                    if watchdog:
                        watchdog.terminate()
                else:
                    save(termination_verified=False)
                    print(f"URGENT owned pod {state['pod_id']} may still bill; watchdog retained",flush=True)
            else:
                print('URGENT: remote logs unverified; ordinary teardown blocked. Emergency watchdog retained.',flush=True)
        finished.set()
    if plan and state.get('ordinary_teardown_blocked'):
        raise RuntimeError('Evaluation driver ended but remote-log backup failed; emergency watchdog owns teardown')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', choices=CHECKPOINTS)
    mode.add_argument('--plan', type=Path)
    mode.add_argument('--evaluate-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--eval-config', help=argparse.SUPPRESS)
    parser.add_argument('--server', help=argparse.SUPPRESS)
    parser.add_argument('--ssh-key', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.evaluate_plan:
        evaluate_frozen(args.evaluate_plan,args.eval_config,args.server,args.ssh_key)
    else:
        main(args.checkpoint or 'nonmoral',args.plan)
