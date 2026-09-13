# ABOUTME: Add owned GPU workers at runtime and drain legacy jobs through explicit ownership.
# ABOUTME: Run: uv run scratch/delegated_harm/scale_live.py init|add|launch|monitor|status|drain|pause|resume --root <path>.
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

from src.infra import runpod
from scratch.delegated_harm.episode_queue import EpisodeQueue
from scratch.delegated_harm.recovery import read
from src.eval.misalignment.delegated_harm.source import save, digest, prepare
from scratch.delegated_harm.launch import account, MODELS


def detached(command, log):
    with open(log, 'a', encoding='utf-8') as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'}).pid


def own_driver(controller):
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and ($_.CommandLine -like '*scripts/run_eval.py*' -or $_.CommandLine -like '*scratch/delegated_harm/run_eval.py*') } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    text = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True)
    processes = json.loads(text or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    found = [p for p in processes if p['ParentProcessId'] == controller['pid']
             and f"2026-09-11_recovery_{controller['arm']}" in p['CommandLine']]
    assert len(found) == 1, 'Must identify this exact owned driver, not another agent process'
    return {'pid': found[0]['ProcessId'], 'identity': runpod._process_identity(found[0]['ProcessId']),
            'command': found[0]['CommandLine']}


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'metadata/handoff.json').exists(), 'Handoff already initialized'
    queue = EpisodeQueue(root / 'queue.sqlite')
    assert not queue.snapshot()['jobs']
    handoff = {'created': time.time(), 'arms': {},
               'policy': 'Ownership is frozen without using outcomes. Legacy drivers may compute overlap while draining; only assigned-worker results enter the combined data.'}
    for arm in ['control', 'da']:
        legacy_root = Path(f'output/delegated_harm/2026-09-11_recovery_{arm}').resolve()
        controller = read(legacy_root / 'controller.json')
        assert controller['status'] == 'evaluating'
        driver = own_driver(controller)
        runs = [p for p in (legacy_root / 'runs').iterdir()
                if not p.name.endswith('-checkpoint') and (p / 'metadata/recovery.json').exists()]
        assert len(runs) == 1
        legacy = runs[0]
        cfg = OmegaConf.load(legacy_root / 'config.yaml')
        prepare(cfg)
        missing = set(cfg.recovery.missing_ids)
        cells = [c for c in read(legacy / 'metadata/schedule.json') if c['id'] in missing]
        initial = Path(cfg.recovery.source_run)
        original = {c['id']: read(initial / 'results/episodes' / f"{c['id']}.json") for c in cells}
        authors = {f.stem: read(f) for f in (legacy / 'metadata/authors').glob('*.json')}
        assert len(authors) == 36 and all(a['accepted'] for a in authors.values())
        owners = {}
        # Initial trace is saved before any model request. Existing first-run traces
        # have earlier timestamps; only attempts started by this recovery are retained.
        cutoff = time.time()
        for cell in cells:
            path = legacy / 'rollouts' / f"{cell['id']}.json"
            if path.exists():
                trace = read(path)
                if controller['billing_started'] <= trace['started'] <= cutoff:
                    owners[cell['id']] = 'legacy-' + arm
        queue.register('legacy-' + arm, arm, {'controller': str(legacy_root / 'controller.json'),
                                            'out_dir': str(legacy), 'driver': driver})
        queue.mode('legacy-' + arm, 'drain')
        queue.enqueue(arm, cells, owners)
        inputs = {'target': read(legacy / 'metadata/protocol.json')['target'],
                  'authors': authors, 'original_records': original, 'cutover': cutoff}
        save(root / f'metadata/inputs-{arm}.json', inputs)
        OmegaConf.save(cfg, root / f'metadata/config-{arm}.yaml')
        handoff['arms'][arm] = {'legacy_root': str(legacy_root), 'legacy_run': str(legacy),
            'driver': driver, 'cutover': cutoff, 'legacy_owned': sorted(owners),
            'launcher': {'pid': controller['pid'], 'identity': runpod._process_identity(controller['pid'])},
            'pod_id': controller['pod_id'],
            'transferred': sorted(missing - owners.keys()), 'initial_completed': len(read(legacy / 'metadata/recovery.json')['preserved_completed_hashes'])}
        print(arm, 'retained with existing GPU', len(owners), 'transferred', len(missing)-len(owners), flush=True)
    save(root / 'metadata/handoff.json', handoff)
    save(root / 'metadata/queue_snapshot.json', queue.snapshot())


def add_workers(root, arm, count):
    assert (root / 'metadata/handoff.json').exists()
    pending = sum(j['arm'] == arm and j['state'] == 'pending'
                  for j in EpisodeQueue(root / 'queue.sqlite').snapshot()['jobs'])
    if not pending:
        print(f'{arm}: no unclaimed episodes; no GPU rented', flush=True)
        return
    for _ in range(count):
        index = 1
        while (root / 'workers' / f'{arm}-h200-{index}').exists():
            index += 1
        worker = f'{arm}-h200-{index}'
        folder = root / 'workers' / worker
        folder.mkdir(parents=True, exist_ok=False)
        cfg = OmegaConf.load(root / f'metadata/config-{arm}.yaml')
        cfg.scaling = {'queue': str((root / 'queue.sqlite').resolve()), 'worker': worker, 'arm': arm,
                       'inputs': str((root / f'metadata/inputs-{arm}.json').resolve())}
        handoff = read(root / 'metadata/handoff.json')
        if handoff.get('protocol_kind') == 'fresh':
            cfg.scaling.update({'kind': 'fresh', 'root': str(root),
                                'target': handoff['target']['hf_path'], 'author_leader': index == 1})
        cfg.output_root = str((folder / 'runs').resolve())
        OmegaConf.save(cfg, folder / 'config.yaml')
        port = int(cfg.get('orchestration', {}).get('port_base', 9300 if arm == 'control' else 9400)) + index
        save(folder / 'allocation.json', {'arm': arm, 'worker': worker, 'port': port})
        pid = detached([sys.executable, '-u', __file__, 'launch', '--root', str(root), '--worker', worker], folder / 'launch.log')
        print('Starting', worker, 'launcher', pid, flush=True)


def launch(root, worker):
    folder = root / 'workers' / worker
    allocation = read(folder / 'allocation.json')
    arm = allocation['arm']
    assert not (folder / 'controller.json').exists()
    cfg = OmegaConf.load(folder / 'config.yaml')
    prepare(cfg)
    target = cfg.scaling.get('target') or MODELS[arm]
    gpu = 'NVIDIA H200'
    quote = next(float(g['securePrice']) for g in account()['gpuTypes'] if g['id'] == gpu)
    hours = 2.5
    assert quote * hours <= 12.5, 'Unexpected H200 price'
    pair = runpod.default_keypair()
    assert pair
    state = {**allocation, 'pid': os.getpid(), 'status': 'provisioning', 'gpu': gpu,
             'max_hours': hours, 'quoted_gpu_hourly_usd': quote}
    save(folder / 'controller.json', state)
    guard = None
    def owned(pod_id):
        nonlocal guard
        state.update(pod_id=pod_id, billing_started=time.time())
        save(folder / 'controller.json', state)
        guard = runpod.start_watchdog(pod_id, int(hours*3600), folder / 'watchdog.log')
        rate = float(runpod.call('GET', f'/pods/{pod_id}').get('costPerHr') or 0)
        assert 0 < rate * hours <= 12.5
        state['actual_pod_hourly_usd'] = rate
        save(folder / 'controller.json', state)
    try:
        name = f'subagents-scale-{worker}-{int(time.time())}'
        for attempt in range(2):
            try:
                runpod.up(name=name, eval=target, gpu=gpu, max_hours=hours, on_provisioned=owned)
                break
            except Exception:
                if state.get('pod_id') or attempt:
                    raise
                matches = [p for p in runpod.active_pods() if p.get('name') == name]
                if matches:
                    # A lost create response must never result in a duplicate rental.
                    assert len(matches) == 1
                    owned(matches[0]['id'])
                    raise RuntimeError('Ambiguous creation found an owned pod; terminating rather than duplicating it')
                state['provision_retry'] = True
                save(folder / 'controller.json', state)
                time.sleep(5)
        if not runpod.wait_bootstrapped(state['pod_id'], timeout_s=2400):
            raise RuntimeError('H200 bootstrap failed')
        pod = runpod.call('GET', f"/pods/{state['pod_id']}")
        host = f"root@{pod['publicIp']}:{pod['portMappings']['22']}"
        state.update(server=host, status='evaluating')
        save(folder / 'controller.json', state)
        command = [sys.executable, 'scratch/delegated_harm/run_eval.py', '--name', 'delegated_harm', '--target', target,
                   '--config', str(folder / 'config.yaml'), '--server', host, '--ssh-key', pair[1],
                   '--port', str(allocation['port']), '--terminate-pod', '--no-push']
        result = subprocess.run(command, check=False)
        state['driver_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Worker exited {result.returncode}')
        state['status'] = 'finished'
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:700])
        raise
    finally:
        try:
            if state.get('pod_id'):
                runpod.teardown(state['pod_id'])
                state.update(terminated=True, finished=time.time())
                if guard:
                    guard.terminate()
        except Exception as exc:
            state.update(status='cleanup_failed', cleanup_error=str(exc)[:700])
            raise
        finally:
            save(folder / 'controller.json', state)


def drain_legacy(root, queue, handoff):
    snapshot = queue.snapshot()
    for arm, info in handoff['arms'].items():
        worker = 'legacy-' + arm
        jobs = [j for j in snapshot['jobs'] if j['owner'] == worker]
        legacy = Path(info['legacy_run'])
        for job in jobs:
            if job['state'] == 'done':
                continue
            path = legacy / 'results/episodes' / f"{job['id']}.json"
            rec = read(path)
            if rec.get('recovery') and rec['status'] != 'running':
                queue.finish(worker, arm, job['id'], path, digest(rec))
        fresh = [j for j in queue.snapshot()['jobs'] if j['owner'] == worker]
        if all(j['state'] == 'done' for j in fresh):
            marker = root / f'metadata/{worker}-drained.json'
            if marker.exists():
                continue
            # Retain all legacy overlap for audit, but ownership—not completion order—
            # determines which observations are accepted by the merger.
            launcher = info['launcher']
            if runpod._process_identity(launcher['pid']) == launcher['identity']:
                subprocess.run(['taskkill', '/PID', str(launcher['pid']), '/T', '/F'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            runpod.teardown(info['pod_id'])
            legacy_controller = Path(info['legacy_root']) / 'controller.json'
            controller = read(legacy_controller)
            controller.update(status='superseded_by_horizontal_scaling', terminated=True,
                              finished=time.time(), horizontal_scaling_root=str(root))
            save(legacy_controller, controller)
            save(marker, {'time': time.time(), 'kept': [j['id'] for j in fresh],
                          'reason': 'All pre-handoff episodes finished; stop legacy queue and cancel unassigned overlap.'})
            queue.mode(worker, 'stopped')
            print('DRAINED', worker, 'kept', len(fresh), flush=True)
        else:
            queue.heartbeat(worker)


def merge_and_score(root, arm):
    handoff = read(root / 'metadata/handoff.json')['arms'][arm]
    legacy = Path(handoff['legacy_run'])
    destination_root = root / 'combined' / arm
    destination = destination_root / 'runs' / legacy.name
    assert not destination.exists(), 'Refuse to overwrite a prior merged run'
    shutil.copytree(legacy, destination)
    cfg = OmegaConf.load(root / f'metadata/config-{arm}.yaml')
    source = Path(cfg.recovery.source_run)
    jobs = [j for j in EpisodeQueue(root / 'queue.sqlite').snapshot()['jobs'] if j['arm'] == arm]
    assert len(jobs) == len(cfg.recovery.missing_ids) and all(j['state'] == 'done' for j in jobs)
    for job in jobs:
        result = Path(job['result_path'])
        rec = read(result)
        assert digest(rec) == job['result_hash']
        if not job['owner'].startswith('legacy-'):
            overlap = destination / 'metadata/legacy_unassigned_attempts'
            for ext in ['json', 'md']:
                candidate = legacy / 'rollouts' / f"{job['id']}.{ext}"
                if candidate.exists():
                    overlap.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, overlap / candidate.name)
        trace_root = result.parents[2]
        for ext in ['json', 'md']:
            trace = trace_root / 'rollouts' / f"{job['id']}.{ext}"
            if trace.exists():
                shutil.copy2(trace, destination / 'rollouts' / trace.name)
        save(destination / 'results/episodes' / result.name, rec)
        save(destination / 'results/rejudged_episodes' / result.name, rec)
    # Source observations are restored from the original immutable local bundle,
    # preventing any legacy overlap from becoming an accepted recovery observation.
    for path in (source / 'results/episodes').glob('*.json'):
        rec = read(path)
        if rec.get('metrics'):
            save(destination / 'results/episodes' / path.name, rec)
            save(destination / 'results/rejudged_episodes' / path.name, rec)
    snapshot = EpisodeQueue(root / 'queue.sqlite').snapshot()
    save(destination / 'metadata/horizontal_scaling.json', {
        'handoff': read(root / 'metadata/handoff.json'), 'queue': snapshot,
        'worker_controllers': [read(p) for p in (root / 'workers').glob('*/controller.json')]})
    recovery = read(destination / 'metadata/recovery.json')
    recovery['horizontal_scaling'] = {'ownership_manifest': 'metadata/horizontal_scaling.json',
                                     'legacy_attempts_preserved': handoff['legacy_owned']}
    save(destination / 'metadata/recovery.json', recovery)
    save(destination_root / 'controller.json', {'arm': arm, 'terminated': True, 'status': 'scoring',
                                               'run_dir': str(destination), 'horizontal_scaling': True})
    for model, json_mode in [('anthropic/claude-sonnet-5', False), ('anthropic/claude-sonnet-4.5', False),
                             ('google/gemini-3-flash-preview', True)]:
        command = [sys.executable, 'scratch/delegated_harm/rescore.py', '--run-dir', str(destination),
                   '--ignore-spending-cap', '--workers', '12', '--judge-max-tokens', '16384', '--judge-model', model]
        if json_mode:
            command.append('--json-mode')
        subprocess.run(command, check=True)
        summary = read(destination / 'results/results.json')
        if summary['unjudged_completed_episodes'] == 0:
            break
    save(destination_root / 'controller.json', {'arm': arm, 'terminated': True, 'status': 'finished',
        'run_dir': str(destination), 'horizontal_scaling': True,
        'unjudged_completed': summary['unjudged_completed_episodes'],
        'completed_episodes': summary['completed_episodes']})


def reconcile_finished_workers(root, snapshot):
    """A lost cleanup response must not indefinitely block already-saved results."""
    candidates = []
    for path in (root / 'workers').glob('*/controller.json'):
        state = read(path)
        if state.get('terminated') or not state.get('pod_id'):
            continue
        worker = next((w for w in snapshot['workers'] if w['id'] == state['worker']), None)
        if worker is None or worker['mode'] != 'stopped':
            continue
        jobs = [j for j in snapshot['jobs'] if j['owner'] == state['worker']]
        if any(j['state'] != 'done' for j in jobs) or runpod._process_identity(state['pid']):
            continue
        candidates.append((path, state, jobs))
    if not candidates:
        return
    # Read-only inventory: no process or pod is stopped by this reconciliation.
    active = {p['id'] for p in runpod.active_pods()}
    for path, state, jobs in candidates:
        if state['pod_id'] in active:
            continue
        for job in jobs:
            assert digest(read(job['result_path'])) == job['result_hash']
        save(root / 'metadata' / f"{state['worker']}-cleanup-reconciliation.json",
             {'checked': time.time(), 'previous_controller': state,
              'pod_absent': True, 'launcher_absent': True, 'verified_results': len(jobs)})
        state.update(status='finished_with_cleanup_error', terminated=True,
                     finished=time.time(), status_reconciled=True)
        save(path, state)
        print('RECONCILED', state['worker'], 'saved results and absent pod', flush=True)


def monitor(root):
    queue = EpisodeQueue(root / 'queue.sqlite')
    handoff = read(root / 'metadata/handoff.json')
    deadline = time.time() + 4 * 3600
    while time.time() < deadline:
        drain_legacy(root, queue, handoff)
        snap = queue.snapshot()
        try:
            reconcile_finished_workers(root, snap)
        except Exception as exc:
            save(root / 'metadata/reconciliation_error.json',
                 {'time': time.time(), 'error': str(exc)[:700]})
            print('Worker cleanup reconciliation failed; will check again:', type(exc).__name__, flush=True)
        save(root / 'metadata/queue_snapshot.json', snap)
        counts = Counter((j['arm'], j['state']) for j in snap['jobs'])
        print('QUEUE', dict(counts), flush=True)
        failures = [read(p) for p in (root / 'workers').glob('*/controller.json') if read(p)['status'] == 'failed']
        if failures:
            save(root / 'metadata/worker_failures.json', failures)
        controllers = [read(p) for p in (root / 'workers').glob('*/controller.json')]
        legacy_states = [read(Path(info['legacy_root']) / 'controller.json') for info in handoff['arms'].values()]
        if all(j['state'] == 'done' for j in snap['jobs']) and controllers and all(
                c.get('terminated') for c in controllers + legacy_states):
            break
        time.sleep(15)
    else:
        raise TimeoutError('Horizontal recovery exceeded coordinator deadline; inspect claims without stealing active work')
    if handoff.get('protocol_kind') == 'fresh':
        from scratch.delegated_harm.fresh import merge_and_score as finish_fresh
        finish_fresh(root)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda arm: merge_and_score(root, arm), ['control', 'da']))
        subprocess.run([sys.executable, 'scratch/delegated_harm/finish_recovery.py',
                        '--control-root', str(root / 'combined/control'), '--da-root', str(root / 'combined/da')], check=True)
    save(root / 'metadata/finished.json', {'finished': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['init', 'add', 'launch', 'monitor', 'status', 'drain', 'pause', 'resume'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm')
    parser.add_argument('--worker')
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'init':
        initialize(root)
    elif args.mode == 'add':
        assert args.arm and args.count > 0
        assert args.arm.replace('-', '').isalnum(), 'Arm must be a simple label'
        add_workers(root, args.arm, args.count)
    elif args.mode == 'launch':
        launch(root, args.worker)
    elif args.mode == 'monitor':
        monitor(root)
    else:
        queue = EpisodeQueue(root / 'queue.sqlite')
        if args.mode == 'drain':
            queue.mode(args.worker, 'drain')
        elif args.mode in ('pause', 'resume'):
            queue.pause(args.mode == 'pause')
        snap = queue.snapshot()
        print(dict(Counter((j['arm'], j['state']) for j in snap['jobs'])))
        print(json.dumps([{'worker': w['id'], 'arm': w['arm'], 'mode': w['mode'],
                           'heartbeat_age_seconds': round(time.time()-w['heartbeat'], 1),
                           'claims': sum(j['owner'] == w['id'] and j['state'] == 'claimed' for j in snap['jobs']),
                           'finished': sum(j['owner'] == w['id'] and j['state'] == 'done' for j in snap['jobs'])}
                          for w in snap['workers']], indent=2))


if __name__ == '__main__':
    main()
