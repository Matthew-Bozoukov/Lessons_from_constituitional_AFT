# ABOUTME: Two independent Lite result sets share one budget, CPU gate and reusable GPU fleet.
# ABOUTME: Pods drain one adapter before switching; recovery preserves each arm's durable outcomes.
from contextlib import ExitStack
from pathlib import Path
import math
import os
import signal
import time

from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini.fleet_host import deadline_value, receipt_deadline
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, lock, read


def members(cfg):
    return [cfg] + [OmegaConf.load(p) for p in cfg.get('following_configs', [])]


def paths(cfg, config_path):
    return [str(config_path)] + list(cfg.get('following_configs', []))


def budget_limit(cfg):
    return cfg.recommended_budget_usd * len(members(cfg))


def view(cfg, owner=None):
    """Combined admission view only; per-arm predictions and scores never mix."""
    owner = read(State(cfg.root).path) if owner is None else owner
    if owner.get('_session_view'):
        return owner
    if not cfg.get('following_configs'):
        return owner
    result = dict(owner, tasks={}, _session_view=True)
    for index, arm in enumerate(members(cfg)):
        path = State(arm.root).path
        data = owner if index == 0 else (read(path) if path.exists() else {})
        result['tasks'].update({f'{index}:{iid}': t for iid, t in data.get('tasks', {}).items()})
        if not data.get('tasks'):
            result['tasks'][f'{index}:uninitialized'] = {'status': 'pending', 'attempts': []}
        if data.get('halt'):
            result['halt'] = data['halt']
    return result


def owner_manifest(cfg):
    return read(Path(cfg.get('fleet_owner_root') or cfg.root) / 'metadata/manifest.json')


def fence_all(cfg):
    from src.eval.capabilities.swebench_mini import fleet
    for arm in members(cfg):
        path = Path(arm.root) / 'metadata/manifest.json'
        if path.exists():
            fleet.fence(arm, read(path))


def recover_worker(cfg, slot):
    import subprocess
    for arm in members(cfg):
        state = State(arm.root)
        for task in read(state.path)['tasks'].values():
            if task['status'] == 'running' and task['attempts'][-1]['worker'].startswith(f'{slot}-'):
                aid = task['attempts'][-1]['id']
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=lasr_attempt=' + aid], text=True).split()
                if ids:
                    subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=90)
        state.recover(worker_prefix=f'{slot}-')


def checkpoints(cfg, config_path, *, required=False):
    from src.eval.capabilities.swebench_mini import fleet
    for arm, path in zip(members(cfg), paths(cfg, config_path)):
        fleet.checkpoint(arm, path, required=required)


def accounting(cfg):
    from src.eval.capabilities.swebench_mini import fleet
    report = fleet.final_accounting(cfg)
    if cfg.get('following_configs'):
        report['scope'] = 'Shared fleet total across both arms; do not sum these copied reports.'
        report['targets'] = [c.target for c in members(cfg)]
        for arm in members(cfg):
            atomic(Path(arm.root) / 'metadata/final-accounting.json', report)
    return report


def execute(cfg, config_path, action, budget):
    """Shared inference only. Supervisor grades both arms after GPU teardown."""
    from src.eval.capabilities.swebench_mini import fleet
    assert os.environ.get('INVOCATION_ID'), 'Use the supervised systemd launch'
    arms = members(cfg)
    assert len(arms) == 2 and not any(c.calibrate for c in arms)
    with ExitStack() as stack:
        for arm in arms:
            stack.enter_context(lock(Path(arm.root) / '.coordinator.lock', nonblocking=True))
        plans = [fleet.preflight(c) for c in arms]
        assert 0 < budget <= budget_limit(cfg)
        for arm, path in zip(arms, paths(cfg, config_path)):
            manifest_path = Path(arm.root) / 'metadata/manifest.json'
            if not manifest_path.exists():
                assert float(plans[0]['balance']['clientBalance']) >= budget
                fleet.initialize(arm, path, budget)
            else:
                manifest = read(manifest_path)
                assert manifest['config'] == OmegaConf.to_container(arm) and manifest['source_hashes'] == fleet.sources(), 'Resume protocol/code drift'
                assert manifest['budget_usd'] == budget, 'Cannot reset the session ledger'
        manifest = owner_manifest(cfg)
        fence_all(cfg)
        fleet.reconcile_rejections(cfg, manifest)
        control_path = Path(cfg.root) / 'metadata/supervisor.json'
        control = read(control_path)
        deadline = control['deadline']
        for index, arm in enumerate(arms):
            with State(arm.root).edit() as data:
                data.update(halt=None, phase='fleet', deadline=deadline - cfg.cpu_finish_reserve_seconds if deadline is not None else None)
                data['breaker_failures_baseline'] = sum(a.get('valid') is False for t in data['tasks'].values() for a in t['attempts'])
            if index:
                child_control = Path(arm.root) / 'metadata/supervisor.json'
                if not child_control.exists():
                    atomic(child_control, dict(control, status='shared_fleet', owner_root=cfg.root))
        def halt(signum, frame):
            for arm in arms:
                with State(arm.root).edit() as data:
                    data['halt'] = f'signal {signum}'
        signal.signal(signal.SIGTERM, halt)
        signal.signal(signal.SIGINT, halt)
        checkpoints(cfg, config_path, required=True)
        try:
            data = view(cfg)
            todo = [iid for iid, t in data['tasks'].items() if t['status'] != 'valid' and len(t['attempts']) < cfg.max_infrastructure_attempts]
            if todo:
                count = min(cfg.replicas, math.ceil(len(todo) / cfg.workers_per_replica))
                atomic(Path(cfg.root) / 'metadata/fleet_plan.json', {
                    'replicas': count, 'workers_per_replica': cfg.workers_per_replica,
                    'targets': [c.target for c in arms], 'budget_usd': budget,
                    'scheduling': 'Alternating initial arm; drain four workers then switch to other arm; longest-first within each arm'})
                fleet.phase(cfg, config_path, todo, count, cfg.rental_seconds, manifest)
        finally:
            fence_all(cfg)
            checkpoints(cfg, config_path)


def finish(cfg, control_path, *, terminal_reason=None):
    """Release the shared fleet, then grade and verify each result independently."""
    from src.eval.capabilities.swebench_mini import fleet, fleet_supervisor as supervisor
    fence_all(cfg)
    accounting(cfg)
    complete = True
    # Owner last: only mark the overall session complete after both results verify.
    for arm in reversed(members(cfg)):
        child_control = Path(arm.root) / 'metadata/supervisor.json'
        if not child_control.exists():
            atomic(child_control, read(control_path))
        for _ in range(arm.get('grading_attempts', 3)):
            result = Path(arm.root) / 'results/results.json'
            if result.exists() and read(result)['status'] == 'complete':
                break
            try:
                with lock(Path(arm.root) / '.coordinator.lock', nonblocking=True):
                    fleet.grade(arm)
            except Exception as exc:
                data = read(child_control)
                data['last_error'] = type(exc).__name__ + ': ' + str(exc)
                atomic(child_control, data)
                time.sleep(arm.get('recovery_backoff_seconds', 30))
        verified = supervisor.publish_until_verified(arm, child_control)
        complete &= verified and read(child_control).get('status') == 'complete'
    data = read(control_path)
    data.update(status='complete' if complete else 'incomplete', finished=time.time())
    if not complete or terminal_reason:
        data['terminal_reason'] = terminal_reason or 'grading_or_publication_incomplete'
    atomic(control_path, data)
    fleet.publish(cfg)
    return complete
