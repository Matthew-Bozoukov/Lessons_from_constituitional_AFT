# ABOUTME: Bounded durable supervision for one authorized Lite campaign, including grading and publication recovery.
# ABOUTME: Never resets spend or completed outcomes; explicit stop and job expiry survive restarts.
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, lock


def explicit_stop(state, control):
    reason = str(state.get('halt') or '')
    return control.get('cancelled', False) or reason.startswith(('signal ', 'user stop'))


def decide(state, control, cfg, now=None):
    """Pure admission decision; restarting this process cannot replenish allowances."""
    now = time.time() if now is None else now
    if explicit_stop(state, control):
        return 'stopped'
    if state.get('tasks') and all(t['status'] == 'valid' for t in state['tasks'].values()):
        return 'finish'
    if now + cfg.get('allocation_min_remaining_seconds', 6480) + cfg.get('cpu_finish_reserve_seconds', 7200) >= control['deadline']:
        return 'inference_window_closed'
    if state.get('halt') and any(x in state['halt'].lower() for x in ('memory', 'disk', 'cleanup')):
        return 'needs_attention'
    if control.get('cycles', 0) >= cfg.get('max_recovery_cycles', 4):
        return 'recovery_exhausted'
    if state.get('tasks') and not any(t['status'] != 'valid' and
            len(t['attempts']) < cfg.max_infrastructure_attempts for t in state['tasks'].values()):
        return 'attempts_exhausted'
    return 'infer'


def publish_until_verified(cfg, control_path):
    from src.eval.capabilities.swebench_mini import fleet
    for _ in range(cfg.get('publication_attempts', 6)):
        control = read(control_path)
        if control.get('cancelled') or time.time() >= control['deadline']:
            break
        try:
            fleet.publish(cfg)
            state = read(Path(cfg.root) / 'metadata/state.json')
            result = read(Path(cfg.root) / 'results/results.json')
            manifest = read(Path(cfg.root) / 'metadata/manifest.json')
            assert not any(fleet.owned(p, manifest) for p in fleet.runpod.active_pods())
            saved = read(fleet.hf_download(manifest['repo'], 'results/results.json',
                         repo_type='dataset', revision=state['hf_commit']))
            assert saved == result, 'HF result round-trip mismatch'
            verified_files = fleet.verify_final_files(cfg, state['hf_commit'])
            control.update(status='complete' if result['status'] == 'complete' else 'incomplete',
                           verified_hf_revision=state['hf_commit'], verified_files=verified_files, finished=time.time())
            atomic(control_path, control)
            # Preserve verification evidence too; its revision points at the fully
            # checked preceding commit and does not require recursive self-hashing.
            fleet.publish(cfg)
            return True
        except Exception as exc:
            control.update(status='publication_retry', last_error=type(exc).__name__ + ': ' + str(exc))
            atomic(control_path, control)
            time.sleep(cfg.get('publication_backoff_seconds', 60))
    return False


def supervise(cfg, config_path, budget):
    from src.eval.capabilities.swebench_mini import fleet
    root = Path(cfg.root)
    control_path = root / 'metadata/supervisor.json'
    with lock(root / '.supervisor.lock', nonblocking=True):
        if not control_path.exists():
            assert budget is not None and 0 < budget <= cfg.recommended_budget_usd
            receipt = read(cfg.receipt)
            deadline = min(time.time() + cfg.get('job_seconds', 21600),
                           datetime.fromisoformat(receipt['stop_at']).timestamp() - 120)
            atomic(control_path, {'status': 'armed', 'budget_usd': budget, 'cycles': 0,
                   'created': time.time(), 'deadline': deadline, 'cancelled': False})
        control = read(control_path)
        assert budget == control['budget_usd'], 'Cannot reset authorized budget'
        def stop(signum, frame):
            control = read(control_path)
            control.update(cancelled=True, status='stopped')
            atomic(control_path, control)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            control = read(control_path)
            state_path = root / 'metadata/state.json'
            state = read(state_path) if state_path.exists() else {}
            decision = decide(state, control, cfg)
            if decision == 'finish':
                manifest = read(root / 'metadata/manifest.json')
                fleet.fence(cfg, manifest)
                fleet.final_accounting(cfg)
                # An existing report with full coverage needs publication only.
                for n in range(cfg.get('grading_attempts', 3)):
                    results = root / 'results/results.json'
                    if results.exists() and read(results)['status'] == 'complete':
                        break
                    try:
                        with lock(root / '.coordinator.lock', nonblocking=True):
                            fleet.grade(cfg)
                    except Exception as exc:
                        control.update(last_error=type(exc).__name__ + ': ' + str(exc))
                        atomic(control_path, control)
                    if n+1 < cfg.get('grading_attempts', 3):
                        time.sleep(cfg.get('recovery_backoff_seconds', 30))
                if not publish_until_verified(cfg, control_path):
                    control = read(control_path)
                    control['status'] = 'publication_pending'
                    atomic(control_path, control)
                return
            if decision != 'infer':
                control.update(status=decision, finished=time.time())
                atomic(control_path, control)
                if state:
                    fleet.fence(cfg, read(root / 'metadata/manifest.json'))
                    if decision == 'stopped':
                        return
                    # Preserve partial official scoring without buying more GPUs.
                    try:
                        fleet.grade(cfg)
                        fleet.final_accounting(cfg)
                        publish_until_verified(cfg, control_path)
                    except Exception as exc:
                        control['last_error'] = type(exc).__name__ + ': ' + str(exc)
                        atomic(control_path, control)
                return
            # A cycle is consumed BEFORE work, including process crashes/reboots.
            control.update(cycles=control['cycles']+1, status='running')
            atomic(control_path, control)
            try:
                fleet.execute(cfg, config_path, 'resume' if state else 'run', budget)
            except (Exception, SystemExit) as exc:
                control = read(control_path)
                control.update(last_error=type(exc).__name__ + ': ' + str(exc))
                atomic(control_path, control)
            if time.time() < control['deadline']:
                time.sleep(cfg.get('recovery_backoff_seconds', 30))


def launch(cfg, config_path, args):
    with lock('/srv/lasr/.lite-launch.lock', nonblocking=True):
        return _launch(cfg, config_path, args)


def _launch(cfg, config_path, args):
    """One explicit request resolves the target, preflights CPU, then arms systemd."""
    from src.eval.capabilities.swebench_mini import fleet
    assert args.target, 'A single HF target is required'
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'lasr-swebench-lite.service'])
    assert active.returncode != 0, 'A model is already running; inspect it rather than replace it'
    slug = args.target.split('/')[-1]
    assert all(c.isalnum() or c in '-_.' for c in slug), 'Unsafe target name'
    root = Path(args.root or '/srv/lasr/runs/' + datetime.now(timezone.utc).strftime('%Y%m%d') + '-' + slug)
    root = root.resolve()
    assert root.is_relative_to('/srv/lasr/runs') and str(root) != '/srv/lasr/runs', 'Campaign root must be below /srv/lasr/runs'
    assert not any(c.isspace() for c in str(root)), 'Campaign path cannot contain whitespace'
    args.root = str(root)
    args.write_config = str(root / 'launch.yaml')
    root.mkdir(parents=True, exist_ok=True)
    state = {}
    if Path(args.write_config).exists():
        saved = OmegaConf.load(args.write_config)
        assert saved.target == args.target
        if args.target_revision:
            assert saved.target_revision == args.target_revision
        state = read(root / 'metadata/state.json') if (root/'metadata/state.json').exists() else {}
        control = read(root/'metadata/supervisor.json') if (root/'metadata/supervisor.json').exists() else {}
        if control.get('status') == 'complete' and control.get('verified_hf_revision'):
            print('Already completed and published:', root)
            return
        assert not explicit_stop(state, control), 'Previously stopped; explicit resume/review required'
        cfg = saved
    else:
        fleet.prepare_target(cfg, args)
    budget = args.budget_usd
    assert budget is not None, 'Pass the authorized --budget-usd backstop'
    assert 0 < budget <= cfg.recommended_budget_usd
    control_path = root/'metadata/supervisor.json'
    if control_path.exists():
        assert read(control_path)['budget_usd'] == budget, 'Resume cannot change the cumulative budget'
    complete_rollouts = bool(state.get('tasks')) and all(t['status'] == 'valid' for t in state['tasks'].values())
    if not complete_rollouts:
        fleet.preflight(cfg)  # Absolutely no rentals before this returns.
    Path('/srv/lasr/lite-launch.env').write_text(
        f'LITE_ACTION=supervise\nLITE_CONFIG={args.write_config}\nLITE_BUDGET_USD={budget}\n')
    Path('/srv/lasr/lite-launch.env').chmod(0o600)
    subprocess.run(['systemctl', 'enable', 'lasr-swebench-lite.service'], check=True)
    subprocess.run(['systemctl', 'start', '--no-block', 'lasr-swebench-lite.service'], check=True)
    print('Submitted:', root, 'cumulative GPU backstop:', budget)


def dispatch(cfg, config_path, args):
    if args.action == 'supervise':
        return supervise(cfg, config_path, args.budget_usd)
    if args.action == 'launch':
        return launch(cfg, config_path, args)
    path = Path(cfg.root) / 'metadata/supervisor.json'
    if path.exists():
        control = read(path)
        control.update(cancelled=True, status='stopped')
        atomic(path, control)
    subprocess.run(['systemctl', 'stop', 'lasr-swebench-lite.service'], check=True)
    subprocess.run(['systemctl', 'disable', 'lasr-swebench-lite.service'], check=True)
