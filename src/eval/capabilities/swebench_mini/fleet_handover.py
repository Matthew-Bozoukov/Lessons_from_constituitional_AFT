# ABOUTME: Explicit, short-lived coordinator handover without restarting serving workers.
# ABOUTME: Verify process birth times and campaign ownership; retain every lease and task attempt.
import os
from pathlib import Path
import subprocess
import time

import psutil

from src.eval.capabilities.swebench_mini.fleet_state import atomic, read


def pending(root, manifest, now=None):
    """A bounded maintenance window, never an extension of a pod's expiry."""
    path = Path(root) / 'metadata/coordinator-handover.json'
    if not path.exists():
        return None
    data = read(path)
    now = time.time() if now is None else now
    if (data.get('status') == 'prepared' and data.get('campaign') == manifest['campaign']
            and data['created'] <= now < data['expires'] <= data['created'] + 180):
        return data
    return None


class ExistingWorker:
    """The subset of Popen used by stop_process, with PID-reuse protection."""
    def __init__(self, pid, created):
        self.pid = pid
        self.created = created
        self.process = psutil.Process(pid)
        assert self.process.create_time() == created, 'Worker PID was reused'

    def poll(self):
        try:
            if (self.process.create_time() != self.created or not self.process.is_running()
                    or self.process.status() == psutil.STATUS_ZOMBIE):
                return 0
        except psutil.NoSuchProcess:
            return 0
        return None

    def wait(self, timeout=None):
        until = time.monotonic() + timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if time.monotonic() >= until:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.05)
        return 0


def validate_worker(record, config_path):
    worker = ExistingWorker(record['worker_pid'], record['worker_created'])
    command = worker.process.cmdline()
    assert worker.poll() is None, 'Handover worker is no longer alive'
    assert '-m' in command and command[command.index('-m') + 1] == 'src.eval.capabilities.swebench_mini.fleet_worker'
    for flag, value in (('--config', str(config_path)), ('--replica', str(record['slot'])),
                        ('--server', record['server'])):
        assert flag in command and command[command.index(flag) + 1] == value, 'Worker identity mismatch'
    assert os.getpgid(worker.pid) == worker.pid, 'Worker must own its process group'
    return worker


def claim(cfg, config_path, manifest, inventory):
    data = pending(cfg.root, manifest)
    if not data:
        return None
    if psutil.pid_exists(data['previous_pid']):
        previous = psutil.Process(data['previous_pid'])
        assert previous.create_time() != data['previous_created'] or previous.status() == psutil.STATUS_ZOMBIE, 'Previous coordinator still alive'
    assert data['budget_usd'] == manifest['budget_usd'], 'Budget amendment mismatch'
    assert not cfg.get('following_configs'), 'Handover currently supports one model'
    assert len(data['workers']) <= cfg.replicas
    from src.eval.capabilities.swebench_mini.fleet import owned
    assert {p['id'] for p in inventory if owned(p, manifest)} == {p['id'] for p in data['workers']}, 'Owned inventory changed during handover'
    state = read(Path(cfg.root) / 'metadata/state.json')
    assert not state.get('halt'), 'Cannot hand over a halted campaign'
    for record in data['workers']:
        saved = next(p for p in state['pods'] if p['slot'] == record['slot'])
        assert all(saved[k] == record[k] for k in ('id', 'worker_pid', 'expires'))
        validate_worker(record, config_path)
    data.update(status='adopted', adopted_by=os.getpid(), adopted_at=time.time())
    atomic(Path(cfg.root) / 'metadata/coordinator-handover.json', data)
    return data['workers']


def monitor(cfg, config_path, record, manifest):
    from src.eval.capabilities.swebench_mini import fleet, fleet_session as session
    worker = validate_worker(record, config_path)
    state = fleet.State(cfg.root)
    slot = record['slot']
    guard = fleet.runpod.start_watchdog(record['id'], max(1, int(record['expires']-time.time())),
        Path(cfg.root) / 'metadata' / f'watchdog-adopted-{slot}.log')
    executor = fleet.SshExec(record['server'], port=cfg.port_base + slot, identity=cfg.ssh_key)
    error = None
    try:
        while worker.poll() is None:
            if read(state.path).get('halt') or time.time() >= record['expires'] - cfg.cleanup_reserve_seconds:
                raise TimeoutError('Adopted replica deadline or campaign halt')
            time.sleep(2)
        remaining = [t for t in read(state.path)['tasks'].values() if t['status'] == 'running'
                     and t['attempts'][-1]['worker'].startswith(f'{slot}-')]
        if remaining:
            raise RuntimeError('Adopted worker exited with unfinished tasks')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            fleet.stop_process(worker)
            session.recover_worker(cfg, slot)
        except Exception as exc:
            error = f'Adopted worker cleanup failed: {exc}'
            with state.edit() as data:
                data['halt'] = error
        try:
            log = executor._ssh('cat /workspace/output/serve/vllm.log', timeout=15)
            (Path(cfg.root) / 'metadata' / f'vllm-{slot}.log').write_text(log)
        except Exception:
            pass
        fleet.runpod.teardown(record['id'])
        with state.edit() as data:
            saved = next(p for p in data['pods'] if p['slot'] == slot)
            saved.update(status='terminated', ended=time.time())
            if error:
                saved['error'] = error
        guard.terminate()
