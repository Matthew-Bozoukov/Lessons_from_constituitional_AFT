# ABOUTME: Durable task leases and append-only attempt directories for the Lite coordinator.
# ABOUTME: Linux advisory locks serialize claims; atomic records preserve outcomes across crashes.
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def lock(path, *, nonblocking=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class State:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'metadata/state.json'

    @contextmanager
    def edit(self):
        with lock(self.root / '.state.lock'):
            data = read(self.path)
            yield data
            atomic(self.path, data)

    def claim(self, worker, allowed, max_attempts, failure_limit, stale_seconds):
        with self.edit() as data:
            if data.get('halt') or time.time() >= data['deadline']:
                return None
            if time.time() - data.get('last_upload', 0) > stale_seconds:
                raise RuntimeError('HF backup stale; stopping inference')
            failures = sum(a.get('valid') is False for t in data['tasks'].values() for a in t['attempts'])
            if failures - data.get('breaker_failures_baseline', 0) >= failure_limit:
                data['halt'] = 'infrastructure failure circuit breaker'
                return None
            for iid, task in data['tasks'].items():
                if iid not in allowed or task['status'] not in ('pending', 'invalid'):
                    continue
                if len(task['attempts']) >= max_attempts:
                    continue
                attempt = {'id': uuid.uuid4().hex, 'worker': worker, 'started': time.time()}
                task['attempts'].append(attempt)
                task['status'] = 'running'
                return iid, attempt['id']
        return None

    def finish(self, iid, aid, result):
        # Commit the result before the state pointer: recovery promotes an orphaned
        # done.json without rerolling a task whose expensive outcome already exists.
        atomic(self.root / 'rollouts' / iid / aid / 'done.json', result)
        with self.edit() as data:
            task = data['tasks'][iid]
            assert task['attempts'][-1]['id'] == aid and task['status'] == 'running'
            task['attempts'][-1].update(result, finished=time.time())
            task['status'] = 'valid' if result['valid'] else 'invalid'

    def recover(self):
        """Only after all prior workers/owned pods have been fenced by the caller."""
        with self.edit() as data:
            for iid, task in data['tasks'].items():
                if task['status'] != 'running':
                    continue
                attempt = task['attempts'][-1]
                done = self.root / 'rollouts' / iid / attempt['id'] / 'done.json'
                result = read(done) if done.exists() else {'valid': False, 'exit_status': 'InterruptedInfrastructure'}
                attempt.update(result, finished=time.time())
                task['status'] = 'valid' if result['valid'] else 'invalid'


def classify(traj, returncode):
    status = traj.get('info', {}).get('exit_status', '')
    # These are model/scaffold outcomes, never selected for a better second try.
    valid = returncode == 0 and status in {'Submitted', 'LimitsExceeded', 'ContextWindowExceededError'}
    return {'valid': valid, 'exit_status': status or 'MissingTrajectory', 'returncode': returncode}
