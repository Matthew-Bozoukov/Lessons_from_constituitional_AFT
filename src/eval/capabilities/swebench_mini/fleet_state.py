# ABOUTME: Durable task leases and append-only attempt directories for the Lite coordinator.
# ABOUTME: Linux advisory locks serialize claims; atomic records preserve outcomes across crashes.
from src.eval.capabilities.swebench_mini.fleet_host import deadline_value
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


@contextmanager
def tool_slot(directory, slots, *, wait_seconds=600, min_available_gib=16):
    """Cross-process admission; kernel locks are released on process death.

    Waiting happens before Docker's execution timer. This limits active commands,
    not conversations, and does not pretend to reserve memory for idle containers.
    """
    assert int(slots) > 0
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while time.monotonic() - started < wait_seconds:
        if available_memory_bytes() >= min_available_gib * 2**30:
            for n in range(int(slots)):
                stream = (directory / str((n + os.getpid()) % int(slots))).open('a')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield time.monotonic() - started
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
        time.sleep(.1)
    raise TimeoutError('CPU tool admission wait exceeded; no command was executed')


def available_memory_bytes():
    # The isolated mini-swe-agent environment deliberately contains no psutil.
    # Linux MemAvailable includes reclaimable cache, unlike MemFree.
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Host MemAvailable unavailable; cannot admit a tool safely')


def failed_replicas(data):
    """One lost server may interrupt all four workers; it is one failure domain."""
    return {str(a['worker']).rsplit('-', 1)[0] if a.get('worker') else 'attempt:' + a['id']
            for task in data['tasks'].values() for a in task['attempts']
            if a.get('valid') is False}


def begin_failure_epoch(data):
    # Retain the old task counter for historical accounting, not breaker admission.
    data['breaker_failures_baseline'] = sum(a.get('valid') is False
        for task in data['tasks'].values() for a in task['attempts'])
    data['breaker_failed_replicas_baseline'] = sorted(failed_replicas(data))


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

    def claim(self, worker, allowed, max_attempts, failure_limit, *, latest_start=None):
        with self.edit() as data:
            if data.get('halt') or time.time() >= deadline_value(data['deadline']):
                return None
            # HF is a replica of durable local state, not a liveness dependency.
            # Admit a task only when its complete time budget fits before expiry.
            if latest_start is not None and time.time() >= latest_start:
                return None
            failures = failed_replicas(data) - set(data.get('breaker_failed_replicas_baseline', []))
            if len(failures) >= failure_limit:
                data['halt'] = 'infrastructure failure circuit breaker'
                data['breaker_evidence'] = {'failed_replicas': sorted(failures),
                    'replica_limit': failure_limit, 'time': time.time()}
                return None
            for iid in allowed:
                task = data['tasks'].get(iid)
                if task is None or task['status'] not in ('pending', 'invalid'):
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

    def recover(self, worker_prefix=None):
        """Only after the selected workers and their containers have been fenced."""
        with self.edit() as data:
            for iid, task in data['tasks'].items():
                if task['status'] != 'running':
                    continue
                attempt = task['attempts'][-1]
                if worker_prefix is not None and not str(attempt['worker']).startswith(worker_prefix):
                    continue
                done = self.root / 'rollouts' / iid / attempt['id'] / 'done.json'
                result = read(done) if done.exists() else {'valid': False, 'exit_status': 'InterruptedInfrastructure'}
                attempt.update(result, finished=time.time())
                task['status'] = 'valid' if result['valid'] else 'invalid'


def classify(traj, returncode):
    status = traj.get('info', {}).get('exit_status', '')
    # These are model/scaffold outcomes, never selected for a better second try.
    valid = returncode == 0 and status in {'Submitted', 'LimitsExceeded', 'ContextWindowExceededError'}
    return {'valid': valid, 'exit_status': status or 'MissingTrajectory', 'returncode': returncode}
