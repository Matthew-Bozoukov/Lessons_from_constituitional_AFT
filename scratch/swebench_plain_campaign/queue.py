# ABOUTME: CPU-side durable two-wave launcher using the standard paired SWE-bench fleet entrypoint.
# ABOUTME: Never resets a ledger or retries a failed wave; completion requires live provider and immutable Hub verification.
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from dotenv import load_dotenv
import httpx
import requests

SERVICE = 'lasr-swebench-lite.service'
BUSY = {'active', 'activating', 'reloading', 'deactivating'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_only_call(operation, *args, **kwargs):
    """Retry only transient reads: three attempts, at most 30 seconds to admit retries.

    Individual requests retain their clients' bounded timeouts. Invariants, auth
    failures and paid submissions never pass through a retry policy.
    """
    deadline = time.monotonic() + 30
    for attempt in range(3):
        try:
            return operation(*args, **kwargs)
        except (requests.RequestException, httpx.HTTPError) as exc:
            response = getattr(exc, 'response', None)
            status = response.status_code if response is not None else None
            transient = (status == 429 or status is not None and status >= 500
                         or status is None and isinstance(exc, (
                             requests.ConnectionError, requests.Timeout,
                             requests.exceptions.ChunkedEncodingError, httpx.TransportError)))
            delay = 2 ** attempt
            if not transient or attempt == 2 or time.monotonic() + delay >= deadline:
                raise
            time.sleep(delay)


def validate_plan(plan):
    """All experiment settings and immutable target pins come from the manifest."""
    assert plan['schema'] == 1 and len(plan['waves']) == 2
    assert 0 < plan['poll_seconds'] <= 60 and plan['startup_grace_seconds'] > 0
    roots, targets = [], []
    for wave in plan['waves']:
        root = Path(wave['root'])
        assert root.is_absolute() and root.resolve().is_relative_to('/srv/lasr/runs')
        assert root != Path('/srv/lasr/runs') and not any(c.isspace() for c in str(root))
        assert '..' not in root.parts and len(wave['targets']) == 2
        assert wave['budget_usd'] > 0
        assert Path(wave['config']).is_absolute()
        assert re.fullmatch('[0-9a-f]{64}', wave['config_sha256'])
        assert sha(wave['config']) == wave['config_sha256'], 'External fleet config drift'
        for target in wave['targets']:
            assert re.fullmatch('[0-9a-f]{40}', target['revision']), 'Require immutable model revisions'
            assert re.fullmatch(r'[\w.-]+/[\w.-]+', target['repo'])
            targets.append(target['repo'])
        roots.append(root)
    assert len(set(targets)) == 4, 'Each model runs once'
    assert not roots[0].is_relative_to(roots[1]) and not roots[1].is_relative_to(roots[0])


def launch_argv(wave):
    a, b = wave['targets']
    return [sys.executable, '-m', 'src.eval.run_eval', '--name', 'swebench_mini', '--fleet',
            '--config', wave['config'], '--target', a['repo'], b['repo'],
            '--target-revision', a['revision'], '--next-target-revision', b['revision'],
            '--run-root', wave['root'], '--budget-usd', str(wave['budget_usd'])]


class Runtime:
    """Paid launches have exactly one boundary; every other method is read-only."""
    def service(self):
        output = subprocess.check_output(['systemctl', 'show', SERVICE, '--property=LoadState',
            '--property=ActiveState', '--property=Result', '--property=ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        assert state.get('LoadState') == 'loaded', 'Fleet service missing'
        env = Path('/srv/lasr/lite-launch.env')
        state['config'] = next((line.split('=', 1)[1] for line in env.read_text().splitlines()
                                if line.startswith('LITE_CONFIG=')), None) if env.exists() else None
        return state

    def launch(self, wave):
        subprocess.run(launch_argv(wave), check=True)

    def inspect(self, wave):
        root = Path(wave['root'])
        control = root / 'metadata/supervisor.json'
        manifest = root / 'metadata/manifest.json'
        if manifest.exists():
            saved = read(manifest)
            assert saved['budget_usd'] == wave['budget_usd'], 'Existing ledger budget differs'
            assert saved['config']['target'] == wave['targets'][0]['repo']
            assert saved['config']['target_revision'] == wave['targets'][0]['revision']
        if control.exists():
            value = read(control)
            assert value['budget_usd'] == wave['budget_usd'], 'Supervisor budget differs'
            return value
        return {}

    def verify(self, wave):
        from omegaconf import OmegaConf
        from src.eval.capabilities.swebench_mini import fleet, fleet_session
        cfg = OmegaConf.load(Path(wave['root']) / 'launch.yaml')
        load_dotenv(cfg.credentials)
        arms = fleet_session.members(cfg)
        assert len(arms) == 2 and cfg.root == wave['root']
        assert arms[1].fleet_owner_root == cfg.root
        evidence = []
        manifests = []
        for arm, target in zip(arms, wave['targets']):
            root = Path(arm.root)
            manifest = read(root / 'metadata/manifest.json')
            state = read(root / 'metadata/state.json')
            control = read(root / 'metadata/supervisor.json')
            result = read(root / 'results/results.json')
            assert (arm.target, arm.target_revision) == (target['repo'], target['revision'])
            assert manifest['config'] == OmegaConf.to_container(arm)
            assert manifest['budget_usd'] == control['budget_usd'] == wave['budget_usd']
            assert control['status'] == 'complete' and not control.get('cancelled')
            assert not control.get('terminal_reason'), 'Terminal wave needs review even with partial publication'
            assert len(state['tasks']) == 300 and all(t['status'] == 'valid' for t in state['tasks'].values())
            assert result['status'] == 'complete' and result['n_total'] == result['n_valid_rollouts'] == result['n_graded'] == 300
            assert len(result['task_results']) == 300 and all(t['graded'] and t['rollout_status'] == 'valid' for t in result['task_results'].values())
            assert set(result['task_results']) == set(state['tasks'])
            revision = control['verified_hf_revision']
            assert re.fullmatch('[0-9a-f]{40}', revision) and control['verified_files'] > 0
            saved = read(read_only_call(fleet.hf_download, manifest['repo'], 'results/results.json',
                         repo_type='dataset', revision=revision))
            assert saved == result, 'Immutable HF readback differs'
            verified = read_only_call(fleet.verify_final_files, arm, revision)
            assert verified > 0
            manifests.append(manifest)
            evidence.append({'target': target, 'root': arm.root, 'repo': manifest['repo'],
                'revision': revision, 'verified_files': verified, 'valid': 300, 'graded': 300,
                'campaign': manifest['campaign'], 'owned_gpus_remaining': []})
        # Hub readback can be slow: the completion decision needs a fresh provider
        # observation after both arms have verified, not a pre-readback snapshot.
        inventory = read_only_call(fleet.runpod.active_pods)
        assert not any(fleet.owned(p, manifest) for manifest in manifests for p in inventory), 'Campaign-owned GPUs remain'
        return {'verified_at': time.time(), 'arms': evidence}


class Queue:
    def __init__(self, plan, state, save, runtime=None):
        self.plan, self.state, self.save = plan, state, save
        self.runtime = runtime or Runtime()

    def step(self):
        """Persist intent before launch; never infer permission to retry terminal work."""
        if self.state['status'] in ('blocked', 'complete'):
            return self.state['status']
        try:
            for index, wave in enumerate(self.plan['waves']):
                entry = self.state['waves'][index]
                if entry['status'] == 'complete':
                    continue
                service = self.runtime.service()
                control = self.runtime.inspect(wave)
                busy = service['ActiveState'] in BUSY
                expected_config = str(Path(wave['root']) / 'launch.yaml')
                if busy:
                    assert service['config'] == expected_config, 'Another fleet owns the CPU service'
                    assert entry['status'] != 'pending', 'Unrecorded fleet launch; reconcile ownership'
                    entry['status'] = 'submitted'
                    self.save(self.state)
                    return 'waiting'
                assert service['ActiveState'] == 'inactive', 'Fleet service failed or has unknown state'
                assert service['Result'] == 'success' and service['ExecMainStatus'] == '0', 'Fleet service did not exit successfully'
                if control.get('status') == 'complete':
                    assert entry['status'] != 'pending', 'Unrecorded existing campaign'
                    evidence = self.runtime.verify(wave)
                    # Recheck after the slow HF readback; an external launcher must not race the transition.
                    final = self.runtime.service()
                    assert final['ActiveState'] == 'inactive' and final['Result'] == 'success'
                    assert final['ExecMainStatus'] == '0' and final['config'] == expected_config
                    entry.update(status='complete', evidence=evidence)
                    self.save(self.state)
                    continue
                if entry['status'] != 'pending':
                    # A crash immediately before spawning is safe to replay only if the standard
                    # launcher has not written ANY launch/ledger state. All other ambiguity stops.
                    pristine = not (Path(wave['root']) / 'launch.yaml').exists() and not (Path(wave['root']) / 'metadata').exists()
                    if entry['status'] == 'submitting' and pristine and not control:
                        pass
                    elif not control and time.time() - entry['submitted_at'] < self.plan['startup_grace_seconds']:
                        return 'waiting'
                    else:
                        raise RuntimeError('Wave stopped without verified completion; no automatic retry')
                else:
                    assert not control and not (Path(wave['root']) / 'launch.yaml').exists() and not (Path(wave['root']) / 'metadata').exists(), 'Existing campaign needs its original queue state'
                if index:
                    # A durable receipt is retained, but ownership/HF are checked live again
                    # after a queue restart before allowing the next paid wave.
                    self.runtime.verify(self.plan['waves'][index - 1])
                    again = self.runtime.service()
                    assert again['ActiveState'] == 'inactive' and again['Result'] == 'success' and again['ExecMainStatus'] == '0'
                assert sha(wave['config']) == wave['config_sha256'], 'Config changed before launch'
                entry.update(status='submitting', submitted_at=time.time())
                self.save(self.state)
                self.runtime.launch(wave)
                entry['status'] = 'submitted'
                self.save(self.state)
                return 'waiting'
            self.state['status'] = 'complete'
            self.save(self.state)
            return 'complete'
        except Exception as exc:
            self.state.update(status='blocked', error=f'{type(exc).__name__}: {exc}', blocked_at=time.time())
            self.save(self.state)
            return 'blocked'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--state', required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('INVOCATION_ID'), 'Run as a durable CPU-side systemd service'
    from src.eval.capabilities.swebench_mini.fleet_state import atomic, lock
    plan = read(args.manifest)
    validate_plan(plan)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    path = Path(args.state).resolve()
    assert path != Path(args.manifest).resolve()
    with lock('/srv/lasr/.plain-wave-queue.lock', nonblocking=True), lock(str(path) + '.lock', nonblocking=True):
        if path.exists():
            state = read(path)
            assert state['manifest_sha256'] == identity, 'Queue inputs changed; never reset existing ledger/state'
        else:
            state = {'manifest_sha256': identity, 'status': 'running', 'created': time.time(),
                     'waves': [{'status': 'pending'} for _ in plan['waves']]}
            atomic(path, state)
        queue = Queue(plan, state, lambda value: atomic(path, value))
        while True:
            status = queue.step()
            print(json.dumps({'status': status, 'waves': [w['status'] for w in state['waves']],
                              'error': state.get('error')}), flush=True)
            if status in ('blocked', 'complete'):
                raise SystemExit(2 if status == 'blocked' else 0)
            time.sleep(plan['poll_seconds'])


if __name__ == '__main__':
    main()
