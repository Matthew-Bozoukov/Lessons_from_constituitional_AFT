# ABOUTME: One-shot continuation after the already-running Lite service drains normally.
# ABOUTME: No GPU rental here; the existing coordinator retains readiness, ownership and budget gates.
import argparse
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf
from scratch.swebench_lite_state import read, atomic, lock
from scratch.swebench_lite import owned, reserved_cost, sources
from src.infra import runpod


UNIT = 'lasr-swebench-lite.service'


def properties():
    raw = subprocess.check_output(['systemctl', 'show', UNIT, '--property=ActiveState,Result,InvocationID'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)


def eligible(state, manifest, cfg, migration):
    if state.get('halt'):
        raise RuntimeError('Campaign halted; never override a user stop or safety breaker')
    assert manifest['source_hashes'] == sources(), 'Code changed after recovery was prepared'
    assert manifest['config'] == OmegaConf.to_container(cfg), 'Configuration changed'
    allowed = set(migration['unfinished_ids'])
    todo = [i for i, t in state['tasks'].items() if t['status'] in ('pending', 'invalid')
            and len(t['attempts']) < cfg.max_infrastructure_attempts]
    assert set(todo) <= allowed, 'Recovery scope expanded'
    assert reserved_cost(state) < manifest['budget_usd'], 'No remaining cumulative budget'
    return todo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scratch/swebench_lite.yaml')
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    load_dotenv(cfg.credentials)
    root = Path(cfg.root)
    proof = root / 'metadata/timeout-recovery-continuation.json'
    with lock(root / '.timeout-recovery.lock', nonblocking=True):
        assert not proof.exists(), 'This one-shot continuation has already been registered'
        audit = {'registered_at': time.time(), 'original_invocation': args.invocation, 'status': 'waiting'}
        atomic(proof, audit)
        try:
            until = time.time() + 7200
            while True:
                props = properties()
                if props['ActiveState'] not in ('active', 'activating', 'deactivating'):
                    break
                assert props.get('InvocationID') == args.invocation, 'A different invocation already started'
                assert time.time() < until, 'Continuation wait expired; no launch'
                time.sleep(15)
            assert props['Result'] == 'success', 'Prior service did not finish normally; inspect instead of restarting'
            state = read(root / 'metadata/state.json')
            manifest = read(root / 'metadata/manifest.json')
            migration = read(root / 'metadata/request-timeout-migration.json')
            todo = eligible(state, manifest, cfg, migration)
            assert not any(owned(p, manifest) for p in runpod.active_pods()), 'Prior GPU cleanup incomplete'
            audit.update(eligible_ids=todo, previous_cost_ceiling=reserved_cost(state))
            if todo:
                # This unit performs preflight, exact source/config checks, verified
                # HF publication and budget reservation BEFORE allocating a GPU.
                subprocess.run(['systemctl', 'start', '--no-block', UNIT], check=True)
                audit.update(status='submitted_once', submitted_at=time.time())
            else:
                audit['status'] = 'nothing_to_retry'
        except BaseException as exc:
            audit.update(status='refused', reason=str(exc), finished_at=time.time())
            atomic(proof, audit)
            raise
        atomic(proof, audit)


if __name__ == '__main__':
    main()
