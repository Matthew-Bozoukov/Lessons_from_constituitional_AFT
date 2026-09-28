# ABOUTME: Retries failed evaluation provisioning without repeating training or rollouts.
# ABOUTME: Shares campaign reservations and API ceiling, and cleans up only its owned pods.
import ctypes
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scratch.da_sep25_campaign.run import CFG, OUT, api_usage, read, runpod, save, spawn


def main(only=None, preflight_fix=False):
    receipt = OUT / ('recovery.json' if only is None else f'recovery-{only}.json')
    assert not receipt.exists(), 'Recovery supervisor already registered'
    state = {'pid': os.getpid(), 'started_epoch': time.time(), 'jobs': {}}
    save(receipt, state)
    processes = {}
    baseline = read(OUT / 'controller.json')['api_usage_start']
    checked = time.time()
    try:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        while True:
            try:
                delta = max(0, api_usage() - baseline)
                checked = time.time()
                state.update(api_usage_delta_upper_bound=delta, api_checked_epoch=checked)
            except Exception:
                assert time.time() - checked < 600, 'API accounting unavailable for 10 minutes'
            assert state.get('api_usage_delta_upper_bound', 0) < float(CFG.api_stop_delta_usd), 'API budget guard reached'
            for pct in (5, 25):
                for kind in ('odcv', 'mask'):
                    if only and f'{kind}{pct}' != only:
                        continue
                    paths = list(OUT.glob(f'{kind}{pct}-attempt*/status.json'))
                    if not paths:
                        continue
                    states = [read(p) for p in paths]
                    latest = max(states, key=lambda s: int(s['key'].rsplit('attempt', 1)[1]))
                    attempt = int(latest['key'].rsplit('attempt', 1)[1])
                    # Only failures before the local evaluation driver started qualify.
                    if latest['phase'] != 'failed' or attempt >= 3:
                        continue
                    if latest.get('eval_pid'):
                        log = OUT / latest['key'] / 'eval.log'
                        safe_preflight = (preflight_fix and log.exists()
                                          and "unknown arguments for this eval: ['concurrency=16', 'prune_images=false']" in log.read_text(encoding='utf-8')
                                          and not list((OUT / latest['key']).rglob('messages_record.txt')))
                        if not safe_preflight:
                            continue
                    if latest.get('owned_pod') and not latest.get('terminated'):
                        continue
                    next_key = f'{kind}{pct}-attempt{attempt + 1}'
                    if next_key in processes:
                        continue
                    active = runpod.call('GET', '/pods')
                    assert not any(p.get('name') == f'nika-da-sep25-{pct}-{kind}' for p in active), 'Ambiguous previous provisioning still active'
                    p = spawn([sys.executable, str(ROOT / 'scratch/da_sep25_campaign/run.py'), kind,
                               str(pct), '--attempt', str(attempt + 1)], OUT / (next_key + '.log'))
                    processes[next_key] = p
                    state['jobs'][next_key] = p.pid
                    save(receipt, state)
            state.update(updated_epoch=time.time(), exit_codes={k: p.poll() for k, p in processes.items()})
            save(receipt, state)
            original = read(OUT / 'controller.json')
            if original.get('phase') in ('finished', 'needs_recovery', 'stopped') and all(p.poll() is not None for p in processes.values()):
                state['phase'] = 'finished'
                save(receipt, state)
                break
            time.sleep(30)
    except BaseException as exc:
        state.update(phase='stopped', error=f'{type(exc).__name__}: {exc}')
        save(receipt, state)
        for key, p in processes.items():
            if p.poll() is None:
                p.terminate()
            path = OUT / key / 'status.json'
            if path.exists():
                owned = read(path).get('owned_pod')
                if owned:
                    runpod.teardown(owned)
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['odcv5', 'odcv25'])
    parser.add_argument('--preflight-fix', action='store_true')
    args = parser.parse_args()
    main(args.only, args.preflight_fix)
