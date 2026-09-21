# ABOUTME: One-shot safe concurrency handoff: stop only with zero in-flight API reservations.
# ABOUTME: Run: uv run --no-sync python -X utf8 -m scratch.dataset_refresh.resume_native_concurrency --config scratch/dataset_refresh/native_lowstakes_full.yaml --root <run-dir> --pid <owned-pid> --log-dir <campaign-dir>
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from filelock import FileLock
from omegaconf import OmegaConf
from scratch.dataset_refresh.run import write_json


def identity(pid):
    command = f'Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}" | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress'
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--log-dir', required=True)
    args = parser.parse_args()
    launch = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    budget = Path(launch['campaign_budget_root'])
    root, logs = Path(args.root), Path(args.log_dir)
    initial = identity(args.pid)
    if not initial or 'scratch.dataset_refresh.run_native_smoke' not in initial['CommandLine'] or args.config not in initial['CommandLine']:
        raise RuntimeError('PID is not the owned full-generation launcher')
    lock = FileLock(str(budget/'spend.lock'), timeout=5)
    start = time.monotonic()
    write_json(logs/'concurrency_handoff.json', dict(status='waiting_for_zero_inflight', old_pid=args.pid, workers=16))
    while time.monotonic() - start < 3600:
        with lock:
            entries = json.loads((budget/'spend.json').read_text())
            if any(e['status'] not in {'reserved', 'settled'} for e in entries):
                raise RuntimeError('Non-settled failure requires investigation; no automatic handoff')
            if not any(e['status'] == 'reserved' for e in entries):
                if identity(args.pid) != initial:
                    raise RuntimeError('Owner process identity changed; refuse termination')
                # The ledger lock blocks every new paid dispatch throughout termination.
                subprocess.run(['taskkill', '/PID', str(args.pid), '/T', '/F'], check=True, capture_output=True)
                count, spent = len(entries), sum(e['charged_or_reserved_usd'] for e in entries)
                break
        time.sleep(0.1)
    else:
        raise TimeoutError('No idle checkpoint within one hour; original run remains untouched')
    command = [sys.executable, '-u', '-X', 'utf8', '-m', 'scratch.dataset_refresh.run_native_smoke',
               '--config', args.config, '--resume', args.root, '--workers', '16',
               '--resume-reason', 'Operational concurrency correction from 4 to native DA 16; stopped at zero in-flight requests under ledger lock; same prompts, candidates, selection and $120 ledger']
    with (logs/'stdout_workers16.log').open('wb') as out, (logs/'stderr_workers16.log').open('wb') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    (logs/'pid.txt').write_text(str(process.pid))
    write_json(logs/'concurrency_handoff.json', dict(status='resumed', old_pid=args.pid, new_pid=process.pid,
        workers=16, settled_calls_at_handoff=count, settled_usd_at_handoff=spent, uncertain_calls=0,
        waited_seconds=round(time.monotonic()-start, 1), command=command))
    print('Resumed with 16 workers; PID='+str(process.pid), flush=True)


if __name__ == '__main__':
    main()
