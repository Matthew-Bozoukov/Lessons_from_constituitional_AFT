# ABOUTME: One-shot concurrency handoff; immediate restart preserves uncertain reservations without retry.
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
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--restart-now', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('Worker override must be between 1 and 32')
    launch = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    budget = Path(launch['campaign_budget_root'])
    root, logs = Path(args.root), Path(args.log_dir)
    initial = identity(args.pid)
    if not initial or 'scratch.dataset_refresh.run_native_smoke' not in initial['CommandLine'] or args.config not in initial['CommandLine']:
        raise RuntimeError('PID is not the owned full-generation launcher')
    lock = FileLock(str(budget/'spend.lock'), timeout=5)
    start = time.monotonic()
    status_path = logs/'concurrency_handoff.json'
    if status_path.exists():
        write_json(logs/'concurrency_handoff_previous.json', json.loads(status_path.read_text()))
    write_json(status_path, dict(status='restarting' if args.restart_now else 'waiting_for_zero_inflight', old_pid=args.pid, workers=args.workers))
    while time.monotonic() - start < 3600:
        with lock:
            entries = json.loads((budget/'spend.json').read_text())
            if any(e['status'] not in {'reserved', 'settled'} for e in entries):
                raise RuntimeError('Non-settled failure requires investigation; no automatic handoff')
            if args.restart_now or not any(e['status'] == 'reserved' for e in entries):
                if identity(args.pid) != initial:
                    raise RuntimeError('Owner process identity changed; refuse termination')
                # The ledger lock blocks every new paid dispatch throughout termination.
                subprocess.run(['taskkill', '/PID', str(args.pid), '/T', '/F'], check=True, capture_output=True)
                interrupted = [e['call_id'] for e in entries if e['status'] == 'reserved']
                write_json(logs/'ledger_at_handoff.json', entries)
                for entry in entries:
                    if entry['call_id'] in interrupted:
                        entry.update(status='interrupted_unknown', interruption_reason='User-requested 32-worker operational restart; reservation retained and request excluded without redispatch')
                write_json(budget/'spend.json', entries)
                count, spent = len(entries), sum(e['charged_or_reserved_usd'] for e in entries)
                break
        time.sleep(0.1)
    else:
        raise TimeoutError('No idle checkpoint within one hour; original run remains untouched')
    command = [sys.executable, '-u', '-X', 'utf8', '-m', 'scratch.dataset_refresh.run_native_smoke',
               '--config', args.config, '--resume', args.root, '--workers', str(args.workers),
               '--resume-reason', f'User-requested operational concurrency change to {args.workers}; preserved settled responses, checkpoints and uncertain reservations, excluded {len(interrupted)} interrupted requests without redispatch; same prompts, selection and $120 ledger']
    with (logs/f'stdout_workers{args.workers}.log').open('wb') as out, (logs/f'stderr_workers{args.workers}.log').open('wb') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    (logs/'pid.txt').write_text(str(process.pid))
    write_json(logs/'concurrency_handoff.json', dict(status='resumed', old_pid=args.pid, new_pid=process.pid,
        workers=args.workers, calls_at_handoff=count, exposure_usd_at_handoff=spent, uncertain_calls=len(interrupted), interrupted_call_ids=interrupted,
        waited_seconds=round(time.monotonic()-start, 1), command=command))
    print(f'Resumed with {args.workers} workers; PID='+str(process.pid), flush=True)


if __name__ == '__main__':
    main()
