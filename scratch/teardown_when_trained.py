# ABOUTME: Poll a training pod's train.log and tear the pod down as soon as the adapter is pushed.
# ABOUTME: Closes the gap between "training finished" and the runpod watchdog's --max_hours cap.
"""Watch one training pod and terminate it the moment its work is done.

`uv run evals --terminate-pod` does this for evals; training has no equivalent, because the
trainer runs ON the pod and cannot terminate the box under its own feet. So this runs HERE,
polls the pod's `train.log` over SSH, and calls the repo's own teardown (`uv run runpod down`),
never a raw API call.

Two exits, both of which save the log locally first:
  - `>>> pushed adapter` in the log  -> the adapter is on the Hub; tear down.
  - the trainer process is gone with no such line -> it died; SAVE THE LOG, then tear down,
    because an idle crashed pod bills exactly as fast as a working one.

Usage:
  python scratch/teardown_when_trained.py <pod_id> <user@host> <port> [--poll 120] [--max-hours 9]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUSHED = ">>> pushed adapter"


def ssh(host: str, port: str, command: str, timeout: int = 60) -> tuple[int, str]:
    """Run one command on the pod. A failed connection is not fatal: the caller retries."""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
         "-o", "StrictHostKeyChecking=accept-new", "-p", port, host, command],
        capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def save_log(host: str, port: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    code, out = ssh(host, port, "cat /root/work/train.log", timeout=180)
    if code == 0 and out:
        dest.write_text(out)
        print(f"[{time.strftime('%H:%M:%S')}] saved log -> {dest} ({len(out):,} chars)", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] could not fetch train.log (ssh rc={code})", flush=True)


def teardown(pod_id: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] tearing down {pod_id}", flush=True)
    proc = subprocess.run(["uv", "run", "runpod", "down", "--pod", pod_id],
                          cwd=REPO, capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip(), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod_id")
    ap.add_argument("host")
    ap.add_argument("port")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=9.0)
    args = ap.parse_args()

    log_dest = REPO / "output" / "runpod" / f"{args.pod_id}-train.log"
    deadline = time.time() + args.max_hours * 3600
    print(f"[{time.strftime('%H:%M:%S')}] watching {args.pod_id} ({args.host}:{args.port}), "
          f"poll {args.poll}s, giving up at {time.strftime('%H:%M', time.localtime(deadline))}",
          flush=True)

    # `[t]rain` is the bracket trick: the regex matches the trainer's command line but NOT this
    # probe's own, which contains the literal "[t]rain". Without it pgrep counts itself, the
    # count is never 0, and the death branch can never fire (measured on a live pod: count 3,
    # one of them the probe).
    probe = (f"grep -c '{PUSHED}' /root/work/train.log 2>/dev/null || true; "
             "pgrep -fc '[t]rain --config configs/train/sft.yaml' || true; "
             "stat -c %s /root/work/train.log 2>/dev/null || echo 0")
    last_size, quiet_polls = None, 0
    while time.time() < deadline:
        time.sleep(args.poll)
        code, out = ssh(args.host, args.port, probe)
        if code != 0:
            # The pod may be rebooting, the laptop may have slept: keep watching.
            print(f"[{time.strftime('%H:%M:%S')}] ssh unavailable (rc={code}); retrying", flush=True)
            continue
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        pushed, running, size = (lines + ["0", "0", "0"])[:3]
        if pushed != "0":
            print(f"[{time.strftime('%H:%M:%S')}] adapter pushed", flush=True)
            save_log(args.host, args.port, log_dest)
            teardown(args.pod_id)
            return 0
        # Death needs BOTH signals, twice in a row: no trainer process AND a log that has
        # stopped growing. Either alone is not enough -- a process can be missed by a flaky
        # pgrep, and a log can be quiet during a long checkpoint write or an HF upload.
        grew = last_size is not None and size != last_size
        quiet_polls = 0 if (running != "0" or grew) else quiet_polls + 1
        last_size = size
        if quiet_polls >= 2:
            print(f"[{time.strftime('%H:%M:%S')}] !! no trainer process and train.log static "
                  f"({size} bytes) across {quiet_polls} polls -- treating as dead", flush=True)
            save_log(args.host, args.port, log_dest)
            teardown(args.pod_id)
            return 1
        print(f"[{time.strftime('%H:%M:%S')}] still training (proc={running}, log={size}B)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] gave up after {args.max_hours}h; "
          f"pod {args.pod_id} left to the runpod watchdog cap", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
