# ABOUTME: Monitor a multi-box ODCV run: per-box progress, vLLM health, and artifact pull-down.
# ABOUTME: Run: uv run python scratch/odcv_monitor.py --once   (or --loop_s 300 to keep watching)

"""Watch an ODCV run spread across rented docker hosts, and get its artifacts to safety.

Rollouts are produced on RENTED boxes that evaporate when the run ends, so the pull-down
here is not a convenience -- it is the thing that stops hours of work dying with an
instance (CLAUDE.md: pull artifacts off CONTINUOUSLY, not at the end).

Reports, per cycle:
  - per box: scenarios completed / 70, status breakdown, non-empty transcript count
  - the empty-transcript alarm: a scenario that finishes `ok` while writing NO transcript
    is the silent failure that once cost a full run; it must be loud, not counted quietly
  - vLLM: running/waiting requests, KV-cache usage, cumulative preemptions (preemptions
    climbing means total concurrency across boxes is above what the GPU can hold)
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import fire
import requests

BOXES = {
    "A": ("184.144.255.144", "17398"),
    "B": ("77.104.167.148", "42779"),
}
KEY = str(Path.home() / ".ssh" / "msm_audit")
METRICS = "https://v95j78dnat6p6q-8000.proxy.runpod.net/metrics"
REMOTE_OUT = "/root/work/output/odcv_bench"


def _ssh(host: str, port: str, cmd: str, timeout: int = 90) -> str:
    """Run one command on a box, returning stdout ('' on failure)."""
    r = subprocess.run(
        ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new",
         "-o", f"ConnectTimeout=20", "-p", port, f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def _vllm() -> str:
    """One-line vLLM health summary from the Prometheus endpoint."""
    try:
        txt = requests.get(METRICS, timeout=30).text
    except Exception as e:
        return f"vLLM metrics unreachable: {type(e).__name__}"
    def g(name: str) -> float:
        m = re.findall(rf"^{name}[^ ]* ([0-9.e+-]+)$", txt, re.M)
        return sum(float(x) for x in m) if m else 0.0
    return (f"vLLM  running={g('vllm:num_requests_running'):.0f} "
            f"waiting={g('vllm:num_requests_waiting'):.0f} "
            f"kv={g('vllm:gpu_cache_usage_perc')*100:.1f}% "
            f"preemptions={g('vllm:num_preemptions_total'):.0f}")


def _pull(local_root: Path) -> None:
    """Copy each box's rollout tree down to the laptop, into per-box subdirs."""
    for tag, (host, port) in BOXES.items():
        dest = local_root / f"box{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-q", "-r", "-i", KEY, "-P", port,
                        f"root@{host}:{REMOTE_OUT}/.", str(dest)],
                       capture_output=True, text=True, timeout=900)


def cycle(local_root: Path, pull: bool = True) -> None:
    """Print one status report for every box plus the server."""
    print(f"\n===== {time.strftime('%H:%M:%S')} =====")
    print(" ", _vllm())
    for tag, (host, port) in BOXES.items():
        done = _ssh(host, port, r"grep -cE '^\[[0-9]+/70\]' /root/work/passes.log || true")
        stat = _ssh(host, port,
                    r"grep -oE '\b(ok\+no_transcript|ok|compose_exit_[0-9]+\+no_container|"
                    r"timeout|error)\b' /root/work/passes.log | sort | uniq -c | sort -rn "
                    r"| tr '\n' ' '")
        tx = _ssh(host, port,
                  "find /root/work/output -name messages_record.txt -size +0 2>/dev/null | wc -l")
        passes = _ssh(host, port, "grep -c 'PASS .* done' /root/work/passes.log || true")
        alive = _ssh(host, port, "pgrep -fc odcv_rollout_cli || true")
        print(f"  box{tag}  lines={done:<5} passes_done={passes:<3} transcripts={tx:<5} "
              f"driver_alive={alive}")
        print(f"        status: {stat}")
        if "ok+no_transcript" in stat:
            print("        *** ALARM: ok+no_transcript -- agent completing without acting ***")
    if pull:
        _pull(local_root)
        n = len(list(local_root.rglob("messages_record.txt")))
        print(f"  pulled -> {local_root}  ({n} transcripts local)")


def main(local_root: str = "output/odcv_bench/lessswap716_multibox",
         once: bool = False, loop_s: float = 300.0, heartbeat: str = "") -> None:
    """Report status, pull artifacts, and refresh the watchdog heartbeat.

    Args:
        local_root: Where pulled rollout trees land on this machine.
        once: Single cycle then exit.
        loop_s: Seconds between cycles when looping.
        heartbeat: If set, touch this file each cycle so the pod watchdog stays satisfied.
    """
    root = Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        cycle(root)
        if heartbeat:
            Path(heartbeat).touch()
        if once:
            return
        time.sleep(loop_s)


if __name__ == "__main__":
    fire.Fire(main)
