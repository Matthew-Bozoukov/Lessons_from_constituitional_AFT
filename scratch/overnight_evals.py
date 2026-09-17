# ABOUTME: Unattended overnight driver: when each delib adapter lands, rent an eval pod and run
# ABOUTME: ODCV-lite then MASK on it, with MASK's --terminate-pod returning the box.
"""Run ODCV-lite + MASK on one trained arm, end to end, without a human.

Ordering is deliberate:
  ODCV first (it needs the served model for every rollout), MASK second WITH --terminate-pod,
  because MASK releases the host early -- the moment generation is on disk, before its judge
  wait (src/eval/misalignment/mask/runner.py:321) -- so the pod dies as early as it can.
  Running MASK first would terminate the pod before ODCV ever started.

ODCV drives Docker on THIS machine at `concurrency: 32`, so two arms must not roll out at the
same time: an ODCV run takes a file lock and holds it until its rollouts are judged. MASK
(API calls only) is free to overlap with the other arm's ODCV.

The adapter's HF id is read from the training log that scratch/teardown_when_trained.py saves
before it tears the training pod down -- the trainer prints `>>> pushed adapter ... to <url>`,
so nothing here has to guess the artifact's name.

Usage:
  python scratch/overnight_evals.py <pod-id-that-trained-it> <label> [--odcv-lock <path>]
"""
from __future__ import annotations

import argparse
import fcntl
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUSHED_RE = re.compile(r">>> pushed adapter.*?to (?:https?://huggingface\.co/)?([\w.\-]+/[\w.\-]+)")
HOST_RE = re.compile(r"host:\s+(\S+@[\d.]+):(\d+)")
POD_RE = re.compile(r"pod:\s+(\S+)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, text=True, **kw)


def wait_ready(pod: str, minutes: int) -> bool:
    """Poll the pod's bootstrap through the repo's own `runpod status`.

    NOT a hand-rolled fetch of the proxy's boot.log: RunPod's proxy answers Python's default
    urllib User-Agent with 403, and the first version of this script swallowed that for 60
    minutes on 2026-09-17 while the pod sat READY and billing.
    """
    for _ in range(minutes * 3):
        time.sleep(20)
        proc = subprocess.run(["uv", "run", "runpod", "status", "--pod", pod],
                              cwd=REPO, capture_output=True, text=True)
        text = proc.stdout + proc.stderr
        if "READY" in text:
            log(f"pod {pod} READY")
            return True
        if proc.returncode != 0:
            log(f"status check failed rc={proc.returncode}: {text.strip()[-200:]}")
    return False


def teardown(pod: str) -> None:
    proc = run(["uv", "run", "runpod", "down", "--pod", pod], capture_output=True)
    log((proc.stdout + proc.stderr).strip()[-300:])


def wait_for_adapter(train_pod: str, timeout_h: float) -> str:
    """The HF id of the adapter the training pod pushed, once its saved log names one."""
    saved = REPO / "output" / "runpod" / f"{train_pod}-train.log"
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        if saved.is_file():
            match = PUSHED_RE.search(saved.read_text())
            if match:
                return match.group(1)
            log(f"{saved.name} exists but names no adapter — training died; nothing to evaluate")
            raise SystemExit(1)
        time.sleep(120)
    raise SystemExit(f"no adapter from {train_pod} within {timeout_h}h")


def rent_eval_pod(label: str, target: str, max_hours: float) -> tuple[str, str, str]:
    """Rent the inference card for `target` and block until its bootstrap says READY."""
    proc = run(["uv", "run", "runpod", "up", "--name", f"jamie-eval-{label}",
                "--eval", target, "--max_hours", str(max_hours)],
               capture_output=True)
    out = proc.stdout + proc.stderr
    host_match, pod_match = HOST_RE.search(out), POD_RE.search(out)
    if not (host_match and pod_match):
        raise SystemExit(f"could not parse `runpod up` output:\n{out[-2000:]}")
    host, port, pod = host_match.group(1), host_match.group(2), pod_match.group(1)
    log(f"pod {pod} at {host}:{port} — waiting for vLLM + weights")
    if wait_ready(pod, minutes=60):
        return pod, host, port
    # Never leave a rented box behind on a failed wait: it bills whether or not we use it.
    teardown(pod)
    raise SystemExit(f"pod {pod} never reported READY; torn down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_pod")
    ap.add_argument("label")
    ap.add_argument("--odcv-lock", default="/tmp/odcv-docker.lock")
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--wait-hours", type=float, default=8.0)
    ap.add_argument("--server", help="user@ip:port of an eval pod that is already READY")
    # Each eval tunnels the pod's vLLM to a LOCAL port. Two evals on one port collide: the
    # second's `ssh -L` fails to bind and its traffic silently reaches the FIRST eval's server
    # (2026-09-17: delib-7's ODCV and MASK both hit delib-sonnet-7's pod -- 0% submissions,
    # 100% failed generations). So every invocation gets its own port.
    ap.add_argument("--odcv-port", type=int, required=True)
    ap.add_argument("--mask-port", type=int, required=True)
    args = ap.parse_args()

    target = wait_for_adapter(args.train_pod, args.wait_hours)
    log(f"adapter: {target}")

    # ODCV holds the Docker lock for its whole run: 32 scenarios in flight is already this
    # laptop's budget, and a second arm's containers would fight it for CPU and sockets.
    # The lock is taken BEFORE renting, so a queued arm never pays for a READY pod that sits
    # waiting for the other arm's ODCV (~2h of idle H200 otherwise).
    lock_path = Path(args.odcv_lock)
    lock_path.touch(exist_ok=True)
    with lock_path.open("w") as lock:
        log("waiting for the ODCV docker lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        log("ODCV lock acquired")
        if args.server:
            # An eval pod that is already READY for this target: use it rather than rent another.
            server = args.server
            log(f"reusing {server}")
        else:
            pod, host, port = rent_eval_pod(args.label, target, args.max_hours)
            server = f"{host}:{port}"
        odcv = run(["uv", "run", "evals", "--name", "odcv",
                    "--config", "configs/eval/odcv/lite.yaml",
                    "--target", target, "--server", server, "--port", str(args.odcv_port)])
        log(f"ODCV finished rc={odcv.returncode}")

    # MASK owns the pod's ending: --terminate-pod tears it down after publication OR failure,
    # and MASK releases it early, before its judge wait.
    mask = run(["uv", "run", "evals", "--name", "mask", "--target", target,
                "--server", server, "--port", str(args.mask_port), "--terminate-pod"])
    log(f"MASK finished rc={mask.returncode}")

    if mask.returncode != 0:
        log(f"!! MASK failed — confirm the pod at {server} is gone: uv run runpod pods")
    return 0 if (odcv.returncode == 0 and mask.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
