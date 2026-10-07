# ABOUTME: Keep asking RunPod for a pod until one actually survives its first 90 seconds.
# ABOUTME: Platform-killed pods ("Exited by Runpod") bill seconds, so retrying is near-free.
"""Acquire one usable eval pod, retrying across cards.

On 2026-10-07 three pods in a row were killed by RunPod within ~6s of creation while an
earlier pod kept running — capacity on the profile's card, not anything about the request.
A pod that dies this way costs seconds of billing, so the right response is to keep asking
rather than to conclude the platform is unusable. Each attempt is checked for an early
EXIT and torn down before the next, so no corpse is left billing or polled for an hour.

Usage:
    ./.venv/bin/python scratch/colosseum_hospital/get_pod.py <name> [max_attempts]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
from src.infra.runpod import call, terminate, up, wait_bootstrapped, _ssh_endpoint

import os
# The arm to serve; overridable so one script covers the control arm too.
TARGET_ENV = "HOSP_TARGET"
# The eval name picks the INFERENCE card from the target profile, so it must match the
# eval that will actually run on this pod -- odcv and mask have their own entries.
EVAL_ENV = "HOSP_EVAL"
TARGET = os.environ.get(TARGET_ENV, "dougalldeepmind/2026-10-05-qwen36-0-da-15")
# The profile's card first, then near-equivalents: capacity is per-card, so a ladder finds
# a host the default alone would not.
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200", "NVIDIA A100 80GB PCIe"]
SETTLE_S = 90


def survives(pod_id: str) -> bool:
    """True if the pod is still alive after SETTLE_S — a platform kill lands in seconds."""
    for t in range(0, SETTLE_S, 10):
        st = call("GET", f"/pods/{pod_id}").get("desiredStatus")
        if st == "EXITED":
            print(f"    t+{t}s EXITED (platform kill)", flush=True)
            return False
        time.sleep(10)
    return True


def main(name: str, max_attempts: int = 12, gpu_only: str | None = None) -> None:
    """Retry until a pod survives, then bootstrap it and write its host to /tmp.

    `gpu_only` pins every attempt to one card instead of walking the ladder.
    """
    cards = [gpu_only] if gpu_only else GPUS
    for attempt in range(1, int(max_attempts) + 1):
        gpu = cards[(attempt - 1) % len(cards)]
        print(f">>> attempt {attempt}/{max_attempts} on {gpu}", flush=True)
        ids: list[str] = []
        try:
            up(name=f"{name}-a{attempt}",
               eval=os.environ.get(EVAL_ENV, "colosseum_hospital"), target=TARGET,
               max_hours=6.0, gpu=gpu, on_provisioned=ids.append)
        except Exception as e:
            print(f"    provision failed: {type(e).__name__} {str(e)[:120]}", flush=True)
            if ids:
                terminate(ids[0])
            time.sleep(20)
            continue
        pod_id = ids[0]
        if not survives(pod_id):
            terminate(pod_id)
            time.sleep(20)
            continue
        print(f">>> {pod_id} survived {SETTLE_S}s on {gpu}; bootstrapping", flush=True)
        ip, port = _ssh_endpoint(pod_id)
        ok = wait_bootstrapped(pod_id, timeout_s=3600)
        Path(f"/tmp/{name}.txt").write_text(f"{pod_id} root@{ip}:{port} bootstrapped={ok}")
        print(f">>> {name} READY pod={pod_id} host=root@{ip}:{port} bootstrapped={ok}", flush=True)
        return
    print(f"!!! {name}: no pod survived in {max_attempts} attempts", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 12,
         sys.argv[3] if len(sys.argv) > 3 else None)
