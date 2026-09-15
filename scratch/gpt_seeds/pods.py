# ABOUTME: List every pod on the (shared) RunPod account: id, name, status, GPUs, $/h. Read-only --
# ABOUTME: it never terminates anything; teardown goes through the driver that created the pod.
# Run: uv run python scratch/gpt_seeds/pods.py
from __future__ import annotations

import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402


def main(prefix: str = "") -> None:
    pods = call("GET", "/pods")
    pods = pods if isinstance(pods, list) else pods.get("pods", [])
    rows = [p for p in pods if p.get("name", "").startswith(prefix)]
    for p in rows:
        print(
            f"{p.get('id'):16s} {p.get('name', ''):32s} {p.get('desiredStatus', ''):10s} "
            f"gpus={p.get('gpuCount')} ${p.get('costPerHr')}/h"
        )
    print(
        f"{len(rows)} pod(s)"
        + (f" with prefix {prefix!r}" if prefix else " on the account")
    )


if __name__ == "__main__":
    fire.Fire(main)
