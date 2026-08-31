# ABOUTME: List every RunPod pod on the account with status and cost, for teardown checks.
# ABOUTME: Exists because `status --pod <id>` on a destroyed pod can report a stale phase.

"""Enumerate live RunPod pods so teardown can be confirmed rather than assumed.

Run:
  uv run python scratch/list_runpod_pods.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

pods = call("GET", "/pods")
items = pods.get("data", pods) if isinstance(pods, dict) else pods
if not items:
    print("0 pods on the account")
else:
    print(f"{len(items)} pod(s):")
    for p in items:
        print(f"  {p.get('id')}  {p.get('desiredStatus')}  {p.get('name')}  "
              f"gpu={p.get('gpuCount')}x{(p.get('machine') or {}).get('gpuTypeId')}  "
              f"costPerHr={p.get('costPerHr')}")
