# ABOUTME: Provisions a RunPod box for the LESS run — nika-prefixed, SSH-enabled, with a
# ABOUTME: fallback GPU ladder because 4x H200 capacity is intermittent.

"""Provision a LESS pod and print its SSH details.

    uv run python scratch/less/provision.py --name nika-less-run --gpus 4

Capacity for 4x H200 reads "Low", so a single hard-coded gpuTypeId turns a transient
stock-out into a stalled run. The ladder below is ordered by suitability rather than
price, and the reason H200 leads is memory: with gradient checkpointing restored the
extractor needs ~70 GB at D's ~1.6k-token rows, but the codebase_resisted validation rows
reach 6,463 tokens, and the activation growth there is what would push an 80 GB H100 over.
H100 is kept as a last resort for the TRAIN split only, where rows are short.

The `nika-` prefix is not cosmetic: the RunPod account is shared, a teammate's pod is
usually running on it, and scratch/less/teardown.py refuses to terminate anything without
that prefix. A pod created outside this script must be named the same way or it cannot be
torn down by the tooling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

REST = "https://rest.runpod.io/v1"

# (gpuTypeId, note) in preference order — memory headroom first, price second.
GPU_LADDER = [
    ("NVIDIA H200", "141GB — fits the 6.5k-token validation rows with room to spare"),
    ("NVIDIA H200 NVL", "143GB — equivalent headroom"),
    ("NVIDIA H100 NVL", "94GB — adequate, tighter on the long rows"),
    ("NVIDIA H100 80GB HBM3", "80GB — train split only; long val rows may OOM"),
]

COUNTRIES = ["US", "CA", "NL", "DE", "FR", "GB", "IE", "BE", "SE", "NO", "FI", "CH", "AT", "ES", "IT"]

START = (
    "mkdir -p /workspace ~/.ssh && printf '%s\\n' \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys "
    "&& chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && "
    "(service ssh start || (apt-get update -qq && apt-get install -y -qq openssh-server "
    "&& service ssh start)) && (cd /workspace && nohup python3 -m http.server 8080 "
    ">/dev/null 2>&1 &) ; sleep infinity"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="nika-less-run")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--disk", type=int, default=400,
                    help="55GB base + 4 checkpoints x 2.4GB Adam state + HF cache + outputs")
    ap.add_argument("--image", default="runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204")
    args = ap.parse_args()

    if not args.name.startswith("nika-"):
        raise SystemExit(f"name {args.name!r} must start with 'nika-' — the account is "
                         f"shared and teardown.py keys off that prefix")

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
               "Content-Type": "application/json"}
    pub = Path(os.path.expanduser("~/.ssh/msm_audit.pub")).read_text().strip()

    for gpu, note in GPU_LADDER:
        payload = {
            "name": args.name, "imageName": args.image,
            "gpuTypeIds": [gpu], "gpuCount": args.gpus,
            "containerDiskInGb": args.disk, "volumeInGb": 0,
            "ports": ["8080/http", "22/tcp"], "cloudType": "SECURE",
            "countryCodes": COUNTRIES,
            "dockerStartCmd": ["bash", "-lc", START],
            "env": {"PUBLIC_KEY": pub, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        }
        r = requests.post(f"{REST}/pods", headers=headers, data=json.dumps(payload), timeout=120)
        if r.ok:
            pod = r.json()
            print(f">>> {args.gpus}x {gpu} — {note}")
            print(f">>> pod {pod.get('id')}  ${pod.get('costPerHr')}/hr")
            print(f">>> poll for SSH, then: uv run python scratch/less/teardown.py "
                  f"--pod {pod.get('id')}")
            return
        print(f"    {args.gpus}x {gpu}: HTTP {r.status_code} — {r.text[:120]}")

    raise SystemExit(
        f"no capacity for {args.gpus} GPUs anywhere on the ladder. Either retry, drop "
        f"--gpus, or finish on the existing single-GPU pod (which already holds the 52GB "
        f"model download and costs less in total, just slower).")


if __name__ == "__main__":
    main()
