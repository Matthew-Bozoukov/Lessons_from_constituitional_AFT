# ABOUTME: Minimal RunPod REST helper for the SAE embed runs — create/status/terminate
# ABOUTME: one pod, ssh details from its runtime ports. Never touches other pods.

"""Usage (from repo root, any env with requests + python-dotenv):

    uv run --project scratch/sae_properties python scratch/sae_properties/podctl.py create \
        --name sae-smoke-8b --gpu "NVIDIA GeForce RTX 4090" [--disk 80] [--count 1]
    ... podctl.py status <pod-id>      # prints state + ssh command when ready
    ... podctl.py terminate <pod-id>   # terminates THAT pod only, then lists what remains
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
BASE = "https://rest.runpod.io/v1"
HEADERS = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
           "Content-Type": "application/json"}
PUBKEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"


def create(args) -> None:
    body = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.count,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,
        "cloudType": args.cloud,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": PUBKEY_PATH.read_text().strip()},
    }
    r = requests.post(f"{BASE}/pods", headers=HEADERS, json=body)
    print(r.status_code, json.dumps(r.json(), indent=2, default=str)[:800])
    if r.ok:
        print(f"\npod id: {r.json().get('id')}  — poll with: podctl.py status {r.json().get('id')}")


def status(args) -> None:
    r = requests.get(f"{BASE}/pods/{args.pod_id}", headers=HEADERS)
    p = r.json()
    print(json.dumps({k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "lastStatusChange")}, default=str))
    ip, port = p.get("publicIp"), (p.get("portMappings") or {}).get("22")
    if ip and port:
        print(f"ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p {port} root@{ip}")
    else:
        print("ssh not ready yet")


def wait(args) -> None:
    import time
    for _ in range(args.tries):
        r = requests.get(f"{BASE}/pods/{args.pod_id}", headers=HEADERS).json()
        ip, port = r.get("publicIp"), (r.get("portMappings") or {}).get("22")
        if ip and port:
            print(f"ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p {port} root@{ip}")
            return
        time.sleep(args.interval)
    sys.exit("ssh never became ready")


def terminate(args) -> None:
    r = requests.delete(f"{BASE}/pods/{args.pod_id}", headers=HEADERS)
    print("terminate:", r.status_code, r.text[:200])
    left = requests.get(f"{BASE}/pods", headers=HEADERS).json()
    print("pods remaining on account:",
          [{"id": p["id"], "name": p["name"], "costPerHr": p["costPerHr"]} for p in left])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--gpu", required=True)
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--disk", type=int, default=80)
    c.add_argument("--cloud", default="SECURE")
    c.add_argument("--image", default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
    c.set_defaults(fn=create)
    s = sub.add_parser("status")
    s.add_argument("pod_id")
    s.set_defaults(fn=status)
    w = sub.add_parser("wait")
    w.add_argument("pod_id")
    w.add_argument("--tries", type=int, default=30)
    w.add_argument("--interval", type=int, default=20)
    w.set_defaults(fn=wait)
    t = sub.add_parser("terminate")
    t.add_argument("pod_id")
    t.set_defaults(fn=terminate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
