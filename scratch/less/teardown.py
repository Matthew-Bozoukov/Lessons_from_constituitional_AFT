# ABOUTME: Terminates ONE named RunPod pod and reports what is left running — pod-scoped
# ABOUTME: by construction, because the account is shared with teammates' pods.

"""Tear down a LESS pod by id, then report the account without touching it.

    uv run python scratch/less/teardown.py --pod <id>
    uv run python scratch/less/teardown.py --list

This deliberately does NOT sweep. A sweep-all teardown on this account would terminate
whatever a teammate is running (there is a `matthewb-*` pod up right now), and the repo's
rule is that we never terminate a resource we did not provision. So the pod id is
required, the call is a single DELETE, and everything else is reported rather than acted
on: after terminating, any surviving pod is listed with its name so a `nika-` orphan is
obvious and a teammate's pod is plainly labelled as theirs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

REST = "https://rest.runpod.io/v1"


def _headers() -> dict:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def report(headers: dict) -> list[dict]:
    pods = requests.get(f"{REST}/pods", headers=headers, timeout=45).json()
    if not pods:
        print(">>> no pods running on the account")
        return []
    print(f">>> {len(pods)} pod(s) still running:")
    for p in pods:
        name = str(p.get("name", ""))
        who = "OURS — orphan?" if name.startswith("nika-") else "not ours, leave alone"
        print(f"    {p.get('id')}  {name!r:<40} ${p.get('costPerHr')}/hr  [{who}]")
    return pods


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pod", help="pod id to terminate")
    ap.add_argument("--list", action="store_true", help="report only, terminate nothing")
    args = ap.parse_args()
    headers = _headers()

    if args.list or not args.pod:
        report(headers)
        if not args.pod:
            raise SystemExit(0 if args.list else "pass --pod <id> to terminate")
        return

    info = requests.get(f"{REST}/pods/{args.pod}", headers=headers, timeout=45)
    name = info.json().get("name", "?") if info.ok else "?"
    if not str(name).startswith("nika-"):
        raise SystemExit(
            f"refusing to terminate {args.pod}: its name is {name!r}, which does not carry "
            f"our `nika-` prefix. If this really is ours, rename it first or delete it by "
            f"hand — an unprefixed pod is far more likely to be a teammate's.")

    r = requests.delete(f"{REST}/pods/{args.pod}", headers=headers, timeout=60)
    print(f">>> terminate {args.pod} ({name}): HTTP {r.status_code}")
    r.raise_for_status()
    report(headers)


if __name__ == "__main__":
    main()
