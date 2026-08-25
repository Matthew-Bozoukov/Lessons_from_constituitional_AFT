# ABOUTME: Rent, address and destroy the vast VM docker hosts an ODCV run drives from.
# ABOUTME: Run: uv run python scratch/odcv_vast_boxes.py up --count 2 --label_prefix nika-odcv-a

"""Provision the docker hosts for an ODCV run, and resolve where they actually are.

    uv run python scratch/odcv_vast_boxes.py up --count 2 --label_prefix nika-odcv-a
    uv run python scratch/odcv_vast_boxes.py ls
    uv run python scratch/odcv_vast_boxes.py addr --instance <id>
    uv run python scratch/odcv_vast_boxes.py down --instance <id>

ODCV needs a REAL VM, not a container. Each scenario is a docker compose project with its
own network, and an unprivileged container cannot create those -- which is why CLAUDE.md
forbids driving ODCV from a RunPod pod. vast's `vms_enabled` machines run
`docker.io/vastai/kvm` (template "Ubuntu 22.04 VM"), a KVM guest where docker runs natively.

Three things here are lifted straight from docs/swebench_run_postmortem.md, which is the
record of a long unattended vast run that lost ~14 hours and most of its output:

  - ADDRESSES MOVE. That run's driver had its public IP remapped mid-run, killing every
    process on it. Worse, `vastai ssh-url` kept returning the STALE address while the API's
    `public_ipaddr` field had the new one. `addr()` therefore reports BOTH and prefers the
    API field, and nothing here caches an address across calls.
  - INSTANCES GET STUCK `loading` WHILE BILLING. One sat in loading/stopped for five hours.
    `wait_ready` polls actual_status and gives up loudly rather than hoping.
  - THE DRIVER IS THE SINGLE POINT OF FAILURE. It holds every rollout produced so far, so
    offers are filtered on reliability and preferred on it, not purely on price.

Every call is scoped to instances this repo labelled. `down` takes ONE id and destroys only
that; nothing here ever sweeps the account.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# "Ubuntu 22.04 VM" (docker.io/vastai/kvm, ubuntu_cli_22.04). 25k+ launches, so failures
# are ours rather than the template's.
VM_TEMPLATE = "b7942f6bbc4374893ff66eb78145bbac"


def _vast(*args: str, raw: bool = False) -> str:
    """One vastai CLI call with the key from .env, never from an ambient login."""
    env = {**os.environ}
    key = env.get("VAST_API_KEY")
    assert key, "VAST_API_KEY missing from .env"
    # The key can carry a trailing \r when .env was authored on Windows; that reaches the
    # API as part of the header value and 401s with a message that names nothing useful.
    env["VAST_API_KEY"] = key.strip().strip('"')
    cmd = ["uvx", "vastai", *args] + (["--raw"] if raw else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout


def _offers(min_cores: int, min_ram_gb: int, min_disk_gb: int,
            min_reliability: float) -> list[dict]:
    """VM-capable offers meeting the floor, cheapest-per-reliable first.

    Filtering is client-side on purpose: combining cpu_ram with cpu_cores_effective in the
    query language silently returns nothing rather than erroring.
    """
    data = json.loads(_vast("search", "offers", "vms_enabled=true rentable=true",
                            "-o", "dph+", raw=True))
    out = [o for o in data
           if (o.get("cpu_cores_effective") or 0) >= min_cores
           and (o.get("cpu_ram") or 0) >= min_ram_gb * 1000
           and (o.get("disk_space") or 0) >= min_disk_gb
           and (o.get("reliability2") or o.get("reliability") or 0) >= min_reliability]
    # Price first, but never take a cheap unreliable host for a driver that holds hours of
    # unreplicated work -- reliability breaks the tie in the other direction.
    out.sort(key=lambda o: (round(o.get("dph_total", 99), 3),
                            -(o.get("reliability2") or 0)))
    return out


def up(count: int = 2, label_prefix: str = "nika-odcv", disk_gb: int = 200,
       min_cores: int = 16, min_ram_gb: int = 32, min_disk_gb: int = 220,
       min_reliability: float = 0.98, dry_run: bool = False) -> None:
    """Rent `count` VM docker hosts, labelled `<prefix>-1..N`.

    Args:
        count: How many boxes.
        label_prefix: MUST start with `nika-` -- the vast account is shared and a label is
            the only thing that says whose an instance is.
        disk_gb: Local disk. ODCV keeps every scenario image (`prune_images: false`), so
            this is sized for image accumulation across 70 cells x N passes, not for source.
        min_cores, min_ram_gb, min_disk_gb: Offer floor. Defaults match the 19-core/49GB
            boxes the 2026-08-18 ODCV run used.
        min_reliability: Host reliability floor; see the postmortem note above.
        dry_run: Show what would be rented and rent nothing.
    """
    assert label_prefix.startswith("nika-"), (
        f"label_prefix {label_prefix!r} must start with 'nika-' -- the vast account is "
        f"shared and teammates' instances must stay distinguishable")
    offers = _offers(min_cores, min_ram_gb, min_disk_gb, min_reliability)
    print(f">>> {len(offers)} offers meet the floor "
          f"({min_cores}+ cores, {min_ram_gb}+GB RAM, {min_disk_gb}+GB disk, "
          f"reliability >= {min_reliability})")
    for o in offers[:count + 3]:
        print(f"    offer {o['id']}  {o.get('cpu_cores_effective')}c "
              f"{round((o.get('cpu_ram') or 0) / 1000)}GB  "
              f"disk {round(o.get('disk_space') or 0)}GB  "
              f"${o.get('dph_total', 0):.3f}/hr  "
              f"rel {o.get('reliability2', 0):.3f}  {o.get('geolocation')}")
    assert len(offers) >= count, f"only {len(offers)} offers meet the floor, need {count}"
    if dry_run:
        print(">>> dry run: nothing rented")
        return

    made = []
    for i, o in enumerate(offers[:count], 1):
        label = f"{label_prefix}-{i}"
        out = _vast("create", "instance", str(o["id"]), "--template_hash", VM_TEMPLATE,
                    "--disk", str(disk_gb), "--ssh", "--direct", "--label", label, raw=True)
        try:
            res = json.loads(out)
            iid = res.get("new_contract") or res.get("id")
        except json.JSONDecodeError:
            print(f"    !! unparseable create response for {label}: {out[:200]}")
            continue
        made.append((label, iid))
        print(f">>> {label}: instance {iid} from offer {o['id']} "
              f"(${o.get('dph_total', 0):.3f}/hr)")
    print("\n>>> wait for ready, then bootstrap:")
    for label, iid in made:
        print(f"    uv run python scratch/odcv_vast_boxes.py addr --instance {iid}")
    print("\n>>> TEARDOWN (one id each, never a sweep):")
    for label, iid in made:
        print(f"    uv run python scratch/odcv_vast_boxes.py down --instance {iid}")


def _instances() -> list[dict]:
    return json.loads(_vast("show", "instances", raw=True))


def ls() -> None:
    """List instances, marking which are ours by label."""
    rows = _instances()
    if not rows:
        print(">>> no vast instances")
        return
    for i in rows:
        label = i.get("label") or ""
        mine = label.startswith("nika-")
        print(f"  {i.get('id')}  {label or '(unlabelled)':<24} "
              f"{i.get('actual_status')}  {i.get('public_ipaddr')}  "
              f"${i.get('dph_total', 0):.3f}/hr  "
              f"{'[OURS]' if mine else '[not ours, leave alone]'}")


def addr(instance: int) -> None:
    """Report where this box actually is, from BOTH sources (they disagree; see module doc)."""
    inst = next((i for i in _instances() if str(i.get("id")) == str(instance)), None)
    assert inst, f"instance {instance} not found on this account"
    api_ip = inst.get("public_ipaddr")
    ports = inst.get("ports") or {}
    mapped = (ports.get("22/tcp") or [{}])[0].get("HostPort")
    print(f"label:         {inst.get('label')}")
    print(f"status:        {inst.get('actual_status')} / {inst.get('intended_status')}")
    print(f"api ip:port:   {api_ip}:{mapped}      <- PREFER THIS")
    try:
        print(f"ssh-url:       {_vast('ssh-url', str(instance)).strip()}   "
              f"<- can be STALE after a remap")
    except Exception as e:  # noqa: BLE001
        print(f"ssh-url:       unavailable ({type(e).__name__})")
    if api_ip and mapped:
        print(f"\nssh -i ~/.ssh/msm_audit -p {mapped} root@{api_ip}")


def wait_ready(instance: int, timeout_min: float = 20.0, poll_s: float = 20.0) -> None:
    """Block until the box reports running, or fail loudly rather than bill in `loading`."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        inst = next((i for i in _instances() if str(i.get("id")) == str(instance)), None)
        if not inst:
            raise SystemExit(f"instance {instance} vanished from the account")
        st = inst.get("actual_status")
        print(f"    {instance}: {st}", flush=True)
        if st == "running":
            addr(instance)
            return
        time.sleep(poll_s)
    raise SystemExit(
        f"instance {instance} never reached running in {timeout_min} min -- it is BILLING "
        f"while stuck (postmortem failure C). Destroy it and rent another:\n"
        f"  uv run python scratch/odcv_vast_boxes.py down --instance {instance}")


def down(instance: int) -> None:
    """Destroy exactly ONE instance, then report what is left without touching it."""
    inst = next((i for i in _instances() if str(i.get("id")) == str(instance)), None)
    if inst and not str(inst.get("label") or "").startswith("nika-"):
        raise SystemExit(
            f"instance {instance} is labelled {inst.get('label')!r}, not ours. Refusing.")
    # -y: the CLI otherwise blocks on an interactive confirmation, which in a script
    # reads as "Aborted." while the instance keeps billing.
    print(_vast("destroy", "instance", str(instance), "-y").strip())
    print(">>> remaining:")
    ls()


if __name__ == "__main__":
    fire.Fire({"up": up, "ls": ls, "addr": addr, "wait_ready": wait_ready, "down": down})
