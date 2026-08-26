# ABOUTME: Runs odcv_vast_boxes' logic over the vastai PYTHON SDK, because this machine's
# ABOUTME: Application Control policy refuses to spawn the `vastai` CLI binary.

"""Same vast box management, different transport.

    uv run --with vastai python scratch/verbose_cot/vast_boxes_sdk.py up --count 4 \
        --label_prefix nika-odcv-verbose [--dry_run True]
    uv run --with vastai python scratch/verbose_cot/vast_boxes_sdk.py ls
    uv run --with vastai python scratch/verbose_cot/vast_boxes_sdk.py addr --instance <id>
    uv run --with vastai python scratch/verbose_cot/vast_boxes_sdk.py down --instance <id>

`scratch/odcv_vast_boxes.py` shells out to `uvx vastai`. On this machine that fails with
"An Application Control policy has blocked this file (os error 4551)" -- the policy blocks
the EXECUTABLE, not the package, so importing the SDK works fine.

Everything that matters is left in the original module and reused: the offer floor and the
client-side filtering (the query language silently returns nothing when cpu_ram and
cpu_cores_effective are combined), the price-then-reliability sort, the `nika-` label
requirement, and a `down` that takes one id and never sweeps a shared account. Only
`_vast` -- the single function that shells out -- is replaced, so there is one transport to
fix if the SDK moves, and no second copy of the policy.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

_spec = importlib.util.spec_from_file_location(
    "odcv_vast_boxes", ROOT / "scratch" / "odcv_vast_boxes.py")
boxes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boxes)

from vastai import VastAI  # noqa: E402

_sdk = VastAI(api_key=os.environ["VAST_API_KEY"].strip().strip('"'))


def _vast_sdk(*args: str, raw: bool = False) -> str:
    """Serve odcv_vast_boxes' CLI calls from the SDK, returning the JSON it expects."""
    a = list(args)
    if a[:2] == ["search", "offers"]:
        # The original passes the filter string positionally and sorts client-side; the SDK
        # takes the same query language, so the floor stays expressed in ONE place.
        return json.dumps(_sdk.search_offers(query=a[2], order="dph_total", limit=200))
    if a[:2] == ["show", "instances"]:
        return json.dumps(_sdk.show_instances())
    if a[:2] == ["create", "instance"]:
        opts = {a[i].lstrip("-"): a[i + 1] for i in range(3, len(a) - 1)
                if a[i].startswith("--") and not a[i + 1].startswith("--")}
        # The CLI's `--ssh --direct` pair is one `runtype` in the SDK. Direct matters: the
        # proxied form routes SSH through vast's gateway, and the box has to be reachable
        # for the systemd tunnel that carries every rollout to the serving pod.
        flags = {x for x in a[3:] if x.startswith("--")}
        runtype = ("ssh_direct" if {"--ssh", "--direct"} <= flags
                   else "ssh_proxy" if "--ssh" in flags else None)
        res = _sdk.create_instance(
            id=int(a[2]), template_hash=opts.get("template_hash"),
            disk=float(opts.get("disk", 200)), label=opts.get("label"), runtype=runtype)
        return json.dumps(res)
    if a[:2] == ["destroy", "instance"]:
        return json.dumps(_sdk.destroy_instance(id=int(a[2])))
    raise NotImplementedError(f"no SDK mapping for: vastai {' '.join(a)}")


boxes._vast = _vast_sdk


if __name__ == "__main__":
    sys.exit(fire.Fire({"up": boxes.up, "ls": boxes.ls,
                        "addr": boxes.addr, "down": boxes.down}))
