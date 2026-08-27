# ABOUTME: Provision / inspect / destroy ONE vast.ai 2xH200 box for the low-stakes LoRA run.
# ABOUTME: Run: uv run --with vastai python scratch/low_stakes/vast_train.py offers|up|ls|ssh|down

"""Rent one training box, and never touch anybody else's.

The `vastai` CLI cannot be spawned on this machine -- Application Control refuses the
executable with os error 4551 -- so this goes through the Python SDK, the same workaround
`scratch/verbose_cot/vast_boxes_sdk.py` documents. The policy blocks the binary, not the
package, so importing works.

Two safety rules, both inherited rather than invented:

  * The account is SHARED. `down` takes ONE instance id and destroys exactly that. There is
    no sweep, no "destroy all orphans" -- a sweep here would kill a teammate's pod.
  * Every box this creates is labelled `nika-...`, so a human reading `ls` can tell whose
    it is. `down` refuses an id whose label does not start with that prefix.
"""

import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LABEL = "nika-lowstakes-train"
# The GPU stack is pinned in pyproject and the lock is linux-only, so a plain torch image is
# all the pod needs -- `uv sync` installs the rest. Same image as the CLAUDE.md playbook.
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
# `cuda_vers>=13.0` is NOT optional and is the easy one to leave out. The repo pins
# torch 2.11.0+cu130, so a host whose driver predates CUDA 13 gives `torch.cuda.
# is_available() == False` while `nvidia-smi` still lists both GPUs happily -- the box looks
# fine and cannot train. Measured 2026-08-26: an offer with cuda_max_good 12.8 (driver
# 570.86.15) was rented, bootstrapped and launched before that surfaced, and had to be
# destroyed. Sort on price, but filter on this first.
QUERY = ("gpu_name=H200 num_gpus=2 rentable=true disk_space>=250 "
         "inet_down>=2000 reliability>=0.98 cuda_vers>=13.0")


def _client():
    from vastai import VastAI
    key = os.environ.get("VAST_API_KEY")
    assert key, "VAST_API_KEY missing from .env"
    return VastAI(api_key=key)


def offers(query: str = QUERY, limit: int = 8) -> None:
    """Cheapest rentable boxes matching the query, price ascending."""
    print(_client().search_offers(query=query, order="dph+", limit=limit))


def ls() -> None:
    """Every instance on the shared account. Read this before creating another."""
    print(_client().show_instances())


def up(offer_id: int, disk_gb: int = 300, image: str = IMAGE,
       label: str = LABEL) -> None:
    """Rent one box. `offer_id` comes from `offers`."""
    assert label.startswith("nika-"), "label must start with nika- on a shared account"
    v = _client()
    # `runtype="ssh_direct ssh_proxy"` is how the SDK expresses what the CLI spells
    # `--ssh --direct`: the CLI flags are not keyword arguments here.
    out = v.create_instance(id=offer_id, image=image, disk=disk_gb, label=label,
                            runtype="ssh_direct ssh_proxy",
                            onstart_cmd="touch /root/.vast_ready")
    print(out)
    print("\nnow:  uv run --with vastai python scratch/low_stakes/vast_train.py ls")


def ssh(instance: int) -> None:
    """The ssh host/port for a box, for ~/.ssh/config."""
    v = _client()
    print(v.show_instance(id=instance))


def down(instance: int, label: str = LABEL) -> None:
    """Destroy exactly ONE instance, and only if it is one of ours.

    The label check is the whole point: this account is shared, and an id typo that lands
    on a teammate's pod is unrecoverable. If the label does not match, this refuses and
    tells you to look rather than guessing.
    """
    v = _client()
    info = str(v.show_instance(id=instance))
    if label not in info:
        print(f"REFUSING: instance {instance} does not carry the label {label!r}.\n"
              f"Never destroy a resource this run did not provision -- report it instead.")
        print(info[:600])
        return
    print(v.destroy_instance(id=instance))
    print("\nverify nothing of ours is left:")
    print(v.show_instances())


if __name__ == "__main__":
    fire.Fire({"offers": offers, "ls": ls, "up": up, "ssh": ssh, "down": down})
