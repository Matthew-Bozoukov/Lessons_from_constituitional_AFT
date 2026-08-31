# ABOUTME: Provision a 2xH200 RunPod pod with SSH for DDP LoRA training, then tear it down.
# ABOUTME: up / ssh / down. The driver rsyncs the repo and launches torchrun over SSH.

"""Stand up a two-GPU RunPod box for a dynamic-batching DDP training run.

Unlike scripts/gpu/runpod_train.py (one H100, public code+data bundle, no credentials on the
pod) this box is driven over SSH exactly like the vast.ai runs: the repo is rsynced, `.env` is
copied, and `torchrun --nproc_per_node=2` runs there. That workflow is what the sibling arms
were trained with, so reusing it keeps the arms comparable.

Two rules carried over from scratch/serve_adapter_runpod.py, both learned the hard way:

  - **Nothing slow happens before the pod is reachable.** The bootstrap only installs an SSH
    key and idles; every heavy step is driven from here so failures are visible immediately
    rather than buried in a restart loop.
  - **CUDA is constrained at scheduling time.** The repo lock pulls torch built for CUDA 13,
    which needs driver >= 580. An older host dies at `_cuda_init` after a ~50GB download.

    uv run python scratch/runpod_train_2xh200.py up
    uv run python scratch/runpod_train_2xh200.py ssh --pod <id>
    uv run python scratch/runpod_train_2xh200.py down --pod <id>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H200"


def up(gpu: str = GPU, count: int = 2, disk_gb: int = 300, cloud: str = "SECURE",
       cuda: str = "13.0", name: str = "train-traits134") -> None:
    """Create the training pod with SSH enabled.

    Args:
        gpu: RunPod GPU type id.
        count: GPUs on the pod.
        disk_gb: Container disk; base weights ~55GB plus checkpoints.
        cloud: SECURE or COMMUNITY.
        cuda: Comma-separated CUDA versions the host driver must satisfy.
        name: Pod name.
    """
    load_dotenv(override=True)
    pub = Path.home() / ".ssh" / "id_ed25519.pub"
    assert pub.is_file(), f"no public key at {pub}"
    pod = call("POST", "/pods", data=json.dumps({
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": count,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "cloudType": cloud,
        "allowedCudaVersions": [c.strip() for c in str(cuda).split(",") if c.strip()],
        # deploy_train.sh reads WANDB_API_KEY out of the CONTAINER env (/proc/1/environ),
        # deliberately, so it never lands in the pod's process list. If the pod is created
        # without it, WANDB_MODE silently falls back to `disabled` and a long run logs
        # nowhere (observed 2026-08-27). Pass it through when the driver has one.
        "env": {k: v for k, v in {
            "PUBLIC_KEY": pub.read_text().strip(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "teaching-claude-why"),
        }.items() if v},
    }))
    print(f">>> pod {pod['id']}  {count}x{gpu}")
    print(f"    ssh   : uv run python scratch/runpod_train_2xh200.py ssh --pod {pod['id']}")
    print(f"    DOWN  : uv run python scratch/runpod_train_2xh200.py down --pod {pod['id']}")


def ssh(pod: str, wait: int = 600) -> None:
    """Print the pod's ssh target once it is running, polling until the port is published.

    Args:
        pod: Pod id.
        wait: Seconds to wait for the pod to expose its SSH port.

    Raises:
        TimeoutError: If no SSH port appears in time.
    """
    deadline = time.time() + wait
    while time.time() < deadline:
        d = call("GET", f"/pods/{pod}")
        status = d.get("desiredStatus")
        for p in (d.get("portMappings") or {}).items():
            pass
        pm = d.get("portMappings") or {}
        ip = d.get("publicIp")
        if pm.get("22") and ip:
            print(f"ssh -p {pm['22']} root@{ip}")
            print(f"HOST={ip} PORT={pm['22']}")
            return
        print(f"  status={status} ip={ip} ports={pm}")
        time.sleep(15)
    raise TimeoutError(f"pod {pod} never published an SSH port")


def down(pod: str) -> None:
    """Terminate the pod and report how many remain.

    Args:
        pod: Pod id.
    """
    call("DELETE", f"/pods/{pod}")
    pods = call("GET", "/pods")
    items = pods.get("data", pods) if isinstance(pods, dict) else pods
    print(f">>> destroyed {pod}; active pods now: {len(items or [])}")


if __name__ == "__main__":
    fire.Fire()
