# ABOUTME: Rent a RunPod GPU, push the feature list to it, pull embeddings.npy back over
# ABOUTME: its HTTP proxy, and terminate it. Lifecycle only; the remote code is podscript.

"""Embed feature strings on a throwaway GPU.

Qwen3-Embedding-8B needs ~16GB of weights, which does not fit on the driving laptop, so
this stage rents a GPU for a few minutes:

    create   -> creates the pod; it installs deps and waits for /workspace/features.done
    push     -> PUT the feature list up; the pod auto-embeds as soon as it lands
    status   -> pod state plus the tail of the most advanced log
    fetch    -> download the vectors once /workspace/DONE exists
    terminate-> TERMINATE. Not optional; the pod bills by the second.

The pod holds no credentials: everything moves over the one :8080 HTTP proxy, so there is
no SSH and no Hugging Face round-trip.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from scratch.llm_feature_discovery.podscript import build_bootstrap
from scratch.llm_feature_discovery.rundir import RunDir
from src.eval.misalignment.internalization.scripts.runpod import call

RUNPOD_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
# 48GB for a 16GB fp16 model is deliberate headroom; it is $0.33/hr against $0.16 for a
# 24GB card, and an OOM after the 16GB download costs more than the difference.
DEFAULT_GPU_TYPE = "NVIDIA RTX A6000"
FETCHED_FILES = ("embeddings.npy", "probe_embeddings.npy", "embed_meta.json", "embed.log")


def _base_url(pod: str) -> str:
    """The pod's HTTP proxy root.

    Args:
        pod: Pod id.

    Returns:
        The proxy base URL.
    """
    return f"https://{pod}-8080.proxy.runpod.net"


def create(name: str = "matthew-bozoukov-feature-embed", gpu: str = DEFAULT_GPU_TYPE,
           disk_gb: int = 80, batch: int = 128) -> str:
    """Create the embedding pod.

    Args:
        name: Pod name, prefixed so it is identifiable on the shared account.
        gpu: RunPod GPU type id.
        disk_gb: Container disk (16GB model + image + HF cache).
        batch: Encoding batch size.

    Returns:
        The pod id.
    """
    public_ssh_key = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    payload = {
        "name": name,
        "imageName": RUNPOD_IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": "SECURE",
        "dockerStartCmd": ["bash", "-lc", build_bootstrap(batch)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": public_ssh_key},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    return pod.get("id") or pod.get("podId", "")


def push(pod: str, run: RunDir) -> int:
    """Upload the run's feature list to the pod, which starts embedding on arrival.

    Args:
        pod: Pod id.
        run: The run directory holding unique_features.txt.

    Returns:
        How many features were uploaded.

    Raises:
        RuntimeError: If the pod round-trips a different line count than was sent.
    """
    base_url = _base_url(pod)
    features_path = run.file("unique_features.txt")
    body = features_path.read_bytes()
    sent = len([x for x in body.decode().splitlines() if x.strip()])
    requests.put(f"{base_url}/features.txt", data=body, timeout=600).raise_for_status()

    round_tripped = requests.get(f"{base_url}/features.txt", timeout=300)
    round_tripped.raise_for_status()
    on_pod = len([x for x in round_tripped.text.splitlines() if x.strip()])
    if on_pod != sent:
        raise RuntimeError(f"upload corrupted: sent {sent} features, pod holds {on_pod}")
    requests.put(f"{base_url}/features.done", data=b"ok", timeout=60).raise_for_status()
    return sent


def status(pod: str) -> dict:
    """Pod state plus the tail of the most advanced log.

    Args:
        pod: Pod id.

    Returns:
        {"desiredStatus", "costPerHr", "log_name", "log_tail"}; log fields are None when
        no log is reachable yet.
    """
    info = call("GET", f"/pods/{pod}")
    for log_name in ("embed.log", "boot.log"):
        resp = requests.get(f"{_base_url(pod)}/{log_name}", timeout=20)
        if resp.ok and resp.text.strip():
            return {"desiredStatus": info.get("desiredStatus"),
                    "costPerHr": info.get("costPerHr"), "log_name": log_name,
                    "log_tail": "\n".join(resp.text.strip().splitlines()[-6:])}
    return {"desiredStatus": info.get("desiredStatus"), "costPerHr": info.get("costPerHr"),
            "log_name": None, "log_tail": None}


def fetch(pod: str, run: RunDir) -> dict:
    """Download the embedding artifacts into the run directory once the pod signals DONE.

    Args:
        pod: Pod id.
        run: The run directory to write into.

    Returns:
        Parsed embed_meta.json.

    Raises:
        RuntimeError: If the pod has not finished embedding.
    """
    base_url = _base_url(pod)
    if not requests.get(f"{base_url}/DONE", timeout=20).ok:
        raise RuntimeError("pod has not written /workspace/DONE yet; check `status`")
    run.ensure()
    for filename in FETCHED_FILES:
        started = time.time()
        resp = requests.get(f"{base_url}/{filename}", timeout=1800)
        resp.raise_for_status()
        run.file(filename).write_bytes(resp.content)
        print(f"{filename}: {len(resp.content) / 1e6:.1f} MB in {time.time() - started:.0f}s")
    return run.read_embed_meta()


def terminate(pod: str) -> list[tuple[str, str]]:
    """Terminate the pod, then report what is still running on the account.

    Args:
        pod: Pod id.

    Returns:
        (id, name) of every pod still running.
    """
    call("DELETE", f"/pods/{pod}")
    still_running = call("GET", "/pods") or []
    return [(p["id"], p.get("name")) for p in still_running]
