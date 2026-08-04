# ABOUTME: Launch/monitor/destroy a RunPod H100 that trains one LoRA arm from a public
# ABOUTME: HF bundle (code+data), no credentials on the pod. Run: uv run python scripts/gpu/runpod_train.py up

"""Train a LoRA arm on a throwaway RunPod H100.

The pod carries NO credentials: it downloads the base model and a public HF dataset
bundle (training code tarball + mixture jsonl), trains, then serves /workspace over
HTTP on :8080 — the adapter is pulled back through the proxy and pushed to HF from the
local machine, where the token lives.

Bootstrap discipline is inherited from scripts/gpu/runpod_capability.py (log server before
anything slow, trainer in the foreground of PID 1, `|| true` so a crash leaves a
readable log instead of a restart loop). See that file for why each line is the way it is.

    uv run python scripts/gpu/runpod_train.py up --bundle LASR-Callum/qwen36-27b-tulu-0-100-train
    uv run python scripts/gpu/runpod_train.py status --pod <id>
    uv run python scripts/gpu/runpod_train.py down --pod <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"


def _bootstrap(base: str, bundle: str, train_config: str, mixture: str) -> str:
    """Pod startup script: fetch bundle, train, expose the adapter over :8080."""
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
# python3 -m pip, NOT bare pip: on this image `pip` and `python3` can resolve to
# different interpreters, and packages land where the trainer cannot import them.
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" datasets accelerate omegaconf fire huggingface_hub hf_transfer
hf download {base} >/dev/null
hf download {bundle} --repo-type dataset --local-dir /workspace/bundle
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
mkdir -p /workspace/repo/data && cp /workspace/bundle/{mixture} /workspace/repo/data/mixture.jsonl
cd /workspace/repo
echo TRAINING_STARTING
(python3 scripts/train/train_lora.py --config {train_config} 2>&1 | tee /workspace/train.log) || true
# Package whatever the trainer wrote (adapter + run_meta) for pull-back over :8080.
tar -czf /workspace/adapter.tar.gz -C /workspace/repo/output . || true
echo TRAINING_DONE
sleep infinity
"""


def up(
    bundle: str,
    train_config: str = "configs/train/lora_qwen36_tulu100.yaml",
    base: str = "Qwen/Qwen3.6-27B",
    gpu: str = DEFAULT_GPU,
    name: str = "train-lora-0-100",
    mixture: str = "mixture.jsonl",
    disk_gb: int = 150,
    image: str = DEFAULT_IMAGE,
    cloud: str = "SECURE",
    countries: str = "US,CA,NL,DE,FR,GB,IE,BE,SE,NO,FI,CH,AT,ES,IT",
) -> str:
    """Create a training pod.

    Args:
        bundle: Public HF dataset repo holding code.tar.gz + the mixture jsonl.
        train_config: Config path inside the code tarball.
        base: Base model to download on the pod.
        gpu: RunPod GPU type id.
        name: Pod name. Prefix it so it is distinguishable on the shared account.
        mixture: Mixture filename inside the bundle; copied to data/mixture.jsonl
            on the pod, which is what every train config points at.
        disk_gb: Container disk (base model ~55GB + HF cache + outputs).
        image: Container image.
        cloud: SECURE or COMMUNITY.
        countries: Placement restriction, "" for anywhere.

    Returns:
        Pod id and the URLs to watch.
    """
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", _bootstrap(base, bundle, train_config, mixture)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    codes = [c.strip().upper() for c in countries.split(",") if c.strip()]
    if codes:
        payload["countryCodes"] = codes
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    return (
        f"pod:       {pod_id}\n"
        f"boot log:  https://{pod_id}-8080.proxy.runpod.net/boot.log\n"
        f"train log: https://{pod_id}-8080.proxy.runpod.net/train.log\n"
        f"adapter:   https://{pod_id}-8080.proxy.runpod.net/adapter.tar.gz (after TRAINING_DONE)\n\n"
        f"Training ~2h after ~25 min setup. THEN TEAR IT DOWN:\n"
        f"  uv run python scripts/gpu/runpod_train.py down --pod {pod_id}"
    )


def status(pod: str) -> str:
    """Report pod state and the last boot/train log lines."""
    import requests

    info = call("GET", f"/pods/{pod}")
    line = f"status: {info.get('desiredStatus')}"
    for log in ("train.log", "boot.log"):
        try:
            r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{log}", timeout=15)
            if r.ok:
                tail = "\n".join(r.text.strip().splitlines()[-3:])
                return f"{line}\n--- {log} tail ---\n{tail}"
        except Exception:  # noqa: BLE001 - keep trying older logs
            continue
    return f"{line}\nno logs reachable yet"


def down(pod: str) -> str:
    """Terminate the pod. Always run this; the pod bills by the second."""
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}"


if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "down": down})
