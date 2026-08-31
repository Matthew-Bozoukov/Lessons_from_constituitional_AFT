# ABOUTME: Serve a Qwen3 embedding model on a RunPod GPU via vLLM's embed task, exposed over
# ABOUTME: the HTTPS proxy so /v1/embeddings can be called from the laptop. up / status / down.

"""Stand up an embedding endpoint.

    uv run python scratch/serve_embed_runpod.py up --model Qwen/Qwen3-Embedding-8B
    uv run python scratch/serve_embed_runpod.py down --pod <id>

Separate from scratch/serve_adapter_runpod.py rather than a flag on it: that script serves a
BASE + LoRA pair for generation, this serves one model with `--task embed` and no adapter,
and folding two services into one bootstrap would make both harder to read.

The hard-won bits are inherited from the sibling script and are NOT optional:
  - vLLM goes into a PYTHON 3.12 venv built with uv, because current flashinfer annotates
    with `array.array[int]` (3.11+ syntax evaluated at import) and the RunPod image ships
    3.10, so the engine dies before loading anything.
  - Credentials are exported with xtrace OFF; /workspace is served over a PUBLIC proxy.
  - The bootstrap is validated as bash before the pod is created, since it bills from start.
  - Embedding is `--runner pooling --convert embed`, NOT `--task embed`: current vLLM
    rejects --task with an argparse error, which leaves the pod alive and idle at
    SERVE_STARTING rather than crashing, so it bills until someone notices (2026-08-17).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fire
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H100 80GB HBM3"


def _bootstrap(model: str, hf_token: str, max_len: int) -> str:
    """Pod startup script: build a py3.12 venv, serve `model` for embeddings, report ready."""
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
set +x
export HF_TOKEN={hf_token}
set -x
export VLLM_USE_FLASHINFER_SAMPLER=0
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
export VENV=/workspace/vllmenv
uv venv $VENV --python 3.12
uv pip install --python $VENV/bin/python -q vllm ninja huggingface_hub
$VENV/bin/hf download {model} >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo SERVE_STARTING
nohup $VENV/bin/python -m vllm.entrypoints.openai.api_server --model {model} --runner pooling --convert embed --served-model-name embed --max-model-len {max_len} --host 0.0.0.0 --port 8000 </dev/null > /workspace/vllm.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://localhost:8000/v1/models > /workspace/models.json && {{ echo VLLM_HEALTHY after ${{i}} polls; break; }}; sleep 15; done
cat /workspace/models.json || echo "NO MODELS RESPONSE"
echo SERVE_READY
sleep infinity
"""


def _validate(script: str) -> None:
    """Refuse to launch on a bootstrap that is not valid bash or lost a command."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in ("--runner pooling", "--convert embed", "uv venv $VENV --python 3.12", "SERVE_READY",
                   "--host 0.0.0.0"):
        assert needle in script, f"bootstrap lost {needle!r}"
    serves = [ln for ln in script.splitlines() if "api_server" in ln]
    assert len(serves) == 1 and serves[0].rstrip().endswith("&"), "serve command truncated"
    print(f">>> bootstrap validated ({len(script.splitlines())} lines)")


def up(model: str = "Qwen/Qwen3-Embedding-8B", max_len: int = 32768,
       gpu: str = GPU, disk_gb: int = 150, cloud: str = "SECURE",
       cuda: str = "13.0") -> None:
    """Create the embedding pod.

    Args:
        model: HF embedding model id.
        max_len: vLLM max_model_len; must exceed the longest document you will send.
        gpu: RunPod GPU type id.
        disk_gb: Container disk.
        cloud: SECURE or COMMUNITY.
        cuda: CUDA versions the host driver must satisfy.
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)
    script = _bootstrap(model, os.environ["HF_TOKEN"], max_len)
    _validate(script)
    pod = call("POST", "/pods", data=json.dumps({
        "name": f"embed-{model.split('/')[-1][:24]}",
        "imageName": IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1,
        "containerDiskInGb": disk_gb, "volumeInGb": 0,
        "ports": ["8000/http", "8080/http"], "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", script],
        "allowedCudaVersions": [c.strip() for c in str(cuda).split(",") if c.strip()],
    }))
    pid = pod["id"]
    print(f">>> pod {pid}  serving {model} for embeddings")
    print(f"    endpoint: https://{pid}-8000.proxy.runpod.net/v1")
    print(f"    boot log: https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"    DOWN    : uv run python scratch/serve_embed_runpod.py down --pod {pid}")


def down(pod: str) -> None:
    """Destroy the pod and confirm the account is clear."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    call("DELETE", f"/pods/{pod}")
    pods = call("GET", "/pods")
    items = pods.get("data", pods) if isinstance(pods, dict) else pods
    print(f">>> destroyed {pod}; active pods now: {len(items or [])}")


if __name__ == "__main__":
    fire.Fire()
