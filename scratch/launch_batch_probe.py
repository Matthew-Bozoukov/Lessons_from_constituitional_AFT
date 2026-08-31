# ABOUTME: Rents one H200 and runs scratch/probe_batch_memory.py on it, to measure the
# ABOUTME: real per-device batch-size wall at the corpus's longest sequence.

"""Measure the batch-size wall on real hardware.

The probe script is shipped base64-encoded inside the bootstrap: it is a single file, and
encoding it sidesteps every quoting hazard of embedding Python in bash inside a Python
f-string (the class of bug that has already truncated one serve command on this project).

    uv run python scratch/launch_batch_probe.py up
    uv run python scratch/launch_batch_probe.py status --pod <id>
    uv run python scratch/launch_batch_probe.py down --pod <id>
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H200"
SEQ_LEN = 8191          # the longest row in the self-reflection mixture
BATCHES = "1,4,6,8,16"


def _bootstrap(hf_token: str, probe_b64: str, seq_len: int, batches: str) -> str:
    """Pod startup: install deps, decode the probe, sweep batch sizes, hold for reading."""
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
# dockerStartCmd REPLACES the image entrypoint, so the image's sshd never runs and an
# exposed 22/tcp just refuses connections. Start it here: without a shell on the pod, a
# one-line bug in the probe costs a whole new pod and another 55GB model download.
(mkdir -p /run/sshd && nohup /usr/sbin/sshd -D </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
set +x
export HF_TOKEN={hf_token}
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "peft==0.20.0" fire hf_transfer huggingface_hub
echo {probe_b64} | base64 -d > /workspace/probe.py
hf download {BASE} >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo PROBE_STARTING
(python3 /workspace/probe.py --seq_len {seq_len} --batches '{batches}' --model {BASE} 2>&1 | tee /workspace/probe.log) || echo PROBE_FAILED
echo PROBE_DONE
sleep infinity
"""


def _validate(script: str) -> None:
    """Parse-check the bootstrap and confirm nothing was lost, before a pod is billed."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in ("base64 -d", "probe.py", "PROBE_DONE", "--seq_len"):
        assert needle in script, f"bootstrap lost {needle!r}"
    print(f">>> bootstrap validated ({len(script.splitlines())} lines)")


def up(seq_len: int = SEQ_LEN, batches: str = BATCHES, gpu: str = GPU,
       disk_gb: int = 200, cloud: str = "SECURE") -> None:
    """Create the probe pod.

    Args:
        seq_len: Sequence length to probe at (the corpus's longest row).
        batches: Comma-separated per-device batch sizes to try.
        gpu: RunPod GPU type — must match the training card for the answer to transfer.
        disk_gb: Container disk (base model ~55GB + cache).
        cloud: SECURE or COMMUNITY.
    """
    load_dotenv(override=True)
    probe = Path(__file__).parent / "probe_batch_memory.py"
    b64 = base64.b64encode(probe.read_bytes()).decode()
    script = _bootstrap(os.environ["HF_TOKEN"], b64, seq_len, batches)
    _validate(script)
    pod = call("POST", "/pods", data=json.dumps({
        "name": "probe-batch-memory",
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http", "22/tcp"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", script],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }))
    pid = pod["id"]
    print(f">>> pod {pid}  {gpu}  probing batches {batches} at seq_len {seq_len:,}")
    print(f"    log : https://{pid}-8080.proxy.runpod.net/probe.log")
    print(f"    boot: https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"    DOWN: uv run python scratch/launch_batch_probe.py down --pod {pid}")


def status(pod: str) -> None:
    """Print the probe's phase and results so far.

    Args:
        pod: Pod id.
    """
    for f in ("probe.log", "boot.log"):
        try:
            r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{f}", timeout=30)
            if r.status_code == 200 and "<html" not in r.text[:200].lower():
                lines = [l for l in r.text.splitlines() if l.strip()]
                print(f"--- {f} ---")
                print("\n".join(lines[-14:]))
                return
        except requests.RequestException:
            pass
    print("no logs yet")


def down(pod: str) -> None:
    """Destroy the pod and print remaining active pods.

    Args:
        pod: Pod id.
    """
    load_dotenv(override=True)
    call("DELETE", f"/pods/{pod}")
    pods = call("GET", "/pods")
    pods = pods if isinstance(pods, list) else pods.get("data", [])
    print(f">>> destroyed {pod}; active pods now: {len(pods)}")
    for p in pods:
        print("   ", p["id"], p.get("desiredStatus"), p.get("name"))


if __name__ == "__main__":
    fire.Fire()
