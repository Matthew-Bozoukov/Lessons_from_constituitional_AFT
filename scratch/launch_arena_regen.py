# ABOUTME: RunPod launcher for the Arena-Hard rerun: regenerates the 110 prompts the target
# ABOUTME: answered empty, for BOTH arms, at a larger token budget, then re-judges them.

"""Rerun the empty Arena-Hard prompts on RunPod.

Everything the pod needs is pulled from one private HF bundle (code, bench tree with the
110 rows already deleted, and the rerun config) — RunPod's `dockerStartCmd` replaces the
image entrypoint, so sshd never starts and there is no shell to fix things from. Results
are served back over :8080.

The generator's resume rule regenerates exactly the deleted rows; the vendored judge caches
on uid and re-judges exactly the deleted judgments.

    uv run python scratch/launch_arena_regen.py up
    uv run python scratch/launch_arena_regen.py status --pod <id>
    uv run python scratch/launch_arena_regen.py down --pod <id>
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
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

BUNDLE = "LASR-Callum/2026-08-09-arena-hard-regen-bundle"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H100 80GB HBM3"


def _bootstrap(hf_token: str, openrouter_key: str) -> str:
    """Pod startup: fetch bundle, install, regenerate the missing rows, judge, expose."""
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -uxo pipefail
# Deliberately NOT `set -e`: this script is PID 1, so any non-zero exit stops the container
# and destroys the log server before the failure can be read. Steps that must not silently
# continue are checked explicitly below.
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
set +x
export HF_TOKEN={hf_token}
export OPENROUTER_API_KEY={openrouter_key}
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# huggingface_hub FIRST: `hf download` below is not in this image, and under `set -e` a
# missing command kills the script -- which, as PID 1, takes the container and the log
# server with it, so the failure surfaces as an unexplained 404 rather than a message.
python3 -m pip install --no-cache-dir -q huggingface_hub hf_transfer
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
hf download {BUNDLE} --repo-type dataset --local-dir /workspace/bundle
mkdir -p /root/work && tar -xzf /workspace/bundle/code.tar.gz -C /root/work
mkdir -p /root/work/src/eval/capabilities/arena_hard/third_party
tar -xzf /workspace/bundle/bench.tar.gz -C /root/work/src/eval/capabilities/arena_hard/third_party
mkdir -p /root/work/scratch && cp /workspace/bundle/arena_hard_pair.py /workspace/bundle/arena_hard_pair_overrides.yaml /root/work/scratch/
cd /root/work
[ -f scratch/arena_hard_pair.py ] || {{ echo "BUNDLE_UNPACK_FAILED"; sleep infinity; }}
git init -q . || true
printf 'HF_TOKEN=%s\\nOPENROUTER_API_KEY=%s\\n' "$HF_TOKEN" "$OPENROUTER_API_KEY" > .env
uv sync 2>&1 | tail -2
mkdir -p output/arena_hard/regen
cp /workspace/bundle/config.yaml output/arena_hard/regen/arena_hard_config.yaml
wc -l src/eval/capabilities/arena_hard/third_party/arena-hard-auto/data/arena-hard-v2.0/model_answer/*.jsonl
echo REGEN_STARTING
(uv run python scratch/arena_hard_pair.py --baseline "LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64" --target "LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64" 2>&1 | tee /workspace/regen.log) || echo REGEN_FAILED
# Serve the refreshed answers and judgments back for pull-through :8080.
tar -czf /workspace/results.tar.gz -C /root/work/src/eval/capabilities/arena_hard/third_party/arena-hard-auto/data/arena-hard-v2.0 model_answer model_judgment || true
echo REGEN_DONE
sleep infinity
"""


def _validate(script: str) -> None:
    """Parse-check and confirm nothing was lost, before a pod bills."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(script)
        path = fh.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in ("arena_hard_pair.py", "--baseline", "--target", "REGEN_DONE",
                   "results.tar.gz", "bench.tar.gz"):
        assert needle in script, f"bootstrap lost {needle!r}"
    # Match the INVOCATION, not every mention: the `cp` line names the file too.
    runs = [ln for ln in script.splitlines() if "python scratch/arena_hard_pair.py" in ln]
    assert len(runs) == 1 and runs[0].rstrip().endswith("REGEN_FAILED"), runs
    print(f">>> bootstrap validated ({len(script.splitlines())} lines)")


def up(gpu: str = GPU, disk_gb: int = 250, cloud: str = "SECURE") -> None:
    """Create the rerun pod.

    Args:
        gpu: RunPod GPU type.
        disk_gb: Container disk (base model ~55GB + bundle + outputs).
        cloud: SECURE or COMMUNITY.
    """
    load_dotenv(override=True)
    script = _bootstrap(os.environ["HF_TOKEN"], os.environ["OPENROUTER_API_KEY"])
    _validate(script)
    pod = call("POST", "/pods", data=json.dumps({
        "name": "arena-hard-regen",
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", script],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }))
    pid = pod["id"]
    print(f">>> pod {pid}  {gpu}")
    print(f"    log    : https://{pid}-8080.proxy.runpod.net/regen.log")
    print(f"    boot   : https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"    results: https://{pid}-8080.proxy.runpod.net/results.tar.gz")
    print(f"    DOWN   : uv run python scratch/launch_arena_regen.py down --pod {pid}")


def status(pod: str) -> None:
    """Print phase and the tail of whichever log is live.

    Args:
        pod: Pod id.
    """
    base = f"https://{pod}-8080.proxy.runpod.net"
    try:
        boot = requests.get(f"{base}/boot.log", timeout=30).text
    except requests.RequestException:
        print("no logs yet"); return
    phase = next((m for m in ("REGEN_DONE", "REGEN_FAILED", "REGEN_STARTING")
                  if m in boot), "booting")
    print(f">>> pod {pod}  phase={phase}")
    for f in ("regen.log", "boot.log"):
        try:
            r = requests.get(f"{base}/{f}", timeout=30)
            if r.status_code == 200 and "<html" not in r.text[:200].lower():
                lines = [l for l in r.text.replace("\r", "\n").splitlines() if l.strip()]
                print(f"--- {f} ---"); print("\n".join(lines[-12:])); return
        except requests.RequestException:
            pass


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


if __name__ == "__main__":
    fire.Fire()
