# ABOUTME: Launch/monitor/destroy a RunPod pod serving Qwen3.6-27B + several LoRA arms
# ABOUTME: for the capability eval. Run: uv run python scripts/runpod_capability.py up

"""One pod, every arm, served concurrently as vLLM LoRA modules.

The existing `src.eval.misalignment.internalization.scripts.runpod` helper merges a *single* adapter into the base
and serves the merged copy. That is the right shape for a one-arm eval and the wrong
shape here: three arms would mean either three pods (three ~55GB downloads, three boots,
three times the bill) or three sequential merges on one pod, which needs ~216GB of disk
for the base plus three merged copies.

vLLM serves a base model with several LoRA adapters attached at once, so all three arms
answer from a single weight load. That is cheaper, faster, and has a methodological
benefit that matters more than either: **every arm is served by the same process, on the
same GPU, with the same build and flags**, so the decoding parity spec §4 requires is a
property of the setup rather than something we have to trust across three separate boots.

No credentials are placed on the pod. The base model and all adapters are public, so the
pod holds no secret and cannot leak one.

    uv run python scripts/runpod_capability.py up
    uv run python scripts/runpod_capability.py status --pod <id>
    uv run python scripts/runpod_capability.py down --pod <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"


def _bootstrap(base: str, modules: dict[str, str], max_rank: int) -> str:
    """Return the pod startup script serving `base` with `modules` as LoRA adapters.

    Everything is teed to /workspace/boot.log and vLLM is started detached, so `status`
    can tell "still downloading" from "died on load" — a pod that failed silently bills
    until somebody notices.
    """
    # Adapters are downloaded up front rather than lazily, so a bad repo id fails during
    # boot (visible in boot.log) instead of on the first request to that one arm.
    downloads = "\n".join(
        f'hf download {repo} --local-dir /workspace/adapters/{name}' for name, repo in modules.items()
    )
    lora_args = " ".join(f"{name}=/workspace/adapters/{name}" for name in modules)
    # `mkdir -p /workspace` MUST come before the redirect, and the redirect must not be
    # under `set -e`. We create the pod with volumeInGb=0, so /workspace is NOT mounted:
    # redirecting into it first makes `tee` fail on line 2 and `set -e` kills the whole
    # bootstrap before anything starts — the container then sits there billing at H100
    # rates with no server, no log, and no SSH. That failure cost ~57 minutes across two
    # pods before the empty directory listing on 8080 gave it away.
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1; \
 mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
# Serve boot.log on 8080 IMMEDIATELY, before anything slow starts. Without this a boot
# that stalls is undiagnosable: RunPod's REST API has no logs endpoint (400), and an
# http-only port mapping gives no SSH to fall back on. The first run of this script cost
# 40 minutes of H100 time establishing exactly that. Costs nothing and answers
# "downloading or dead?" from a browser.
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# Use the native sampler rather than FlashInfer's JIT-compiled one: one less thing to
# build at startup, and it removes an entire class of boot failure.
export VLLM_USE_FLASHINFER_SAMPLER=0
# CLAUDE.md gotcha 3: this allocator setting is what keeps 80GB workloads on this model
# family from dying to fragmentation rather than to genuine exhaustion.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# `ninja` is NOT optional: FlashInfer JIT-builds its sampling kernels during engine
# init and shells out to ninja, so without it vLLM dies with exit 127 ("command not
# found") behind a generic "Engine core initialization failed". Belt and braces, we
# also disable the FlashInfer sampler below, which skips the JIT step entirely.
pip install --no-cache-dir -q "vllm>=0.8.5" "transformers>=4.51.3" peft huggingface_hub hf_transfer ninja
mkdir -p /workspace/adapters
{downloads}
echo ADAPTERS_DOWNLOADED
# vLLM runs in the FOREGROUND as the last command, deliberately. `dockerStartCmd` is
# PID 1: if this script backgrounds vLLM and then exits, PID 1 dies and the container
# reaps every child with it — the pod stays "RUNNING" with nothing listening on any
# port, which is indistinguishable from "still downloading" unless you happen to notice
# that the log server died too. That cost ~70 minutes and ~$3.50 on one pod.
# Keeping vLLM in the foreground pins PID 1 for the life of the server, which also keeps
# the boot-log server alive.
#
# Window 16384, not 8192: a thinking model generates its trace inside max_tokens, and
# at an 8192 window the budget ceiling (8192 - longest prompt) truncated 4% of base-arm
# MMLU generations. KV at 16k fits fine within utilization 0.85 at eval concurrency.
# Memory: 27B in bf16 is ~54GB of the H100's 79GB before any LoRA or KV cache. At
# utilization 0.92 the run OOMed during CUDA *graph capture* with 116MB free, so
# `--enforce-eager` skips graph capture entirely and 0.85 leaves genuine headroom. Eager
# costs some decode throughput, which is irrelevant here: we generate a few hundred
# answers once, and a server that starts slowly beats one that does not start.
vllm serve {base} \
  --served-model-name base \
  --enable-lora --max-lora-rank {max_rank} --max-loras {len(modules)} \
  --lora-modules {lora_args} \
  --port 8000 --host 0.0.0.0 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --reasoning-parser qwen3 --trust-remote-code \
  </dev/null 2>&1 | tee /workspace/vllm.log || true
# `|| true` matters: `set -o pipefail` makes a failed vLLM pipeline trip `set -e`, the
# script exits, and RunPod restart-loops the whole bootstrap — re-downloading 55GB each
# time and burning the log. Holding here instead keeps the failure readable.
echo "VLLM EXITED — see /workspace/vllm.log"
sleep infinity
"""


def up(
    config: str = "configs/capability_eval.yaml",
    gpu: str = DEFAULT_GPU,
    name: str = "capability-eval",
    disk_gb: int = 200,
    image: str = DEFAULT_IMAGE,
    cloud: str = "SECURE",
    countries: str = "US,CA,NL,DE,FR,GB,IE,BE,SE,NO,FI,CH,AT,ES,IT",
) -> str:
    """Create a pod serving every arm in the config that has an adapter.

    Args:
        config: Capability-eval config; supplies the base model and the arm->adapter map.
        gpu: RunPod GPU type id.
        name: Pod name.
        disk_gb: Container disk. Holds the base model plus the adapters.
        image: Container image.
        cloud: SECURE or COMMUNITY.
        countries: Comma-separated ISO country codes to restrict placement to. Defaults
            to US/Canada and Western Europe — close to both HuggingFace's CDN and to us,
            which matters because every boot pulls ~55GB of base weights and a distant
            datacenter turns that into the dominant cost of the run. Pass "" to let
            RunPod place the pod anywhere.

    Returns:
        Pod id, base URL, and the served name for each arm.
    """
    cfg = OmegaConf.load(config)
    modules = {str(a.name): str(a.adapter) for a in cfg.arms if a.adapter}
    if not modules:
        raise SystemExit(f"No arms in {config} have an adapter set.")

    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        # 8080 carries boot.log (see _bootstrap); 22/tcp is the SSH fallback. Both are
        # cheap insurance against a silent boot failure billing at H100 rates.
        "ports": ["8000/http", "8080/http", "22/tcp"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", _bootstrap(str(cfg.base_model), modules, 32)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    codes = [c.strip().upper() for c in countries.split(",") if c.strip()]
    if codes:
        payload["countryCodes"] = codes
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
    served = "\n".join(f"    {n:16} <- {r}" for n, r in modules.items())
    region = ",".join(codes) if codes else "anywhere"
    return (
        f"pod:      {pod_id}\n"
        f"region:   {region}\n"
        f"base_url: {url}\n"
        f"boot log: https://{pod_id}-8080.proxy.runpod.net/boot.log\n"
        f"arms served as LoRA modules:\n{served}\n\n"
        f"Boot takes ~20-30 min (base download ~55GB, adapters, vLLM load). Poll:\n"
        f"  uv run python scripts/runpod_capability.py status --pod {pod_id}\n\n"
        f"THEN TEAR IT DOWN — it bills by the second:\n"
        f"  uv run python scripts/runpod_capability.py down --pod {pod_id}"
    )


def status(pod: str) -> str:
    """Report pod state and whether the OpenAI endpoint is answering yet."""
    import requests

    info = call("GET", f"/pods/{pod}")
    url = f"https://{pod}-8000.proxy.runpod.net/v1/models"
    line = f"status: {info.get('desiredStatus')}  machine: {info.get('machine', {}).get('gpuTypeId')}"
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            names = [m["id"] for m in r.json().get("data", [])]
            return f"{line}\nSERVING: {names}"
        return f"{line}\nendpoint HTTP {r.status_code} (still booting)"
    except Exception as exc:  # noqa: BLE001 - any failure here means "not up yet"
        return f"{line}\nendpoint not reachable yet ({type(exc).__name__})"


def down(pod: str) -> str:
    """Terminate the pod. Always run this; the pod bills by the second."""
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}"


def list_pods() -> str:
    """List active pods, so a forgotten one cannot bill quietly."""
    pods = call("GET", "/pods")
    rows = pods if isinstance(pods, list) else pods.get("data", [])
    if not rows:
        return "no active pods"
    return "\n".join(f"{p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}" for p in rows)


if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "down": down, "list": list_pods})
