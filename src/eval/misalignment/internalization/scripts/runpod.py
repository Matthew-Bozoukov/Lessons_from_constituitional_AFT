# ABOUTME: Provision, monitor, and TEAR DOWN a RunPod GPU serving a merged LoRA checkpoint.
# ABOUTME: RunPod proxies pod ports over HTTPS, so arm B needs no SSH tunnel - just a base_url.

"""Stand up arm B's serving endpoint on RunPod.

    export RUNPOD_API_KEY=rpa_...            # or put it in .env
    uv run python -m src.eval.misalignment.internalization.scripts.runpod up --gpu "NVIDIA H100 80GB HBM3"
    uv run python -m src.eval.misalignment.internalization.scripts.runpod status --pod <id>
    uv run python -m src.eval.misalignment.internalization.scripts.runpod down --pod <id>      # ALWAYS do this

The pod boots, installs vLLM + peft, merges `adapter` into `base` with
`src.eval.misalignment.internalization.scripts.merge_lora`, and serves the merged weights on port 8000. Merging rather
than serving the adapter at runtime is deliberate: vLLM's LoRA path is unproven for Qwen3.6's
hybrid vision-language architecture, and a merged checkpoint is just an ordinary model.

The endpoint is reachable at `https://<pod-id>-8000.proxy.runpod.net/v1` — pass that as
`target.base_url` and no tunnel is involved.

COST: a pod bills by the second from the moment it starts, including the ~25 minutes it
spends downloading and merging. `down` is not optional; `status` prints the running spend.
"""

from __future__ import annotations

import json
import os
import time

import fire
import requests

# The REST client moved to src/infra/runpod.py (2026-08-27) so `uv run chat` and this
# module share one; re-exported here because scripts/gpu/* and scratch/* import it from here.
from src.infra.runpod import REST, call  # noqa: F401

# Weights (~55GB bf16) + the merged copy (~55GB) + room for the image and HF cache.
DEFAULT_DISK_GB = 250
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"



def _bootstrap(base: str, adapter: str, served_name: str) -> str:
    """Return the pod's startup script.

    Everything is logged to /workspace/boot.log and the server is started detached, so
    `status` can distinguish "still merging" from "died during merge" - a pod that failed
    silently would otherwise bill until someone noticed.
    """
    # NO credentials are placed on the pod. The base model and the adapter are both public
    # HF repos, so no token is needed - and a pod that holds no secret cannot leak one.
    # Overriding dockerStartCmd REPLACES the image entrypoint, which is what normally
    # installs PUBLIC_KEY and starts sshd. Without re-doing it here the pod has no SSH and
    # a failed boot is undiagnosable - which is exactly how the first two attempts were lost.
    return f"""set -euxo pipefail
exec > >(tee -a /workspace/boot.log) 2>&1
mkdir -p ~/.ssh && [ -n "${{PUBLIC_KEY:-}}" ] && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true
(apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1; \
 mkdir -p /run/sshd && /usr/sbin/sshd -D &) || echo "sshd unavailable"
export HF_HOME=/workspace/hf
pip install --no-cache-dir -q "vllm>=0.8.5" "transformers>=4.51.3" peft accelerate fire huggingface_hub
python - <<'PY'
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
base = "{base}"
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, "{adapter}")
model = model.merge_and_unload()
model.save_pretrained("/workspace/merged")
AutoTokenizer.from_pretrained(base).save_pretrained("/workspace/merged")
try:
    AutoProcessor.from_pretrained(base).save_pretrained("/workspace/merged")
except Exception as e:
    print("no processor:", e)
print("MERGE_COMPLETE")
PY
nohup vllm serve /workspace/merged \
  --served-model-name {served_name} \
  --port 8000 --host 0.0.0.0 \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 --trust-remote-code \
  </dev/null >/workspace/vllm.log 2>&1 &
echo BOOTSTRAP_DONE
"""


def up(
    gpu: str = "NVIDIA H100 80GB HBM3",
    base: str = "Qwen/Qwen3.6-27B",
    adapter: str = "matboz/qwen3.6-27b-difficult-advice-tulu-lora",
    served_name: str = "qwen36-difficult-advice",
    name: str = "internalization-arm-b",
    disk_gb: int = DEFAULT_DISK_GB,
    image: str = DEFAULT_IMAGE,
    cloud: str = "SECURE",
) -> str:
    """Create a pod that merges the adapter and serves it on port 8000.

    Args:
        gpu: GPU type id. `gpus` lists what is available with 80GB+.
        base: Base model repo id.
        adapter: LoRA repo id to merge in.
        served_name: --served-model-name, which becomes `target.model`.
        name: Pod name.
        disk_gb: Container disk. Must hold the base AND the merged copy.
        image: Container image.
        cloud: SECURE or COMMUNITY.

    Returns:
        The pod id and the base_url to point arm B at.
    """
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", _bootstrap(base, adapter, served_name)],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
    return (
        f"pod:      {pod_id}\n"
        f"base_url: {url}\n"
        f"model:    {served_name}\n\n"
        f"Boot takes ~25 min (download ~55GB, merge, load). Poll with:\n"
        f"  uv run python -m src.eval.misalignment.internalization.scripts.runpod status --pod {pod_id}\n\n"
        f"Then run arm B:\n"
        f"  uv run python -m src.eval.misalignment.internalization.cli run --config qwen36_lora.yaml \\\n"
        f"    --base-url {url} --model {served_name}\n\n"
        f"THEN TEAR IT DOWN - it bills by the second:\n"
        f"  uv run python -m src.eval.misalignment.internalization.scripts.runpod down --pod {pod_id}"
    )


def train_up(
    gpu: str = "NVIDIA H100 80GB HBM3",
    name: str = "tulu-control-sft",
    disk_gb: int = 200,
    image: str = DEFAULT_IMAGE,
    cloud: str = "SECURE",
    pubkey_path: str = "~/.ssh/id_ed25519.pub",
) -> str:
    """Create an SSH-enabled pod for a QLoRA training run.

    Deliberately does NOT set `dockerStartCmd`. RunPod's official images start sshd from
    their own entrypoint when `PUBLIC_KEY` is set and then idle; overriding the entrypoint
    replaces that, so the container brings up no sshd and exits as soon as the override
    finishes. Dependencies are installed over SSH instead (see `train_setup`), which also
    means a bad install command costs a re-run rather than a re-provision.

    Args:
        gpu: GPU type id.
        name: Pod name.
        disk_gb: Container disk; must hold the base model plus checkpoints.
        image: Container image.
        cloud: SECURE or COMMUNITY.
        pubkey_path: Public key authorised for SSH on the pod.

    Returns:
        The pod id and how to reach it.
    """
    key_file = os.path.expanduser(pubkey_path)
    if not os.path.exists(key_file):
        raise RuntimeError(f"No public key at {key_file}. Pass --pubkey-path.")
    with open(key_file) as fh:
        pubkey = fh.read().strip()

    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "cloudType": cloud,
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "PUBLIC_KEY": pubkey},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pod_id = pod.get("id") or pod.get("podId", "")
    return (
        f"pod: {pod_id}\n\n"
        f"Wait for SSH, then rsync the repo to /root/work:\n"
        f"  uv run python -m src.eval.misalignment.internalization.scripts.runpod ssh_addr --pod {pod_id}\n\n"
        f"TEAR IT DOWN when the adapter is pulled - it bills by the second:\n"
        f"  uv run python -m src.eval.misalignment.internalization.scripts.runpod down --pod {pod_id}"
    )


def ssh_addr(pod: str) -> str:
    """Print the SSH host/port for a pod, and a ready-to-paste ssh command.

    Args:
        pod: Pod id.

    Returns:
        The ssh connection details, or a note that SSH is not mapped yet.
    """
    info = call("GET", f"/pods/{pod}")
    for m in info.get("portMappings") or []:
        # portMappings is either a list of dicts or a {privatePort: publicPort} map.
        if isinstance(m, dict) and str(m.get("privatePort")) == "22":
            host, port = info.get("publicIp", ""), m.get("publicPort")
            return f"host: {host}\nport: {port}\nssh:  ssh -p {port} root@{host}"
    mappings = info.get("portMappings")
    if isinstance(mappings, dict) and "22" in mappings:
        host, port = info.get("publicIp", ""), mappings["22"]
        return f"host: {host}\nport: {port}\nssh:  ssh -p {port} root@{host}"
    return (
        f"status: {info.get('desiredStatus', '?')}\n"
        f"SSH not mapped yet. Poll again in ~30s.\n"
        f"raw portMappings: {json.dumps(mappings)}"
    )


def status(pod: str) -> str:
    """Report pod state and whether the vLLM endpoint is answering yet.

    Args:
        pod: Pod id.

    Returns:
        A status line, including whether the model endpoint is live.
    """
    info = call("GET", f"/pods/{pod}")
    url = f"https://{pod}-8000.proxy.runpod.net/v1"
    ready, detail = False, "not answering yet (still downloading, merging, or loading)"
    try:
        resp = requests.get(f"{url}/models", timeout=15)
        if resp.ok:
            ready = True
            detail = ", ".join(m["id"] for m in resp.json().get("data", []))
    except requests.RequestException as e:
        detail = f"not reachable: {type(e).__name__}"
    return (
        f"pod:     {pod}\n"
        f"status:  {info.get('desiredStatus', '?')} / {info.get('lastStatusChange', '')}\n"
        f"cost:    ${info.get('costPerHr', 0)}/hr\n"
        f"url:     {url}\n"
        f"serving: {'YES -> ' + detail if ready else 'no  -> ' + detail}"
    )


def wait(pod: str, timeout_min: int = 45, interval_s: int = 60) -> str:
    """Block until the endpoint serves, or the timeout expires.

    Args:
        pod: Pod id.
        timeout_min: Give up after this long.
        interval_s: Poll interval.

    Returns:
        The final status.
    """
    url = f"https://{pod}-8000.proxy.runpod.net/v1/models"
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=15).ok:
                return f"READY: {url}"
        except requests.RequestException:
            pass
        print(f"  not ready, retrying in {interval_s}s...", flush=True)
        time.sleep(interval_s)
    return f"TIMEOUT after {timeout_min} min. Check the pod's /workspace/boot.log before re-trying."


def gpus(min_gb: int = 80) -> str:
    """List GPU types with at least `min_gb` of memory, with prices."""
    rows = call("GET", "/gputypes")
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    out = []
    for g in rows:
        mem = g.get("memoryInGb") or 0
        if mem >= min_gb:
            out.append(
                f"  {str(g.get('id')):36s} {mem:>4}GB  "
                f"secure ${g.get('securePrice') or 0:.2f}/hr  "
                f"community ${g.get('communityPrice') or 0:.2f}/hr"
            )
    return "\n".join(out) or f"No GPU types with >= {min_gb}GB returned."


def down(pod: str) -> str:
    """Terminate a pod. Billing continues until this succeeds.

    Args:
        pod: Pod id.

    Returns:
        Confirmation.
    """
    call("DELETE", f"/pods/{pod}")
    return f"terminated {pod}. Confirm with `list` that nothing is still running."


def list_pods() -> str:
    """List every pod on the account, so nothing is left billing unnoticed."""
    rows = call("GET", "/pods")
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    if not rows:
        return "no pods running"
    return "\n".join(
        f"  {p.get('id')}  {p.get('name'):28s} {p.get('desiredStatus'):10s} ${p.get('costPerHr', 0)}/hr"
        for p in rows
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "up": up,
            "train_up": train_up,
            "ssh_addr": ssh_addr,
            "status": status,
            "wait": wait,
            "down": down,
            "gpus": gpus,
            "list": list_pods,
        }
    )
