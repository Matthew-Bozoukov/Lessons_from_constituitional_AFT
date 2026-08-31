# ABOUTME: Launch a 2xH200 DDP LoRA training run on vast.ai, with a smoke gate, durable
# ABOUTME: log upload to HF, and label-matched self-destruct so nothing bills unattended.

"""Train a LoRA arm on vast.ai across 2xH200.

    uv run python scratch/launch_train_vast_h200x2.py up --config <path> --label <unique>
    uv run python scratch/launch_train_vast_h200x2.py status --instance <id>
    uv run python scratch/launch_train_vast_h200x2.py down --instance <id>

Structure is lifted from scratch/launch_train_h200x4.py (RunPod): the pod pulls code from a
small public HF bundle, runs a 2-step `--smoke` through the SAME torchrun path before the
paid run, and nothing about the run depends on the laptop surviving. Two additions, both
because this variant is meant to be walked away from:

  - **Logs are uploaded to HF before teardown.** Destroying the instance destroys its disk,
    so the boot/smoke/train logs go to the bundle repo first. Evidence outlives the box.
  - **The instance destroys ITSELF, matched by label.** The RunPod launcher deliberately
    keeps teardown manual because self-teardown once failed silently there. That rule is
    right when a human is watching; it is the wrong trade for an unattended run, where the
    failure mode is billing all night. The instance finds itself in the account listing by
    the unique label baked into this script — no guessing at container env vars, and no
    "destroy everything on the account", which would kill a bystander instance.

Self-destruct fires after BOTH success and failure, because a crashed run that bills until
morning is the exact outcome this is here to prevent — the logs are already on HF by then.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = "https://console.vast.ai/api/v0"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
BASE = "Qwen/Qwen3.6-27B"
NPROC = 2


def _headers(key: str) -> dict:
    """Auth header for the vast REST API."""
    return {"Authorization": f"Bearer {key}"}


def _bootstrap(hf_token: str, wandb_key: str, vast_key: str, bundle_repo: str,
               mixture_repo: str, mixture_file: str, config: str, label: str,
               nproc: int, base: str, self_destruct: bool) -> str:
    """Build the onstart script.

    Args:
        hf_token: HF token — adapter push and log upload.
        wandb_key: W&B key for the loss curve.
        vast_key: vast API key, used ONLY for the self-destruct call.
        bundle_repo: Public HF repo holding code.tar.gz.
        mixture_repo: HF repo holding the mixture (may be private).
        mixture_file: Mixture filename in that repo.
        config: Train config path inside the tarball.
        label: Unique instance label; the self-destruct matches on it.
        nproc: torchrun ranks.
        base: Base model.
        self_destruct: Whether to destroy the instance when the run ends.

    Returns:
        The bash script vast runs as onstart.
    """
    destruct = f"""
echo SELF_DESTRUCT_STARTING
python3 - <<'PYEOF' || echo SELF_DESTRUCT_FAILED
import os, requests
k = os.environ["VAST_API_KEY"]
h = {{"Authorization": f"Bearer {{k}}"}}
r = requests.get("{API}/instances/", headers=h, timeout=60).json()
me = [i for i in r.get("instances", []) if i.get("label") == "{label}"]
print("matched instances:", [i["id"] for i in me])
for i in me:
    d = requests.delete(f"{API}/instances/{{i['id']}}/", headers=h, timeout=60)
    print("destroy", i["id"], d.status_code)
PYEOF
""" if self_destruct else "\necho SELF_DESTRUCT_DISABLED\n"

    return f"""#!/bin/bash
mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
cd /workspace
export HF_HOME=/workspace/hf
# xtrace OFF around credentials so they never land in boot.log, which is uploaded to HF.
set +x
export HF_TOKEN={hf_token}
export HUGGINGFACE_API_KEY={hf_token}
export WANDB_API_KEY={wandb_key}
export VAST_API_KEY={vast_key}
set -x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/repo
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" datasets accelerate omegaconf fire huggingface_hub wandb requests
hf download {base} >/dev/null
hf download {bundle_repo} --repo-type dataset --local-dir /workspace/bundle
hf download {mixture_repo} --repo-type dataset --local-dir /workspace/mix
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
mkdir -p /workspace/repo/data && cp /workspace/mix/{mixture_file} /workspace/repo/data/mixture.jsonl
cd /workspace/repo
wc -l data/mixture.jsonl
echo SMOKE_STARTING
WANDB_MODE=disabled torchrun --nproc_per_node={nproc} scripts/train/train_lora.py --config {config} --smoke 2>&1 | tee /workspace/smoke.log
echo SMOKE_OK
echo TRAINING_STARTING
(torchrun --nproc_per_node={nproc} scripts/train/train_lora.py --config {config} 2>&1 | tee /workspace/train.log) || echo TRAINING_FAILED
tar -czf /workspace/adapter.tar.gz -C /workspace/out . || true
echo TRAINING_DONE
echo UPLOADING_LOGS
python3 - <<'PYEOF' || echo LOG_UPLOAD_FAILED
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ("boot.log", "smoke.log", "train.log"):
    p = f"/workspace/{{f}}"
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"logs/{label}/{{f}}",
                        repo_id="{bundle_repo}", repo_type="dataset")
        print("uploaded", f)
PYEOF
echo LOGS_UPLOADED
{destruct}
sleep infinity
"""


def _validate(script: str, nproc: int, config: str, self_destruct: bool) -> None:
    """Refuse to launch on a bootstrap that is not valid bash or lost a command.

    An instance bills from the moment it starts, so everything checkable happens here.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"onstart is not valid bash:\n{r.stderr}"
    needles = [f"--nproc_per_node={nproc}", f"--config {config}", "--smoke",
               "SMOKE_OK", "TRAINING_DONE", "LOGS_UPLOADED", "mixture.jsonl"]
    if self_destruct:
        needles.append("SELF_DESTRUCT_STARTING")
    for needle in needles:
        assert needle in script, f"onstart lost {needle!r}"
    runs = [ln for ln in script.splitlines() if "train_lora.py" in ln]
    assert len(runs) == 2, f"expected smoke + full train invocation, found {len(runs)}"
    for ln in runs:
        assert ln.rstrip().endswith(("smoke.log", "TRAINING_FAILED")), f"truncated: {ln}"
    print(f">>> onstart validated: {len(script.splitlines())} lines, 2 torchrun calls, "
          f"self_destruct={self_destruct}")


def up(config: str, label: str, bundle_repo: str, mixture_repo: str, mixture_file: str,
       gpu_name: str = "H200", nproc: int = NPROC, disk_gb: int = 400,
       image: str = IMAGE, base: str = BASE, self_destruct: bool = True,
       max_dph: float = 12.0) -> None:
    """Rent the cheapest matching multi-GPU box and start the run on it.

    Args:
        config: Train config path inside the code tarball.
        label: Unique label for this run; the self-destruct matches on it.
        bundle_repo: Public HF repo with code.tar.gz; logs are uploaded here too.
        mixture_repo: HF repo holding the mixture.
        mixture_file: Mixture filename inside it.
        gpu_name: vast GPU model filter.
        nproc: GPUs, and torchrun ranks.
        disk_gb: Container disk — base model ~55GB plus cache and checkpoints.
        image: Container image.
        base: Base model.
        self_destruct: Destroy the instance when the run ends. Leave ON for unattended runs.
        max_dph: Refuse offers above this $/hr, so a price spike cannot silently cost more.
    """
    load_dotenv(override=True)
    key = os.environ["VAST_API_KEY"]

    q = (f"gpu_name={gpu_name} num_gpus={nproc} rentable=true "
         f"disk_space>={disk_gb} inet_down>=500 reliability>=0.98")
    out = subprocess.run(["uvx", "vastai", "search", "offers", q, "-o", "dph+",
                          "--raw", "--api-key", key], capture_output=True, text=True)
    assert out.returncode == 0, f"offer search failed:\n{out.stderr}"
    offers = json.loads(out.stdout)
    assert offers, f"no offers matching: {q}"
    best = offers[0]
    dph = float(best["dph_total"])
    assert dph <= max_dph, f"cheapest offer ${dph:.2f}/hr exceeds max_dph ${max_dph:.2f}"
    print(f">>> offer {best['id']}: {nproc}x{gpu_name} ${dph:.2f}/hr "
          f"{best.get('geolocation')} reliability {best.get('reliability2', 0):.3f}")

    script = _bootstrap(os.environ["HF_TOKEN"], os.environ["WANDB_API_KEY"], key,
                        bundle_repo, mixture_repo, mixture_file, config, label,
                        nproc, base, self_destruct)
    _validate(script, nproc, config, self_destruct)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        onstart = f.name
    cmd = ["uvx", "vastai", "create", "instance", str(best["id"]), "--image", image,
           "--disk", str(disk_gb), "--onstart", onstart, "--label", label,
           "--raw", "--api-key", key]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"create failed:\n{res.stdout}\n{res.stderr}"
    info = json.loads(res.stdout)
    print(f">>> created: {info}")
    print(f"    label   : {label}")
    print(f"    est cost: ${dph:.2f}/hr")
    print(f"    status  : uv run python scratch/launch_train_vast_h200x2.py status "
          f"--instance <id>")
    print(f"    DOWN    : uv run python scratch/launch_train_vast_h200x2.py down "
          f"--instance <id>")


def _instances(key: str) -> list[dict]:
    """All instances on the account."""
    r = requests.get(f"{API}/instances/", headers=_headers(key), timeout=60)
    r.raise_for_status()
    return r.json().get("instances", [])


def status(instance: str = "", label: str = "", tail: int = 15) -> None:
    """Print instance state and the tail of the onstart log.

    Args:
        instance: Instance id. Omit to match on label instead.
        label: Label to match when no id is given.
        tail: Lines of log to show.
    """
    load_dotenv(override=True)
    key = os.environ["VAST_API_KEY"]
    ins = _instances(key)
    if instance:
        ins = [i for i in ins if str(i["id"]) == str(instance)]
    elif label:
        ins = [i for i in ins if i.get("label") == label]
    if not ins:
        print(">>> no matching instance (already destroyed, or never created)")
        return
    for i in ins:
        print(f">>> {i['id']}  {i.get('actual_status')}  "
              f"${float(i.get('dph_total', 0)):.2f}/hr  label={i.get('label')}")
        out = subprocess.run(["uvx", "vastai", "logs", str(i["id"]), "--api-key", key],
                             capture_output=True, text=True)
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        phase = "booting"
        for m in ("SMOKE_STARTING", "SMOKE_OK", "TRAINING_STARTING", "TRAINING_FAILED",
                  "TRAINING_DONE", "LOGS_UPLOADED", "SELF_DESTRUCT_STARTING"):
            if any(m in ln for ln in lines):
                phase = m
        print(f"    phase={phase}")
        print("\n".join(lines[-tail:]))


def down(instance: str = "", label: str = "") -> None:
    """Destroy an instance and verify the account is clear.

    Args:
        instance: Instance id.
        label: Label to match when no id is given.
    """
    load_dotenv(override=True)
    key = os.environ["VAST_API_KEY"]
    targets = _instances(key)
    if instance:
        targets = [i for i in targets if str(i["id"]) == str(instance)]
    elif label:
        targets = [i for i in targets if i.get("label") == label]
    for i in targets:
        r = requests.delete(f"{API}/instances/{i['id']}/", headers=_headers(key), timeout=60)
        print(f">>> destroy {i['id']}: {r.status_code}")
    time.sleep(5)
    left = _instances(key)
    print(f">>> active instances now: {len(left)}")
    for i in left:
        print(f"    {i['id']} {i.get('label')} {i.get('actual_status')}")


if __name__ == "__main__":
    fire.Fire()
