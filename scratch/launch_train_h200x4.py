# ABOUTME: Publishes a code bundle and launches a 4xH200 RunPod DDP LoRA training run for
# ABOUTME: any arm, with a smoke gate through the same torchrun path before the paid run.

"""Launch a LoRA training arm on 4xH200.

The pod pulls its code from a small HF bundle and its data from the already-public mixture
repo, so nothing here depends on the pod being able to reach GitHub. Two deliberate
choices, both learned the hard way:

  - **The smoke run gates the full run inside the same boot script.** `--smoke` trains 2
    steps through the identical DDP path; if it fails the pod stops before spending real
    money, and its log says why. A smoke run driven from the laptop instead would have to
    survive the laptop.
  - **Nothing self-terminates.** Pod self-teardown has failed silently on this account
    before (a pod idled ~9 minutes past a successful push), so `down` is a separate,
    verified step and `status` prints what the run is actually doing.

    uv run python scratch/launch_memself_h200x4.py bundle
    uv run python scratch/launch_memself_h200x4.py up
    uv run python scratch/launch_memself_h200x4.py status --pod <id>
    uv run python scratch/launch_memself_h200x4.py down --pod <id>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.internalization.scripts.runpod import call  # noqa: E402

BUNDLE_REPO = "LASR-Callum/2026-08-08-train-h200x4-code-selfreflect"
MIXTURE_REPO = "LASR-Callum/2026-08-06-qwen36-table2-80-selfreflect-20-10k-train"
MIXTURE_FILE = "mixture.jsonl"
CONFIG = "configs/train/lora_qwen36_table2_80_selfreflect_20.yaml"
BASE = "Qwen/Qwen3.6-27B"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU = "NVIDIA H200"
NPROC = 4

# Same exclusions as the previous bundle: without them the tarball picks up the vendored
# SWE-bench fixtures and every __pycache__, turning ~0.5 MB into 164 MB per pod start.
SKIP = ("__pycache__", ".git", "third_party", "swebench_mini", ".pytest_cache")


def bundle(repo: str = BUNDLE_REPO, stage: str = "/tmp/train_bundle",
           mixture_repo: str = MIXTURE_REPO, config: str = CONFIG,
           base: str = BASE, constitution: str = "claude_distilled_09_principles_mid_20260804",
           date_generated: str = "2026-08-07") -> None:
    """Pack the trainer's import closure and publish it as a public HF dataset repo.

    Args:
        repo: Destination HF dataset repo id.
        stage: Scratch directory to assemble in.
        mixture_repo: Mixture this bundle is meant to train on; named on the card.
        config: Train config inside the tarball; named on the card.
        base: Base model; named on the card.
        constitution: Constitution the mixture's synthetic half targets.
        date_generated: ISO date for the card.
    """
    load_dotenv(override=True)
    from huggingface_hub import HfApi

    S = Path(stage)
    shutil.rmtree(S, ignore_errors=True)
    S.mkdir(parents=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "unknown"

    def _lean(info):
        parts = Path(info.name).parts
        return None if any(s in parts for s in SKIP) or info.name.endswith(".pyc") else info

    with tarfile.open(S / "code.tar.gz", "w:gz") as t:
        for p in ("scripts/train/train_lora.py", "src", "configs"):
            t.add(p, arcname=p, filter=_lean)
    size = (S / "code.tar.gz").stat().st_size / 1e6

    (S / "README.md").write_text(f"""---
license: apache-2.0
tags: [sft, lora, qwen3.6, training-bundle]
---

# Training bundle — 4xH200 LoRA run

`code.tar.gz` only: the trainer, `src/` and `configs/`. The mixture is pulled separately
from [`{mixture_repo}`](https://huggingface.co/datasets/{mixture_repo}).

| field | value |
|---|---|
| `experiment` | Code bundle for a 1-epoch LoRA SFT of {base}; see the mixture repo for composition |
| `date_generated` | {date_generated} |
| `constitution` | `{constitution}` — via the mixture's synthetic side; see the mixture card |
| `source_repo` | `teaching_claude_why_replication` @ `{sha}` |
| `models` | trains `{base}` |
| `generation_config` | n/a — this repo carries code, not generated data |
| `schema` | `code.tar.gz`: `scripts/train/train_lora.py`, `src/`, `configs/` |
| `provenance` | `uv run python scratch/launch_memself_h200x4.py bundle` |

Training config inside the tarball: `{config}`.
""")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(S), repo_id=repo, repo_type="dataset",
                      commit_message=f"code bundle @ {sha[:8]}")
    print(f">>> https://huggingface.co/datasets/{repo}   code.tar.gz {size:.1f} MB")


def _bootstrap(hf_token: str, wandb_key: str, bundle_repo: str = BUNDLE_REPO,
               mixture_repo: str = MIXTURE_REPO, mixture_file: str = MIXTURE_FILE,
               config: str = CONFIG, nproc: int = NPROC, base: str = BASE) -> str:
    """Pod startup script: fetch, smoke-gate, then train under torchrun.

    Args:
        hf_token: Token for the adapter push and checkpoint mirroring.
        wandb_key: Weights & Biases key; the config logs the loss curve there.
        bundle_repo: Public HF dataset repo holding code.tar.gz.
        mixture_repo: HF dataset repo holding the mixture. May be PRIVATE — HF_TOKEN is
            exported below, so the download authenticates.
        mixture_file: Mixture filename inside that repo.
        config: Train config path inside the code tarball.
        nproc: GPUs per node; torchrun ranks.
        base: Base model to pre-download.

    Returns:
        The bash script the pod runs as its entrypoint.
    """
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
# xtrace OFF around the credentials. /workspace is served over the pod's PUBLIC HTTPS
# proxy on :8080, so anything `set -x` echoes into boot.log is world-readable for the
# life of the pod -- these two exports would publish both tokens.
set +x
export HF_TOKEN={hf_token}
export HUGGINGFACE_API_KEY={hf_token}
export WANDB_API_KEY={wandb_key}
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# PYTHONPATH: every rank imports `src.*`, and without this torchrun's children start in a
# cwd that is not on sys.path -> ModuleNotFoundError several minutes into the boot.
export PYTHONPATH=/workspace/repo
# Deliberately ONE line each. A backslash continuation inside this f-string is eaten by Python,
# not passed to bash, so what looks like a continued command here is a different command
# there -- the failure mode that once truncated a serve command into a foreground hang.
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" datasets accelerate omegaconf fire huggingface_hub hf_transfer wandb
hf download {base} >/dev/null
hf download {bundle_repo} --repo-type dataset --local-dir /workspace/bundle
hf download {mixture_repo} --repo-type dataset --local-dir /workspace/mix
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
mkdir -p /workspace/repo/data && cp /workspace/mix/{mixture_file} /workspace/repo/data/mixture.jsonl
cd /workspace/repo
wc -l data/mixture.jsonl
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo SMOKE_STARTING
# The smoke run uses the SAME torchrun path as the real run: it proves DDP, the mask gate
# and the 4-rank device map before any long spend. `set -e` makes a failure here abort the
# script, leaving a readable log instead of an 8-hour run that was broken from step 1.
WANDB_MODE=disabled torchrun --nproc_per_node={nproc} scripts/train/train_lora.py --config {config} --smoke 2>&1 | tee /workspace/smoke.log
echo SMOKE_OK
echo TRAINING_STARTING
(torchrun --nproc_per_node={nproc} scripts/train/train_lora.py --config {config} 2>&1 | tee /workspace/train.log) || echo TRAINING_FAILED
# /workspace/out, NOT /workspace/repo/output: every train config writes its output_dir
# under /workspace/out, so the old path tarred an empty directory and `|| true` hid it.
# This is the fallback if the HF push fails; the pod holds a token and pushes normally.
tar -czf /workspace/adapter.tar.gz -C /workspace/out . || true
ls -la /workspace/adapter.tar.gz || true
echo TRAINING_DONE
sleep infinity
"""


def _validate(script: str, nproc: int = NPROC, config: str = CONFIG) -> None:
    """Refuse to launch on a bootstrap that is not valid bash or lost a command.

    A pod bills from the moment it starts, so every check that can happen before `POST
    /pods` happens here. `bash -n` parses without executing; the marker check catches a
    command silently mangled by f-string interpolation, and the one-line check catches a
    backslash continuation eaten by Python before bash ever sees it.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bootstrap is not valid bash:\n{r.stderr}"
    for needle in (f"--nproc_per_node={nproc}", f"--config {config}", "--smoke",
                   "SMOKE_OK", "TRAINING_DONE", "mixture.jsonl"):
        assert needle in script, f"bootstrap lost {needle!r}"
    # Every torchrun must sit on ONE line: a swallowed continuation would drop the flags
    # after the break and silently train a different run than the one reviewed.
    runs = [ln for ln in script.splitlines() if "train_lora.py" in ln]
    assert len(runs) == 2, f"expected smoke + full train invocation, found {len(runs)}"
    for ln in runs:
        assert ln.rstrip().endswith(("smoke.log", "TRAINING_FAILED")), f"truncated: {ln}"
    print(f">>> bootstrap validated: {len(script.splitlines())} lines, 2 torchrun calls")


def up(name: str = "train-selfreflect-20-80-h200x4", disk_gb: int = 400,
       cloud: str = "SECURE", gpu: str = GPU, image: str = IMAGE,
       nproc: int = NPROC, bundle_repo: str = BUNDLE_REPO,
       mixture_repo: str = MIXTURE_REPO, mixture_file: str = MIXTURE_FILE,
       config: str = CONFIG, base: str = BASE) -> None:
    """Create the 4-GPU training pod.

    Args:
        name: Pod name; prefixed so it is identifiable on the shared account.
        disk_gb: Container disk — base model ~55GB, HF cache, checkpoints.
        cloud: SECURE or COMMUNITY.
        gpu: RunPod GPU type id.
        image: Container image.
        nproc: GPUs per node; torchrun ranks and RunPod gpuCount.
        bundle_repo: Public HF repo holding code.tar.gz.
        mixture_repo: HF repo holding the mixture (may be private).
        mixture_file: Mixture filename inside that repo.
        config: Train config path inside the tarball.
        base: Base model.
    """
    load_dotenv(override=True)
    script = _bootstrap(os.environ["HF_TOKEN"], os.environ["WANDB_API_KEY"],
                        bundle_repo, mixture_repo, mixture_file, config, nproc, base)
    _validate(script, nproc, config)
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": nproc,
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["8080/http"],
        "cloudType": cloud,
        "dockerStartCmd": ["bash", "-lc", script],
        "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    }
    pod = call("POST", "/pods", data=json.dumps(payload))
    pid = pod["id"]
    print(f">>> pod {pid}  {gpu} x{nproc}")
    print(f"    boot : https://{pid}-8080.proxy.runpod.net/boot.log")
    print(f"    smoke: https://{pid}-8080.proxy.runpod.net/smoke.log")
    print(f"    train: https://{pid}-8080.proxy.runpod.net/train.log")
    print(f"    DOWN : uv run python scratch/launch_memself_h200x4.py down --pod {pid}")


def status(pod: str, tail: int = 12) -> None:
    """Print the run's phase and the tail of whichever log is live.

    Args:
        pod: Pod id.
        tail: Lines of log to show.
    """
    load_dotenv(override=True)
    base = f"https://{pod}-8080.proxy.runpod.net"
    boot = requests.get(f"{base}/boot.log", timeout=30).text if _ok(f"{base}/boot.log") else ""
    phase = "booting"
    for marker in ("SMOKE_STARTING", "SMOKE_OK", "TRAINING_STARTING",
                   "TRAINING_FAILED", "TRAINING_DONE"):
        if marker in boot:
            phase = marker
    print(f">>> pod {pod}  phase={phase}")
    for f in ("train.log", "smoke.log", "boot.log"):
        if _ok(f"{base}/{f}"):
            txt = requests.get(f"{base}/{f}", timeout=30).text
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            print(f"--- {f} ({len(lines)} lines) ---")
            print("\n".join(lines[-tail:]))
            break


def _ok(url: str) -> bool:
    """True if the URL serves a real file rather than the directory listing or a 404."""
    try:
        r = requests.get(url, timeout=30)
        return r.status_code == 200 and "<html" not in r.text[:200].lower()
    except requests.RequestException:
        return False


def down(pod: str) -> None:
    """Destroy the pod and print the account's remaining active pods as verification.

    Args:
        pod: Pod id to terminate.
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
