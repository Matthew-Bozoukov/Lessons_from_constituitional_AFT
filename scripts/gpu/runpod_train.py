# ABOUTME: Launch/monitor/destroy a RunPod H100 that trains one LoRA arm from a public
# ABOUTME: HF bundle (code+data), no credentials on the pod. Run: uv run python scripts/gpu/runpod_train.py up

"""Train a LoRA arm on a throwaway RunPod H100.

The pod carries NO credentials: it downloads the base model and a public HF dataset
bundle (training code tarball + mixture jsonl), trains with the bundle passed as the
trainer's data_repo/data_file overrides (anonymous read — the bundle is public) and
push=false, then serves /workspace over HTTP on :8080 — the adapter is pulled back
through the proxy and pushed to HF from the local machine, where the token lives.

Bootstrap discipline is inherited from scripts/gpu/runpod_arena_hard.py (log server before
anything slow, trainer in the foreground of PID 1, `|| true` so a crash leaves a
readable log instead of a restart loop). See that file for why each line is the way it is.

    uv run python scripts/gpu/runpod_train.py up --bundle LASR-Callum/2026-07-31-qwen36-27b-tulu-0-100-train-mixture
    uv run python scripts/gpu/runpod_train.py status --pod <id>
    uv run python scripts/gpu/runpod_train.py down --pod <id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.infra.runpod import GPU, ProvisionSpec, call, provision_runpod  # noqa: E402

DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
DEFAULT_GPU = GPU   # one definition, in src/infra/runpod.py


def _bootstrap(base: str, bundle: str, train_config: str, mixture: str,
               gpu_count: int = 1) -> str:
    """Pod startup script: fetch bundle, train, expose the adapter over :8080.

    `gpu_count` > 1 launches the trainer under torchrun for DDP, which is how every
    dynamic-batching arm in configs/train/lora_qwen36_t2_9284_*.yaml was trained (their
    `# Run:` line is the same torchrun invocation). At 1 the launcher is plain python3,
    exactly as before.
    """
    launch = (f"torchrun --nproc_per_node={gpu_count}" if gpu_count > 1 else "python3")
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
# python3 -m pip, NOT bare pip: on this image `pip` and `python3` can resolve to
# different interpreters, and packages land where the trainer cannot import them.
# wandb is in this list even though WANDB_MODE=disabled above means nothing is ever
# logged. Several train configs declare `report_to: ["wandb"]` (they were written on boxes
# that had it), and transformers builds WandbCallback from that field BEFORE consulting
# WANDB_MODE -- so without the package the run dies at trainer construction with
# "WandbCallback requires wandb to be installed", ~25 minutes into a paid pod's boot.
# Installing it keeps those configs runnable verbatim, which matters when a pod arm has to
# stay byte-comparable to a sibling trained elsewhere; disabled mode keeps it a no-op with
# no account, no network and no auth.
# python-dotenv is here for the same class of reason as wandb: `src/huggingface.py::hf_token`
# calls `load_dotenv()` at import, so the trainer cannot even be imported without it. That
# call was ADDED on 2026-08-20 to fix a Windows driver that got a bare 401 from the Hub --
# a fix for one workflow that silently broke this one, because this pip list predates it and
# nothing re-checked it. It cost a pod on 2026-08-26, ~25 minutes in, with
# `ModuleNotFoundError: No module named 'dotenv'` after the base model had downloaded.
# PyYAML is transitively present today via transformers; named explicitly so a future
# dependency trim cannot take it away silently.
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" datasets accelerate omegaconf fire huggingface_hub hf_transfer wandb python-dotenv PyYAML
hf download {base} >/dev/null
hf download {bundle} code.tar.gz --repo-type dataset --local-dir /workspace/bundle
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
cd /workspace/repo
# The repo is not pip-installed on the pod (only its deps are), so `import src.*`
# resolves via the working directory, which script execution alone does not add.
export PYTHONPATH=/workspace/repo
echo TRAINING_STARTING
# The mixture comes straight from the (public) bundle repo via the trainer's own HF
# data path, so training_meta records the real repo@revision; push=false because the
# pod has no token — the driver pushes the pulled-back adapter.
set +e
({launch} scripts/train/train_lora.py --config {train_config} data_repo={bundle} data_file={mixture} push=false 2>&1 | tee /workspace/train.log)
TRAIN_RC=${{PIPESTATUS[0]}}
set -e
# Package what the trainer wrote (adapter + run_meta) for pull-back over :8080. The output
# root is READ FROM THE CONFIG, never assumed: these configs set output_dir to an absolute
# /workspace/out/<arm> path, so the old hardcoded /workspace/repo/output tarred nothing and
# produced a 20-byte archive next to a perfectly good adapter.
OUTDIR=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('{train_config}').get('output_dir') or 'output')")
echo "packaging adapter from $OUTDIR (trainer rc=$TRAIN_RC)"
tar -czf /workspace/adapter.tar.gz -C "$OUTDIR" . || true
# Say which happened. The trainer used to run under `|| true`, so a crash still reached the
# DONE marker and any watcher polling for it read a dead run as a success.
if [ "$TRAIN_RC" -ne 0 ]; then echo TRAINING_FAILED; fi
echo TRAINING_DONE
sleep infinity
"""


def up(
    bundle: str,
    train_config: str = "configs/train/2026-07-31_lora_qwen36_tulu_100.yaml",
    base: str = "Qwen/Qwen3.6-27B",
    gpu: str = DEFAULT_GPU,
    gpu_count: int = 1,
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
        gpu: RunPod GPU type id. NOTE the arms in configs/train/lora_qwen36_t2_9284_*
            CANNOT train on an H100 80GB: ModelProfile records a measured negative bound
            (a 1x~8k fwd+bwd OOMs 7.36 GiB short) and those mixtures carry rows at 8,191
            tokens. Pass an H200 type for them.
        gpu_count: GPUs on the pod. >1 runs the trainer under torchrun (DDP), matching how
            the dynamic-batching arms were trained.
        name: Pod name. Prefix it so it is distinguishable on the shared account.
        mixture: Mixture filename inside the bundle; passed to the trainer as its
            data_file override (the bundle repo is its data_repo).
        disk_gb: Container disk (base model ~55GB + HF cache + outputs).
        image: Container image.
        cloud: SECURE or COMMUNITY.
        countries: Placement restriction, "" for anywhere.

    Returns:
        Pod id and the URLs to watch.
    """
    pod_id = provision_runpod(
        ProvisionSpec(gpu=gpu, count=gpu_count, disk_gb=disk_gb, cloud=cloud, image=image,
                      cuda="", countries=countries),
        name=name,
        start_script=_bootstrap(base, bundle, train_config, mixture, gpu_count),
        ports=("8080/http", "22/tcp"),
    )
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
