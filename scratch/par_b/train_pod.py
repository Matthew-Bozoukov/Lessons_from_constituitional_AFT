# ABOUTME: Trains the post-action-retrospection (design B) 716 arm on ONE credential-free RunPod 2xH200 pod
# ABOUTME: with torchrun DDP (the da716 protocol), then pulls the adapter back and pushes it.

"""Train the PAR-716 arm on a throwaway 2xH200 RunPod pod (fork of the t10 driver).

    uv run python scratch/par_b/train_pod.py bundle
    uv run python scratch/par_b/train_pod.py up                      # seed 0, torchrun DDP
    uv run python scratch/par_b/train_pod.py up --gpus 2 --name nika-par716-seeds \
        --configs configs/train/lora_qwen36_t2_9284_par716_s1_dynbatch_1xh200.yaml,configs/train/lora_qwen36_t2_9284_par716_s2_dynbatch_1xh200.yaml
                                                                     # seeds 1+2, one per GPU
    uv run python scratch/par_b/train_pod.py status --pod <id>
    uv run python scratch/par_b/train_pod.py pull --pod <id>
    uv run python scratch/par_b/train_pod.py push
    uv run python scratch/less/teardown.py --pod <id>

A single-arm fork of scratch/less/train_arms.py. What differs, and why:
  * ONE arm, launched with `torchrun --nproc_per_node=2` rather than one process per
    GPU. The da716 organism this arm is compared against was trained "2xH200 DDP with
    token-budgeted dynamic batching, global batch 16, route_step over 2 ranks"
    (configs/train/lora_qwen36_t2_9284_da716_dynbatch_2xh200.yaml), and the batching
    protocol is part of what makes two arms comparable (docs/LOG.md 2026-08-16: the 4xH200
    batch-1 arm cannot serve as a control for the dynbatch arms). Same protocol here.
  * H200 only, for the same reason as train_arms.py: ModelProfile carries a MEASURED
    dynamic-batching budget for H200 and only a negative bound for H100.
Everything else -- credential-free pod, public code bundle, public data repo resolved
through the trainer's own data_repo path, adapter pulled over :8080 and pushed from the
machine that holds the token, `nika-` pod prefix so teardown.py can tell it apart on the
shared account -- is inherited unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.huggingface import card_markdown, hf_api, push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REST = "https://rest.runpod.io/v1"
BUNDLE = "LASR-Callum/2026-08-26-par716-arm-code-bundle"
BASE = "Qwen/Qwen3.6-27B"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"
GPU_LADDER = [("NVIDIA H200", "141GB"), ("NVIDIA H200 NVL", "143GB")]
COUNTRIES = [
    "US",
    "CA",
    "NL",
    "DE",
    "FR",
    "GB",
    "IE",
    "BE",
    "SE",
    "NO",
    "FI",
    "CH",
    "AT",
    "ES",
    "IT",
]

TRAIN_CONFIG = "configs/train/lora_qwen36_t2_9284_par716_dynbatch_2xh200.yaml"
ADAPTER_REPO = "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-r64-dynbatch"
OUT_NAME = "train_t2_9284_par716_dynbatch"  # basename of the config's output_dir
DEST = "output/adapters/par716"

# Every arm this driver knows, keyed by the basename of the config's output_dir:
# (train config, adapter repo, seed, launch). The seed-0 arm ran as 2-rank DDP (the da716
# protocol); the seed replicates run ONE process per GPU, two to a pod (`up --configs a,b`).
# Under dynamic batching both launches hand the same 16-example step to the optimizer.
ARMS = {
    OUT_NAME: (TRAIN_CONFIG, ADAPTER_REPO, 0, "torchrun --nproc_per_node=2"),
    "train_t2_9284_par716_s1_dynbatch": (
        "configs/train/lora_qwen36_t2_9284_par716_s1_dynbatch_1xh200.yaml",
        "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-s1-r64-dynbatch",
        1,
        "CUDA_VISIBLE_DEVICES=<gpu> python3 (one process on one H200)",
    ),
    # Arm 1 of the PAR coherence experiment (2026-08-28): same 716 rows, trained turn rewritten
    # so the reasoning ends on a first-person decision and the reply enacts it. Same 2-rank DDP
    # launch as seed 0 so the two arms are paired on protocol as well as data.
    "train_t2_9284_par716coh_dynbatch": (
        "configs/train/lora_qwen36_t2_9284_par716coh_dynbatch_2xh200.yaml",
        "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716coh-r64-dynbatch",
        0,
        "torchrun --nproc_per_node=2",
    ),
    "train_t2_9284_par716_s2_dynbatch": (
        "configs/train/lora_qwen36_t2_9284_par716_s2_dynbatch_1xh200.yaml",
        "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-s2-r64-dynbatch",
        2,
        "CUDA_VISIBLE_DEVICES=<gpu> python3 (one process on one H200)",
    ),
}

# Everything `scripts/train/train_lora.py` imports, plus the config. An explicit allowlist,
# never a directory walk: this tarball goes to a PUBLIC repo, and a walk would sweep up .env.
CODE = [
    "pyproject.toml",
    "scripts/train/train_lora.py",
    "src/__init__.py",
    "src/utils.py",
    "src/huggingface.py",
    "src/model_profile.py",
    "src/train/__init__.py",
    "src/train/train_lora.py",
    "src/train/masking.py",
    "src/train/mask_gate.py",
    "src/train/dynamic_batching.py",
    *[arm[0] for arm in ARMS.values()],
]


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
        "Content-Type": "application/json",
    }


def _bootstrap(configs: list[str], gpus: int) -> str:
    """Pod startup: fetch code, train (torchrun DDP for one config; one process per GPU for
    several), expose the adapters on :8080."""
    if len(configs) == 1:
        launch = (
            f"(torchrun --nproc_per_node={gpus} scripts/train/train_lora.py "
            f"--config {configs[0]} push=false > /workspace/train.log 2>&1) || true"
        )
    else:
        # One trainer per GPU, all backgrounded, then wait for THOSE PIDS. Each gets its own
        # log; `TRAINING_EXIT_<i>` in boot.log marks which ones came back. A bare `wait`
        # never returns here: the `exec > >(tee ...)` below is itself a background job of
        # this shell (bit us on 2026-08-27 -- the tar step never ran; see docs/GOTCHAS.md).
        launch = (
            "\n".join(
                f"(CUDA_VISIBLE_DEVICES={i} python3 scripts/train/train_lora.py --config {c} "
                f"push=false > /workspace/train_{i}.log 2>&1 </dev/null; echo TRAINING_EXIT_{i}) &\n"
                f"PID_{i}=$!"
                for i, c in enumerate(configs)
            )
            + "\nwait "
            + " ".join(f"$PID_{i}" for i in range(len(configs)))
            + " || true"
        )
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb is installed but muted: the config declares report_to: [wandb] as part of the
# scientific record, and transformers raises on an uninstalled reporter, but a
# credential-free pod has nothing to log with.
export WANDB_MODE=disabled
# python3 -m pip, NOT bare pip: on this image `pip` and `python3` can resolve to different
# interpreters, and packages land where the trainer cannot import them.
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" \
    datasets accelerate omegaconf fire huggingface_hub hf_transfer wandb python-dotenv
hf download {BASE} >/dev/null
hf download {BUNDLE} code.tar.gz --repo-type dataset --local-dir /workspace/bundle
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
cd /workspace/repo
# The repo is not pip-installed on the pod (only its deps are), so `import src.*` resolves
# via the working directory, which script execution alone does not add.
export PYTHONPATH=/workspace/repo
echo TRAINING_STARTING
# `|| true` so a crash leaves a readable log and the tar below still runs.
{launch}
# Adapter and run metadata only. `checkpoint-*` holds optimizer state that is several GB
# and is not wanted back on the driver.
tar -czf /workspace/adapter.tar.gz -C /workspace/out --exclude='checkpoint-*' . || true
echo TRAINING_DONE
sleep infinity
"""


def bundle(repo: str = BUNDLE) -> str:
    """Publish code.tar.gz (the allowlist above) to a public HF dataset repo."""
    missing = [f for f in CODE if not (ROOT / f).is_file()]
    assert not missing, f"missing from the code allowlist: {missing}"
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "code.tar.gz"
        with tarfile.open(tar_path, "w:gz") as t:
            for f in CODE:
                t.add(ROOT / f, arcname=f)
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo="code.tar.gz",
            repo_id=repo,
            repo_type="dataset",
        )
    card = card_markdown(
        {
            "title": "Training code bundle for the post-action-retrospection (design B) 716 arm",
            "experiment": (
                "Code-only bundle for a credential-free RunPod pod that trains the post-action-retrospection "
                "difficult-advice arm: the trainer, its src/ import closure and the train "
                "config. The training DATA is not here -- the mixture is a public dataset the "
                "trainer resolves through its normal data_repo path, sha-pinned in the config."
            ),
            "date_generated": "2026-08-26",
            "constitution": (
                "none -- this repo holds code. The constitution the training data connects to "
                "is declared on the mixture repo the config pins."
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": f"trains a LoRA adapter over {BASE}",
            "generation_config": json.dumps(
                {"files": CODE, "train_config": TRAIN_CONFIG}
            ),
            "schema": "code.tar.gz -- a tarball to extract at the repository root.",
            "provenance": "uv run python scratch/par_b/train_pod.py bundle",
        }
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    return f"pushed code.tar.gz ({len(CODE)} files) -> hf.co/datasets/{repo}"


def up(
    gpus: int = 2,
    name: str = "nika-par716-arm",
    disk_gb: int = 200,
    gpu: str | None = None,
    configs: str = TRAIN_CONFIG,
) -> str:
    """Provision the pod and start training.

    Args:
        gpus: torchrun world size for a single config (2 is the da716 protocol); with
            several configs, the number of GPUs = the number of configs, one trainer each.
        configs: Comma-separated train configs (each must be in ARMS / the code bundle).
            One config -> torchrun DDP over `gpus`; N configs -> N trainers, one per GPU.
        name: Pod name. MUST start with `nika-`: the account is shared and teardown.py
            keys off that prefix to tell our pods from teammates'.
        disk_gb: Container disk. ~55GB base + HF cache + output tree.
        gpu: Pin one gpuTypeId instead of walking the ladder.
    """
    if not name.startswith("nika-"):
        raise SystemExit(
            f"name {name!r} must start with 'nika-' -- the account is shared "
            f"and teardown.py keys off that prefix"
        )
    cfgs = [
        c.strip()
        for c in (
            configs if isinstance(configs, (list, tuple)) else str(configs).split(",")
        )
        if c.strip()
    ]
    unknown = [c for c in cfgs if c not in {arm[0] for arm in ARMS.values()}]
    assert not unknown, f"config(s) not in ARMS / the code bundle: {unknown}"
    if len(cfgs) > 1:
        assert len(cfgs) == gpus, (
            f"{len(cfgs)} configs need exactly {len(cfgs)} GPUs "
            f"(one trainer per GPU), got gpus={gpus}"
        )
    ladder = [(gpu, "pinned")] if gpu else GPU_LADDER
    for gpu_id, note in ladder:
        payload = {
            "name": name,
            "imageName": IMAGE,
            "gpuTypeIds": [gpu_id],
            "gpuCount": gpus,
            "containerDiskInGb": disk_gb,
            "volumeInGb": 0,
            "ports": ["8080/http", "22/tcp"],
            "cloudType": "SECURE",
            "countryCodes": COUNTRIES,
            "dockerStartCmd": ["bash", "-lc", _bootstrap(cfgs, gpus)],
            "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        }
        r = requests.post(
            f"{REST}/pods", headers=_headers(), data=json.dumps(payload), timeout=120
        )
        if r.ok:
            pod = r.json()
            pid = pod.get("id")
            return (
                f">>> {gpus}x {gpu_id} ({note})  pod {pid}  ${pod.get('costPerHr')}/hr\n"
                f"boot:    https://{pid}-8080.proxy.runpod.net/boot.log\n"
                f"train:   https://{pid}-8080.proxy.runpod.net/train.log\n"
                f"adapter: https://{pid}-8080.proxy.runpod.net/adapter.tar.gz "
                f"(after TRAINING_DONE)\n\n"
                f"THEN TEAR IT DOWN:\n"
                f"  uv run python scratch/less/teardown.py --pod {pid}"
            )
        print(f"    {gpus}x {gpu_id}: HTTP {r.status_code} -- {r.text[:160]}")
    raise SystemExit(f"no capacity for {gpus} GPUs on the H200 ladder. Retry later.")


def status(pod: str, lines: int = 8) -> str:
    """Pod state plus the tail of the logs the pod is serving."""
    info = requests.get(f"{REST}/pods/{pod}", headers=_headers(), timeout=30).json()
    out = [
        f"status: {info.get('desiredStatus')}  gpus: {info.get('gpuCount')}  "
        f"${info.get('costPerHr')}/hr"
    ]
    for log in ("boot.log", "train.log", "train_0.log", "train_1.log"):
        try:
            r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{log}", timeout=20)
            tail = (
                "\n".join(r.text.strip().splitlines()[-lines:]) if r.ok else "(not yet)"
            )
        except Exception as e:  # noqa: BLE001 -- a log that is not up yet is normal
            tail = f"(unreachable: {type(e).__name__})"
        out.append(f"--- {log} ---\n{tail}")
    return "\n".join(out)


def pull(pod: str, dest: str = DEST) -> str:
    """Fetch adapter.tar.gz over the pod's :8080 proxy and unpack it."""
    d = ROOT / dest
    d.mkdir(parents=True, exist_ok=True)
    url = f"https://{pod}-8080.proxy.runpod.net/adapter.tar.gz"
    tgz = d / "adapter.tar.gz"
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with tgz.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with tarfile.open(tgz) as t:
        t.extractall(d)
    tgz.unlink()
    found = sorted(p.relative_to(d) for p in d.rglob("adapter_config.json"))
    assert found, f"no adapter found under {d} -- check train.log before tearing down"
    return f"pulled {len(found)} adapter(s) -> {d}\n" + "\n".join(
        f"  {p}" for p in found
    )


def push(dest: str = DEST, private: bool = False, only: str = "") -> str:
    """Push every pulled adapter under `dest` to its arm's HF repo (card + training stamp).

    Args:
        dest: Where `pull` unpacked the tarball(s); one subdir per arm (its output_dir name).
        private: Repo visibility.
        only: Comma-separated output-dir basenames (keys of ARMS) to push. Default: every
            arm found under `dest`.
    """
    d = ROOT / dest
    # fire hands `--only a,b` over as a tuple already; a quoted string arrives as str.
    parts = only if isinstance(only, (list, tuple)) else str(only).split(",")
    wanted = {str(k).strip() for k in parts if str(k).strip()} if only else set(ARMS)
    urls = []
    for out_name, (train_config, repo, seed, launch) in ARMS.items():
        if out_name not in wanted or not (d / out_name).is_dir():
            continue
        adapters = sorted((d / out_name).glob("*/adapter"))
        assert adapters, f"no adapter under {d / out_name}"
        adapter = adapters[
            -1
        ]  # the newest timestamped run, if a retry left more than one
        meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
        assert meta.get("thinking") is True, (
            f"training_meta says thinking={meta.get('thinking')!r}; the 716 PAR rows carry "
            f"real reasoning traces and the arm must be stamped thinking: true"
        )
        assert meta.get("train_config") == train_config, (
            f"{adapter} was trained from {meta.get('train_config')!r}, not {train_config!r}"
        )
        ds = meta.get("dataset", {})
        replicate = (
            ""
            if seed == 0
            else (
                f" SEED-{seed} REPLICATE of that arm (LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-"
                "r64-dynbatch, seed 0): same data sha, same recipe, different LoRA init and shuffle "
                "order; ONE trainer on ONE H200 (no DDP) -- under dynamic batching the optimizer "
                "step is the same 16 examples either way."
            )
        )
        coherent = out_name == "train_t2_9284_par716coh_dynbatch"
        url = push_run_dir(
            adapter,
            repo,
            {
                "experiment": (
                    "LoRA SFT adapter -- ARM 1 OF THE PAR COHERENCE EXPERIMENT (2026-08-28): the "
                    "PAR-716 organism retrained on the SAME 716 five-turn rows with only the trained "
                    "turn rewritten so the private reasoning ends on a first-person decision and the "
                    "reply enacts it (LASR-Callum/2026-08-28-post-action-retrospection-716-coherent). "
                    "Recipe, seed, Table-2 half, launch (2-rank DDP, dynamic batching) identical to "
                    "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-r64-dynbatch; the trained text is the "
                    "only variable."
                ) if coherent else (
                    f"LoRA SFT adapter -- {Path(train_config).stem}. THE POST-ACTION-"
                    "RETROSPECTION ARM (design B): Qwen3.6-27B trained on 9,284 Table-2 rows + "
                    "716 five-turn rows -- a difficult-advice prompt, a bare refusal (Sonnet, "
                    "form instructed, never trained), the person's pushback, and the trained "
                    "turn doing the reasoning the refusal skipped under DA's own rewrite "
                    "contract; only the last assistant turn is in the loss. Same organism as "
                    "the da716 arm (LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch) "
                    "in every respect but the 716 rows' shape." + replicate
                ),
                "date_generated": "2026-08-28" if coherent else ("2026-08-26" if seed == 0 else "2026-08-27"),
                "constitution": (
                    "constitutions/claude_distilled_12_principles_mid/constitution.md "
                    f"(9 principles), the same as difficult advice's; inherited from "
                    f"{ds.get('repo')}"
                ),
                "source_repo": f"{origin_url()} @ {git_sha()}",
                "models": f"base: {BASE}",
                "generation_config": json.dumps(
                    {
                        "seed": seed,
                        "thinking": True,
                        "epochs": 1,
                        "lr": 1.0e-4,
                        "batch_size": 1,
                        "grad_accum": 16,
                        "max_seq_len": 8192,
                        "dynamic_batching": "token budget from ModelProfile.train_memory "
                        "(H200 8000)",
                        "ddp_ranks": 2 if seed == 0 else 1,
                        "launch": launch,
                        "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
                    }
                ),
                "schema": "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
                "{thinking, train_config, base_model, dataset{repo,file,revision}, "
                "git_sha, timestamp}",
                "provenance": (
                    f"{launch} scripts/train/train_lora.py --config {train_config} push=false "
                    "(on pod), then uv run python scratch/par_b/train_pod.py push"
                ),
                "dataset": f"hf.co/datasets/{ds.get('repo')}@{ds.get('revision')} "
                f"({ds.get('file')})",
                "comparison_arm": (
                    "LASR-Callum/qwen3.6-27b-lora-t2-9284-par716-r64-dynbatch (paired parent) and "
                    "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch"
                ) if coherent else "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
            },
            private=private,
            repo_type="model",
        )
        urls.append(f"{out_name} -> {url}")
    assert urls, f"nothing to push under {d} (looked for {sorted(wanted)})"
    return "\n".join(urls)


if __name__ == "__main__":
    fire.Fire(
        {"bundle": bundle, "up": up, "status": status, "pull": pull, "push": push}
    )
