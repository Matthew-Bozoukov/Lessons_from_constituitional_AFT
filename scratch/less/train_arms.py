# ABOUTME: Trains the LESS top-10% arm and its random-220 control on ONE credential-free
# ABOUTME: RunPod pod, one arm per GPU, then pulls the adapters back and pushes them to HF.

"""Train both arms of the LESS top-10% experiment on a single pod.

    uv run python scratch/less/train_arms.py bundle
    uv run python scratch/less/train_arms.py up --gpus 2
    uv run python scratch/less/train_arms.py status --pod <id>
    uv run python scratch/less/train_arms.py pull --pod <id>
    uv run python scratch/less/train_arms.py push
    uv run python scratch/less/teardown.py --pod <id>

ONE POD, BOTH ARMS, and that is the whole reason this exists rather than two calls to
scripts/gpu/runpod_train.py. Each arm is 220 rows x 4 epochs = 56 optimizer steps, which is
minutes of compute; the ~55GB base-model download dominates the wall clock by an order of
magnitude. Two pods would pay that download twice and still run no faster. In one pod the
download lands once in a shared HF_HOME and the two trainers read it concurrently, pinned to
separate devices with CUDA_VISIBLE_DEVICES.

DDP across GPUs would be the wrong parallelism here: 14 optimizer steps per epoch is too few
for the coordination to pay for itself, and CLAUDE.md's rule for training several models at
once (devote N_GPUS//N_MODELS to each) lands on exactly one GPU per arm.

The pod carries NO credentials, inherited from scripts/gpu/runpod_train.py. Both mixtures
are public HF datasets, so the trainer resolves them anonymously through its normal
data_repo path -- the bundle here holds only code. Trainers run with push=false and the
adapters come back over the :8080 proxy to be pushed from this machine, where the token is.

Bootstrap discipline is also inherited from that file: log server before anything slow,
`|| true` so a crash leaves a readable log instead of a restart loop, and the shell stays
PID 1 waiting on the trainers.
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
BUNDLE = "LASR-Callum/2026-08-19-less-top10-arms-code-bundle"
BASE = "Qwen/Qwen3.6-27B"
IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"

# H200 only. ModelProfile carries a MEASURED positive entry for H200 and, for H100 80GB, a
# measured NEGATIVE bound (a 1x~8k fwd+bwd OOMed 7.4GiB short). These rows are far shorter
# than that, but both arms must run on the same card for the comparison to hold, and a
# silent fallback to a different one would make them differ in more than their data.
GPU_LADDER = [("NVIDIA H200", "141GB"), ("NVIDIA H200 NVL", "143GB")]

COUNTRIES = ["US", "CA", "NL", "DE", "FR", "GB", "IE", "BE", "SE", "NO", "FI", "CH", "AT",
             "ES", "IT"]

# (label, train config, destination adapter repo). The two configs differ in four keys:
# data_repo, data_revision, output_dir, hf_repo. Everything else is byte-identical.
ARMS = [
    ("arm", "configs/train/2026-08-19_lora_qwen36_less_top_10_220_rank64.yaml",
     "LASR-Callum/2026-08-19-qwen36-lora-less-top10-220-rank-64", "train_less_top10_220"),
    ("control", "configs/train/2026-08-19_lora_qwen36_less_random_220_control_rank64.yaml",
     "LASR-Callum/2026-08-19-qwen36-lora-random-220-control-rank-64", "train_random220_control"),
]

# Everything `python3 scripts/train/train_lora.py` imports, plus the two configs. An
# explicit allowlist, never a directory walk: this tarball goes to a PUBLIC repo, and a
# walk would sweep up .env.
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
    *[cfg for _, cfg, _, _ in ARMS],
]


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json"}


def _bootstrap(gpus: int) -> str:
    """Pod startup: fetch code, train both arms, expose the adapters on :8080.

    Arms are assigned round-robin to devices, and the arms sharing a device are chained
    SEQUENTIALLY inside one background job. That distinction is the whole safety property
    when gpus < len(ARMS): two 27B bf16 models on one card do not fit, so `--gpus 1` must
    produce one chain of two runs, never two concurrent ones.
    """
    by_device: dict[int, list[tuple[str, str]]] = {}
    for i, (label, cfg, _, _) in enumerate(ARMS):
        by_device.setdefault(i % gpus, []).append((label, cfg))

    launches = []
    for dev, arms in sorted(by_device.items()):
        chain = " ; ".join(
            f"CUDA_VISIBLE_DEVICES={dev} python3 scripts/train/train_lora.py "
            f"--config {cfg} push=false > /workspace/train_{label}.log 2>&1 || true"
            for label, cfg in arms)
        # Stagger each device's start: the trainers otherwise materialise their 27B bf16
        # models simultaneously, and the transient host-RAM peak during load is the one
        # place a multi-process pod can die for a reason unrelated to GPU memory.
        wait = "" if dev == 0 else f"sleep {120 * dev}; "
        launches.append(f"({wait}{chain}) &")
    parallel = "\n".join(launches)
    return f"""mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
set -euxo pipefail
(cd /workspace && nohup python3 -m http.server 8080 </dev/null >/dev/null 2>&1 &) || true
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb is installed but muted: the configs declare report_to: [wandb] as part of the
# scientific record, and transformers raises on an uninstalled reporter, but a
# credential-free pod has nothing to log with.
export WANDB_MODE=disabled
# python3 -m pip, NOT bare pip: on this image `pip` and `python3` can resolve to different
# interpreters, and packages land where the trainer cannot import them.
python3 -m pip install --no-cache-dir -q "transformers>=5.14" "trl>=0.27" "peft==0.20.0" \
    datasets accelerate omegaconf fire huggingface_hub hf_transfer wandb python-dotenv
# ONCE, before either trainer starts: this is the download both arms share and the whole
# reason they are on one pod.
hf download {BASE} >/dev/null
hf download {BUNDLE} code.tar.gz --repo-type dataset --local-dir /workspace/bundle
mkdir -p /workspace/repo && tar -xzf /workspace/bundle/code.tar.gz -C /workspace/repo
cd /workspace/repo
# The repo is not pip-installed on the pod (only its deps are), so `import src.*` resolves
# via the working directory, which script execution alone does not add.
export PYTHONPATH=/workspace/repo
echo TRAINING_STARTING
{parallel}
# `set -e` would abort here on a nonzero trainer, before the adapters are packaged.
wait || true
# Adapters and run metadata only. `checkpoint-*` holds optimizer state that is several GB
# per arm and is not wanted back on the driver.
tar -czf /workspace/adapters.tar.gz -C /workspace/out --exclude='checkpoint-*' . || true
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
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo="code.tar.gz",
                        repo_id=repo, repo_type="dataset")
    card = card_markdown({
        "title": "Training code bundle for the LESS top-10% arms",
        "experiment": (
            "Code-only bundle for a credential-free RunPod pod that trains both arms of "
            "the LESS top-10% difficult-advice experiment: the trainer, its src/ import "
            "closure and the two train configs. The training DATA is not here — both "
            "mixtures are public datasets the trainer resolves through its normal "
            "data_repo path, sha-pinned in the configs."),
        "date_generated": "2026-08-19",
        "constitution": (
            "none — this repo holds code. The constitution the training data connects to "
            "is declared on the two mixture repos the configs pin."),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"trains LoRA adapters over {BASE}",
        "generation_config": json.dumps({"files": CODE, "arms": [a[1] for a in ARMS]}),
        "schema": "code.tar.gz — a tarball to extract at the repository root.",
        "provenance": "uv run python scratch/less/train_arms.py bundle",
    })
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    return f"pushed code.tar.gz ({len(CODE)} files) -> hf.co/datasets/{repo}"


def up(gpus: int = 2, name: str = "nika-less-top10-arms", disk_gb: int = 220,
       gpu: str | None = None) -> str:
    """Provision the pod and start both arms.

    Args:
        gpus: GPUs in the pod, one per arm. 1 runs the arms sequentially on device 0.
        name: Pod name. MUST start with `nika-`: the account is shared and teardown.py
            keys off that prefix to tell our pods from teammates'.
        disk_gb: Container disk. ~55GB base + HF cache + 2 output trees.
        gpu: Pin one gpuTypeId instead of walking the ladder.
    """
    if not name.startswith("nika-"):
        raise SystemExit(f"name {name!r} must start with 'nika-' — the account is shared "
                         f"and teardown.py keys off that prefix")
    ladder = [(gpu, "pinned")] if gpu else GPU_LADDER
    for gpu_id, note in ladder:
        payload = {
            "name": name, "imageName": IMAGE,
            "gpuTypeIds": [gpu_id], "gpuCount": gpus,
            "containerDiskInGb": disk_gb, "volumeInGb": 0,
            "ports": ["8080/http", "22/tcp"], "cloudType": "SECURE",
            "countryCodes": COUNTRIES,
            "dockerStartCmd": ["bash", "-lc", _bootstrap(gpus)],
            "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        }
        r = requests.post(f"{REST}/pods", headers=_headers(),
                          data=json.dumps(payload), timeout=120)
        if r.ok:
            pod = r.json()
            pid = pod.get("id")
            return (
                f">>> {gpus}x {gpu_id} ({note})  pod {pid}  ${pod.get('costPerHr')}/hr\n"
                f"boot:    https://{pid}-8080.proxy.runpod.net/boot.log\n"
                + "".join(f"{label}:{' ' * (8 - len(label) - 1)}"
                          f"https://{pid}-8080.proxy.runpod.net/train_{label}.log\n"
                          for label, _, _, _ in ARMS)
                + f"adapters: https://{pid}-8080.proxy.runpod.net/adapters.tar.gz "
                  f"(after TRAINING_DONE)\n\n"
                  f"THEN TEAR IT DOWN:\n"
                  f"  uv run python scratch/less/teardown.py --pod {pid}")
        print(f"    {gpus}x {gpu_id}: HTTP {r.status_code} — {r.text[:160]}")
    raise SystemExit(f"no capacity for {gpus} GPUs on the H200 ladder. Retry, or run "
                     f"--gpus 1 (both arms sequentially on one card, same total GPU-hours).")


def status(pod: str, lines: int = 6) -> str:
    """Pod state plus the tail of every log the pod is serving."""
    info = requests.get(f"{REST}/pods/{pod}", headers=_headers(), timeout=30).json()
    out = [f"status: {info.get('desiredStatus')}  gpus: {info.get('gpuCount')}"]
    for log in ("boot.log", *[f"train_{label}.log" for label, _, _, _ in ARMS]):
        try:
            r = requests.get(f"https://{pod}-8080.proxy.runpod.net/{log}", timeout=20)
            tail = "\n".join(r.text.strip().splitlines()[-lines:]) if r.ok else "(not yet)"
        except Exception as e:  # noqa: BLE001 — a log that is not up yet is normal
            tail = f"(unreachable: {type(e).__name__})"
        out.append(f"--- {log} ---\n{tail}")
    return "\n".join(out)


def pull(pod: str, dest: str = "output/adapters/less_top10") -> str:
    """Fetch adapters.tar.gz over the pod's :8080 proxy and unpack it."""
    d = ROOT / dest
    d.mkdir(parents=True, exist_ok=True)
    url = f"https://{pod}-8080.proxy.runpod.net/adapters.tar.gz"
    tgz = d / "adapters.tar.gz"
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with tgz.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with tarfile.open(tgz) as t:
        t.extractall(d)
    tgz.unlink()
    found = sorted(p.relative_to(d) for p in d.rglob("adapter_config.json"))
    assert found, f"no adapter found under {d} — check the train logs before tearing down"
    return f"pulled {len(found)} adapter(s) -> {d}\n" + "\n".join(f"  {p}" for p in found)


def push(dest: str = "output/adapters/less_top10", private: bool = False) -> str:
    """Push each pulled adapter to its HF repo with the card and its training stamp.

    private=False by DEFAULT, unlike train_lora's own push: push_run_dir defaults to
    private and the 2026-08-17 less-swap adapter had to be flipped by hand afterwards.
    """
    d = ROOT / dest
    out = []
    for label, cfg, repo, out_name in ARMS:
        adapters = sorted((d / out_name).glob("*/adapter"))
        assert adapters, f"{label}: no adapter under {d / out_name}"
        # The newest timestamped run, if a retry left more than one.
        adapter = adapters[-1]
        meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
        assert meta.get("thinking") is True, (
            f"{label}: training_meta says thinking={meta.get('thinking')!r}; both arms "
            f"train on real reasoning traces and must be stamped thinking: true")
        ds = meta.get("dataset", {})
        url = push_run_dir(adapter, repo, {
            "experiment": (
                f"LoRA SFT adapter — {Path(cfg).stem}. "
                + ("THE LESS ARM: base Qwen3.6-27B trained on the 220 highest-influence "
                   "rows (top 10%) of the difficult-advice pool by LESS score_max."
                   if label == "arm" else
                   "THE CONTROL ARM: base Qwen3.6-27B trained on 220 rows drawn at random "
                   "(seed 1) from the same 2,203-row pool.")
                + " The two arms differ only in which 220 rows they saw; every "
                  "hyperparameter, the batching protocol and the loss path are identical, "
                  "so a difference between them is attributable to the selection. Trained "
                  "one arm per GPU on a single pod so both share one base-model download."),
            "date_generated": "2026-08-19",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md — "
                f"inherited from the training data ({ds.get('repo')})"),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": f"base: {BASE}",
            "generation_config": json.dumps({
                "seed": 0, "thinking": True, "epochs": 4, "lr": 1.0e-4,
                "batch_size": 1, "grad_accum": 16, "max_seq_len": 8192,
                "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
            }),
            "schema": "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
                      "{thinking, train_config, base_model, dataset{repo,file,revision}, "
                      "git_sha, timestamp}",
            "provenance": (f"uv run train --config {cfg} push=false (on pod), then "
                           "uv run python scratch/less/train_arms.py push"),
            "dataset": f"hf.co/datasets/{ds.get('repo')}@{ds.get('revision')} "
                       f"({ds.get('file')})",
            "paired_arm": [r for _, _, r, _ in ARMS if r != repo][0],
        }, private=private, repo_type="model")
        out.append(f"{label}: {url}")
    return "\n".join(out)


if __name__ == "__main__":
    fire.Fire({"bundle": bundle, "up": up, "status": status, "pull": pull, "push": push})
