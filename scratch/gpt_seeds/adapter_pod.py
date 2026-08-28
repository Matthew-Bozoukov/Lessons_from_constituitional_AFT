# ABOUTME: GPT seed replicates: unpack the pulled adapter tarball, pick checkpoint-600 (the run's
# ABOUTME: expected end, as seed 0), stamp training_meta.json the way seed 0's was, push to HF.
"""Adapter round-trip for the GPT seed replicates (seeds 42, 69).

    uv run python scratch/gpt_seeds/adapter_pod.py extract --seed 42
    uv run python scratch/gpt_seeds/adapter_pod.py push --seed 42

The watchdog (scratch/low_stakes/train_watchdog.py) leaves `output/adapters/gptresp685_s<seed>/
adapter.tar.gz`. Seed 0's published adapter is checkpoint-600 of 624 steps because the final
1-example step cannot be split over 2 DDP ranks (see its card); `extract` enforces the same
choice -- it REFUSES a tarball whose end state differs from seed 0's (a completed `adapter/`
dir, or no checkpoint-600), because then the replicate would not be the same protocol.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.huggingface import hf_api, push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"
BUNDLE = "LASR-Callum/2026-08-28-gptresp685-seeds-bundle"
MIXTURE = "t2_9284_gptresp685_10k.jsonl"
SEED0 = "LASR-Callum/qwen3.6-27b-lora-t2-9284-gptresp685-paired-r64"
CHECKPOINT = 600
TOTAL_STEPS = 624
# What seed 0's repo carries, and therefore what a replicate carries: the adapter, the
# tokenizer it was trained with, the args. Optimizer/scheduler/rng state stay local.
KEEP = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "trainer_state.json",
)


def _spec(seed: int) -> dict:
    assert seed in (42, 69), seed
    return {
        "config": f"configs/train/lora_qwen36_t2_9284_gptresp685_paired_s{seed}_2xh200.yaml",
        "repo": f"LASR-Callum/qwen3.6-27b-lora-t2-9284-gptresp685-paired-r64-seed{seed}",
        "dest": ROOT / f"output/adapters/gptresp685_s{seed}",
    }


def extract(seed: int) -> str:
    """Unpack adapter.tar.gz, verify the seed-0 end state, assemble `adapter/` with a stamp."""
    spec = _spec(seed)
    d = spec["dest"]
    tgz = d / "adapter.tar.gz"
    assert tgz.exists(), f"{tgz} missing -- did the watchdog pull?"
    raw = d / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    with tarfile.open(tgz) as t:
        t.extractall(raw)
    finals = [p.parent for p in raw.rglob("training_meta.json")]
    ckpts = sorted(p for p in raw.rglob("checkpoint-*") if p.is_dir())
    print("checkpoints:", [c.name for c in ckpts], " final adapter dirs:", finals)
    assert not finals, (
        f"the trainer wrote a FINAL adapter ({finals}); seed 0 never did (crash at 624/624). "
        "The protocol differs -- stop and compare before pushing."
    )
    ck = raw.rglob(f"checkpoint-{CHECKPOINT}")
    ck = next(iter(sorted(ck)), None)
    assert ck is not None, f"no checkpoint-{CHECKPOINT} in the tarball"
    state = json.loads((ck / "trainer_state.json").read_text())
    assert state["global_step"] == CHECKPOINT, state["global_step"]
    assert state["max_steps"] == TOTAL_STEPS, state["max_steps"]
    last = [h for h in state["log_history"] if "loss" in h][-1]
    lr = [h for h in state["log_history"] if "learning_rate" in h][-1]["learning_rate"]

    out = d / "adapter"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for name in KEEP:
        src = ck / name
        if src.exists():
            shutil.copy2(src, out / name)
    for must in ("adapter_config.json", "adapter_model.safetensors"):
        assert (out / must).exists(), f"{must} missing from checkpoint-{CHECKPOINT}"
    api = hf_api()
    bundle_sha = api.dataset_info(BUNDLE).sha
    meta = {
        "thinking": True,
        "train_config": spec["config"],
        "base_model": BASE,
        "seed": seed,
        "dataset": {"repo": BUNDLE, "file": MIXTURE, "revision": bundle_sha},
        "checkpoint": {
            "global_step": CHECKPOINT,
            "total_steps": TOTAL_STEPS,
            "epoch": round(state["epoch"], 4),
            "lr_at_checkpoint": lr,
            "last_logged_loss": last.get("loss"),
            "why_not_final": (
                f"run crashed at step {TOTAL_STEPS}/{TOTAL_STEPS}: route_step cannot split a "
                "final 1-example batch across 2 DDP ranks (9,969 rows at global batch 16 leaves "
                f"remainder 1). checkpoint-{CHECKPOINT} is {100 * CHECKPOINT / TOTAL_STEPS:.1f}% "
                "of the epoch -- the SAME end state as seed 0 (see its card), kept on purpose so "
                "the replicates share seed 0's protocol exactly."
            ),
        },
        "backfilled_by": "scratch/gpt_seeds/adapter_pod.py extract (checkpoint stamp, as seed 0)",
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["checkpoint"], indent=1))
    return f"assembled {out} from {ck.relative_to(raw)}; last loss {last.get('loss')}"


def push(seed: int, private: bool = False) -> str:
    """Push `adapter/` (checkpoint-600 + stamp) to the seed's HF repo with the card."""
    spec = _spec(seed)
    out = spec["dest"] / "adapter"
    meta = json.loads((out / "training_meta.json").read_text())
    assert meta["thinking"] is True and meta["seed"] == seed
    ds = meta["dataset"]
    ck = meta["checkpoint"]
    return push_run_dir(
        out,
        spec["repo"],
        {
            "title": f"qwen3.6-27b LoRA -- generator ablation ARM C (GPT-written answers), SEED {seed} replicate",
            "experiment": (
                f"LoRA SFT adapter -- {Path(spec['config']).stem}. Seed-{seed} REPLICATE of the "
                f"GPT-responder paired arm ({SEED0} is seed 0): Qwen3.6-27B on 9,284 Table-2 rows + "
                "685 difficult-advice rows whose assistant turn was drafted by openai/gpt-5.6-luna "
                "and revised by openai/gpt-5.6-terra. Identical data (byte-for-byte), code (seed 0's "
                "own bundle), LoRA shape, schedule and global batch; only the seed (LoRA init + "
                "shuffle order) differs. Purpose: the between-seed error on the arm's ODCV number."
            ),
            "date_generated": meta["timestamp"][:10],
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md, via the corpus "
                "LASR-Callum/2026-08-25-difficult-advice-gpt-responder-716"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()} (branch worktree-gpt-seeds)",
            "models": (
                f"base: {BASE}; synth rows by openai/gpt-5.6-luna (draft) + openai/gpt-5.6-terra "
                "(revision); prompts by anthropic/claude-haiku-4.5 + anthropic/claude-sonnet-5 (the "
                "baseline's, reused verbatim)"
            ),
            "generation_config": json.dumps(
                {
                    "seed": seed,
                    "thinking": True,
                    "epochs": 1,
                    "lr": 1.0e-4,
                    "lr_scheduler": "cosine",
                    "batch_size": 1,
                    "grad_accum": 16,
                    "max_seq_len": 8192,
                    "dynamic_batching": "token budget from ModelProfile.train_memory (H200 8000)",
                    "ddp_ranks": 2,
                    "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
                }
            ),
            "schema": (
                "PEFT LoRA adapter (safetensors) + tokenizer + training_args.bin + trainer_state.json "
                "(loss history) + training_meta.json {thinking, train_config, base_model, seed, "
                "dataset{repo,file,revision}, checkpoint{global_step,total_steps,epoch,"
                "lr_at_checkpoint,why_not_final}, git_sha, timestamp}"
            ),
            "provenance": (
                f"uv run python scripts/gpu/runpod_train.py up --bundle {BUNDLE} --train_config "
                f"{spec['config']} --gpu 'NVIDIA H200' --gpu_count 2 --mixture {MIXTURE} "
                "(torchrun --nproc_per_node=2 scripts/train/train_lora.py on the pod, push=false); "
                "scratch/low_stakes/train_watchdog.py pulled adapter.tar.gz; "
                f"scratch/gpt_seeds/adapter_pod.py extract/push --seed {seed}"
            ),
            "dataset": f"hf.co/datasets/{ds['repo']}@{ds['revision']} ({ds['file']}; sha256-identical "
            "to LASR-Callum/2026-08-25-gptresp685-paired-bundle@c7994ec2, seed 0's pin)",
            "checkpoint": (
                f"checkpoint-{ck['global_step']} of {ck['total_steps']} steps (epoch {ck['epoch']}, "
                f"lr {ck['lr_at_checkpoint']:.2e}, last logged loss {ck['last_logged_loss']}). "
                f"{ck['why_not_final']}"
            ),
            "comparison_arms": (
                f"seed 0: {SEED0}; A (Sonnet): LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch; "
                "B (grok): LASR-Callum/qwen3.6-27b-lora-t2-9284-grokresp703-paired-r64"
            ),
        },
        private=private,
        repo_type="model",
        front_matter={
            "base_model": BASE,
            "tags": [
                "lora",
                "sft",
                "qwen3.6",
                "generator-ablation",
                "difficult-advice",
                "seed-replicate",
            ],
        },
    )


if __name__ == "__main__":
    fire.Fire({"extract": extract, "push": push})
