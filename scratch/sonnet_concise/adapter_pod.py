# ABOUTME: Pull arm C's adapter back from its credential-free RunPod pod and push it to HF
# ABOUTME: with the card + training stamp; the pod itself is driven by scripts/gpu/runpod_train.py.

"""Arm C (length-capped Sonnet) adapter round-trip.

    uv run python scripts/gpu/runpod_train.py up --bundle LASR-Callum/2026-08-26-table2-9284-sonnet-concise-703-paired-train \\
        --train_config configs/train/2026-08-24_lora_qwen36_table2_9284_sonnet_concise_703_paired.yaml \\
        --gpu "NVIDIA H200" --gpu_count 2 --mixture t2_9284_sonnetconcise703_10k.jsonl --name kunwar-train-sonnetconcise703
    uv run python scratch/sonnet_concise/adapter_pod.py pull --pod <id>     # after TRAINING_DONE in boot.log
    uv run python scratch/sonnet_concise/adapter_pod.py push
    uv run python scripts/gpu/runpod_train.py down --pod <id>

`pull` is scratch/trait10_curiosity/train_pod.py's, unchanged (it is generic); `push` carries
this arm's card. Same protocol as arms A/B: 2xH200 torchrun DDP, dynamic batching, r=64.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from scratch.trait10_curiosity.train_pod import pull as _pull  # noqa: E402
from src.infra.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

TRAIN_CONFIG = "configs/train/2026-08-24_lora_qwen36_table2_9284_sonnet_concise_703_paired.yaml"
ADAPTER_REPO = "LASR-Callum/2026-08-26-qwen36-lora-table2-9284-sonnet-concise-703-paired-rank-64"
BASE = "Qwen/Qwen3.6-27B"
DEST = "output/adapters/sonnetconcise703"


def pull(pod: str, dest: str = DEST) -> str:
    """Fetch adapter.tar.gz over the pod's :8080 proxy and unpack it under `dest`."""
    return _pull(pod, dest)


def push(dest: str = DEST, private: bool = False) -> str:
    """Push the pulled adapter to its HF repo with the card and its training stamp."""
    d = ROOT / dest
    adapters = sorted(p for p in d.rglob("adapter_config.json"))
    assert adapters, f"no adapter under {d}"
    adapter = adapters[
        -1
    ].parent  # the newest timestamped run, if a retry left more than one
    meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
    assert meta.get("thinking") is True, (
        f"training_meta says thinking={meta.get('thinking')!r}; the 703 capped rows carry "
        f"real reasoning traces and the arm must be stamped thinking: true"
    )
    ds = meta.get("dataset", {})
    return push_run_dir(
        adapter,
        ADAPTER_REPO,
        {
            "experiment": (
                f"LoRA SFT adapter -- {Path(TRAIN_CONFIG).stem}. ARM C, the LENGTH CONTROL of the "
                "generator ablation: Qwen3.6-27B on 9,284 Table-2 rows + 703 difficult-advice rows "
                "whose assistant turn is the baseline's own Haiku draft rewritten by Sonnet 5 under a "
                "one-sentence cap at grok's median lengths (reasoning ~220 words, reply ~270). "
                "Same 703 questions, same Table2 half, same seed/LoRA/schedule as arm B "
                "(LASR-Callum/2026-08-24-qwen36-lora-table2-9284-grok-responder-703-paired-rank-64); differs from the "
                "da716 baseline only in the rewrite's length. Read C's ODCV against A (16.3%) and "
                "B (7.8%): near B means length carried the drop, near A means the generator did."
            ),
            "date_generated": meta.get("timestamp", "2026-08-26")[:10],
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md "
                f"(sha fe2ed96093d68a87...), inherited from {ds.get('repo')}"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": f"base: {BASE}; synth half written by anthropic/claude-haiku-4.5 (draft) + "
            "anthropic/claude-sonnet-5 (length-capped rewrite)",
            "generation_config": json.dumps(
                {
                    "seed": 0,
                    "thinking": True,
                    "epochs": 1,
                    "lr": 1.0e-4,
                    "batch_size": 1,
                    "grad_accum": 16,
                    "max_seq_len": 8192,
                    "dynamic_batching": "token budget from ModelProfile.train_memory (H200 8000)",
                    "ddp_ranks": 2,
                    "lora": {"r": 64, "alpha": 128, "dropout": 0.05},
                }
            ),
            "schema": "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
            "{thinking, train_config, base_model, dataset{repo,file,revision}, "
            "git_sha, timestamp}",
            "provenance": (
                f"torchrun --nproc_per_node=2 scripts/train/train_lora.py --config "
                f"{TRAIN_CONFIG} push=false (on a RunPod 2xH200 pod via "
                "scripts/gpu/runpod_train.py), then "
                "uv run python scratch/sonnet_concise/adapter_pod.py push"
            ),
            "dataset": f"hf.co/datasets/{ds.get('repo')}@{ds.get('revision')} ({ds.get('file')})",
            "comparison_arms": (
                "A: LASR-Callum/2026-08-14-qwen36-lora-table2-9284-difficult-advice-716-rank-64-dynbatch; "
                "B: LASR-Callum/2026-08-24-qwen36-lora-table2-9284-grok-responder-703-paired-rank-64; "
                "D: LASR-Callum/2026-08-25-qwen36-lora-table2-9284-gpt-responder-685-paired-rank-64"
            ),
        },
        private=private,
        repo_type="model",
    )


if __name__ == "__main__":
    fire.Fire({"pull": pull, "push": push})
