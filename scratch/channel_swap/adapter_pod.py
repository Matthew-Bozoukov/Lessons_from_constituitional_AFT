# ABOUTME: Pull a channel-swap arm's adapter back from its credential-free RunPod pod and push it to
# ABOUTME: HF with the card + training stamp; the pod itself is driven by scripts/gpu/runpod_train.py.
"""Channel-swap arms: adapter round-trip.

    uv run python scripts/gpu/runpod_train.py up --bundle <arm repo> --train_config <arm config> \\
        --gpu "NVIDIA H200" --gpu_count 2 --mixture <arm jsonl> --name kunwar-swap-<arm>
    uv run python scratch/channel_swap/adapter_pod.py pull --arm gtrace_sreply703 --pod <id>
    uv run python scratch/channel_swap/adapter_pod.py push --arm gtrace_sreply703
    uv run python scripts/gpu/runpod_train.py down --pod <id>

`pull` is scratch/trait10_curiosity/train_pod.py's (generic); `push` carries each arm's card.
Same protocol as arms A/B/C: 2xH200 torchrun DDP, dynamic batching, r=64.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv()  # the worktree has no .env; python-dotenv walks up to the main checkout's
sys.path.insert(0, str(ROOT))

from scratch.trait10_curiosity.train_pod import pull as _pull  # noqa: E402
from src.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"
ARMS = {
    "gtrace_sreply703": {
        "config": "configs/train/lora_qwen36_t2_9284_gtrace_sreply703_paired_2xh200.yaml",
        "repo": "LASR-Callum/qwen3.6-27b-lora-t2-9284-gtrace-sreply703-paired-r64",
        "dest": "output/adapters/gtrace_sreply703",
        "what": "grok-4.6's reasoning trace + Sonnet 5's reply",
        "models": (
            "base: Qwen/Qwen3.6-27B; synth trace channel by x-ai/grok-4.6; synth reply channel by "
            "anthropic/claude-haiku-4.5 (draft) + anthropic/claude-sonnet-5 (rewrite)"
        ),
    },
    "strace_greply703": {
        "config": "configs/train/lora_qwen36_t2_9284_strace_greply703_paired_2xh200.yaml",
        "repo": "LASR-Callum/qwen3.6-27b-lora-t2-9284-strace-greply703-paired-r64",
        "dest": "output/adapters/strace_greply703",
        "what": "Sonnet 5's reasoning trace + grok-4.6's reply",
        "models": (
            "base: Qwen/Qwen3.6-27B; synth trace channel by anthropic/claude-haiku-4.5 (draft) + "
            "anthropic/claude-sonnet-5 (rewrite); synth reply channel by x-ai/grok-4.6"
        ),
    },
}


def pull(arm: str, pod: str) -> str:
    """Fetch adapter.tar.gz over the pod's :8080 proxy and unpack it under the arm's dest."""
    return _pull(pod, ARMS[arm]["dest"])


def push(arm: str, private: bool = False) -> str:
    """Push the pulled adapter to its HF repo with the card and its training stamp."""
    spec = ARMS[arm]
    d = ROOT / spec["dest"]
    adapters = sorted(p for p in d.rglob("adapter_config.json"))
    assert adapters, f"no adapter under {d}"
    adapter = adapters[-1].parent
    meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
    assert meta.get("thinking") is True, (
        f"training_meta says thinking={meta.get('thinking')!r}"
    )
    ds = meta.get("dataset", {})
    return push_run_dir(
        adapter,
        spec["repo"],
        {
            "experiment": (
                f"LoRA SFT adapter -- {Path(spec['config']).stem}. CHANNEL-SWAP arm of the generator "
                f"ablation: Qwen3.6-27B on 9,284 Table-2 rows + 703 difficult-advice rows whose assistant "
                f"turn is {spec['what']}, recombined row-for-row from arms A and B on the same 703 questions. "
                "Same Table2 half, seed, LoRA shape and schedule as A/B/C. Read the two swaps as a 2x2 with "
                "A (da716 16.3%) and B (7.8%) on the same 65 ODCV cells: the swap near B names the channel "
                "that carries grok's effect (docs/LOG.md 2026-08-27)."
            ),
            "date_generated": meta.get("timestamp", "2026-08-27")[:10],
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md "
                f"(sha fe2ed96093d68a87...), inherited from {ds.get('repo')}"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": spec["models"],
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
            "schema": (
                "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
                "{thinking, train_config, base_model, dataset{repo,file,revision}, git_sha, timestamp}"
            ),
            "provenance": (
                f"torchrun --nproc_per_node=2 scripts/train/train_lora.py --config {spec['config']} "
                "push=false (RunPod 2xH200 via scripts/gpu/runpod_train.py), then "
                f"uv run python scratch/channel_swap/adapter_pod.py push --arm {arm}"
            ),
            "dataset": f"hf.co/datasets/{ds.get('repo')}@{ds.get('revision')} ({ds.get('file')})",
            "comparison_arms": (
                "A: LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch; "
                "B: LASR-Callum/qwen3.6-27b-lora-t2-9284-grokresp703-paired-r64; "
                "C: LASR-Callum/qwen3.6-27b-lora-t2-9284-sonnetconcise703-paired-r64; "
                "other swap: " + [v["repo"] for k, v in ARMS.items() if k != arm][0]
            ),
        },
        private=private,
        repo_type="model",
    )


if __name__ == "__main__":
    fire.Fire({"pull": pull, "push": push})
