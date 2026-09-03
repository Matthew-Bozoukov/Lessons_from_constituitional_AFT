# ABOUTME: Push the principle-scoped 702 seed adapters to the Hub, through the naming gate.
# ABOUTME: Run: uv run python scratch/da716_seeds/push_adapters.py --seed 42
#
# The pod trains with push=false (it holds no token), so the adapter is pulled back over
# :8080 and published from the machine that has one. `gate_push` runs first, so a name a
# reader cannot date or place never reaches the Hub.

from __future__ import annotations

import json
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

SEEDS = {
    42: dict(
        local="output/adapters/chunkonly_s42/20260831_182411/adapter",
        repo="2026-08-31-qwen36-lora-table2-9284-difficult-advice-principle-scoped-702-rank-64-seed-42",
    ),
    69: dict(
        local="output/adapters/chunkonly_s69/20260831_182410/adapter",
        repo="2026-08-31-qwen36-lora-table2-9284-difficult-advice-principle-scoped-702-rank-64-seed-69",
    ),
}


def main(seed: int = 42, private: bool = False) -> None:
    """Publish one seed's adapter with the card the contract requires."""
    load_dotenv(ROOT / ".env")
    from huggingface_hub import HfApi

    from src.infra.huggingface import card_markdown, gate_push, hf_repo_id

    spec = SEEDS[int(seed)]
    local = ROOT / spec["local"]
    meta = json.loads((local / "training_meta.json").read_text())
    assert meta["thinking"] is True, (
        "this arm trains on real traces; the stamp must say so"
    )

    fields = {
        "experiment": (
            f"Seed replicate ({seed}) of the principle-scoped 702 difficult-advice arm — the "
            "project's difficult-advice BASELINE (docs/BASELINES.md). Table2 9,284 filtered + "
            "702 difficult-advice rows (7.03%) whose two refine stages were never shown the "
            "whole constitution. Trained so the baseline carries seed-to-seed variance: with "
            "one seed it could not be ranked against any sibling."
        ),
        "date_generated": "2026-08-31",
        "constitution": (
            "`claude_distilled_12_principles_mid` (9 principles). NOTE this arm's generator saw "
            "only ONE principle per stage — the constitution's preamble (the priority / "
            "conflict-resolution section) reached it through no stage at all."
        ),
        "source_repo": (
            "https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT — training "
            f"config `{meta['train_config']}` (its pre-rename name, as run)"
        ),
        "models": "Base `Qwen/Qwen3.6-27B`; corpus written by anthropic/claude-haiku-4.5 + anthropic/claude-sonnet-5 via OpenRouter.",
        "generation_config": (
            f"seed {seed}; LoRA r=64 alpha=128, lr 1e-4 cosine, warmup 0.05, global batch 16, "
            "1 epoch, dynamic batching, max_seq_len 8192, 2xH200 DDP, 625 steps. Identical to "
            "seed 0 but for seed/output_dir/hub_model_id."
        ),
        "schema": "PEFT LoRA adapter + `training_meta.json` (thinking stamp + the sha-pinned dataset).",
        "provenance": (
            f"`scripts/gpu/runpod_train.py up --bundle {meta['dataset']['repo']}` then "
            "`scratch/da716_seeds/push_adapters.py`. Data pinned at "
            f"{meta['dataset']['repo']}@{meta['dataset']['revision'][:12]} :: {meta['dataset']['file']}."
        ),
    }
    repo_id = hf_repo_id(spec["repo"])
    gate_push(repo_id, fields, what="adapter push")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=card_markdown(fields).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["README.md"],
    )
    print(f"pushed -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    fire.Fire(main)
