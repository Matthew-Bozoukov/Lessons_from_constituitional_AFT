# ABOUTME: Pushes the advice-framed training mixture to HF with a dashboard-ready card.
# ABOUTME: 9,284 Table2 + 716 advice-framed difficult-advice rows, rendered text format.
"""Publish the advice mixture with the required dataset-card fields.

Run: uv run python scratch/push_advice_mixture.py --dry_run
     uv run python scratch/push_advice_mixture.py
"""
import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

from src.huggingface import card_markdown, hf_token, training_data_tags

MIX = "output/advice_mixture/20260828_125534/t2_9284_advice716_10k.jsonl"
REPO = "matboz/2026-08-28-t2-9284-advice716-train"
FILE = "t2_9284_advice716_10k.jsonl"
CONSTITUTION = "constitutions/claude_distilled_12_principles_mid/constitution.md"


def _card() -> str:
    """Build the dataset card with configs+tags front-matter and required fields."""
    stats = json.loads(Path(MIX + ".stats.json").read_text())
    fm = {
        "configs": [{"config_name": "default", "data_files": FILE, "default": True}],
        "tags": training_data_tags("mixture", "t2-9284-advice716", CONSTITUTION,
                                   extra=("stage:final",)),
    }
    fields = {
        "title": "Advice-framed difficult-advice mixture (9,284 + 716)",
        "experiment": (
            "Training mixture for the advice-framed arm. The 716-row difficult-advice corpus "
            "behind qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64, lightly rewritten so "
            "the user asks whether to take the tempting shortcut and the assistant purely "
            "advises (framing only; wording/stance preserved). Plus the same 9,284 Table2 "
            "rows. A downstream difference against the synthdoc-716 arm isolates the "
            "advice-vs-action framing of the difficult-advice half."),
        "date_generated": "2026-08-28",
        "constitution": CONSTITUTION,
        "source_repo": "teaching_claude_why_replication",
        "models": ("advice rewrite: openai/gpt-5.6-terra (temp 0.3); difficult-advice content "
                   "and Table2 half unchanged from the synthdoc-716 mixture"),
        "generation_config": "light rewrite of user+reasoning+answer, temperature 0.3, seed n/a (deterministic selection)",
        "schema": (f"JSONL. `source` — difficult_advice_v2 | a Table2 source; `text` — rendered "
                   f"Qwen chat string (<|im_start|>{{role}}...<|im_end|>, assistant turns carry "
                   f"<think>...</think>); `scenario_id`,`trait_id` on the 716 synth rows. "
                   f"{stats['total']} rows = {stats['advice']} advice + {stats['table2']} Table2 "
                   f"({100 * stats['synth_fraction']:.2f}% synth)."),
        "provenance": ("scratch/advice_light_rewrite.py over the 716 corpus, then "
                       "scratch/build_advice_mixture.py to merge with the da716 Table2 half."),
        "note": ("Advice framing only — the tempting shortcut and the assistant's stance "
                 "(decline it) are unchanged from the source corpus. One row (t1_b01_s007) was "
                 "recovered from the rendered mixture rather than stage_7."),
    }
    return card_markdown(fields, fm)


def main(dry_run: bool = False) -> None:
    """Push the mixture and card to HF.

    Args:
        dry_run: Print the card and stop.
    """
    load_dotenv()
    card = _card()
    print(card)
    if dry_run:
        print("\n[dry run] nothing uploaded")
        return
    api = HfApi(token=hf_token())
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    stats_path = MIX + ".stats.json"
    api.create_commit(
        repo_id=REPO, repo_type="dataset",
        commit_message="advice-framed mixture: 9,284 Table2 + 716 advice difficult-advice",
        operations=[
            CommitOperationAdd("README.md", card.encode()),
            CommitOperationAdd(FILE, MIX),
            CommitOperationAdd(FILE + ".stats.json", stats_path),
        ])
    print(f"\n>>> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    fire.Fire(main)
