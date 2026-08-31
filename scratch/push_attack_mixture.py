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

MIX = "output/attack_mixture/20260828_131455/t2_9284_attack716_10k.jsonl"
REPO = "matboz/2026-08-28-t2-9284-attack716-train"
FILE = "t2_9284_attack716_10k.jsonl"
CONSTITUTION = "constitutions/claude_distilled_12_principles_mid/constitution.md"


def _card() -> str:
    """Build the dataset card with configs+tags front-matter and required fields."""
    stats = json.loads(Path(MIX + ".stats.json").read_text())
    fm = {
        "configs": [{"config_name": "default", "data_files": FILE, "default": True}],
        "tags": training_data_tags("mixture", "t2-9284-attack716", CONSTITUTION,
                                   extra=("stage:final",)),
    }
    fields = {
        "title": "Attack-variant robustness mixture (9,284 + 716)",
        "experiment": (
            "Training mixture for the attack-variant robustness arm. 179 difficult-advice "
            "scenarios x four user-prompt framings that all press for the same norm-violating "
            "shortcut (original, hidden, incremental, authority); the assistant reply is held "
            "constant across a scenario's four framings (the correct refusal). Plus the same "
            "9,284 Table2 rows. Trains the model to hold its line when the ask is reframed."),
        "date_generated": "2026-08-28",
        "constitution": CONSTITUTION,
        "source_repo": "teaching_claude_why_replication",
        "models": ("attack framings: openai/gpt-5.6-terra (temp 0.7); assistant replies and "
                   "the Table2 half unchanged"),
        "generation_config": "three attack framings generated per scenario, temperature 0.7; original is corpus text",
        "variants": "179 scenarios x {original, hidden, incremental, authority} = 716 rows",
        "schema": (f"JSONL. `source` — difficult_advice_v2_attack | a Table2 source; `variant` on synth rows; `text` — rendered "
                   f"Qwen chat string (<|im_start|>{{role}}...<|im_end|>, assistant turns carry "
                   f"<think>...</think>); `scenario_id`,`trait_id` on the 716 synth rows. "
                   f"{stats['total']} rows = {stats['attack']} attack + {stats['table2']} Table2 "
                   f"({100 * stats['synth_fraction']:.2f}% synth)."),
        "provenance": ("scratch/make_attack_variants.py to generate framings, then "
                       "scratch/build_attack_mixture.py to merge with the da716 Table2 half."),
        "note": ("Robustness set: the assistant reply is authored against the original "
                 "framing, so on the three attack rows it is the correct target but does not "
                 "reference the specific manipulation. Scenarios are the 179-scenario "
                 "theme-balanced sample (t6 over-represented, t7 under)."),
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
        commit_message="attack-variant mixture: 9,284 Table2 + 716 four-framing difficult-advice",
        operations=[
            CommitOperationAdd("README.md", card.encode()),
            CommitOperationAdd(FILE, MIX),
            CommitOperationAdd(FILE + ".stats.json", stats_path),
        ])
    print(f"\n>>> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    fire.Fire(main)
