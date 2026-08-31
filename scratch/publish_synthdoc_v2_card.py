# ABOUTME: Makes the synthdoc-v2 difficult-advice corpus discoverable in the research dashboard.
# ABOUTME: Writes the card's configs+tags front-matter and a stats sidecar; uploads both.
"""The corpus rows are already on the Hub; this publishes the metadata the dashboard needs.

dashboard/lib/trainingData.ts discovers a corpus by ONE listing call:
`/api/datasets?author=LASR-Callum&filter=training-data` with cardData expanded. To appear
and be streamable a repo needs three things, and this repo was missing two:

  1. the `training-data` + `kind:` / `pipeline:` / `constitution:` tags   (present, but the
     constitution facet said `unknown`)
  2. a `configs:` block whose DEFAULT entry names the rows file          (missing -- without
     it the explorer has no file to stream)
  3. a stats sidecar (`stats.json`) for the record count and category breakdown  (missing)

Run: uv run python scratch/publish_synthdoc_v2_card.py --dry_run
     uv run python scratch/publish_synthdoc_v2_card.py
"""
import collections
import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

from src.huggingface import card_markdown, hf_token, training_data_tags
from src.utils import read_jsonl

REPO = "LASR-Callum/synthdoc-v2-difficult-advice"
RUN = Path("output/synthdoc_v2/20260803_211524")
ROWS_FILE = "stage_7_sft.jsonl"
CONSTITUTION = "constitutions/claude_distilled_12_principles_mid/constitution.md"


def _stats() -> dict:
    """Record count and per-trait breakdown, in the shape statsFromSidecar() reads.

    Returns:
        A sidecar dict with `rows` and a `by_source` map the dashboard renders as categories.
    """
    rows = list(read_jsonl(RUN / ROWS_FILE))
    by = collections.Counter(r["metadata"]["trait_id"] for r in rows)
    names = {}
    for r in rows:
        names.setdefault(r["metadata"]["trait_id"], r["metadata"].get("trait_name", ""))
    return {
        "rows": len(rows),
        "by_source": {f"{tid} {names[tid]}".strip(): {"examples": n}
                      for tid, n in sorted(by.items())},
    }


def _card(stats: dict) -> str:
    """The dataset card: required provenance fields plus the dashboard's front-matter."""
    manifest = json.loads((RUN / "manifest.json").read_text())
    cfg = manifest["config"]
    models = ", ".join(f"{stage}={spec['model']}" for stage, spec in cfg["models"].items())
    gen = ", ".join(
        f"{stage}(temp={spec.get('temperature')}, max_tokens={spec.get('max_tokens')})"
        for stage, spec in cfg["models"].items())
    front_matter = {
        "configs": [{"config_name": "default", "data_files": ROWS_FILE, "default": True}],
        "tags": training_data_tags("synth", "difficult_advice", CONSTITUTION),
    }
    fields = {
        "title": "synthdoc-v2 — difficult advice (constitution-grounded SFT corpus)",
        "experiment": (
            "Constitution-grounded difficult-advice corpus: a user faces an ethically "
            "ambiguous situation and the assistant reasons about its values before replying. "
            "The base corpus every ablation and restoration arm in the series derives from."),
        "date_generated": "2026-08-03",
        "constitution": CONSTITUTION,
        "source_repo": f"teaching_claude_why_replication @ {manifest['git_sha']}",
        "models": models,
        "generation_config": f"seed={cfg['seed']}, {gen}",
        "schema": (
            "messages[] (system, user, assistant with `content` and `reasoning_content`); "
            "metadata{scenario_id, trait_id, trait_name, trait_text, domain, shortcut, "
            "situation}"),
        "provenance": (
            "uv run synth run --config configs/data/synth/difficult_advice.yaml — all seven "
            "pipeline stages are published here; stage_7_sft.jsonl is the trainable output."),
        "rows": f"{stats['rows']} ({cfg['n_traits']} traits x "
                f"{cfg['scenarios_per_trait']} scenarios, minus generation losses)",
        "constitution_sha256": manifest["constitution_sha256"],
    }
    return card_markdown(fields, front_matter)


def main(dry_run: bool = False, create_pr: bool = False) -> None:
    """Write the card and sidecar, and upload them unless dry_run.

    Args:
        dry_run: Print what would be uploaded and stop.
        create_pr: Open a pull request instead of committing to main. Required when the
            token has PR-only rights on the target org.
    """
    load_dotenv()
    stats = _stats()
    card = _card(stats)
    print(card)
    print(json.dumps(stats, indent=1)[:400])
    if dry_run:
        print("\n[dry run] nothing uploaded")
        return

    api = HfApi(token=hf_token())
    res = api.create_commit(
        repo_id=REPO, repo_type="dataset", create_pr=create_pr or None,
        commit_message="dashboard: card configs + constitution tag + stats sidecar",
        commit_description=(
            "Makes the corpus discoverable and streamable in the research dashboard:\n"
            "- configs: default entry naming stage_7_sft.jsonl (the file the explorer streams)\n"
            "- constitution tag corrected from `unknown` to the manifest's constitution\n"
            "- stats.json sidecar: 2203 rows, per-trait breakdown"),
        operations=[
            CommitOperationAdd("README.md", card.encode()),
            CommitOperationAdd("stats.json", json.dumps(stats, indent=2).encode()),
        ])
    print(f"  {'PR opened' if create_pr else 'committed'}: {getattr(res, 'pr_url', None) or res}")
    print(f"\n>>> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    fire.Fire(main)
