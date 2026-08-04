# ABOUTME: HF-push epilogue for eval runs: enforce the CLAUDE.md dataset-card fields and
# ABOUTME: upload a run directory to a dated HF dataset repo. Owned by run_eval.py, not evals.

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

# The mandatory card fields from CLAUDE.md "Required metadata in the dataset card".
# `constitution` is deliberately included: write "none" explicitly, never omit it.
REQUIRED_FIELDS = ("experiment", "date_generated", "constitution", "source_repo",
                   "models", "generation_config", "schema", "provenance")


def card_markdown(fields: dict) -> str:
    """Render the dataset card, refusing incomplete metadata (pure; unit-tested offline)."""
    missing = [f for f in REQUIRED_FIELDS if not str(fields.get(f, "")).strip()]
    assert not missing, (f"dataset card is missing required fields {missing} — "
                         "write `constitution: none` explicitly if it connects to none")
    lines = ["# " + fields.get("title", fields["experiment"]), "", "| field | value |", "| --- | --- |"]
    for key in REQUIRED_FIELDS:
        value = str(fields[key]).replace("\n", " ")
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


def push_run_dir(out_dir: Path, repo_id: str, fields: dict, private: bool = True) -> str:
    """Upload an eval run directory (with its card) to an HF dataset repo.

    Args:
        out_dir: The per-target run directory (rollouts, results.json, run_meta.json).
        repo_id: Dated repo per the naming rule, e.g. org/2026-08-03-mmlu-<model_key>.
        fields: Card fields; all REQUIRED_FIELDS must be present and non-empty.
        private: Create the repo private (default) — flip deliberately, not by accident.

    Returns:
        The repo URL.
    """
    card = card_markdown(fields)  # validate before any network call
    api = HfApi(token=os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    (out_dir / "README.md").write_text(card)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
