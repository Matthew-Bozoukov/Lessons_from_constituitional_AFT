# ABOUTME: THE Hugging Face module: one token resolution (reads and pushes alike), one
# ABOUTME: card contract (CLAUDE.md's fields — datasets, adapters, caches), one upload mechanic.

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

# The mandatory card fields from CLAUDE.md "Required metadata in the dataset card".
# The contract is uniform across artifact types — a LoRA adapter's `constitution` and
# `generation_config` (training hyperparams) matter as much as a dataset's; cards are
# always DERIVED from real run metadata, never filler.
# `constitution` is deliberately included: write "none" explicitly, never omit it.
REQUIRED_FIELDS = ("experiment", "date_generated", "constitution", "source_repo",
                   "models", "generation_config", "schema", "provenance")


def hf_api() -> HfApi:
    """The one token resolution for every push: HUGGINGFACE_API_KEY or HF_TOKEN."""
    return HfApi(token=os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN"))


def hf_download(repo_id: str, filename: str, repo_type: str = "model", **kwargs) -> str:
    """hf_hub_download with the shared token resolution.

    The bare helper reads only HF_TOKEN/cached logins; this also honours
    HUGGINGFACE_API_KEY, so private-repo READS (a private adapter's training_meta.json,
    a private cache entry) work wherever pushes do.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id, filename, repo_type=repo_type,
        token=os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN"),
        **kwargs)


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


def push_run_dir(out_dir: Path, repo_id: str, fields: dict, private: bool = True,
                 repo_type: str = "dataset") -> str:
    """Upload a run directory (with its card) to an HF repo.

    Args:
        out_dir: The directory to upload (eval run dir, adapter dir, ...).
        repo_id: Dated repo per the naming rule, e.g. org/2026-08-03-mmlu-<model_key>
            (adapter repos keep their model-key naming).
        fields: Card fields; all REQUIRED_FIELDS must be present and non-empty.
        private: Create the repo private (default) — flip deliberately, not by accident.
        repo_type: "dataset" (default) or "model" (adapters).

    Returns:
        The repo URL.
    """
    card = card_markdown(fields)  # validate before any network call
    api = hf_api()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    (out_dir / "README.md").write_text(card)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type=repo_type)
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def push_files(paths: list[Path], repo_id: str, fields: dict, private: bool = True) -> str:
    """Upload named files (with their card) to an HF dataset repo, keeping basenames.

    The checkpoint-push flavour: a staged pipeline pushes exactly the files each stage
    produced, never its whole working directory — pushing a directory would drag every
    later stage's artifacts into an earlier stage's repo on re-upload.

    Args:
        paths: Files to upload; each lands at its basename in the repo.
        repo_id: Dated repo per the naming rule.
        fields: Card fields; all REQUIRED_FIELDS must be present and non-empty.
        private: Create the repo private (default).

    Returns:
        The repo URL.
    """
    card = card_markdown(fields)  # validate before any network call
    missing = [str(p) for p in paths if not Path(p).is_file()]
    assert not missing, f"push_files: not files: {missing}"
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    for p in paths:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=Path(p).name,
                        repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
