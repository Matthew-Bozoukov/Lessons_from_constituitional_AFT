# ABOUTME: THE Hugging Face module: one token resolution (reads and pushes alike), one
# ABOUTME: card contract (CLAUDE.md's fields — datasets, adapters, caches), one upload mechanic.

from __future__ import annotations

import os
import re

import yaml
from pathlib import Path

from huggingface_hub import HfApi

# The mandatory card fields from CLAUDE.md "Required metadata in the dataset card".
# The contract is uniform across artifact types — a LoRA adapter's `constitution` and
# `generation_config` (training hyperparams) matter as much as a dataset's; cards are
# always DERIVED from real run metadata, never filler.
# `constitution` is deliberately included: write "none" explicitly, never omit it.
REQUIRED_FIELDS = ("experiment", "date_generated", "constitution", "source_repo",
                   "models", "generation_config", "schema", "provenance")


def hf_token() -> str | None:
    """THE token resolution, reads and pushes alike: HUGGINGFACE_API_KEY or HF_TOKEN.

    The bare hub helpers read only HF_TOKEN/cached logins; everything in this repo
    resolves through here instead, so private-repo READS (a private adapter's
    training_meta.json, a private cache entry) work wherever pushes do.
    """
    return os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")


def hf_api() -> HfApi:
    """The one HfApi for every push, on the shared token resolution."""
    return HfApi(token=hf_token())


def hf_download(repo_id: str, filename: str, repo_type: str = "model", **kwargs) -> str:
    """hf_hub_download with the shared token resolution (see hf_token)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type=repo_type, token=hf_token(),
                           **kwargs)


def hf_snapshot(repo_id: str, **kwargs) -> str:
    """snapshot_download with the shared token resolution (see hf_token)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, token=hf_token(), **kwargs)


def card_front_matter(configs: list[dict]) -> str:
    """Render the README YAML front-matter declaring a repo's `configs:` (pure).

    Each entry: {"config_name": str, "data_files": str, "default": bool?}. Declared
    configs are how a multi-file dataset repo stays loadable: `load_dataset(repo)`
    fetches only the default config's files, and the dataset viewer stops globbing
    every jsonl into one schema. An empty list renders bare markers (no `configs:`
    key — the hub rejects an empty sequence).
    """
    lines = ["---"]
    if configs:
        lines.append("configs:")
        for c in configs:
            lines.append(f"- config_name: {c['config_name']}")
            lines.append(f"  data_files: {c['data_files']}")
            if c.get("default"):
                lines.append("  default: true")
    lines.append("---")
    return "\n".join(lines) + "\n"


def card_markdown(fields: dict, front_matter: dict | None = None) -> str:
    """Render the dataset card, refusing incomplete metadata (pure; unit-tested offline).

    Required fields render first; any extra fields (e.g. an adapter's `dataset` pin)
    follow in insertion order. `title` only names the heading, it never becomes a row.

    `front_matter` (e.g. {"tags": ["eval-run", "eval:odcv"]}) renders as a YAML block
    ahead of the markdown — the Hub indexes it, so `/api/datasets?filter=<tag>` finds
    the repo. This is HF's canonical discovery mechanism; the dashboard's eval-run
    picker relies on the `eval-run` + `eval:<name>` + `model:<key>` tags.
    """
    missing = [f for f in REQUIRED_FIELDS if not str(fields.get(f, "")).strip()]
    assert not missing, (f"dataset card is missing required fields {missing} — "
                         "write `constitution: none` explicitly if it connects to none")
    lines = []
    if front_matter:
        lines += ["---", yaml.safe_dump(front_matter, sort_keys=False).strip(), "---"]
    lines += ["# " + fields.get("title", fields["experiment"]), "", "| field | value |", "| --- | --- |"]
    extras = [k for k in fields if k not in REQUIRED_FIELDS and k != "title"]
    for key in (*REQUIRED_FIELDS, *extras):
        value = str(fields[key]).replace("\n", " ")
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


def pick_data_file(files: list[str], filename: str | None = None) -> str:
    """Choose the one data file from a dataset repo's file list (pure; unit-tested offline).

    Args:
        files: Every file in the repo (rfilenames).
        filename: Explicit choice; required when the repo holds more than one .jsonl.

    Returns:
        The chosen filename, never a guess: an ambiguous repo is a hard error.
    """
    if filename:
        assert filename in files, (
            f"data_file {filename!r} is not in the dataset repo (files: {sorted(files)})")
        return filename
    candidates = [f for f in files if f.endswith(".jsonl")]
    assert len(candidates) == 1, (
        f"expected exactly one .jsonl in the dataset repo, found {sorted(candidates)} — "
        "declare data_file: <name> to pick one")
    return candidates[0]


def resolve_dataset(repo_id: str, filename: str | None = None,
                    revision: str | None = None) -> tuple[str, dict]:
    """Pin an HF dataset repo to an exact revision and fetch its data file.

    The one way training data reaches a trainer: never a local path — the repo id is
    the provenance, and the revision it resolves to is recorded so the run is
    reproducible even after the dataset repo moves on.

    Args:
        repo_id: HF dataset repo id (`org/name`). Local paths are refused.
        filename: File inside the repo; required when it holds more than one .jsonl.
        revision: Branch/tag/sha to pin; defaults to the repo's current head.

    Returns:
        (local_path, {"repo", "file", "revision"}) with `revision` the resolved
        commit sha, ready for training_meta.json.
    """
    assert _HF_REPO_ID.match(repo_id) and not repo_id.endswith((".json", ".jsonl")) \
        and not Path(repo_id).exists(), (
        f"{repo_id!r} is not an HF dataset repo id (org/name): training data loads from "
        "Hugging Face, never from a local file (CLAUDE.md: artifacts live on HF)")
    info = hf_api().repo_info(repo_id, repo_type="dataset", revision=revision)
    chosen = pick_data_file([s.rfilename for s in info.siblings], filename)
    local = hf_download(repo_id, chosen, repo_type="dataset", revision=info.sha)
    return local, {"repo": repo_id, "file": chosen, "revision": info.sha}


def push_run_dir(out_dir: Path, repo_id: str, fields: dict, private: bool = False,
                 repo_type: str = "dataset", front_matter: dict | None = None) -> str:
    """Upload a run directory (with its card) to an HF repo.

    Args:
        out_dir: The directory to upload (eval run dir, adapter dir, ...).
        repo_id: Dated repo per the naming rule, e.g. org/2026-08-03-mmlu-<model_key>
            (adapter repos keep their model-key naming).
        fields: Card fields; all REQUIRED_FIELDS must be present and non-empty.
        private: PUBLIC by default (2026-08-24: the dashboard reads eval repos token-less); pass private=True deliberately for anything sensitive.
        repo_type: "dataset" (default) or "model" (adapters).

    Returns:
        The repo URL.
    """
    card = card_markdown(fields, front_matter)  # validate before any network call
    api = hf_api()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    # Explicit utf-8: cards are full of em-dashes, and upload_folder reads this file back
    # with encoding="utf8" to extract the front matter. On Windows the default encoding is
    # cp1252, so an unqualified write_text round-trips into a UnicodeDecodeError that
    # surfaces from deep inside huggingface_hub — after create_repo has already run, which
    # leaves an empty repo behind and reads as a hub problem rather than an encoding one.
    (out_dir / "README.md").write_text(card, encoding="utf-8")
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
