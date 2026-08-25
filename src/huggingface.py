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

# Hub-indexed card tags are HF's discovery mechanism: `/api/datasets?filter=<tag>` finds a
# repo by them, token-less, with no registry to keep in sync. Eval repos carry `eval-run`
# (src/eval/run_eval.py); every TRAINING CORPUS carries this family, stamped by the
# publishers (synth's StageCache, mix's push, properties/ablate) through
# `training_data_tags`, and the dashboard's /datasets explorer lists exactly these.
TRAINING_DATA_TAG = "training-data"
TRAINING_DATA_KINDS = ("synth", "mixture", "ablation", "fixture")


def hf_token() -> str | None:
    """THE token resolution, reads and pushes alike: HUGGINGFACE_API_KEY or HF_TOKEN.

    The bare hub helpers read only HF_TOKEN/cached logins; everything in this repo
    resolves through here instead, so private-repo READS (a private adapter's
    training_meta.json, a private cache entry) work wherever pushes do.

    `.env` is loaded HERE rather than being inherited from whichever import happened to
    call `load_dotenv()` first. Until 2026-08-20 that was a side effect of importing
    `src.endpoints.openrouter`, so an entry point that only needed the Hub — a push
    script, a card refresh — got a bare `401 Unauthorized` from `create_repo` after doing
    all of its work. `load_dotenv` does not override an env var that is already set, so
    calling it on every resolution is free and cannot shadow a deliberate export.
    """
    from dotenv import load_dotenv

    load_dotenv()
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


def tag_safe(text: str) -> str:
    """Make a value usable inside a Hub tag: no whitespace or slashes, bounded length."""
    return re.sub(r"[\s/]+", "-", str(text).strip()).strip("-")[:64] or "unknown"


def constitution_slug(value: str | None) -> str:
    """The `constitution:<slug>` tag value for a card's constitution field (pure).

    Card fields name the constitution as a repo path, sometimes followed by prose
    (`constitutions/<name>/constitution.md — the constitution the pool ...`); the tag
    carries the `<name>` alone. `none` stays `none` — a corpus that connects to no
    constitution says so explicitly (CLAUDE.md), and the tag repeats it.
    """
    # Hand-written cards wrap the name in backticks or a markdown link, or name the
    # file (`claude_approved_constitution.md`) rather than the folder.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", str(value or "")).replace("`", "").strip()
    if not text or text.lower().startswith("none"):
        return "none"
    m = (re.search(r"constitutions/([A-Za-z0-9._-]+?)(?:/|\s|$)", text)
         or re.search(r"\b(claude_distilled_[A-Za-z0-9_]+)", text)
         or re.search(r"([A-Za-z0-9._-]+)/constitution\.md\b", text)   # <dir>/constitution.md
         or re.search(r"\b([A-Za-z0-9_-]+)\.md\b", text))
    if m:
        return m.group(1)
    # Free text: the name up to the first dash-clause, parenthesis or colon. Hyphens
    # inside a name ("Claude-approved constitution") are part of it, not separators.
    return tag_safe(re.split(r"\s*(?:—|\(|:)\s*|\s-\s", text, maxsplit=1)[0])[:40]


def training_data_tags(kind: str, pipeline: str, constitution: str | None, *,
                       smoke: bool = False, extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    """The card tags every training corpus carries (pure; unit-tested offline).

    Args:
        kind: One of TRAINING_DATA_KINDS — which publisher produced the rows.
        pipeline: The producing recipe: a synth document type (`difficult_advice`), a
            mixture config stem (`qwen36_less_top10`), an ablation tag.
        constitution: The card's constitution field, reduced by `constitution_slug`.
        smoke: Smoke runs are tagged so the dashboard can fold them away by default.
        extra: Further `key:value` facets (a mixture's `stage:final`, say).

    Returns:
        `[training-data, kind:<kind>, pipeline:<pipeline>, constitution:<slug>, ...]`.
    """
    assert kind in TRAINING_DATA_KINDS, f"kind must be one of {TRAINING_DATA_KINDS}: {kind!r}"
    tags = [TRAINING_DATA_TAG, f"kind:{kind}", f"pipeline:{tag_safe(pipeline)}",
            f"constitution:{constitution_slug(constitution)}"]
    if smoke:
        tags.append("smoke")
    return tags + [str(t) for t in extra]


def _front_matter_block(front_matter: dict) -> str:
    """One renderer for a card's YAML front-matter, so every publisher's block parses alike."""
    return "---\n" + yaml.safe_dump(front_matter, sort_keys=False).strip() + "\n---\n"


def card_front_matter(configs: list[dict], tags: tuple[str, ...] | list[str] = ()) -> str:
    """Render the README YAML front-matter declaring a repo's `configs:` and `tags:` (pure).

    Each config: {"config_name": str, "data_files": str, "default": bool?}. Declared
    configs are how a multi-file dataset repo stays loadable: `load_dataset(repo)`
    fetches only the default config's files, and the dataset viewer stops globbing
    every jsonl into one schema — and the default config's `data_files` is the file
    the dashboard streams. Empty inputs render bare markers (no `configs:` key — the
    hub rejects an empty sequence).
    """
    front_matter: dict = {}
    if configs:
        front_matter["configs"] = [dict(c) for c in configs]
    if tags:
        front_matter["tags"] = [str(t) for t in tags]
    return _front_matter_block(front_matter) if front_matter else "---\n---\n"


def card_markdown(fields: dict, front_matter: dict | None = None) -> str:
    """Render the dataset card, refusing incomplete metadata (pure; unit-tested offline).

    Required fields render first; any extra fields (e.g. an adapter's `dataset` pin)
    follow in insertion order. `title` only names the heading, it never becomes a row.

    `front_matter` (e.g. {"tags": ["eval-run", "eval:odcv"]}) renders as a YAML block
    ahead of the markdown — the Hub indexes it, so `/api/datasets?filter=<tag>` finds
    the repo. This is HF's canonical discovery mechanism; the dashboard's eval-run
    picker relies on the `eval-run` + `eval:<name>` + `model:<key>` tags, and its
    /datasets explorer on `training_data_tags` plus a `configs:` default data file.
    """
    missing = [f for f in REQUIRED_FIELDS if not str(fields.get(f, "")).strip()]
    assert not missing, (f"dataset card is missing required fields {missing} — "
                         "write `constitution: none` explicitly if it connects to none")
    lines = [_front_matter_block(front_matter).rstrip("\n")] if front_matter else []
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


def resolve_run_dir(repo_id: str, revision: str | None = None,
                    local_dir: str | Path | None = None) -> tuple[str, dict]:
    """Pin a whole dataset repo to an exact revision and fetch it as a directory.

    `resolve_dataset` is the one-jsonl flavour and is what a training mixture needs. A run
    directory is not one file: an eval run is a tree of per-rollout transcripts plus its
    score and metadata files, and every consumer walks it with `rglob`. This is the same
    contract — repo id in, exact sha out — for that shape.

    Args:
        repo_id: HF dataset repo id (`org/name`).
        revision: Branch/tag/sha to pin; defaults to the repo's current head.
        local_dir: Materialise into this directory instead of the shared HF cache. Worth
            passing on Windows, where the cache cannot use symlinks and a cached tree is
            awkward to read by path.

    Returns:
        (local directory, {"repo", "revision"}) with `revision` the resolved commit sha.
    """
    assert _HF_REPO_ID.match(repo_id) and not Path(repo_id).exists(), (
        f"{repo_id!r} is not an HF dataset repo id (org/name)")
    info = hf_api().repo_info(repo_id, repo_type="dataset", revision=revision)
    path = hf_snapshot(repo_id, repo_type="dataset", revision=info.sha,
                       **({"local_dir": str(local_dir)} if local_dir else {}))
    return path, {"repo": repo_id, "revision": info.sha}


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


def push_files(paths: list[Path], repo_id: str, fields: dict, private: bool = True,
               front_matter: dict | None = None) -> str:
    """Upload named files (with their card) to an HF dataset repo, keeping basenames.

    The checkpoint-push flavour: a staged pipeline pushes exactly the files each stage
    produced, never its whole working directory — pushing a directory would drag every
    later stage's artifacts into an earlier stage's repo on re-upload.

    Args:
        paths: Files to upload; each lands at its basename in the repo.
        repo_id: Dated repo per the naming rule.
        fields: Card fields; all REQUIRED_FIELDS must be present and non-empty.
        private: Create the repo private (default).
        front_matter: Card YAML front-matter — a training corpus passes its
            `configs:` (default data file) and `training_data_tags` here.

    Returns:
        The repo URL.
    """
    card = card_markdown(fields, front_matter)  # validate before any network call
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
