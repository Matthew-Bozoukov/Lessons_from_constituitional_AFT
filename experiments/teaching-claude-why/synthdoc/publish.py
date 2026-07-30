# ABOUTME: Publish a dataset to Hugging Face in the shape the Research Log visualizer reads,
# ABOUTME: and build the AGENTS.md-mandated dataset card that every upload in this repo carries.

"""Publishing to Hugging Face, in the repository's house style.

Two things live here:

1. :func:`dataset_card` - the card required by the root ``AGENTS.md``. Every
   upload from this repository carries ``experiment``, ``date_generated``,
   ``constitution``, ``source_repo``, ``models``, ``generation_config``,
   ``schema`` and ``provenance``. ``constitution`` is mandatory and may be the
   literal string ``none``, but never omitted - it is the field a future reader
   needs most and the one most easily lost.

2. :func:`publish_petri_run` / :func:`publish_dialogue_dataset` - upload a bundle
   in the layout the visualizer's build expects:

   .. code-block:: text

       <repo>/
         README.md                  the dataset card
         manifest.json              SMALL. The only file the site build fetches.
         transcripts/<id>.json      one per transcript, fetched lazily in-browser
         chunks/chunk-NNN.json      dialogue records, paged lazily in-browser
         data/, results/, artifacts/, assets/   canonical export, byte-identical

   The split matters: ``manifest.json`` is what the static build bakes in, so it
   must stay small; everything under ``transcripts/`` and ``chunks/`` is fetched
   by the browser only when a reader opens that item. Uploading the bulk without
   the shards would work but would force the visualizer to choose between a slow
   build and a slow page.

Uploads reuse :mod:`huggingface_hub` exactly as :mod:`synthdoc.snapshots` does.
The token is read from the environment by ``huggingface_hub`` itself and is
never read, logged, or echoed here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import git_sha

#: Repo names are `<YYYY-MM-DD>-<short-experiment-description>`, by AGENTS.md.
REPO_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Files the visualizer build reads. Everything else is linked, not fetched.
MANIFEST_NAME = "manifest.json"


def _slug(value: str) -> str:
    """Match the visualizer's `slugify`, so shard names line up on both sides."""
    out = re.sub(r"[^a-z0-9]+", "-", str(value).lower().strip())
    return out.strip("-")[:80]


def validate_repo_name(repo_id: str) -> str:
    """Check a repo id against the naming convention.

    Args:
        repo_id: Either ``name`` or ``org/name``.

    Returns:
        The repo id unchanged.

    Raises:
        ValueError: If the name part does not start with an ISO date followed by
            a kebab-case description.
    """
    name = repo_id.split("/")[-1]
    if not REPO_NAME_RE.match(name):
        raise ValueError(
            f"Repo name {name!r} does not follow <YYYY-MM-DD>-<short-experiment-description>. "
            "The date is the date the data was GENERATED, not uploaded. "
            "Example: 2026-07-29-msm-philosophy-spec-focused-discovery"
        )
    return repo_id


@dataclass
class CardFields:
    """The metadata every upload from this repository must carry.

    Attributes:
        experiment: Which experiment produced this, in one sentence.
        date_generated: ISO date the data was produced.
        constitution: The constitution, spec or model spec this connects to, by
            name and link. Pass the literal ``"none"`` if it genuinely connects
            to none - passing an empty value is an error, not a shortcut.
        source_repo: Repository URL or path the generating code lives in.
        models: Model id -> revision/pin (or a description of the pin).
        generation_config: Sampling settings: temperature, top_p, max_tokens, seeds.
        schema: Field name -> what that field means.
        provenance: Exact script and arguments that regenerate the data.
        source_commit: Commit the generating code was at. Defaults to HEAD.
        extra: Anything else worth recording; rendered after the required table.
    """

    experiment: str
    date_generated: str
    constitution: str
    source_repo: str
    models: dict[str, str]
    generation_config: dict[str, Any]
    schema: dict[str, str]
    provenance: str
    source_commit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [
            name
            for name in (
                "experiment",
                "date_generated",
                "constitution",
                "source_repo",
                "provenance",
            )
            if not str(getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError(
                "Dataset card is missing required field(s): "
                + ", ".join(missing)
                + ". `constitution` must be stated explicitly - write 'none' if "
                "the data genuinely connects to no spec."
            )
        for name in ("models", "generation_config", "schema"):
            if not getattr(self, name):
                raise ValueError(
                    f"Dataset card field {name!r} is empty. State it, even if the "
                    "answer is a single entry."
                )
        try:
            _date.fromisoformat(self.date_generated)
        except ValueError as exc:
            raise ValueError(
                f"date_generated {self.date_generated!r} is not an ISO date"
            ) from exc
        if not self.source_commit:
            self.source_commit = git_sha()


def _table(rows: Iterable[tuple[str, str]]) -> str:
    body = "\n".join(f"| `{k}` | {v} |" for k, v in rows)
    return f"| field | value |\n| --- | --- |\n{body}\n"


def _fenced(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```\n"


def dataset_card(
    fields: CardFields,
    title: str,
    body: str = "",
    data_files: Sequence[tuple[str, str]] = (),
) -> str:
    """Render the dataset card (`README.md`) for an upload.

    Args:
        fields: The required metadata.
        title: Card heading.
        body: Optional prose inserted after the metadata, e.g. the run's
            research note.
        data_files: ``(split, path)`` pairs declared in the YAML header so the
            Hub's dataset viewer can load them.

    Returns:
        Markdown, ready to write to `README.md` in the repo root.
    """
    header = ["---"]
    if data_files:
        header += ["configs:", "- config_name: default", "  data_files:"]
        header += [f"  - split: {split}\n    path: {path}" for split, path in data_files]
    # Machine-readable copy of the required fields, so a consumer does not have
    # to parse the prose table.
    header += [
        "annotations_creators:",
        "- machine-generated",
        "tags:",
        "- research-log",
        f"- experiment-date-{fields.date_generated}",
        "---",
    ]

    required = _table(
        [
            ("experiment", fields.experiment),
            ("date_generated", fields.date_generated),
            ("constitution", fields.constitution),
            ("source_repo", f"{fields.source_repo} @ `{fields.source_commit}`"),
            (
                "models",
                ", ".join(f"`{k}` ({v})" for k, v in fields.models.items()),
            ),
            ("generation_config", "see below"),
            ("schema", "see below"),
            ("provenance", f"`{fields.provenance}`"),
        ]
    )

    parts = [
        "\n".join(header),
        f"\n# {title}\n",
        body.strip() + "\n" if body.strip() else "",
        "## Required metadata\n",
        required,
        "\n## generation_config\n",
        _fenced(fields.generation_config),
        "\n## schema\n",
        _table([(k, v) for k, v in fields.schema.items()]),
        "\n## provenance\n",
        f"Regenerate with:\n\n```bash\n{fields.provenance}\n```\n",
    ]
    if fields.extra:
        parts += ["\n## Additional detail\n", _fenced(fields.extra)]
    parts += [
        "\n## Layout\n",
        "`manifest.json` is the small index the Research Log visualizer bakes\n"
        "into its static build. Per-item payloads under `transcripts/` and\n"
        "`chunks/` are fetched by the browser only when a reader opens that\n"
        "item. The canonical export directories are byte-identical to the\n"
        "originals and are what you should cite.\n",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    # Passing token=None lets huggingface_hub resolve HF_TOKEN / the stored
    # login itself. The value is never read into this process's own variables.
    return HfApi(token=token)


def _ensure_repo(api, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON") from exc
    return records


def _messages_for(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversation", "turns", "dialogue"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    if "prompt" in record or "response" in record:
        return [
            {"role": "user", "content": str(record.get("prompt", ""))},
            {"role": "assistant", "content": str(record.get("response", ""))},
        ]
    return []


def _stage(staging: Path, relative: str, payload: Any) -> int:
    """Write one file into the staging tree and return its size."""
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    target.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _upload(staging: Path, repo_id: str, private: bool, message: str, dry_run: bool) -> str:
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    count = sum(1 for f in staging.rglob("*") if f.is_file())
    if dry_run:
        listing = "\n".join(
            f"  {f.relative_to(staging).as_posix()}  {f.stat().st_size:>9,} B"
            for f in sorted(staging.rglob("*"))
            if f.is_file()
        )
        return (
            f"DRY RUN - would upload {count} file(s), {total:,} B to "
            f"https://huggingface.co/datasets/{repo_id}\n{listing}"
        )
    api = _api()
    _ensure_repo(api, repo_id, private)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    return (
        f"Uploaded {count} file(s), {total:,} B to "
        f"https://huggingface.co/datasets/{repo_id}"
    )


def publish_petri_run(
    export_dir: Path | str,
    repo_id: str,
    card: CardFields,
    *,
    private: bool = False,
    dry_run: bool = False,
    staging_dir: Path | str | None = None,
) -> str:
    """Publish a Petri export bundle in the shape the visualizer reads.

    The input is the export directory described in
    ``dashboard/docs/CLAUDE_CODE_PETRI_EXPORT_GUIDE.md``: ``index.md``,
    ``data/scenarios.jsonl``, ``results/transcripts.jsonl``,
    ``results/scores.json`` and optional ``artifacts/`` and ``assets/``. Those
    files are uploaded unchanged; the shards and the manifest are derived.

    Args:
        export_dir: The export bundle.
        repo_id: Target dataset repo, ``org/<YYYY-MM-DD>-<description>``.
        card: Required dataset-card metadata.
        private: Create the repo private. Public datasets need no reader token.
        dry_run: Stage and report without touching the Hub.
        staging_dir: Where to assemble the upload. Defaults to a temp directory.

    Returns:
        A human-readable result line.

    Raises:
        FileNotFoundError: If the bundle is missing `results/transcripts.jsonl`.
        ValueError: If the repo name breaks the naming convention.
    """
    import shutil
    import tempfile

    validate_repo_name(repo_id)
    export = Path(export_dir)
    transcripts_path = export / "results" / "transcripts.jsonl"
    if not transcripts_path.exists():
        raise FileNotFoundError(
            f"{transcripts_path} not found. publish_petri_run expects the export "
            "layout from dashboard/docs/CLAUDE_CODE_PETRI_EXPORT_GUIDE.md."
        )

    owned_temp = staging_dir is None
    staging = Path(staging_dir or tempfile.mkdtemp(prefix="petri-publish-"))
    try:
        if staging.exists() and owned_temp:
            pass
        staging.mkdir(parents=True, exist_ok=True)

        # 1. Canonical export, copied verbatim. This is what a citation points at.
        for sub in ("data", "results", "artifacts", "assets"):
            source = export / sub
            if source.is_dir():
                shutil.copytree(source, staging / sub, dirs_exist_ok=True)

        transcripts = _read_jsonl(transcripts_path)
        scenarios_path = export / "data" / "scenarios.jsonl"
        scores_path = export / "results" / "scores.json"

        # 2. Per-transcript shards: the lazy half of the split.
        index: list[dict[str, Any]] = []
        for record in transcripts:
            name = f"{_slug(record['id'])}.json"
            size = _stage(staging, f"transcripts/{name}", record)
            index.append(
                {
                    "id": str(record["id"]),
                    "file": name,
                    "scenario_id": str(record.get("scenario_id", "")),
                    "category": str(record.get("category", "uncategorized")),
                    "outcome": str(record.get("outcome", "unknown")),
                    "scores": record.get("scores", {}) or {},
                    "tags": list(record.get("tags", []) or []),
                    "message_count": len(_messages_for(record)),
                    "size_bytes": size,
                }
            )

        # 3. The small manifest: the only file the site build downloads.
        manifest = {
            "manifest_version": 1,
            "kind": "petri-run",
            "experiment": card.experiment,
            "date_generated": card.date_generated,
            "constitution": card.constitution,
            "source_repo": {"url": card.source_repo, "commit": card.source_commit},
            "models": card.models,
            "generation_config": card.generation_config,
            "provenance": card.provenance,
            "scenarios": _read_jsonl(scenarios_path) if scenarios_path.exists() else [],
            "scores": json.loads(scores_path.read_text(encoding="utf-8"))
            if scores_path.exists()
            else {},
            "transcripts": index,
            "transcript_count": len(index),
            "total_transcript_bytes": sum(item["size_bytes"] for item in index),
        }
        manifest_bytes = _stage(staging, MANIFEST_NAME, manifest)
        if manifest_bytes > 512_000:
            # Not fatal, but the manifest is baked into every page of the static
            # site, so it growing without bound is the one regression that
            # silently undoes this whole design.
            print(
                f"WARNING: {MANIFEST_NAME} is {manifest_bytes:,} B. It is baked "
                "into the static build; consider moving more per-item detail "
                "into the transcript shards."
            )

        # 4. The card.
        note = ""
        index_md = export / "index.md"
        if index_md.exists():
            text = index_md.read_text(encoding="utf-8")
            note = text.split("---", 2)[-1].strip() if text.startswith("---") else text
        _stage(
            staging,
            "README.md",
            dataset_card(
                card,
                title=f"Petri audit: {card.experiment}",
                body=note,
                data_files=(("transcripts", "results/transcripts.jsonl"),),
            ),
        )

        return _upload(
            staging, repo_id, private, f"Publish Petri run {repo_id}", dry_run
        )
    finally:
        if owned_temp and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def publish_dialogue_dataset(
    jsonl_path: Path | str,
    repo_id: str,
    card: CardFields,
    *,
    chunk_size: int = 50,
    private: bool = False,
    dry_run: bool = False,
    staging_dir: Path | str | None = None,
) -> str:
    """Publish a dialogue JSONL as a visualizer-readable dataset.

    Records are uploaded whole (``data/dialogues.jsonl``) and also pre-chunked
    into ``chunks/chunk-NNN.json``, which is what the dataset browser pages
    through without ever loading the whole corpus.

    Args:
        jsonl_path: Source JSONL, one dialogue record per line.
        repo_id: Target dataset repo, ``org/<YYYY-MM-DD>-<description>``.
        card: Required dataset-card metadata.
        chunk_size: Records per lazily-fetched chunk.
        private: Create the repo private.
        dry_run: Stage and report without touching the Hub.
        staging_dir: Where to assemble the upload.

    Returns:
        A human-readable result line.
    """
    import shutil
    import tempfile

    validate_repo_name(repo_id)
    source = Path(jsonl_path)
    records = _read_jsonl(source)

    owned_temp = staging_dir is None
    staging = Path(staging_dir or tempfile.mkdtemp(prefix="dataset-publish-"))
    try:
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, _ensure_parent(staging / "data" / "dialogues.jsonl"))

        chunks: list[str] = []
        total = 0
        for start in range(0, len(records), chunk_size):
            name = f"chunks/chunk-{start // chunk_size:03d}.json"
            total += _stage(staging, name, records[start : start + chunk_size])
            chunks.append(name)

        turns = [len(_messages_for(r)) for r in records]
        roles: dict[str, int] = {}
        splits: dict[str, int] = {}
        categories: dict[str, int] = {}
        for record in records:
            for message in _messages_for(record):
                role = str(message.get("role", "unknown"))
                roles[role] = roles.get(role, 0) + 1
            meta = record.get("metadata") or {}
            split = str(meta.get("split", record.get("split", "unspecified")))
            category = str(meta.get("category", record.get("category", "uncategorized")))
            splits[split] = splits.get(split, 0) + 1
            categories[category] = categories.get(category, 0) + 1

        manifest = {
            "manifest_version": 1,
            "kind": "dialogue-dataset",
            "experiment": card.experiment,
            "date_generated": card.date_generated,
            "constitution": card.constitution,
            "source_repo": {"url": card.source_repo, "commit": card.source_commit},
            "models": card.models,
            "generation_config": card.generation_config,
            "provenance": card.provenance,
            "dataset": {
                "source_file": "data/dialogues.jsonl",
                "format": "jsonl",
                "record_count": len(records),
                "chunk_size": chunk_size,
                "chunks": chunks,
                "total_bytes": total,
                "stats": {
                    "average_turns": round(sum(turns) / len(turns), 1) if turns else 0,
                    "role_counts": roles,
                    "splits": splits,
                    "categories": categories,
                },
            },
        }
        _stage(staging, MANIFEST_NAME, manifest)
        _stage(
            staging,
            "README.md",
            dataset_card(
                card,
                title=f"Dialogue dataset: {card.experiment}",
                data_files=(("train", "data/dialogues.jsonl"),),
            ),
        )
        return _upload(staging, repo_id, private, f"Publish dataset {repo_id}", dry_run)
    finally:
        if owned_temp and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def card_from_file(path: Path | str) -> CardFields:
    """Load :class:`CardFields` from a JSON or YAML file.

    Keeping the card in a file next to the run config is what makes an upload
    reproducible: the metadata is reviewed in git, not retyped at the shell.

    Args:
        path: JSON or YAML file with the card fields as top-level keys.

    Returns:
        The parsed card.
    """
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    known = {f for f in CardFields.__dataclass_fields__}
    extra = {k: v for k, v in data.items() if k not in known}
    kept = {k: v for k, v in data.items() if k in known}
    kept.setdefault("extra", {})
    kept["extra"] = {**kept["extra"], **extra}
    return CardFields(**kept)
