# ABOUTME: Per-stage corpus snapshots: an explicit parquet schema identical across
# ABOUTME: stages, full-fidelity JSONL, and non-blocking HuggingFace pushes.

from __future__ import annotations

import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .core.types import Document

# Turn struct, fixed so every split has the same shape and the dataset viewer diffs cleanly.
TURN_TYPE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.string()),
        ("thinking", pa.string()),
        ("tool_calls", pa.string()),
    ]
)


def build_schema(axis_names: Sequence[str], filter_fields: Sequence[str]) -> pa.Schema:
    """Build the snapshot schema.

    The schema is declared explicitly rather than inferred so that every stage -
    including stage_00, where no filter has run - has identical columns. Inference
    would give stage_00 a null filter_scores column and stage_NN a struct, and the
    two splits would stop being comparable.

    Args:
        axis_names: The axes this run samples; become the `axes` struct fields.
        filter_fields: Union of every configured filter's score fields.

    Returns:
        The pyarrow schema.
    """
    axes_type = pa.struct([(name, pa.string()) for name in sorted(axis_names)]) \
        if axis_names else pa.struct([("_none", pa.string())])
    scores_type = pa.struct([(name, pa.float64()) for name in sorted(filter_fields)]) \
        if filter_fields else pa.struct([("_none", pa.float64())])
    return pa.schema(
        [
            ("doc_id", pa.string()),
            ("scenario_hash", pa.string()),
            ("stage_idx", pa.int32()),
            ("stage_name", pa.string()),
            ("input_doc_id", pa.string()),
            ("spec_id", pa.string()),
            ("chunk_ids", pa.list_(pa.string())),
            ("grouping_strategy", pa.string()),
            ("doc_type", pa.string()),
            ("axes", axes_type),
            ("turns", pa.list_(TURN_TYPE)),
            ("generator_model", pa.string()),
            ("prompt_hash", pa.string()),
            ("n_tokens", pa.int64()),
            ("cost_usd", pa.float64()),
            ("n_turns", pa.int32()),
            ("n_words", pa.int32()),
            ("cached", pa.bool_()),
            ("error", pa.string()),
            ("filter_scores", scores_type),
            ("filter_verdict", pa.string()),
            ("dropped_by", pa.string()),
        ]
    )


def to_row(doc: Document, axis_names: Sequence[str], filter_fields: Sequence[str]) -> dict[str, Any]:
    """Flatten a Document into a snapshot row.

    Args:
        doc: The document.
        axis_names: Axis struct fields to populate.
        filter_fields: Filter score struct fields to populate.

    Returns:
        A dict matching build_schema().
    """
    last = doc.lineage[-1] if doc.lineage else None
    axes = {name: str(doc.scenario.axes.get(name, "")) for name in sorted(axis_names)} \
        if axis_names else {"_none": ""}
    scores = {name: doc.filter_scores.get(name) for name in sorted(filter_fields)} \
        if filter_fields else {"_none": None}
    return {
        "doc_id": doc.doc_id,
        "scenario_hash": doc.scenario.scenario_hash,
        "stage_idx": doc.stage_idx,
        "stage_name": doc.stage_name,
        "input_doc_id": doc.input_doc_id or doc.doc_id,
        "spec_id": doc.scenario.spec_id,
        "chunk_ids": doc.scenario.chunk_ids,
        "grouping_strategy": doc.scenario.grouping_strategy,
        "doc_type": doc.scenario.doc_type,
        "axes": axes,
        "turns": [t.to_dict() for t in doc.turns],
        "generator_model": last.model if last else "",
        "prompt_hash": last.prompt_hash if last else "",
        "n_tokens": doc.n_tokens,
        "cost_usd": doc.cost_usd_total,
        "n_turns": len(doc.turns),
        "n_words": len(doc.text().split()),
        "cached": bool(last.cached) if last else False,
        "error": doc.error,
        "filter_scores": scores,
        "filter_verdict": doc.filter_verdict,
        "dropped_by": doc.dropped_by,
    }


@dataclass
class SnapshotConfig:
    """Snapshot destinations and behaviour.

    Attributes:
        backend: "huggingface" to push, "local" to stay on disk.
        org: HF namespace.
        repo: HF repo name template; "{run_id}" is substituted.
        private: Create the dataset repo private.
        push_every_stage: Push after each stage rather than only at the end.
        also_local: Always write local parquet (strongly recommended; it is the
            source of truth when a push fails).
        write_jsonl: Also write full-fidelity JSONL including lineage.
    """

    backend: str = "local"
    org: str = ""
    repo: str = "synthdoc-{run_id}"
    private: bool = True
    push_every_stage: bool = True
    also_local: bool = True
    write_jsonl: bool = True


class SnapshotWriter:
    """Writes complete corpus snapshots, one per stage.

    Every stage writes the WHOLE corpus, not a delta. That is what lets any stage be
    re-run in isolation and any two stages be diffed as corpora.
    """

    def __init__(
        self,
        run_dir: Path | str,
        cfg: SnapshotConfig,
        run_id: str,
        axis_names: Sequence[str],
        filter_fields: Sequence[str],
    ) -> None:
        """Initialize.

        Args:
            run_dir: Local run directory.
            cfg: Snapshot config.
            run_id: Run identifier.
            axis_names: Axes sampled by this run.
            filter_fields: Union of configured filter score fields.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.run_id = run_id
        self.axis_names = list(axis_names)
        self.filter_fields = list(filter_fields)
        self.schema = build_schema(self.axis_names, self.filter_fields)
        self.stages: list[str] = []
        self._threads: list[threading.Thread] = []
        self.push_errors: list[str] = []

    @property
    def repo_id(self) -> str:
        """Full HF dataset repo id for this run."""
        return f"{self.cfg.org}/{self.cfg.repo.format(run_id=self.run_id)}"

    def write(self, stage_name: str, corpus: Sequence[Document]) -> Path:
        """Write one complete stage snapshot locally and optionally push it.

        Args:
            stage_name: e.g. stage_01_revised.
            corpus: Every document at this stage, including failed and filtered ones.

        Returns:
            Path to the local parquet file.
        """
        rows = [to_row(d, self.axis_names, self.filter_fields) for d in corpus]
        table = pa.Table.from_pylist(rows, schema=self.schema)
        path = self.run_dir / f"{stage_name}.parquet"
        pq.write_table(table, path, compression="zstd")

        if self.cfg.write_jsonl:
            jsonl = self.run_dir / f"{stage_name}.jsonl"
            with jsonl.open("w") as fh:
                for doc in corpus:
                    fh.write(json.dumps(doc.to_dict()) + "\n")

        if stage_name not in self.stages:
            self.stages.append(stage_name)
        if self.cfg.backend == "huggingface" and self.cfg.push_every_stage:
            self._push_async(path, f"data/{stage_name}.parquet")
        return path

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        """Write and optionally push the run manifest.

        Args:
            manifest: Config, git sha, seeds, thresholds, agreement statistics.

        Returns:
            Path to the local manifest.json.
        """
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        if self.cfg.backend == "huggingface":
            self._push_async(path, "manifest.json")
            readme = self.run_dir / "README.md"
            readme.write_text(self._readme(manifest))
            self._push_async(readme, "README.md")
        return path

    def _readme(self, manifest: dict[str, Any]) -> str:
        """Build the dataset card, declaring one split per stage."""
        entries = "\n".join(
            f"  - split: {s}\n    path: data/{s}.parquet" for s in self.stages
        )
        return (
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            f"{entries}\n"
            "---\n\n"
            f"# synthdoc run `{self.run_id}`\n\n"
            "Synthetic document corpus for spec finetuning. One split per pipeline\n"
            "stage; the schema is identical across splits, and `doc_id` is stable\n"
            "across stages, so stage-over-stage comparison is a join on `doc_id`.\n\n"
            "Filtered-out documents are retained with `filter_verdict = \"drop\"`\n"
            "rather than deleted, so the filter's effect on the corpus is visible.\n\n"
            "## Stages\n\n"
            + "\n".join(f"- `{s}`" for s in self.stages)
            + "\n\n## Run manifest\n\n```json\n"
            + json.dumps(
                {k: manifest.get(k) for k in ("run_id", "git_sha", "seed", "spec", "counts")},
                indent=2,
                default=str,
            )
            + "\n```\n"
        )

    def _push_async(self, local: Path, remote: str) -> None:
        """Upload a file in a background thread; failures warn and never kill a run."""
        thread = threading.Thread(
            target=self._push, args=(local, remote), name=f"hf-push-{remote}", daemon=False
        )
        thread.start()
        self._threads.append(thread)

    def _push(self, local: Path, remote: str) -> None:
        """Upload one file to the run's dataset repo."""
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=self.cfg.private,
                exist_ok=True,
            )
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"synthdoc {self.run_id}: {remote}",
            )
        except Exception as e:
            msg = f"HF push failed for {remote}: {type(e).__name__}: {e}"
            self.push_errors.append(msg)
            warnings.warn(f"{msg}. Local parquet in {self.run_dir} remains the source of truth.")

    def finish(self, timeout: float = 900.0) -> list[str]:
        """Wait for outstanding pushes.

        Args:
            timeout: Total seconds to wait across all threads.

        Returns:
            Any push error messages collected.
        """
        for thread in self._threads:
            thread.join(timeout=timeout)
        return list(self.push_errors)


def load_snapshot(path: Path | str) -> list[Document]:
    """Load a full-fidelity snapshot from its JSONL sidecar.

    Args:
        path: Path to either the parquet or the jsonl for a stage.

    Returns:
        The documents, with lineage intact.

    Raises:
        FileNotFoundError: If the JSONL sidecar is missing (parquet drops lineage).
    """
    p = Path(path)
    jsonl = p.with_suffix(".jsonl")
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{jsonl} not found. Re-running a stage in isolation needs the JSONL "
            "sidecar; set snapshots.write_jsonl: true."
        )
    return [Document.from_dict(json.loads(line)) for line in jsonl.read_text().splitlines() if line.strip()]
