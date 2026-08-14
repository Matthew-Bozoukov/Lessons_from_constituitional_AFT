# ABOUTME: Per-stage dataset cache: every stage's complete output is written locally and
# ABOUTME: pushed to one HF dataset repo, so a run can resume from any stage.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as jsonl and verify the file round-trips.

    Args:
        path: Destination file.
        rows: Records to write.

    Returns:
        The path written.
    """
    # encoding is explicit everywhere here: `ensure_ascii=False` emits non-ASCII, and an
    # unqualified open() uses the platform locale codec -- cp1252 on Windows. Snapshots
    # written that way round-trip locally and then fail to decode on HF and on the Linux
    # GPU box, which is the only place they are ever consumed.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    back = [json.loads(line) for line in path.open(encoding="utf-8")]
    assert len(back) == len(rows), f"{path} is truncated: {len(back)} != {len(rows)}"
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a jsonl file into a list of records."""
    return [json.loads(line) for line in path.open(encoding="utf-8")]


class StageCache:
    """Local snapshots plus an optional mirror in one HF dataset repo.

    Each stage writes `stage_<n>_<name>.jsonl`. On resume, a stage whose file already
    exists is skipped and its output reused, so an interrupted run costs nothing to
    restart and any single stage can be re-run alone by deleting its file.

    Attributes:
        run_dir: Local directory holding this run's snapshots.
        repo_id: HF dataset repo to mirror into, or None for local-only.
    """

    def __init__(self, run_dir: Path, repo_id: str | None, private: bool = False,
                 token: str | None = None, card_fields: dict | None = None) -> None:
        """Set up the cache.

        Args:
            run_dir: Local run directory.
            repo_id: HF dataset repo id, or None to skip publishing.
            card_fields: CLAUDE.md card fields, uploaded as the repo README on first
                creation (every upload carries a card, the cache repo included).
            private: Create the HF repo private.
            token: HF token; falls back to HF_TOKEN.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.private = private
        self.token = (token or os.environ.get("HUGGINGFACE_API_KEY")
                      or os.environ.get("HF_TOKEN"))
        self.card_fields = card_fields
        self._api = None

    def _hf(self):
        """Return a lazily-created HfApi, creating the repo (with its card) on first use."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
            fresh = not self._api.repo_exists(self.repo_id, repo_type="dataset")
            self._api.create_repo(self.repo_id, repo_type="dataset", exist_ok=True,
                                  private=self.private)
            # Every upload carries a card (CLAUDE.md) — the cache repo included.
            if fresh and self.card_fields:
                from src.huggingface import card_markdown

                self._api.upload_file(
                    path_or_fileobj=card_markdown(self.card_fields).encode(),
                    path_in_repo="README.md", repo_id=self.repo_id,
                    repo_type="dataset")
        return self._api

    def path(self, index: int, name: str) -> Path:
        """Return the local path for a stage's snapshot."""
        return self.run_dir / f"stage_{index}_{name}.jsonl"

    def has(self, index: int, name: str) -> bool:
        """Return True when this stage's snapshot already exists locally."""
        return self.path(index, name).exists()

    def load(self, index: int, name: str) -> list[dict[str, Any]]:
        """Read a stage's cached output."""
        return read_jsonl(self.path(index, name))

    def save(self, index: int, name: str, rows: list[dict[str, Any]]) -> Path:
        """Write a stage's output locally and mirror it to HF when configured.

        Args:
            index: Stage number.
            name: Stage name.
            rows: Records to persist.

        Returns:
            The local path written.
        """
        dest = write_jsonl(self.path(index, name), rows)
        if self.repo_id:
            self._hf().upload_file(
                path_or_fileobj=str(dest),
                path_in_repo=dest.name,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"stage {index}: {name} ({len(rows)} records)",
            )
        return dest

    def save_json(self, name: str, payload: Any) -> Path:
        """Write a small JSON artefact (manifest, usage) and mirror it."""
        dest = self.run_dir / name
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.mirror(dest)

    def mirror(self, path: Path) -> Path:
        """Publish a file already written into the run dir, if there is a repo.

        For artefacts the run produces as text rather than JSON -- the pattern-frequency
        table, the judged-label sidecar. Those are the starting point for any "why did
        this corpus behave that way" analysis, and until 2026-08-13 they were written
        locally and never mirrored, so they existed only until the run dir was cleaned up.
        """
        path = Path(path)
        if self.repo_id and path.exists():
            self._hf().upload_file(
                path_or_fileobj=str(path), path_in_repo=path.name,
                repo_id=self.repo_id, repo_type="dataset",
                commit_message=f"update {path.name}",
            )
        return path
