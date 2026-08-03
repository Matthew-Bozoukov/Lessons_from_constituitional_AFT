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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    back = [json.loads(line) for line in path.open()]
    assert len(back) == len(rows), f"{path} is truncated: {len(back)} != {len(rows)}"
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a jsonl file into a list of records."""
    return [json.loads(line) for line in path.open()]


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
                 token: str | None = None) -> None:
        """Set up the cache.

        Args:
            run_dir: Local run directory.
            repo_id: HF dataset repo id, or None to skip publishing.
            private: Create the HF repo private.
            token: HF token; falls back to HF_TOKEN.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.private = private
        self.token = token or os.environ.get("HF_TOKEN")
        self._api = None

    def _hf(self):
        """Return a lazily-created HfApi, creating the repo on first use."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
            self._api.create_repo(self.repo_id, repo_type="dataset", exist_ok=True,
                                  private=self.private)
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
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if self.repo_id:
            self._hf().upload_file(
                path_or_fileobj=str(dest), path_in_repo=name,
                repo_id=self.repo_id, repo_type="dataset",
                commit_message=f"update {name}",
            )
        return dest
