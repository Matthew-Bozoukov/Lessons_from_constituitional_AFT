# ABOUTME: Per-stage dataset cache: every stage's complete output is written locally and
# ABOUTME: pushed to one HF dataset repo (stages/ + final dataset.jsonl, declared configs).

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def dataset_card(card_fields: dict | None, stage_files: list[str],
                 has_dataset: bool, tags: tuple[str, ...] | list[str] = ()) -> str:
    """Render the repo README: `configs:` + `tags:` YAML front-matter + the card table (pure).

    The front-matter is the synth->mixture contract's discovery layer: `dataset` is the
    DEFAULT config (so `load_dataset(repo)` fetches dataset.jsonl and nothing else) and
    every stage snapshot is its own named config under stages/ — which also keeps the
    dataset viewer working, since without declared configs it globs every jsonl in the
    repo and chokes on the stages' differing schemas. `tags` are the Hub-indexed
    `training_data_tags` the dashboard discovers the corpus by (src/huggingface.py).
    """
    from src.huggingface import card_front_matter, card_markdown

    def stage_no(f: str) -> int:
        m = re.match(r"stage_(\d+)_", f)
        return int(m.group(1)) if m else 0

    configs = []
    if has_dataset:
        configs.append({"config_name": "dataset", "data_files": "dataset.jsonl",
                        "default": True})
    configs += [{"config_name": f.removesuffix(".jsonl"), "data_files": f"stages/{f}"}
                for f in sorted(stage_files, key=stage_no)]
    return card_front_matter(configs, tags) + (card_markdown(card_fields)
                                               if card_fields else "")


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

    Repo layout (the synth->mixture contract; the local run dir stays FLAT):
        dataset.jsonl    the final dataset — what build_mixture consumes (default config)
        stages/          every stage snapshot, one named config each
        manifest.json    pipeline-wide metadata; README.md carries the configs block
    Every upload refreshes the README's `configs:` front-matter in the SAME commit, so
    the declared configs never lag the files.

    Attributes:
        run_dir: Local directory holding this run's snapshots.
        repo_id: HF dataset repo to mirror into, or None for local-only.
    """

    def __init__(self, run_dir: Path, repo_id: str | None, private: bool = False,
                 token: str | None = None, card_fields: dict | None = None,
                 tags: tuple[str, ...] | list[str] = ()) -> None:
        """Set up the cache.

        Args:
            run_dir: Local run directory.
            repo_id: HF dataset repo id, or None to skip publishing.
            card_fields: CLAUDE.md card fields, uploaded as the repo README on first
                creation (every upload carries a card, the cache repo included).
            tags: Card front-matter tags (`training_data_tags`), refreshed with every
                README so the corpus is discoverable from the Hub.
            private: Create the HF repo private.
            token: HF token; falls back to the shared resolution (src.huggingface.hf_token).
        """
        from src.huggingface import hf_token

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.private = private
        self.token = token or hf_token()
        self.card_fields = card_fields
        self.tags = list(tags)
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
                self._api.upload_file(
                    path_or_fileobj=self._readme().encode(),
                    path_in_repo="README.md", repo_id=self.repo_id,
                    repo_type="dataset")
        return self._api

    def _readme(self) -> str:
        """The repo README, regenerated from what the local run dir actually holds."""
        return dataset_card(self.card_fields,
                            [p.name for p in self.run_dir.glob("stage_*.jsonl")],
                            (self.run_dir / "dataset.jsonl").exists(), self.tags)

    def _commit(self, files: list[tuple[Path, str]], message: str) -> None:
        """One commit: the given (local, repo_path) files plus the refreshed README."""
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp))
               for lp, rp in files]
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=self._readme().encode()))
        self._hf().create_commit(repo_id=self.repo_id, repo_type="dataset",
                                 operations=ops, commit_message=message)

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
            self._commit([(dest, f"stages/{dest.name}")],
                         f"stage {index}: {name} ({len(rows)} records)")
        return dest

    def publish_final(self, rows: list[dict[str, Any]]) -> Path:
        """Publish the completed run's final records as the repo's default dataset.

        Writes `dataset.jsonl` locally and uploads it at the repo ROOT — the one file
        build_mixture (and any other consumer) reads; the same commit marks it as the
        default config, so `load_dataset(repo)` returns it and nothing else.
        """
        dest = write_jsonl(self.run_dir / "dataset.jsonl", rows)
        if self.repo_id:
            self._commit([(dest, "dataset.jsonl")],
                         f"dataset: {len(rows)} records (final)")
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
