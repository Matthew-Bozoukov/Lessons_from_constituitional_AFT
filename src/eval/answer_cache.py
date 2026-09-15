# ABOUTME: Content-addressed per-model answer sets on Hugging Face, shared by evals whose
# ABOUTME: generations are reusable across comparisons (arena_hard, later mmlu).

"""The HF answer cache behind the "push important artifacts to HF" policy.

An entry is one model's answers to one exact exam: keyed by (eval, model, thinking mode,
prompt-subset hash, generation-params hash). Any arm — reference or target — whose entry
exists is never generated again on any machine; with lazy serving (ServedTarget boots on
first `base_url` access) a fully cached arm never even starts vLLM.

This module is imported by the evals that follow the answers pattern and used inside
their `run()` — it is NOT part of the run() contract, and run_eval.py knows nothing
about it. Behaviour evals (psychosis, agentic_misalignment) have no use for it: caching
their responses would be caching the experiment itself.

Backends, chosen by the `repo` string:
- `hf:org/name` — an HF dataset repo (long-lived, one per eval, entries as folders).
- anything else — a local directory: offline tests, smoke runs, and deliberate
  air-gapped work. Same layout, same validation.

A per-invocation `mirror` directory gives read-through behaviour: pushes land there
too, so when the reference arm fills the cache, the target arms of the same
invocation read it back without a network round-trip.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ANSWERS = "answers.jsonl"
META = "answers_meta.json"


def gen_hash(params: dict) -> str:
    """Hash the generation params that make answers comparable across arms.

    Everything that changes what an answer would be belongs in here — temperature,
    top_p, max_tokens, prompt template variants. Two entries with different gen hashes
    answered under different conditions and must never be judged against each other.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class CacheKey:
    """Identity of one model's answers to one exact exam."""

    model_key: str    # TargetSpec.model_key (filesystem-safe)
    mode: str         # think | nothink | default — cross-mode reuse is meaningless
    subset_hash: str  # the eval's prompt-subset hash
    gen_hash: str     # gen_hash() of the generation params

    @property
    def path(self) -> str:
        return f"{self.model_key}/{self.mode}/{self.subset_hash}-{self.gen_hash}"


class AnswerCache:
    """One eval's answer store: probe/fetch/push with meta validation on every read."""

    def __init__(self, repo: str, mirror: Path | None = None):
        """
        Args:
            repo: `hf:<name>` for an HF dataset repo (the org is .env's HF_ORG, as
                for every other repo this project owns), else a local directory path.
            mirror: Per-invocation read-through directory (see module docstring).
        """
        from src.infra.huggingface import gate_push, hf_repo_id

        self.hf_repo = hf_repo_id(repo[3:]) if repo.startswith("hf:") else None
        if self.hf_repo:
            gate_push(self.hf_repo, what="answer cache")
        self.local = None if self.hf_repo else Path(repo)
        self.mirror = mirror

    # --- internals -------------------------------------------------------------------

    def _api(self):
        from src.infra.huggingface import hf_api

        return hf_api()

    def _mirror_dir(self, key: CacheKey) -> Path | None:
        return self.mirror / key.path if self.mirror else None

    def _validated(self, entry_dir: Path, key: CacheKey) -> Path:
        """Refuse an entry whose sidecar disagrees with the key it was looked up under."""
        meta = json.loads((entry_dir / META).read_text())
        for field, want in (("mode", key.mode), ("subset_hash", key.subset_hash),
                            ("gen_hash", key.gen_hash)):
            got = meta.get(field)
            assert got == want, (
                f"cache entry {key.path}: sidecar {field}={got!r} != expected {want!r} — "
                "the entry was stored under the wrong key or built by different code; "
                "refusing to use it")
        assert (entry_dir / ANSWERS).exists(), f"cache entry {key.path} has no {ANSWERS}"
        return entry_dir

    # --- public API ------------------------------------------------------------------

    def probe(self, key: CacheKey) -> bool:
        """True when an entry exists (mirror, local dir, or HF) — no download."""
        mirror = self._mirror_dir(key)
        if mirror and (mirror / META).exists():
            return True
        if self.local is not None:
            return (self.local / key.path / META).exists()
        return self._api().file_exists(self.hf_repo, f"{key.path}/{META}",
                                       repo_type="dataset")

    def fetch(self, key: CacheKey, dest: Path) -> Path:
        """Materialize an entry into `dest` (validated), returning the entry directory."""
        dest.mkdir(parents=True, exist_ok=True)
        source = self._mirror_dir(key)
        if not (source and (source / META).exists()):
            if self.local is not None:
                source = self.local / key.path
            else:
                from src.infra.huggingface import hf_download

                for name in (ANSWERS, META):
                    hf_download(self.hf_repo, f"{key.path}/{name}",
                                repo_type="dataset", local_dir=str(dest))
                # hf_hub_download recreates the key path under local_dir.
                source = dest / key.path
        for name in (ANSWERS, META):
            if (source / name) != (dest / name):
                shutil.copy2(source / name, dest / name)
        return self._validated(dest, key)

    def push(self, key: CacheKey, src_dir: Path, card_fields: dict,
             refresh: bool = False) -> None:
        """Store an entry (answers + sidecar), refusing silent overwrites.

        Args:
            key: The entry's identity; the sidecar in `src_dir` must agree with it.
            src_dir: Directory holding answers.jsonl + answers_meta.json.
            card_fields: CLAUDE.md-required dataset-card fields, applied on repo
                creation (HF backend only).
            refresh: Overwrite an existing entry — a deliberate regeneration, never a
                default.
        """
        self._validated(src_dir, key)  # never push an entry that would refuse to load
        if self.probe(key) and not refresh:
            raise RuntimeError(
                f"cache entry {key.path} already exists — pass cache.refresh=true only "
                "for a deliberate regeneration; silently overwriting would invalidate "
                "every comparison already judged against it")
        mirror = self._mirror_dir(key)
        if mirror:
            mirror.mkdir(parents=True, exist_ok=True)
            for name in (ANSWERS, META):
                shutil.copy2(src_dir / name, mirror / name)
        if self.local is not None:
            entry = self.local / key.path
            entry.mkdir(parents=True, exist_ok=True)
            for name in (ANSWERS, META):
                shutil.copy2(src_dir / name, entry / name)
            return
        from src.infra.huggingface import card_markdown

        api = self._api()
        if not api.repo_exists(self.hf_repo, repo_type="dataset"):
            api.create_repo(self.hf_repo, repo_type="dataset", private=True)
            api.upload_file(path_or_fileobj=card_markdown(card_fields).encode(),
                            path_in_repo="README.md", repo_id=self.hf_repo,
                            repo_type="dataset")
        for name in (ANSWERS, META):
            api.upload_file(path_or_fileobj=str(src_dir / name),
                            path_in_repo=f"{key.path}/{name}", repo_id=self.hf_repo,
                            repo_type="dataset")
