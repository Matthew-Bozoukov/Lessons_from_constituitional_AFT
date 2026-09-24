# ABOUTME: `extend_from`: a synth run that grows a published corpus instead of replacing it --
# ABOUTME: the prior corpus seeds the scenario stage's dedupe and rides into dataset.jsonl.
"""Extending a published corpus.

A recipe's corpus is generated once and then found to be too small for the next arm
(a 25% share needs more rows than a 15% one). Re-running the whole recipe throws the
paid rows away; a second run merged by hand leaves no record and dedupes nothing across
the two. `extend_from: <org>/<repo>[@<revision>]` makes the second run the record:

1. the prior corpus's scenarios seed the scenario stage's ban list and its embedding
   gate, so a new scenario too close to an old one is refused exactly as one too close
   to another new one is;
2. the prior `dataset.jsonl` rows are carried into this run's `dataset.jsonl` ahead of
   its own, so the published repo's default config is the union and `uv run mix` reads
   it as any other corpus;
3. the manifest and card name the prior repo, its pinned revision and its row count.

What it refuses, up front and before any spend: a prior from another style-type or
another constitution (a union of two recipes is not a corpus), and a run whose
`id_prefix` would let its scenario ids collide with the prior's (every id-keyed join
would silently take the wrong row).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Prior:
    """A published corpus this run extends, pinned to one revision."""

    repo: str
    revision: str
    rows: list[dict]                     # dataset.jsonl, verbatim
    scenarios: list[dict]                # the scenario snapshot: `situation`, `domain`, ...
    scenario_file: str
    run_id: str = ""
    git_sha: str = ""
    pipeline: str = ""
    constitution_sha256: str = ""
    ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def situations(self) -> list[str]:
        return [str(s.get("situation") or "") for s in self.scenarios]


def parse_spec(spec: str) -> tuple[str, str | None]:
    """`org/repo@rev` -> (`org/repo`, `rev`); `org/repo` -> (`org/repo`, None)."""
    spec = str(spec).strip()
    repo, _, rev = spec.partition("@")
    assert repo.count("/") == 1 and not repo.endswith((".json", ".jsonl")) \
        and not Path(repo).exists(), (
        f"extend_from={spec!r} is not an HF dataset repo id (org/name[@revision]): a "
        "corpus is extended from Hugging Face, never from a local file")
    return repo, (rev or None)


def _read(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_prior(spec: str) -> Prior:
    """Fetch the corpus named by `extend_from` at an exact revision.

    Reads `dataset.jsonl` (the rows carried forward), the scenario snapshot the recipe
    deduped against (the post-dedupe one when the prior ran that stage, else the raw
    scenario stage) and `manifest.json` (style-type, constitution, git sha).
    """
    from src.infra.huggingface import hf_api, hf_download

    repo, rev = parse_spec(spec)
    info = hf_api().repo_info(repo, repo_type="dataset", revision=rev)
    sha = info.sha
    files = {s.rfilename for s in info.siblings}
    assert "dataset.jsonl" in files, (
        f"{repo}@{sha[:8]} has no dataset.jsonl: only a COMPLETED synth run can be "
        "extended (a halted one holds stage snapshots and no final dataset)")
    snap = next((f for f in sorted(files) if f.endswith("_dedupe_scenarios.jsonl")), None) \
        or next((f for f in sorted(files) if f.endswith("_write_scenarios.jsonl")), None)
    assert snap, f"{repo}@{sha[:8]} carries no scenario snapshot under stages/ to dedupe against"

    def get(name: str) -> str:
        return hf_download(repo, name, repo_type="dataset", revision=sha)

    rows = _read(get("dataset.jsonl"))
    scenarios = _read(get(snap))
    manifest = json.load(open(get("manifest.json"), encoding="utf-8")) \
        if "manifest.json" in files else {}
    ids = frozenset(str((r.get("metadata") or {}).get("scenario_id") or "") for r in rows)
    return Prior(repo=repo, revision=sha, rows=rows, scenarios=scenarios,
                 scenario_file=snap, run_id=str(manifest.get("run_id", "")),
                 git_sha=str(manifest.get("git_sha", "")),
                 pipeline=str(manifest.get("pipeline", "")),
                 constitution_sha256=str(manifest.get("constitution_sha256", "")),
                 ids=ids)


def check_compatible(prior: Prior, cfg: dict, constitution_sha256: str) -> None:
    """Refuse, before any spend, a prior this run must not be merged with.

    Raises:
        ValueError: style-type or constitution differ, or `id_prefix` is missing or
            already used by the prior's ids.
    """
    style = str(cfg.get("pipeline", ""))
    if prior.pipeline and prior.pipeline != style:
        raise ValueError(
            f"extend_from: {prior.repo} is a `{prior.pipeline}` corpus; this run is "
            f"`{style}`. A union of two style-types is not a corpus.")
    if prior.constitution_sha256 and prior.constitution_sha256 != constitution_sha256:
        raise ValueError(
            f"extend_from: {prior.repo} was generated against a different constitution "
            f"(sha256 {prior.constitution_sha256[:12]} vs {constitution_sha256[:12]}). "
            "Regenerate rather than extend.")
    prefix = str(cfg.get("id_prefix", ""))
    if not prefix:
        raise ValueError(
            "extend_from needs `id_prefix` (e.g. --overrides id_prefix=r2_): scenario ids "
            "restart at t1_b00_s000 in every run, so without a prefix this run's ids would "
            "collide with the prior's and every id-keyed join would take the wrong row.")
    clash = sorted(i for i in prior.ids if i.startswith(prefix))
    if clash:
        raise ValueError(
            f"extend_from: id_prefix={prefix!r} is already in use by {len(clash)} of "
            f"{prior.repo}'s rows ({clash[0]} ...); choose one no prior run used.")
    if not prior.ids or "" in prior.ids:
        raise ValueError(
            f"extend_from: {prior.repo}'s dataset rows do not all carry "
            "metadata.scenario_id; disjointness cannot be checked.")


def merged_rows(prior: Prior, rows: list[dict]) -> list[dict]:
    """Prior rows first, then this run's; refuses an id present in both."""
    dup = sorted(str((r.get("metadata") or {}).get("scenario_id") or "")
                 for r in rows if str((r.get("metadata") or {}).get("scenario_id") or "") in prior.ids)
    if dup:
        raise ValueError(f"extend_from: {len(dup)} of this run's ids already exist in "
                         f"{prior.repo} ({dup[0]} ...)")
    return list(prior.rows) + list(rows)
