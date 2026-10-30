# ABOUTME: `extend_from`: a synth run that grows a published corpus instead of replacing it --
# ABOUTME: the prior seeds the dedupe and the deal, rides into dataset.jsonl, and sizes the run (plan).
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
3. the manifest and card name the prior repo, its pinned revision and its row count;
4. a rotation axis that declares `continue_from_prior: true` (da.yaml: sectors) deals
   each unit the labels it has met least across the prior's runs (`prior_deals`, read
   off its rows), so the run does not hand every unit the same first labels again.

How big to make it is `plan_extension`: rows each unit still needs for the pools it must
fill, inflated by the loss the prior's own extension runs measured (`prior_runs`,
`loss_rates`), and printed as the exact `total_scenarios=...,trait_weights.<unit>=...`
overrides to launch with.

What it refuses, up front and before any spend: a prior from another style-type or
another constitution (a union of two recipes is not a corpus), and a run whose
`id_prefix` would let its scenario ids collide with the prior's (every id-keyed join
would silently take the wrong row).
"""
from __future__ import annotations

import json
import re
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
    manifest: dict = field(default_factory=dict)     # the prior's own manifest.json

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


def prior_scenarios(rows: list[dict], snapshot: list[dict]) -> list[dict]:
    """Every scenario the prior corpus holds: its scenario snapshot plus the scenario of
    each dataset row the snapshot does not cover.

    A corpus that was itself extended carries ALL its runs' rows in dataset.jsonl but only
    the LAST run's `stages/` (each run overwrites them). Measured 2026-10-30: at
    2026-10-02-da-synth @ 305914d5 the snapshot holds the second run's 642 scenarios and
    none of the first run's, so a third run seeded from the snapshot alone would be free
    to rewrite any of the first run's 646. A row's own metadata (`situation`, `domain`,
    `shortcut`) closes that gap; the snapshot still adds what a run wrote but dropped
    downstream.

    Raises:
        ValueError: a row the snapshot does not cover carries no `situation`, so nothing
            could stop this run from repeating it.
    """
    out = list(snapshot)
    have = {str(s.get("scenario_id") or "") for s in snapshot}
    bare = []
    for r in rows:
        md = r.get("metadata") or {}
        sid = str(md.get("scenario_id") or "")
        if sid in have:
            continue
        if not str(md.get("situation") or "").strip():
            bare.append(sid)
            continue
        out.append({k: md[k] for k in ("scenario_id", "trait_id", "domain", "situation", "shortcut")
                    if k in md})
        have.add(sid)
    if bare:
        raise ValueError(
            f"extend_from: {len(bare)} prior rows ({bare[0]!r} ...) are not in the scenario "
            "snapshot and carry no metadata.situation, so the dedupe cannot cover them.")
    return out


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

    def get(name: str) -> str:
        return hf_download(repo, name, repo_type="dataset", revision=sha)

    manifest = json.load(open(get("manifest.json"), encoding="utf-8")) \
        if "manifest.json" in files else {}
    # A prior whose records came from `load_source_run` wrote no scenarios: its prompts
    # are another corpus's, and the loader (not the diversity gate) keeps the extension
    # disjoint from it, so there is nothing to dedupe against and no snapshot to want.
    assert snap or loads_source_ids(manifest.get("config") or {}), \
        f"{repo}@{sha[:8]} carries no scenario snapshot under stages/ to dedupe against"
    rows = _read(get("dataset.jsonl"))
    scenarios = prior_scenarios(rows, _read(get(snap))) if snap else []
    ids = frozenset(str((r.get("metadata") or {}).get("scenario_id") or "") for r in rows)
    return Prior(repo=repo, revision=sha, rows=rows, scenarios=scenarios,
                 scenario_file=snap or "", run_id=str(manifest.get("run_id", "")),
                 git_sha=str(manifest.get("git_sha", "")),
                 pipeline=str(manifest.get("pipeline", "")),
                 constitution_sha256=str(manifest.get("constitution_sha256", "")),
                 ids=ids, manifest=manifest)


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
    if loads_source_ids(cfg):
        # A run whose records come from `load_source_run` keeps the SOURCE's scenario ids
        # (the join back to the prompts' corpus is the point of the arm), so no prefix is
        # minted; disjointness comes from the loader setting aside every id the prior
        # already answered, and merged_rows still refuses any that slips through.
        if not prior.ids or "" in prior.ids:
            raise ValueError(
                f"extend_from: {prior.repo}'s dataset rows do not all carry "
                "metadata.scenario_id; disjointness cannot be checked.")
        return
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


def call_id(scenario_id: str) -> str:
    """The writer call a scenario came from: its id without the per-scenario `_s<j>`."""
    return re.sub(r"_[a-z]+\d+$", "", str(scenario_id))


def run_prefix(scenario_id: str, unit_id: str) -> str:
    """The `id_prefix` of the run a scenario came from ('' for a corpus's first run)."""
    sid, head = str(scenario_id), f"{unit_id}_b"
    at = sid.find(head)
    assert at >= 0, f"{sid!r} is not a scenario id of unit {unit_id!r}"
    return sid[:at]


def prior_deals(prior: Prior, axes: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    """What the prior corpus's runs were DEALT on each axis: `axis -> unit -> label -> calls`.

    Read off what the corpus holds, not off its configs: run 2 of 2026-10-02-da-synth was
    launched from a hand-edited config whose sector list was rotated by 18, so the order
    in da.yaml says nothing about what that run dealt. Every scenario carries its call's
    label (`write_scenarios` stamps it, `export_sft` keeps it as metadata), so a call is
    counted once from whichever of its scenarios survived; a call none of whose scenarios
    survived is not seen, which is the only gap.

    Raises:
        ValueError: two scenarios of one call carry different labels.
    """
    records = [*prior.scenarios, *((r.get("metadata") or {}) for r in prior.rows)]
    label_of: dict[tuple[str, str], tuple[str, str]] = {}
    for s in records:
        sid, unit = str(s.get("scenario_id") or ""), str(s.get("trait_id") or "")
        if not sid or not unit:
            continue
        for axis in axes:
            if s.get(axis) is None:
                continue
            key, value = (axis, call_id(sid)), (unit, str(s[axis]))
            if label_of.setdefault(key, value) != value:
                raise ValueError(f"call {key[1]} carries two {axis} labels: "
                                 f"{label_of[key][1]!r} and {value[1]!r}")
    out: dict[str, dict[str, dict[str, int]]] = {axis: {} for axis in axes}
    for (axis, _call), (unit, lab) in sorted(label_of.items()):
        by_label = out[axis].setdefault(unit, {})
        by_label[lab] = by_label.get(lab, 0) + 1
    return out


# --------------------------------------------------------------------------------------
# Planning an extension: how many scenarios to ask for, per unit
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorRun:
    """One run of a corpus: how many scenarios it ASKED for per unit, and what it kept."""

    prefix: str
    requested: dict[str, int]
    kept: dict[str, int]
    extended: bool          # its dedupe was seeded with an earlier corpus


def requested_by_unit(cfg: dict, unit_ids: list[str]) -> dict[str, int]:
    """Scenarios a run of `cfg` asks the writer for, per unit (the plan, not the yield)."""
    from .stage_operators import scenario_batches

    out = {u: 0 for u in unit_ids}
    for ti, _bi, n in scenario_batches(len(unit_ids), cfg, unit_ids):
        out[unit_ids[ti]] += n
    return out


def kept_by_run(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Rows a corpus holds per run (`id_prefix`) and unit."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        md = r.get("metadata") or {}
        unit = str(md["trait_id"])
        by_unit = out.setdefault(run_prefix(md["scenario_id"], unit), {})
        by_unit[unit] = by_unit.get(unit, 0) + 1
    return out


def _unit_order(unit: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", unit))


def prior_runs(prior: Prior) -> list[PriorRun]:
    """Every run behind a corpus, newest first, walked back through `extend_from`.

    Each run's manifest records the config it was LAUNCHED with (overrides applied), so
    what it asked for is recomputed from that, and what it kept is counted off the rows
    that carry its `id_prefix`.
    """
    from src.infra.huggingface import hf_download

    units = sorted({str((r.get("metadata") or {})["trait_id"]) for r in prior.rows}, key=_unit_order)
    kept = kept_by_run(prior.rows)
    runs, manifest = [], prior.manifest
    while manifest:
        cfg = manifest.get("config") or {}
        prefix = str(cfg.get("id_prefix") or "")
        runs.append(PriorRun(prefix=prefix, requested=requested_by_unit(cfg, units),
                             kept=kept.get(prefix, {}), extended=bool(manifest.get("extend_from"))))
        src = manifest.get("extend_from")
        manifest = json.load(open(hf_download(src["repo"], "manifest.json", repo_type="dataset",
                                              revision=src["revision"]), encoding="utf-8")) if src else {}
    missing = sorted(set(kept) - {r.prefix for r in runs})
    assert not missing, f"rows with id prefixes {missing} belong to no run in the manifest chain"
    return runs


def loss_rates(runs: list[PriorRun]) -> dict[str, float]:
    """Per unit, the share of requested scenarios that did not become rows.

    Everything between the ask and the corpus counts: scenarios the similarity gate
    refused that two make-up rounds did not replace, prompts the provider refused, drafts
    the lint retries could not save. Taken from the runs whose dedupe was seeded with an
    earlier corpus when there are any, because the gate refuses more the bigger the prior
    it is seeded with (2026-10-02: 24 rejections and no shortfall unseeded, 80 and a
    shortfall of 8 seeded with 650), and an extension is always one of those.
    """
    pool = [r for r in runs if r.extended] or list(runs)
    assert pool, "no prior run to measure a loss rate from"
    units = sorted({u for r in pool for u in r.requested})
    out = {}
    for u in units:
        asked = sum(r.requested.get(u, 0) for r in pool)
        got = sum(r.kept.get(u, 0) for r in pool)
        assert asked > 0, f"no prior run asked for unit {u}"
        out[u] = max(0.0, 1 - got / asked)
    return out


def plan_extension(unused: dict[str, int], pools: int, per_pool: int,
                   loss: dict[str, float], *, tokens_per_row: dict[str, float],
                   prior_tokens: int, budget_tokens: int, pool_pct: float,
                   goal_share: float) -> dict:
    """Scenarios to request per unit so the corpus can fill `pools` new, disjoint mixtures.

    Each new pool needs `per_pool` rows of every unit that no earlier mixture used; the
    rows already unused count towards that, and what is missing is inflated by the unit's
    measured loss. The asks become `trait_weights` over `total_scenarios`, which
    `scenario_batches` apportions back to exactly these numbers.

    Args:
        unused: Rows per unit the corpus holds that no earlier mixture took.
        pools: New disjoint pools wanted (2: da-15 B and C).
        per_pool: Rows per unit each pool must hold (75: the 72-73 a 15% fill takes, plus
            the slack its round-robin needs to skip a row that would overshoot).
        loss: `loss_rates` per unit.
        tokens_per_row: Mean supervised tokens of a row, per unit, measured on the prior.
        prior_tokens: Supervised tokens the prior corpus holds.
        budget_tokens: The base mixture's supervised tokens (the size of every arm).
        pool_pct: The share one pool's mixture fills (15).
        goal_share: The share of `budget_tokens` the grown corpus must exceed (0.45).

    Raises:
        ValueError: `per_pool` rows of every unit cannot hold one pool's token share, or
            the grown corpus would not reach `goal_share`.
    """
    import math

    units = sorted(unused)
    need = {u: max(0, pools * per_pool - int(unused[u])) for u in units}
    request = {u: math.ceil(need[u] / (1 - loss[u])) if need[u] else 0 for u in units}
    pool_tokens = sum(per_pool * tokens_per_row[u] for u in units)
    fill = budget_tokens * pool_pct / 100
    if pool_tokens < fill:
        raise ValueError(f"{per_pool} rows a unit hold ~{pool_tokens:,.0f} supervised tokens, "
                         f"less than one {pool_pct}% fill ({fill:,.0f}); raise per_pool")
    grown = prior_tokens + sum(need[u] * tokens_per_row[u] for u in units)
    if grown / budget_tokens <= goal_share:
        raise ValueError(f"the grown corpus would hold {grown / budget_tokens:.1%} of the "
                         f"budget, not more than {goal_share:.0%}; raise per_pool")
    # Every unit carries a weight, a unit with nothing to add a 0: `scenario_batches`
    # refuses a table that does not cover the constitution's units.
    overrides = [f"total_scenarios={sum(request.values())}",
                 *(f"trait_weights.{u}={n}" for u, n in request.items())]
    return {"need": need, "request": request, "loss": {u: round(loss[u], 4) for u in units},
            "total_scenarios": sum(request.values()), "trait_weights": request,
            "new_rows": sum(need.values()),
            "expected_tokens": round(grown), "expected_share": round(grown / budget_tokens, 4),
            "pool_tokens": round(pool_tokens), "fill_tokens": round(fill),
            "overrides": ",".join(overrides)}


def loads_source_ids(cfg: dict) -> bool:
    """True when this run's records (and so their ids) come from another run's records."""
    return any(str(st.get("kind")) == "load_source_run" for st in (cfg.get("stages") or []))


def merged_rows(prior: Prior, rows: list[dict]) -> list[dict]:
    """Prior rows first, then this run's; refuses an id present in both."""
    dup = sorted(str((r.get("metadata") or {}).get("scenario_id") or "")
                 for r in rows if str((r.get("metadata") or {}).get("scenario_id") or "") in prior.ids)
    if dup:
        raise ValueError(f"extend_from: {len(dup)} of this run's ids already exist in "
                         f"{prior.repo} ({dup[0]} ...)")
    return list(prior.rows) + list(rows)
