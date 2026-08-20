# ABOUTME: The config-driven pipeline engine: builds Stage objects from the config's
# ABOUTME: `stages:` list and owns caching, checkpoints, ablation, budget, manifest, estimates.

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.utils import git_sha, timestamp

from .constitution import full_text
from .stage_runtime import Checkpoint, Ctx, Stage, Usage, measured_per_stage, price_of
from .hf_cache import StageCache
from .stage_operators import OPERATORS


def build_stages(cfg: dict) -> list[Stage]:
    """Materialise the config's `stages:` list via the operator registry.

    The engine attaches the generic ablation null-op here: a stage entry with
    `ablate_with: {field: source_field}` gets a field-copy null-operation; entries
    without it cannot be ablated.
    """
    assert cfg.get("stages"), "config must declare a `stages:` list"
    out = []
    for sc in cfg["stages"]:
        kind = sc.get("kind")
        if kind not in OPERATORS:
            raise ValueError(f"stage {sc.get('name')!r}: unknown kind {kind!r}. "
                             f"Operators: {sorted(OPERATORS)}")
        st = OPERATORS[kind](sc, cfg)
        if "ablate_with" in sc:
            copy_map = dict(sc["ablate_with"])
            st = Stage(**{**st.__dict__,
                          "ablate_fn": lambda rs, m=copy_map:
                          [{**r, **{f: r[src] for f, src in m.items()}} for r in rs]})
        out.append(st)
    names = [s.name for s in out]
    assert len(set(names)) == len(names), f"duplicate stage names: {names}"
    return out


def snapshot_positions(stages: list[Stage]) -> dict[str, int]:
    """Map stage name -> its `stage_<n>_<name>.jsonl` position, skipping observers.

    Positions and names are the on-disk contract that keeps completed run dirs and HF
    mirrors resumable, so an observer must not consume one -- that is what makes a
    mid-pipeline corpus check free to add. An observer maps to the position of the last
    real stage before it: the snapshot holding the records it inspects.

    Takes built Stages rather than the raw config so `Stage.observer` is the ONE place
    the fact lives.
    """
    out: dict[str, int] = {}
    pos = 0
    for st in stages:
        if not st.observer:
            pos += 1
        out[st.name] = pos
    return out


def _validate_ablate(ablate: list[str], stage_list: list[Stage]) -> None:
    """Fail fast on ablation typos or attempts to ablate a load-bearing stage."""
    by_name = {s.name: s for s in stage_list}
    unknown = [a for a in ablate if a not in by_name]
    if unknown:
        raise ValueError(f"ablate names not in this pipeline's stages: {unknown}. "
                         f"Stages: {[s.name for s in stage_list]}")
    fixed = [a for a in ablate if by_name[a].ablate_fn is None]
    if fixed:
        raise ValueError(f"stage(s) {fixed} declare no `ablate_with` null-operation "
                         f"and cannot be ablated")


def run(cfg: dict, smoke: bool = False, resume: str | None = None) -> dict:
    """Run the config's pipeline, caching every stage snapshot and mirroring to HF.

    Args:
        cfg: Run config: `pipeline:` (label), `stages:` (the pipeline itself),
            `models:`, prompts, and the document-type knobs the stages read.
        smoke: Merge the config's `smoke:` overrides and route to the smoke HF repo.
        resume: Existing run directory to continue; completed snapshots are reused and
            checkpointed stages pick up per item.

    Returns:
        A manifest dict describing the run.
    """
    started = time.time()
    ts = timestamp()
    original_cfg = cfg
    cfg = json.loads(json.dumps(cfg))  # stages must never mutate the caller's config
    if smoke:
        cfg.update(cfg.get("smoke") or {})

    if resume:
        run_dir = Path(resume)
        assert run_dir.exists(), f"resume dir does not exist: {run_dir}"
        print(f">>> resuming into {run_dir}")
    else:
        run_dir = Path(cfg["output_dir"]) / (f"smoke_{ts}" if smoke else ts)
    repo = cfg.get("hf_repo_smoke") if smoke else cfg.get("hf_repo")
    from src.utils import origin_url
    # A config may enrich (or correct) any card field via a top-level `card:` map —
    # for arms whose experiment, model mix, or config filename the auto-built defaults
    # cannot infer (e.g. `pipeline` != filename, or a per-stage model split). The
    # config's values win, so `card.provenance` fixes the run command when the two
    # names differ, and `card.experiment`/`card.models` carry the real detail.
    card = {
        "experiment": f"synth `{cfg['pipeline']}` run — per-stage "
                      "snapshots (resumable generation cache)",
        "date_generated": ts,
        "constitution": str(cfg["constitution"]),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": str(cfg.get("model")
                      or "per-stage models — see manifest.json"),
        "generation_config": "see manifest.json (full run config, "
                              "sampling settings, per-stage usage)",
        "schema": "stage_<n>_<name>.jsonl snapshots + manifest.json",
        "provenance": "uv run synth run --config "
                      f"configs/data/synth/{cfg['pipeline']}.yaml",
    }
    card.update({k: str(v) for k, v in (cfg.get("card") or {}).items()})
    cache = StageCache(run_dir, repo, private=bool(cfg.get("hf_private", False)),
                       card_fields=card)

    workers = int(cfg.get("workers", 8))
    budget = float(cfg.get("budget_usd", 0)) or None
    usage = Usage()
    ctx = Ctx(cfg=cfg, usage=usage, workers=workers, run_dir=run_dir, smoke=smoke,
              vars={"constitution": full_text(cfg["constitution"])}, cache=cache)

    stage_list = build_stages(cfg)
    ablate = [str(a) for a in (cfg.get("ablate") or [])]
    _validate_ablate(ablate, stage_list)

    records: list[dict] = []
    durations: dict[str, float] = {}
    counts: dict[str, int] = {}
    positions = snapshot_positions(stage_list)
    for st in stage_list:
        pos = positions.get(st.name)
        label = (f"check ({st.name})" if st.observer else f"stage {pos} ({st.name})")
        if st.skip and st.skip(ctx, records):
            print(f">>> {label}: not applicable -- skipped")
            continue
        # No snapshot, no position and no cache for an observer: it produces nothing the
        # pipeline consumes, so re-running it is cheap and always tells the truth about
        # the records actually in hand.
        if not st.observer and cache.has(pos, st.name):
            records = cache.load(pos, st.name)
            if st.on_cached:
                st.on_cached(ctx, records)
            print(f">>> {label}: reused {len(records)} cached records")
        else:
            # Stop BEFORE starting to spend more, never after the money is gone -- a
            # run that finishes its last paid stage over budget still completes and
            # keeps everything it paid for.
            if st.paid and budget is not None and usage.usd > budget:
                raise RuntimeError(
                    f"budget_usd=${budget:.2f} exceeded (${usage.usd:.2f}) before "
                    f"{label}. Snapshots up to this stage are in {run_dir}; raise "
                    f"budget_usd and re-run to resume.")
            ckpt = Checkpoint(run_dir / f"stage_{pos}_{st.name}.partial.jsonl",
                              key=st.checkpoint_key) if st.checkpoint_key else None
            t0 = time.time()
            if st.name in ablate:
                records = st.ablate_fn(records)
                print(f">>> {label}: ABLATED -- null-operation applied to "
                      f"{len(records)} records")
            else:
                records = st.fn(ctx, records, ckpt)
            durations[st.name] = round(time.time() - t0, 1)
            if not st.observer:
                cache.save(pos, st.name, records)
            print(f">>> {label}: {len(records)} records")
        counts[st.name] = len(records)
        if st.preview and records:
            print(f"    FIRST: {st.preview(records[0])[:220]}")
        if ctx.stop:
            # An intermediate corpus check asking to halt before the expensive stages
            # after it. Everything paid for is on disk; the manifest is still written.
            print(f"\n!!! run halted at {label}: {ctx.stop}")
            break

    # A COMPLETED run publishes its final records as the repo's default dataset —
    # `dataset.jsonl` at the root is the synth->mixture contract; a halted run has no
    # final dataset and leaves only its stages/ snapshots.
    if not ctx.stop and records:
        cache.publish_final(records)

    manifest = {
        "run_id": ts,
        "pipeline": cfg.get("pipeline", "unnamed"),
        "git_sha": git_sha(),
        "smoke": smoke,
        # Which spec actually conditioned this corpus — the config path alone is not
        # provenance, since the file behind it can change between runs.
        "constitution_sha256": hashlib.sha256(
            ctx.vars["constitution"].encode()).hexdigest(),
        "config": original_cfg,
        "effective": (cfg.get("smoke") or {}) if smoke else {},
        "ablated": ablate,
        "halted": ctx.stop,
        "dataset": None if ctx.stop else "dataset.jsonl",
        "counts": counts,
        "usage": usage.as_dict(),
        "wall_clock_s": round(time.time() - started, 1),
        "stage_seconds": durations,
        "workers": workers,
        "hf_repo": repo,
        "run_dir": str(run_dir),
        **ctx.manifest_extra,
    }
    cache.save_json("manifest.json", manifest)

    print(f"\n>>> {counts.get(stage_list[-1].name, 0)} final records in {run_dir}")
    print(f">>> spend ${usage.usd:.2f} | {manifest['wall_clock_s']}s")
    if repo:
        print(f">>> https://huggingface.co/datasets/{repo}")
    return manifest


def corpus_gate_failed(manifest: dict) -> bool:
    """True when any corpus check declaring `on_fail: error` or `stop` did not pass.

    Gating is an exit code, never an exception inside the stage: raising there would
    abort `run` before the manifest is written, throwing away the provenance and usage
    tally of a run that has already paid for every generation stage -- to report a
    diagnostic. `stop` differs from `error` only in when the run ends, not in what it
    keeps.
    """
    checks = manifest.get("corpus_checks") or {}
    return any(c.get("on_fail") in ("error", "stop") and c.get("pass") is False
               for c in checks.values())


def exit_if_gate_failed(manifest: dict) -> None:
    """Turn a failed corpus gate into the process exit status (both entrypoints end
    this way), and say what survived.

    Raises:
        SystemExit: 1 when a check declaring `on_fail: error` or `stop` did not pass.
    """
    if not corpus_gate_failed(manifest):
        return
    failed = sorted(name for name, c in (manifest.get("corpus_checks") or {}).items()
                    if c.get("pass") is False)
    print(f">>> corpus check(s) {failed} FAILED and declare on_fail "
          f"error/stop. Everything the run produced is on disk and on HF -- "
          f"snapshots, reports and the manifest; only the exit status reflects it.")
    raise SystemExit(1)


# --- the estimator ------------------------------------------------------------------


def n_units(cfg: dict) -> int:
    """How many units stage 1 will emit, derived from the config rather than declared.

    `n_traits` is a hand-maintained hint and has gone stale before. The `chunking:` flag
    makes that fatal rather than merely untidy: `whole` yields one unit and `bullet`
    yields dozens, so a declared count would misprice most methods. Chunking is offline
    and free, so the real number is always available; a declared `n_traits` that
    disagrees is treated as a config bug rather than silently overridden.
    """
    from .constitution import units_from_config

    units, _ = units_from_config(cfg)
    limit = cfg.get("max_traits")
    n = min(len(units), int(limit)) if limit else len(units)
    declared = cfg.get("n_traits")
    # `only_traits` restricts the run below the document's unit count on purpose, so the
    # hint (which tracks the document) is not compared against the restricted count.
    if declared is not None and not limit and not cfg.get("only_traits"):
        assert int(declared) == n, (
            f"n_traits: {declared} in the config, but {cfg['constitution']} yields {n} "
            f"units under chunking {cfg.get('chunking') or 'principle'!r}. Fix n_traits "
            "(or drop it -- it is only a hint; the count is derived).")
    return n


def n_examples(cfg: dict) -> int:
    """Final training examples a full run yields."""
    if "cells" in cfg:
        return sum(int(n) for n in cfg["cells"].values() if int(n) > 0)
    if "total_scenarios" in cfg:
        return int(cfg["total_scenarios"])
    return n_units(cfg) * int(cfg["scenarios_per_trait"])


def arm_shares(cfg: dict) -> dict[str, dict[str, float]]:
    """Normalised arm proportions the config's `assign` stages will produce.

    An `assign` stage declares `fields: {reply_quality: {good: 0.5, flawed: 0.5}}`, which
    is the cell-less config's equivalent of cell counts: it says what share of the corpus
    each arm gets. The estimator needs it to price a `when:`-scoped stage and to know how
    much of the corpus a gate sits in front of.
    """
    out: dict[str, dict[str, float]] = {}
    for sc in cfg.get("stages", []):
        # Either its own `kind: assign` stage, or an `assign:` block folded into a paid
        # stage that branches on the label it produces.
        spec = sc if sc.get("kind") == "assign" else (sc.get("assign") or {})
        for field, weights in (spec.get("fields") or {}).items():
            total = sum(float(w) for w in weights.values()) or 1.0
            out[field] = {str(k): float(w) / total for k, w in weights.items()}
    return out


def arm_population(cfg: dict, n: float) -> list[tuple[dict, float]]:
    """Expected record counts over the cross product of every assigned arm.

    A scalar will not do. Gates fire in sequence, and the first one changes the mix: with
    a 50/50 split and an 80% flawed gate, what reaches the second gate is 1040 flawed to
    1300 good, so the good arm is no longer half the corpus. Pricing the second gate as
    if it were overstates the survivors by several percent -- and the survivors are what
    the two most expensive stages are billed over.
    """
    pop: list[tuple[dict, float]] = [({}, float(n))]
    for field, weights in arm_shares(cfg).items():
        pop = [({**a, field: v}, c * w) for a, c in pop for v, w in weights.items()]
    return pop


def _in_scope(spec: dict | list | None, arm: dict) -> bool:
    """Whether a `when:` filter admits an arm combination (no filter = all of them).

    A list is a conjunction, mirroring `selected` -- the arm population is the cross
    product of every assigned field, so a two-condition scope prices to exactly the
    slice it covers (e.g. flawed x one weak author = 1/6 of the corpus).
    """
    if not spec:
        return True
    conds = spec if isinstance(spec, list) else [spec]
    return all(c["field"] not in arm
               or str(arm[c["field"]]) in [str(x) for x in c["in"]]
               for c in conds)


def _apply_gate(pop: list[tuple[dict, float]], sc: dict) -> list[tuple[dict, float]]:
    """Shrink the arms a `filter` stage sits in front of by its declared yield.

    `expected_keep` mirrors the stage's own shape: a scalar for a single contract, and a
    per-arm map where `keep.cases` gives each arm its own -- a gate whose two arms expect
    opposite outcomes has two different yields, and averaging them misprices both.
    """
    keep = sc.get("expected_keep", 1.0)
    if isinstance(keep, dict):
        by = sc["keep"]["by"]
        return [(a, c * float(keep.get(str(a.get(by)), 1.0))) for a, c in pop]
    return [(a, c * float(keep) if _in_scope(sc.get("when"), a) else c) for a, c in pop]


def _scoped_pop(spec: dict | None, pop: list[tuple[dict, float]]) -> float:
    """How many records a `when:`-scoped stage covers, given the live arm population."""
    return sum(c for a, c in pop if _in_scope(spec, a))


def n_final_examples(cfg: dict) -> int:
    """Documents a full run is expected to KEEP, after every `filter` stage's yield.

    `n_examples` counts what the config PLANS. Where a recipe's labels are found rather
    than assigned -- a rater deciding whether a reply really fell short -- some planned
    records are gated out, so the two numbers differ and only this one belongs in a
    cost-per-example. The yield is the config's declared `expected_keep` prior until a
    smoke run measures the real drop rate.
    """
    ablate = set(cfg.get("ablate") or [])
    if "cells" not in cfg:
        # No cells: one scalar population, shrunk by each gate in proportion to how much
        # of the corpus that gate sits in front of.
        pop = arm_population(cfg, n_examples(cfg))
        for sc in cfg["stages"]:
            if sc.get("keep") and sc["name"] not in ablate:
                pop = _apply_gate(pop, sc)
        return int(round(sum(c for _a, c in pop)))
    counts = {c: float(v) for c, v in cfg["cells"].items() if int(v) > 0}
    for sc in cfg["stages"]:
        if sc.get("keep") and sc["name"] not in ablate:
            keep = float(sc.get("expected_keep", 1.0))
            for c in _cells_in_scope(sc.get("when"), cfg, f"stage {sc['name']!r}"):
                counts[c] *= keep
    return int(round(sum(counts.values())))


def _cells_in_scope(spec: dict, cfg: dict, where: str) -> list[str]:
    """The enabled cells a `when:` filter admits, for pricing purposes."""
    from .model_eval_model_cells import CELLS

    enabled = [c for c, n in cfg["cells"].items() if int(n) > 0]
    if not spec:
        return enabled
    field, wanted = spec["field"], [str(v) for v in spec["in"]]
    getters = {"cell": lambda c: c,
               "response_kind": lambda c: CELLS[c].response_kind,
               "attribution": lambda c: CELLS[c].attribution}
    assert field in getters, (
        f"{where}: cannot price `when.field: {field}` -- expected one of "
        f"{sorted(getters)}")
    return [c for c in enabled if str(getters[field](c)) in wanted]


def _scoped_docs(sc: dict, cfg: dict, n_docs: float, counts: dict[str, float],
                 pop: list[tuple[dict, float]] | None = None) -> float:
    """How many records a per-record stage actually calls for, honouring its `when:`.

    A stage scoped to some cells must not be priced as if it ran over the whole corpus
    -- the first-turn and follow-up stages of the natural-turn configs each cover a
    subset, and the difference is tens of dollars.

    `counts` is the cell -> effective-record-count map, which shrinks as `filter` stages
    are passed: a stage after a gate runs over the survivors, not the plan.
    """
    pop = pop if pop is not None else []
    if not sc.get("when"):
        return sum(counts.values()) if counts else n_docs
    if "cells" not in cfg:
        return _scoped_pop(sc["when"], pop)
    return sum(counts[c]
               for c in _cells_in_scope(sc["when"], cfg, f"stage {sc['name']!r}"))


def _calls(cfg: dict) -> dict[str, int]:
    """Exact API-call counts per model key, derived from the stage kinds, ablation-aware."""
    from .model_eval_model_cells import CELLS

    ablate = set(cfg.get("ablate") or [])
    calls: dict[str, float] = {}
    n_docs = n_examples(cfg)
    # A cell-based config that generates its own scenarios runs per-record stages over
    # TWO different populations: the scenario pool before `plan_cells` allocates cells,
    # and the planned documents after it. Pricing everything at the document count
    # misprices the prompt-writing stages by the whole ratio between them.
    planned = not any(sc["kind"] == "plan_cells" for sc in cfg["stages"])
    n_pool = int(cfg.get("total_scenarios") or n_docs)
    # Effective record count per cell, which SHRINKS as `filter` stages are passed. A
    # found-lapse recipe gates records out before its two expensive stages, so pricing
    # those over the planned counts overstates the run by the whole drop rate. The yield
    # is a prior the config declares (`expected_keep`), exactly like `assumed_tokens`;
    # a smoke run's real drop percentages are what replace it.
    counts: dict[str, float] = {c: float(v) for c, v in cfg.get("cells", {}).items()
                                if int(v) > 0}
    # Cell-less: the arm distribution, shrunk by each gate as it is passed.
    pop = arm_population(cfg, n_docs)
    # A gate is any stage carrying a `keep:` contract -- its own `filter` stage, or a paid
    # stage that drops on the strength of what it just produced. The second kind is why
    # the gate is applied on the NEXT iteration rather than this one: such a stage is
    # billed for every record it was handed, and only its successors see the survivors.
    pending_gate: dict | None = None
    for sc in cfg["stages"]:
        kind, name = sc["kind"], sc["name"]
        if pending_gate is not None:
            if counts:  # cell-based: a gate scopes to whole cells
                keep = float(pending_gate.get("expected_keep", 1.0))
                for c in _cells_in_scope(pending_gate.get("when"), cfg,
                                         f"stage {pending_gate['name']!r}"):
                    counts[c] *= keep
            else:
                pop = _apply_gate(pop, pending_gate)
            pending_gate = None
        if kind == "plan_cells":
            planned = True
        if sc.get("keep") and name not in ablate:
            pending_gate = sc
        if name in ablate:
            continue
        if kind == "scenarios":
            from .stage_operators import scenario_batches
            n = len(scenario_batches(n_units(cfg), cfg))
        elif kind == "scenarios_weighted":
            from .constitution import units_from_config
            from .stage_operators import plan_weighted_batches
            units = units_from_config(cfg)[0]
            n = len(plan_weighted_batches([u.as_trait() for u in units], cfg))
        elif kind in ("llm_json", "llm_tagged"):
            n = _scoped_docs(sc, cfg, sum(c for _a, c in pop), counts, pop) \
                if planned else n_pool
        elif kind == "perturb_pairs":
            unknown = sorted(set(counts) - set(CELLS))
            if unknown:
                raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                                 f"Registered: {sorted(CELLS)}")
            n = sum(v for c, v in counts.items()
                    if CELLS[c].response_kind == "flawed")
        elif kind == "revise_cells":
            unknown = sorted(set(counts) - set(CELLS))
            if unknown:
                raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                                 f"Registered: {sorted(CELLS)}")
            # One rewrite per verdict-carrying document; control passes through free.
            n = sum(v for c, v in counts.items() if CELLS[c].verdicts)
        elif kind == "corpus_check":
            # Without this branch a judged corpus property would fall into the
            # deterministic `else` below and estimate at zero -- the exact trap that
            # makes a stage look free right up until the bill arrives. Attributed per
            # model, not to the stage's: pattern_scan's scan and classify passes differ
            # by ~3x in tokens per call and by model tier.
            from .check_corpus import corpus_check_calls_by_model
            for key, n in corpus_check_calls_by_model(sc, n_docs).items():
                calls[key] = calls.get(key, 0) + n
            continue
        elif kind == "generate_cells":
            unknown = sorted(set(counts) - set(CELLS))
            if unknown:
                raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                                 f"Registered: {sorted(CELLS)}")
            for c, v in counts.items():
                key = CELLS[c].model_key
                calls[key] = calls.get(key, 0) + v
            continue
        else:  # deterministic/free kinds
            continue
        calls[sc["model"]] = calls.get(sc["model"], 0) + n
    return {k: int(round(v)) for k, v in calls.items()}


def estimate(cfg: dict, measured_manifest: str | None = None) -> dict[str, Any]:
    """Estimate the USD cost of a full run of the config's pipeline.

    Args:
        cfg: Run config (its `ablate:` list, if any, is priced out). Token priors come
            from each model block's `assumed_tokens`.
        measured_manifest: Optional manifest.json from a smoke run; per-call token
            counts then come from its real per-stage usage. The scenarios kind is
            rescaled to the full batch size.

    Returns:
        A per-stage breakdown plus the total.
    """
    meas: dict[str, dict[str, float]] = {}
    if measured_manifest:
        meas, manifest = measured_per_stage(measured_manifest)
        if any(sc["kind"] == "scenarios" for sc in cfg["stages"]):
            # A smoke run asks for fewer scenarios per call than a full run. Output
            # scales with the batch size, so rescale. `scenarios_per_trait` is only
            # the legacy fallback for a config that never sets `scenarios_per_call`
            # -- a `total_scenarios`-only config (courtroom) need not define it.
            per_call = int(cfg.get("scenarios_per_call")
                           or cfg.get("scenarios_per_trait", 0))
            eff = manifest.get("effective", {})
            smoke_pc = int(eff.get("scenarios_per_call",
                                   manifest["config"].get("scenarios_per_call", 0)))
            if "scenarios" in meas and per_call and smoke_pc and smoke_pc != per_call:
                meas["scenarios"]["out_per_call"] *= per_call / smoke_pc

    calls = _calls(cfg)
    rows = []
    total = 0.0
    for key in sorted(k for k, n in calls.items() if n > 0):
        block = cfg["models"][key]
        model = block["model"]
        if key in meas:
            tin, tout = meas[key]["in_per_call"], meas[key]["out_per_call"]
            source = "measured"
        else:
            tin, tout = block["assumed_tokens"]["in"], block["assumed_tokens"]["out"]
            source = "assumed"
        price = price_of(model)
        usd = calls[key] * (tin / 1e6 * price["in"] + tout / 1e6 * price["out"])
        total += usd
        rows.append({
            "stage": key, "model": model, "calls": calls[key],
            "tokens_in_per_call": round(tin), "tokens_out_per_call": round(tout),
            "source": source, "usd": round(usd, 2),
        })

    n_docs = n_final_examples(cfg)
    return {
        "pipeline": cfg.get("pipeline", "unnamed"),
        "ablated": list(cfg.get("ablate") or []),
        "planned_documents": n_examples(cfg),
        "final_training_examples": n_docs,
        "per_stage": rows,
        "total_usd": round(total, 2),
        "usd_per_example": round(total / n_docs, 4) if n_docs else 0.0,
        "note": ("Priced from a measured smoke run."
                 if meas else
                 "Priced from built-in assumptions. Run --smoke, then pass its "
                 "manifest.json via --measured for a real estimate."),
    }
