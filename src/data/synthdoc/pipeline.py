# ABOUTME: The config-driven pipeline engine: builds Stage objects from the config's
# ABOUTME: `stages:` list and owns caching, checkpoints, ablation, budget, manifest, estimates.

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from src.utils import git_sha, timestamp

from .constitution import full_text
from .core import PRICES, Checkpoint, Ctx, Stage, Usage, measured_per_stage
from .hf_cache import StageCache
from .operators import OPERATORS


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
    from src.hf_publish import origin_url
    cache = StageCache(run_dir, repo, private=bool(cfg.get("hf_private", False)),
                       card_fields={
                           "experiment": f"synthdoc `{cfg['pipeline']}` run — per-stage "
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
                                         f"configs/data/synthdoc/{cfg['pipeline']}.yaml",
                       })

    workers = int(cfg.get("workers", 8))
    budget = float(cfg.get("budget_usd", 0)) or None
    usage = Usage()
    ctx = Ctx(cfg=cfg, usage=usage, workers=workers, run_dir=run_dir, smoke=smoke,
              vars={"constitution": full_text(cfg["constitution"])})

    stage_list = build_stages(cfg)
    ablate = [str(a) for a in (cfg.get("ablate") or [])]
    _validate_ablate(ablate, stage_list)

    records: list[dict] = []
    durations: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, st in enumerate(stage_list, start=1):
        label = f"stage {i} ({st.name})"
        if st.skip and st.skip(ctx, records):
            print(f">>> {label}: not applicable -- skipped")
            continue
        if cache.has(i, st.name):
            records = cache.load(i, st.name)
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
            ckpt = Checkpoint(run_dir / f"stage_{i}_{st.name}.partial.jsonl",
                              key=st.checkpoint_key) if st.checkpoint_key else None
            t0 = time.time()
            if st.name in ablate:
                records = st.ablate_fn(records)
                print(f">>> {label}: ABLATED -- null-operation applied to "
                      f"{len(records)} records")
            else:
                records = st.fn(ctx, records, ckpt)
            durations[st.name] = round(time.time() - t0, 1)
            cache.save(i, st.name, records)
            print(f">>> {label}: {len(records)} records")
        counts[st.name] = len(records)
        if st.preview and records:
            print(f"    FIRST: {st.preview(records[0])[:220]}")

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


# --- the estimator ------------------------------------------------------------------


def n_examples(cfg: dict) -> int:
    """Final training examples a full run yields."""
    if "cells" in cfg:
        return sum(int(n) for n in cfg["cells"].values() if int(n) > 0)
    if "total_scenarios" in cfg:
        return int(cfg["total_scenarios"])
    return int(cfg.get("n_traits", 8)) * int(cfg["scenarios_per_trait"])


def _calls(cfg: dict) -> dict[str, int]:
    """Exact API-call counts per model key, derived from the stage kinds, ablation-aware."""
    from .cells import CELLS

    ablate = set(cfg.get("ablate") or [])
    calls: dict[str, int] = {}
    n_docs = n_examples(cfg)
    for sc in cfg["stages"]:
        kind, name = sc["kind"], sc["name"]
        if name in ablate:
            continue
        if kind == "scenarios":
            per_trait = int(cfg["scenarios_per_trait"])
            per_call = int(cfg.get("scenarios_per_call", per_trait))
            n = int(cfg.get("n_traits", 8)) * math.ceil(per_trait / per_call)
        elif kind == "scenarios_weighted":
            from .constitution import segment as _segment
            from .operators import plan_weighted_batches
            n = len(plan_weighted_batches(_segment(cfg["constitution"])[0], cfg))
        elif kind in ("llm_json", "llm_tagged"):
            n = n_docs
        elif kind == "perturb_pairs":
            enabled = {c: int(v) for c, v in cfg["cells"].items() if int(v) > 0}
            unknown = sorted(set(enabled) - set(CELLS))
            if unknown:
                raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                                 f"Registered: {sorted(CELLS)}")
            n = sum(v for c, v in enabled.items()
                    if CELLS[c].response_kind == "flawed")
        elif kind == "generate_cells":
            enabled = {c: int(v) for c, v in cfg["cells"].items() if int(v) > 0}
            unknown = sorted(set(enabled) - set(CELLS))
            if unknown:
                raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                                 f"Registered: {sorted(CELLS)}")
            for c, v in enabled.items():
                key = CELLS[c].model_key
                calls[key] = calls.get(key, 0) + v
            continue
        else:  # deterministic/free kinds
            continue
        calls[sc["model"]] = calls.get(sc["model"], 0) + n
    return calls


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
            # A smoke run asks for 1 scenario per call; a full run asks for
            # `per_call`. Output scales with the batch size, so rescale.
            per_trait = int(cfg["scenarios_per_trait"])
            per_call = int(cfg.get("scenarios_per_call", per_trait))
            eff = manifest.get("effective", {})
            smoke_pc = int(eff.get("scenarios_per_call",
                                   manifest["config"].get("scenarios_per_call", 0)))
            if "scenarios" in meas and smoke_pc and smoke_pc != per_call:
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
        price = PRICES.get(model, {"in": 0.0, "out": 0.0})
        usd = calls[key] * (tin / 1e6 * price["in"] + tout / 1e6 * price["out"])
        total += usd
        rows.append({
            "stage": key, "model": model, "calls": calls[key],
            "tokens_in_per_call": round(tin), "tokens_out_per_call": round(tout),
            "source": source, "usd": round(usd, 2),
        })

    n_docs = n_examples(cfg)
    return {
        "pipeline": cfg.get("pipeline", "unnamed"),
        "ablated": list(cfg.get("ablate") or []),
        "final_training_examples": n_docs,
        "per_stage": rows,
        "total_usd": round(total, 2),
        "usd_per_example": round(total / n_docs, 4) if n_docs else 0.0,
        "note": ("Priced from a measured smoke run."
                 if meas else
                 "Priced from built-in assumptions. Run --smoke, then pass its "
                 "manifest.json via --measured for a real estimate."),
    }
