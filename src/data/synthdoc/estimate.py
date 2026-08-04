# ABOUTME: Cost models for a full synthdoc or MEM run, priced per stage from OpenRouter
# ABOUTME: rates. Uses a smoke run's measured per-stage token counts when one is supplied.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .stages import PRICES

# Per-call token assumptions used only when no measured run is supplied. Prompt sizes are
# dominated by what each stage injects: stages 4 and 6 carry the whole constitution.
ASSUMED: dict[str, dict[str, int]] = {
    "scenarios": {"in": 700, "out": 1400},
    "draft": {"in": 500, "out": 700},
    "refine": {"in": 2000, "out": 900},
    "respond": {"in": 1200, "out": 1600},
    "rewrite": {"in": 3200, "out": 1800},
}


def _calls(stage: str, n_traits: int, n_scenarios: int, per_trait: int,
           per_call: int) -> int:
    """Return how many API calls a stage makes at full scale."""
    if stage == "scenarios":
        return n_traits * math.ceil(per_trait / per_call)
    return n_scenarios


def _measured(manifest_path: str) -> tuple[dict[str, dict[str, float]], dict]:
    """Return per-stage per-call averages from a completed run's manifest.

    Args:
        manifest_path: Path to manifest.json.

    Returns:
        (mapping stage -> {in_per_call, out_per_call}, the manifest itself).
    """
    m = json.loads(Path(manifest_path).read_text())
    by_stage = m["usage"].get("by_stage", {})
    out = {}
    for stage, b in by_stage.items():
        calls = max(b["calls"], 1)
        out[stage] = {"in_per_call": b["prompt_tokens"] / calls,
                      "out_per_call": b["completion_tokens"] / calls}
    return out, m


def estimate(cfg: dict, measured_manifest: str | None = None) -> dict[str, Any]:
    """Estimate the USD cost of a full run.

    Args:
        cfg: Run config.
        measured_manifest: Optional manifest.json from a smoke run. Per-call token counts
            then come from that run's real per-stage usage, which is far more accurate
            than the built-in assumptions.

    Returns:
        A per-stage breakdown plus the total.

    Raises:
        AssertionError: If the measured run used a different scenarios-per-call batch
            size, which would misprice the scenario stage.
    """
    n_traits = int(cfg.get("n_traits", 8))
    per_trait = int(cfg["scenarios_per_trait"])
    per_call = int(cfg.get("scenarios_per_call", per_trait))
    n_scen = n_traits * per_trait

    meas: dict[str, dict[str, float]] = {}
    if measured_manifest:
        meas, manifest = _measured(measured_manifest)
        eff = manifest.get("effective", {})
        smoke_per_call = int(eff.get("scenarios_per_call",
                                     manifest["config"].get("scenarios_per_call", 0)))
        if "scenarios" in meas and smoke_per_call and smoke_per_call != per_call:
            # A smoke run asks for 1 scenario per call; a full run asks for `per_call`.
            # Output scales with the batch size, so rescale rather than mislead.
            meas["scenarios"]["out_per_call"] *= per_call / smoke_per_call

    rows = []
    total = 0.0
    for stage in ("scenarios", "draft", "refine", "respond", "rewrite"):
        model = cfg["models"][stage]["model"]
        calls = _calls(stage, n_traits, n_scen, per_trait, per_call)
        if stage in meas:
            tin, tout = meas[stage]["in_per_call"], meas[stage]["out_per_call"]
            source = "measured"
        else:
            tin, tout = ASSUMED[stage]["in"], ASSUMED[stage]["out"]
            source = "assumed"
        price = PRICES.get(model, {"in": 0.0, "out": 0.0})
        usd = calls * (tin / 1e6 * price["in"] + tout / 1e6 * price["out"])
        total += usd
        rows.append({
            "stage": stage, "model": model, "calls": calls,
            "tokens_in_per_call": round(tin), "tokens_out_per_call": round(tout),
            "source": source, "usd": round(usd, 2),
        })

    return {
        "n_traits": n_traits,
        "scenarios_per_trait": per_trait,
        "scenarios_per_call": per_call,
        "n_scenarios": n_scen,
        "final_training_examples": n_scen,
        "per_stage": rows,
        "total_usd": round(total, 2),
        "usd_per_example": round(total / n_scen, 4) if n_scen else 0.0,
        "note": ("Priced from a measured smoke run; the scenario stage's output is "
                 "rescaled to the full batch size."
                 if meas else
                 "Priced from built-in assumptions. Run --smoke, then pass its "
                 "manifest.json via --measured for a real estimate."),
    }


# Per-call token assumptions for the MEM stages, keyed by model_key (= the per-stage
# usage key). Prompts are dominated by the injected constitution plus a full transcript.
# control/critique were measured on the 2026-08-04 smoke run (manifest in
# output/mem/smoke_20260804_132908); perturb is still a guess until its pass lands.
ASSUMED_MEM: dict[str, dict[str, int]] = {
    "control": {"in": 12000, "out": 2200},
    "critique": {"in": 12100, "out": 4000},
    "perturb": {"in": 2500, "out": 1500},
}


def estimate_mem(cfg: dict, measured_manifest: str | None = None) -> dict[str, Any]:
    """Estimate the USD cost of a MEM run.

    Call counts are exact, not assumed: the planning stage is deterministic and free, so
    each model_key's calls are just the summed counts of the cells it serves (plus one
    perturbation call per flawed-cell document).

    Args:
        cfg: MEM run config (must carry a `cells` block).
        measured_manifest: Optional manifest.json from a smoke run; per-call token
            counts then come from its real per-model_key usage.

    Returns:
        A per-stage breakdown plus the total.
    """
    from .stages import CELLS

    cells = {c: int(n) for c, n in cfg["cells"].items() if int(n) > 0}
    unknown = sorted(set(cells) - set(CELLS))
    if unknown:
        raise ValueError(f"unregistered cell(s) enabled: {unknown}. "
                         f"Registered: {sorted(CELLS)}")

    calls: dict[str, int] = {}
    for c, n in cells.items():
        key = CELLS[c].model_key
        calls[key] = calls.get(key, 0) + n
    n_flawed = sum(n for c, n in cells.items() if CELLS[c].response_kind == "flawed")
    if n_flawed:
        calls["perturb"] = n_flawed

    meas = _measured(measured_manifest)[0] if measured_manifest else {}
    rows = []
    total = 0.0
    for key in sorted(calls):
        model = cfg["models"][key]["model"]
        if key in meas:
            tin, tout = meas[key]["in_per_call"], meas[key]["out_per_call"]
            source = "measured"
        else:
            tin, tout = ASSUMED_MEM[key]["in"], ASSUMED_MEM[key]["out"]
            source = "assumed"
        price = PRICES.get(model, {"in": 0.0, "out": 0.0})
        usd = calls[key] * (tin / 1e6 * price["in"] + tout / 1e6 * price["out"])
        total += usd
        rows.append({
            "stage": key, "model": model, "calls": calls[key],
            "tokens_in_per_call": round(tin), "tokens_out_per_call": round(tout),
            "source": source, "usd": round(usd, 2),
        })

    n_docs = sum(cells.values())
    return {
        "cells": cells,
        "final_training_examples": n_docs,
        "per_stage": rows,
        "total_usd": round(total, 2),
        "usd_per_example": round(total / n_docs, 4) if n_docs else 0.0,
        "note": ("Priced from a measured smoke run."
                 if meas else
                 "Priced from built-in assumptions. Run --smoke, then pass its "
                 "manifest.json via --measured for a real estimate."),
    }
