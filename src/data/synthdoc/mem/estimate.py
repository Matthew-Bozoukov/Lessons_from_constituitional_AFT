# ABOUTME: Cost model for a MEM run, priced per cell family from OpenRouter rates.
# ABOUTME: Call counts are exact (the planning stage is deterministic and free).

from __future__ import annotations

from typing import Any

from ..core import PRICES, measured_per_stage

# Per-call token assumptions keyed by model_key (= the per-stage usage key). Prompts are
# dominated by the injected constitution plus a full transcript. control/critique/reflect
# were measured on the 2026-08-04 smoke runs (manifests under output/mem/).
ASSUMED: dict[str, dict[str, int]] = {
    "control": {"in": 12000, "out": 2200},
    "critique": {"in": 12100, "out": 4000},
    "reflect": {"in": 12500, "out": 3500},
    "perturb": {"in": 2500, "out": 1500},
}


def estimate(cfg: dict, measured_manifest: str | None = None) -> dict[str, Any]:
    """Estimate the USD cost of a MEM run.

    Each model_key's calls are the summed counts of the cells it serves, plus one
    perturbation call per flawed-cell document.

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

    meas = measured_per_stage(measured_manifest)[0] if measured_manifest else {}
    rows = []
    total = 0.0
    for key in sorted(calls):
        model = cfg["models"][key]["model"]
        if key in meas:
            tin, tout = meas[key]["in_per_call"], meas[key]["out_per_call"]
            source = "measured"
        else:
            tin, tout = ASSUMED[key]["in"], ASSUMED[key]["out"]
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
