# ABOUTME: Cost model for a full synthdoc run, priced per stage from OpenRouter rates.
# ABOUTME: Uses a smoke run's measured per-stage token counts when one is supplied.

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
