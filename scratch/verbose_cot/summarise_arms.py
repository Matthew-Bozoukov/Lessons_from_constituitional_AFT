# ABOUTME: Pull the two verbose-CoT arms' own ODCV results into one JSON: headline numbers,
# ABOUTME: the paired rows-vs-tokens difference, and per-scenario detail for the artifact.
# Run: uv run python scratch/verbose_cot/summarise_arms.py [--out <path>]

"""Scope is exactly the two verbose-CoT LoRAs and nothing else.

Both arms ran the SAME 30 incentivized cells, which makes a per-scenario PAIRED
difference available. That matters because the two marginal CIs overlap heavily, and
overlapping intervals are not a test: a paired comparison can show a real effect the
overlap hides, or — as here — show that there is nothing to find.

Bootstrap note: the arms are resampled TOGETHER by scenario, one index draw applied to
both. Drawing each arm separately would throw away the pairing the shared cell set buys
and widen the interval on the difference for no reason.

The `published` base figure quoted alongside is the reference the harness already records
inside each arm's own results.json, not a separate run pulled in for comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VIOLATION_THRESHOLD, scenario_violation_rate,
)

RUN_ROOT = ROOT / "output" / "odcv_verbose" / "root"
ARMS = {
    "rows": ("qwen3_6-27b-lora-da716-verbose-rows-r64",
             "Row-matched: difficult advice held at 7.16% of rows, so its share of "
             "trainable tokens rises to 47.6%"),
    "tokens": ("qwen3_6-27b-lora-da-verbose-tokenmatched-r64",
               "Token-matched: difficult advice's share of trainable tokens held at the "
               "baseline, which costs rows"),
}


def _results(model_key: str) -> dict:
    """The newest combined run's results.json for one arm."""
    combined = sorted((RUN_ROOT / model_key).glob("combined3x_*"))[-1]
    return json.loads((combined / "results.json").read_text(encoding="utf-8"))


def _paired_diff(a: dict[str, list[float]], b: dict[str, list[float]],
                 scenarios: list[str], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap the mean per-scenario difference in violation rate, a minus b."""
    ra = np.array([scenario_violation_rate(a[s]) for s in scenarios]) * 100
    rb = np.array([scenario_violation_rate(b[s]) for s in scenarios]) * 100
    diff = ra - rb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenarios), size=(n_boot, len(scenarios)))
    draws = diff[idx].mean(axis=1)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {
        "mean_diff_pp": round(float(diff.mean()), 1),
        "ci95": [round(lo, 1), round(hi, 1)],
        # Whether it crosses zero IS the verdict, so state it rather than imply it.
        "crosses_zero": bool(lo <= 0 <= hi),
        "n_rows_worse": int((diff > 0).sum()),
        "n_tokens_worse": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
    }


def main(out: str = "output/odcv_verbose/arms_summary.json") -> None:
    docs = {name: _results(key) for name, (key, _) in ARMS.items()}
    sev = {name: doc["per_scenario_medians"]["incentivized"] for name, doc in docs.items()}

    shared = sorted(set(sev["rows"]) & set(sev["tokens"]))
    for name, d in sev.items():
        assert set(d) == set(shared), f"{name} does not run the shared cell set"

    arms = {}
    for name, doc in docs.items():
        o = doc["ours"]["overall"]
        arms[name] = {
            "model_key": ARMS[name][0], "label": ARMS[name][1],
            "mr_pct": o["mr_pct"], "mr_ci95": o["mr_ci95"],
            "mean_severity": o["mean_severity"],
            "severity_ci95": o["severity_ci95"],
            "n_scenarios": o["n_scenarios"], "n_rollouts": o["n_rollouts"],
            "judges": doc["judges"],
        }

    per_scenario = []
    for s in shared:
        row = {"scenario": s}
        for name, d in sev.items():
            row[name] = {"rate": round(scenario_violation_rate(d[s]), 3),
                         "severities": d[s], "n": len(d[s])}
        # A scenario is only informative about the DIFFERENCE if the arms disagree on it,
        # and only informative about the CI if it varies across its own rollouts.
        row["agrees"] = row["rows"]["rate"] == row["tokens"]["rate"]
        row["mixed"] = any(0 < row[n]["rate"] < 1 for n in sev)
        per_scenario.append(row)
    per_scenario.sort(key=lambda r: -(r["rows"]["rate"] + r["tokens"]["rate"]))

    consistency = {
        name: {
            "always": sum(1 for r in per_scenario if r[name]["rate"] == 1),
            "never": sum(1 for r in per_scenario if r[name]["rate"] == 0),
            "mixed": sum(1 for r in per_scenario if 0 < r[name]["rate"] < 1),
        } for name in sev
    }

    doc = {
        "variant": "incentivized",
        "shared_scenarios": len(shared),
        "violation_threshold": VIOLATION_THRESHOLD,
        "base_published_incentivized_mr_pct":
            docs["rows"]["published"]["incentivized"]["mr_pct"],
        "arms": arms,
        "paired_rows_vs_tokens": _paired_diff(sev["rows"], sev["tokens"], shared),
        "consistency": consistency,
        "per_scenario": per_scenario,
    }
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"shared incentivized cells: {len(shared)}\n")
    for name, a in arms.items():
        print(f"  {name:7s} MR {a['mr_pct']:5.1f}%  CI {str(a['mr_ci95']):14s} "
              f"sev {a['mean_severity']:.2f}  ({a['n_rollouts']} rollouts)")
    print(f"  base    MR {doc['base_published_incentivized_mr_pct']:5.1f}%  (published "
          "reference recorded in each arm's own results.json)\n")
    p = doc["paired_rows_vs_tokens"]
    print(f"  paired rows - tokens: {p['mean_diff_pp']:+.1f} pp  CI {p['ci95']}  "
          f"{'no detectable difference' if p['crosses_zero'] else 'SIGNIFICANT'}")
    print(f"    scenarios where they differ: "
          f"{p['n_rows_worse'] + p['n_tokens_worse']}/{len(shared)}")
    for name, c in consistency.items():
        print(f"    {name:7s} always {c['always']:2d} / never {c['never']:2d} "
              f"/ mixed {c['mixed']:2d}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    fire.Fire(main)
