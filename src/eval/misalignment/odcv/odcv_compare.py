# ABOUTME: Paired comparison of two ODCV-Bench arms scored on an identical scenario set,
# ABOUTME: with a scenario-level paired bootstrap and McNemar test on the violation flips.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD, misalignment_rate  # noqa: E402
from src.eval.misalignment.odcv.stats import arm_difference, mcnemar_exact  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402


def _cells(results_path: Path) -> dict[str, float]:
    """Flatten one arm's per-scenario medians into {"variant/scenario": median}."""
    res = json.loads(results_path.read_text())
    med = res["per_scenario_medians"]
    return {f"{v}/{s}": score for v in VARIANTS for s, score in med[v].items()}


def main(
    treatment: str,
    control: str,
    out: str = "output/odcv_bench/comparison",
) -> None:
    """Compare two judged ODCV arms on their shared scenario cells.

    Args:
        treatment: results.json of the arm under test (e.g. the fine-tune).
        control: results.json of the control arm (e.g. the base model).
        out: Output directory for the comparison artifacts.
    """
    t_cells, c_cells = _cells(Path(treatment)), _cells(Path(control))
    shared = sorted(set(t_cells) & set(c_cells))
    assert shared, "arms share no scenario cells"
    only_t, only_c = sorted(set(t_cells) - set(c_cells)), sorted(set(c_cells) - set(t_cells))

    t_scores = [t_cells[k] for k in shared]
    c_scores = [c_cells[k] for k in shared]

    # McNemar on discordant pairs: which arm flipped each scenario's violation flag.
    t_only = [k for k in shared
              if t_cells[k] >= VIOLATION_THRESHOLD > c_cells[k]]
    c_only = [k for k in shared
              if c_cells[k] >= VIOLATION_THRESHOLD > t_cells[k]]
    b, c = len(t_only), len(c_only)
    p_exact = mcnemar_exact(b, c)

    boot = arm_difference({k: t_cells[k] for k in shared}, {k: c_cells[k] for k in shared})
    summary = {
        "n_shared_cells": len(shared),
        "cells_only_in_treatment": only_t,
        "cells_only_in_control": only_c,
        "treatment": {"mr_pct": round(misalignment_rate(t_scores), 1),
                      "mean_severity": round(float(np.mean(t_scores)), 2)},
        "control": {"mr_pct": round(misalignment_rate(c_scores), 1),
                    "mean_severity": round(float(np.mean(c_scores)), 2)},
        "paired": boot,
        "mcnemar": {
            "violation_only_in_treatment": b,
            "violation_only_in_control": c,
            "concordant": len(shared) - b - c,
            "p_exact_two_sided": round(p_exact, 5),
            "flipped_to_safe": c_only,
            "flipped_to_violation": t_only,
        },
    }

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(out_dir, {"treatment": treatment, "control": control},
                   extra={"timestamp": timestamp()})

    t, c_, p = summary["treatment"], summary["control"], summary["paired"]
    print("=" * 68)
    print(f"PAIRED COMPARISON over {len(shared)} shared scenario cells")
    print("=" * 68)
    print(f"  treatment MR {t['mr_pct']:>5}%   sev {t['mean_severity']}")
    print(f"  control   MR {c_['mr_pct']:>5}%   sev {c_['mean_severity']}")
    print(f"  MR difference : {p['mr_diff_pp']:+.1f} pp  CI95 {p['mr_diff_ci95']}  "
          f"p={p['mr_diff_p_two_sided']}")
    print(f"  Sev difference: {p['sev_diff']:+.2f}     CI95 {p['sev_diff_ci95']}  "
          f"p={p['sev_diff_p_two_sided']}")
    m = summary["mcnemar"]
    print(f"  McNemar: {m['violation_only_in_control']} flipped to SAFE, "
          f"{m['violation_only_in_treatment']} flipped to VIOLATION, "
          f"p={m['p_exact_two_sided']}")
    if only_t or only_c:
        print(f"  NOTE unmatched cells dropped: {len(only_t)} treatment-only, "
              f"{len(only_c)} control-only")
    print(f">>> {out_dir/'comparison.json'}")

