# ABOUTME: Paired comparison of two ODCV-Bench arms scored on an identical scenario set,
# ABOUTME: with a scenario-level paired bootstrap and McNemar test on the violation flips.

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np

from src.eval.misalignment.odcv import VARIANTS, VIOLATION_THRESHOLD, misalignment_rate  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402


def _cells(results_path: Path) -> dict[str, float]:
    """Flatten one arm's per-scenario medians into {"variant/scenario": median}."""
    res = json.loads(results_path.read_text())
    med = res["per_scenario_medians"]
    return {f"{v}/{s}": score for v in VARIANTS for s, score in med[v].items()}


def _paired_bootstrap(pairs: list[tuple[float, float]], n_boot: int, seed: int) -> dict:
    """Bootstrap the treatment-minus-control difference, resampling scenarios.

    Args:
        pairs: One (treatment, control) median pair per scenario cell.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Point estimates and 95% CIs for the MR difference and severity difference.
    """
    arr = np.asarray(pairs, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"bad shape {arr.shape}"
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = arr[idx]
    assert draws.shape == (n_boot, n, 2), f"bad shape {draws.shape}"

    viol = draws >= VIOLATION_THRESHOLD
    mr_diff = 100.0 * (viol[:, :, 0].mean(axis=1) - viol[:, :, 1].mean(axis=1))
    sev_diff = draws[:, :, 0].mean(axis=1) - draws[:, :, 1].mean(axis=1)
    return {
        "mr_diff_pp": round(
            100.0 * ((arr[:, 0] >= VIOLATION_THRESHOLD).mean()
                     - (arr[:, 1] >= VIOLATION_THRESHOLD).mean()), 1),
        "mr_diff_ci95": [round(float(np.percentile(mr_diff, 2.5)), 1),
                         round(float(np.percentile(mr_diff, 97.5)), 1)],
        "mr_diff_p_two_sided": round(2 * min((mr_diff >= 0).mean(), (mr_diff <= 0).mean()), 4),
        "sev_diff": round(float(arr[:, 0].mean() - arr[:, 1].mean()), 2),
        "sev_diff_ci95": [round(float(np.percentile(sev_diff, 2.5)), 2),
                          round(float(np.percentile(sev_diff, 97.5)), 2)],
        "sev_diff_p_two_sided": round(2 * min((sev_diff >= 0).mean(), (sev_diff <= 0).mean()), 4),
    }


def main(
    treatment: str,
    control: str,
    out: str = "output/odcv_bench/comparison",
    n_boot: int = 10000,
    seed: int = 0,
) -> None:
    """Compare two judged ODCV arms on their shared scenario cells.

    Args:
        treatment: results.json of the arm under test (e.g. the fine-tune).
        control: results.json of the control arm (e.g. the base model).
        out: Output directory for the comparison artifacts.
        n_boot: Bootstrap resamples.
        seed: RNG seed.
    """
    t_cells, c_cells = _cells(Path(treatment)), _cells(Path(control))
    shared = sorted(set(t_cells) & set(c_cells))
    assert shared, "arms share no scenario cells"
    only_t, only_c = sorted(set(t_cells) - set(c_cells)), sorted(set(c_cells) - set(t_cells))

    pairs = [(t_cells[k], c_cells[k]) for k in shared]
    t_scores = [p[0] for p in pairs]
    c_scores = [p[1] for p in pairs]

    # McNemar on discordant pairs: which arm flipped each scenario's violation flag.
    t_only = [k for k in shared
              if t_cells[k] >= VIOLATION_THRESHOLD > c_cells[k]]
    c_only = [k for k in shared
              if c_cells[k] >= VIOLATION_THRESHOLD > t_cells[k]]
    b, c = len(t_only), len(c_only)
    # Exact two-sided binomial test on the discordant pairs under p=0.5.
    from math import comb
    n_disc = b + c
    p_exact = (
        min(1.0, 2 * sum(comb(n_disc, i) for i in range(0, min(b, c) + 1)) / 2**n_disc)
        if n_disc else 1.0
    )

    boot = _paired_bootstrap(pairs, n_boot, seed)
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
    write_run_meta(out_dir, {"treatment": treatment, "control": control, "seed": seed},
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


if __name__ == "__main__":
    fire.Fire(main)
