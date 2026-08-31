# ABOUTME: Summarises repeated ODCV trajectories as rollout-level violations.
# ABOUTME: Reports the requested count/n and a CI conditional on the frozen cells.

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np

from src.eval.misalignment.odcv import VIOLATION_THRESHOLD


def main(results: str, expected_cells: int = 70, expected_rollouts: int = 5) -> None:
    """Write rollout MR and a within-cell bootstrap interval beside results.json."""
    path = Path(results)
    payload = json.loads(path.read_text())
    by_cell: dict[str, list[float]] = {}
    for variant, units in payload["per_scenario_medians"].items():
        for unit, score in units.items():
            scenario, rollout = unit.rsplit("/", 1)
            assert rollout.startswith("rollout_")
            by_cell.setdefault(f"{variant}/{scenario}", []).append(float(score))

    assert 0 < len(by_cell) <= expected_cells, (len(by_cell), expected_cells)
    assert all(0 < len(scores) <= expected_rollouts for scores in by_cell.values())
    binary = {
        cell: np.asarray([score >= VIOLATION_THRESHOLD for score in scores], dtype=float)
        for cell, scores in sorted(by_cell.items())
    }
    n_misaligned = int(sum(values.sum() for values in binary.values()))
    n_rollouts = int(sum(values.size for values in binary.values()))

    # Cells remain fixed. Within each cell, resample its observed trajectories.
    rng = np.random.default_rng(0)
    draw_sums = np.zeros(100_000)
    for values in binary.values():
        idx = rng.integers(0, values.size, size=(100_000, values.size))
        draw_sums += values[idx].sum(axis=1)
    draws = draw_sums / n_rollouts
    lo, hi = np.percentile(draws, [2.5, 97.5]) * 100
    summary = {
        "estimand": "percent rollout misalignment conditional on the observed frozen ODCV cells",
        "violation_threshold": VIOLATION_THRESHOLD,
        "planned_cells": expected_cells,
        "observed_cells": len(binary),
        "target_rollouts_per_cell": expected_rollouts,
        "observed_rollouts_per_cell": {
            "min": min(values.size for values in binary.values()),
            "max": max(values.size for values in binary.values()),
        },
        "num_misaligned_rollouts": n_misaligned,
        "n_rollouts": n_rollouts,
        "misalignment_rate": n_misaligned / n_rollouts,
        "misalignment_rate_pct": round(100 * n_misaligned / n_rollouts, 2),
        "conditional_rollout_bootstrap_ci95_pct": [round(float(lo), 2), round(float(hi), 2)],
        "bootstrap_draws": 100_000,
        "bootstrap_seed": 0,
    }
    out = path.with_name("repeated_rollout_summary.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f">>> {out}")


if __name__ == "__main__":
    fire.Fire(main)
