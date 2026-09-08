# ABOUTME: Coverage of src/eval/stats.interval WITH vs WITHOUT the rollout-noise floor,
# ABOUTME: over the ODCV regimes that matter for it: few rollouts per cell, and unequal R.
# Run: uv run python scratch/stats/noise_floor_coverage.py [--reps 2000]

"""Does flooring the SE at the within-cell noise term help, and what does it cost?

The floor only binds when a sample's observed spread of CELL MEANS falls below what the
rollout draws alone can support. That is rare at R=5 and common at R=2 on a binary outcome,
which is exactly ODCV's regime, so the question is empirical: does it fix under-coverage
where it binds, and does it over-widen where it does not?

Both arms of the comparison run the SAME code path on the SAME collapsed table. "Without" is
produced by blanking `within_cell_var`, which is precisely the information the floor uses
and nothing else does -- so the two differ in the floor and in nothing else.

Reported per regime: coverage (target 95%), mean half-width, and how often the floor bound.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.eval.stats import collapse, interval

_spec = importlib.util.spec_from_file_location(
    "simcov", Path(__file__).with_name("simulate_coverage.py"))
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)
draw, logistic, truth, DESIGN, J, VARIANTS = (
    _sim.draw, _sim.logistic, _sim.truth, _sim.DESIGN, _sim.J, _sim.VARIANTS)

# Scenario/checkpoint ingredients held at the sim's "base" and "near-zero" shapes; what varies
# here is R, because that is what the floor responds to. `R="mixed"` draws each cell's rollout
# count from {1, 2, 3} -- the unequal-R case a dropped-and-recovered ODCV pass produces.
REGIMES = {
    "base, R=5 (the sim's usual)":  dict(D=("mixture", .6, -1.5, .5, 1.0, .5), sC=.30, R=5),
    "base, R=2 (ODCV's passes=2)":  dict(D=("mixture", .6, -1.5, .5, 1.0, .5), sC=.30, R=2),
    "base, R mixed 1-3":            dict(D=("mixture", .6, -1.5, .5, 1.0, .5), sC=.30, R="mixed"),
    "near-zero, R=2":               dict(D=("mixture", .85, -4.5, .5, -1.5, .5), sC=.30, R=2),
    "near-zero, R mixed 1-3":       dict(D=("mixture", .85, -4.5, .5, -1.5, .5), sC=.30, R="mixed"),
    "high-agreement scenarios, R=2": dict(D=("bernoulli", .5), sC=.05, R=2),
    # The adversarial case for a spread-based SE: scenarios are IDENTICAL (sigma_b = 0), so
    # the observed spread of cell means is pure rollout noise and a sample can show almost
    # none of it. This is case B of the 0.5-vs-{0,1} example, generated rather than staged.
    "identical scenarios (sigma_b=0), R=2": dict(D=("normal", 0.0, 0.0), sC=.0, R=2),
    "identical scenarios (sigma_b=0), R=5": dict(D=("normal", 0.0, 0.0), sC=.0, R=5),
}
A_SPEC = ("normal", 0, .35)


def rows_for(rng, p, held_a: float) -> list[dict]:
    """One single-arm experiment: J scenarios x 2 variants x R rollouts, one fixed checkpoint."""
    D = draw(rng, p["D"], J)
    out = []
    for j in range(J):
        for k, g in zip(VARIANTS, (.5, -.5)):
            pi = logistic(D[j] + held_a + g + rng.normal(0, p["sC"]))
            r = int(rng.integers(1, 4)) if p["R"] == "mixed" else p["R"]
            for t in range(r):
                out.append({"checkpoint": "m0", "scenario": f"s{j}", "variant": k,
                            "pass": t, "value": float(rng.random() < pi)})
    return out


def blanked(table):
    """The same table with the within-cell information removed: the pre-floor estimator."""
    return dataclasses.replace(table, within_cell_var=np.full_like(table.within_cell_var, np.nan))


def main(reps: int = 2000, truth_draws: int = 2_000_000, seed: int = 0) -> None:
    print(f"{'regime':<32} {'cover':>7} {'cover':>7} {'width':>8} {'width':>8} {'floor':>7}")
    print(f"{'':32} {'no floor':>7} {'floor':>7} {'no floor':>8} {'floor':>8} {'binds':>7}")
    print("-" * 76)
    for label, spec in REGIMES.items():
        p = {**spec, "A": A_SPEC}
        rng = np.random.default_rng(seed)
        held_a = float(draw(rng, A_SPEC, 1)[0])
        mu, mu_err = truth(rng, p, draws=truth_draws, held_a=held_a)
        cov = [0, 0]
        width = [0.0, 0.0]
        bound = 0
        for _ in range(reps):
            table = collapse(rows_for(rng, p, held_a), DESIGN)
            r_new = interval(table, bounds=(0.0, 1.0))
            r_old = interval(blanked(table), bounds=(0.0, 1.0))
            for i, r in enumerate((r_old, r_new)):
                cov[i] += int(r.lo <= mu <= r.hi)
                width[i] += (r.hi - r.lo) / 2
            bound += int("floored" in r_new.method)
        n = float(reps)
        print(f"{label:<32} {100 * cov[0] / n:6.1f}% {100 * cov[1] / n:6.1f}% "
              f"{width[0] / n:8.4f} {width[1] / n:8.4f} {100 * bound / n:6.1f}%")
    print(f"\n{reps} replicates per regime; truth from {truth_draws:,} draws "
          f"(MC error ~{mu_err:.2e}). J={J} scenarios, one fixed checkpoint, both variants.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--truth-draws", type=int, default=2_000_000)
    sys.exit(main(**vars(ap.parse_args())))
