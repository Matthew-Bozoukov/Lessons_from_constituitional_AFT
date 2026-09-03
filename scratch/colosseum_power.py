# ABOUTME: What effect size this design can actually resolve, from the OBSERVED seed variance.
# ABOUTME: Run: uv run python scratch/colosseum_power.py <control_dir> [<treatment_dir>]

"""How many seeds would Experiment 1 need?

The design fixed 40 seeds before anything ran, which was an assumption rather than a
calculation. This turns it into one, using the variance the runs actually produced
instead of a guess.

The quantity being tested is a difference of differences: each arm's within-seed effect
of the private objective (collusion cell minus baseline cell), contrasted between arms.
Both stages are paired on seed, and the second pairing is what makes the interval
tractable — if the two arms' per-seed deltas correlate, because they faced the same
ticket sets and the same cost matrices, the contrast is tighter than either arm's own
interval suggests. That correlation is measured here rather than assumed, whenever both
arms are available.

Reports the minimum detectable effect at the current n, and the n needed for effects
expressed as a FRACTION OF THE CONTROL ARM'S OWN COLLUSION EFFECT — because "the
treatment removes half the collusion" is the question, and an absolute reward-unit target
means nothing without knowing how much collusion there was to remove.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Two-sided alpha 0.05, power 0.80: (z_{1-a/2} + z_{1-b})^2 = (1.96 + 0.8416)^2
Z_SUM_SQ = (1.959964 + 0.841621) ** 2
Z_SUM = 1.959964 + 0.841621


def _deltas(run_dir: Path, measure: str = "coalition_advantage") -> dict[int, float]:
    """Per-seed within-arm effect of the private objective (treated cell minus baseline)."""
    meas = json.loads((run_dir / "results" / "per_seed.json").read_text())[measure]
    cells = sorted(meas)
    assert len(cells) == 2, f"{run_dir.name} has cells {cells}; need exactly two"
    baseline, treated = cells if cells[0] == "baseline" else cells[::-1]
    shared = sorted(set(meas[treated]) & set(meas[baseline]))
    return {int(s): meas[treated][s] - meas[baseline][s] for s in shared}


def main(argv: list[str]) -> None:
    control = _deltas(Path(argv[1]))
    c = np.array(list(control.values()), float)
    n = c.size
    print(
        f"control arm: n={n} seeds, per-seed delta mean={c.mean():+.3f} sd={c.std(ddof=1):.3f}"
    )

    if len(argv) > 2 and (Path(argv[2]) / "results" / "per_seed.json").is_file():
        treatment = _deltas(Path(argv[2]))
        shared = sorted(set(control) & set(treatment))
        c = np.array([control[s] for s in shared], float)
        t = np.array([treatment[s] for s in shared], float)
        n = len(shared)
        rho = float(np.corrcoef(c, t)[0, 1])
        sd_diff = float((t - c).std(ddof=1))
        print(
            f"treatment arm: n={n}, per-seed delta mean={t.mean():+.3f} sd={t.std(ddof=1):.3f}"
        )
        print(f"seed-pairing correlation between arms: rho={rho:+.3f}")
        print(f"sd of the PAIRED between-arm difference: {sd_diff:.3f}")
    else:
        # Only one arm so far: assume the other has the same spread and no correlation,
        # which is the PESSIMISTIC case — real pairing can only tighten it.
        sd_diff = float(c.std(ddof=1)) * np.sqrt(2)
        rho = None
        print(
            f"(treatment arm not available; assuming equal spread and rho=0 — pessimistic)"
        )
        print(f"sd of the between-arm difference: {sd_diff:.3f}")

    se = sd_diff / np.sqrt(n)
    mde = Z_SUM * se
    print(
        f"\nat n={n}: SE={se:.3f}, minimum detectable effect (80% power, a=0.05) = {mde:.3f}"
    )

    control_effect = abs(float(np.mean(list(control.values()))))
    print(
        f"\nthe control arm's OWN collusion effect is {control_effect:.3f} reward units."
    )
    print("seeds needed to detect the treatment removing:")
    for frac in (1.0, 0.75, 0.5, 0.25):
        target = control_effect * frac
        if target <= 0:
            continue
        need = Z_SUM_SQ * (sd_diff**2) / (target**2)
        print(
            f"  {frac:>5.0%} of it ({target:6.3f} units) -> n = {need:7.0f} seeds"
            + ("   <-- within reach" if need <= 200 else "")
        )


if __name__ == "__main__":
    main(sys.argv)
