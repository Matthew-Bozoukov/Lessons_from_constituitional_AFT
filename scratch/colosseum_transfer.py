# ABOUTME: The transfer ratio — Experiment 1's effect over Experiment 2's, bootstrapped on
# ABOUTME: a shared seed draw. Run: uv run python scratch/colosseum_transfer.py <4 run dirs>

"""Does the trained disposition survive contact with a peer?

The ratio of the multi-agent effect to the single-agent one is what Experiments 1 and 2
exist to produce together: near 1 means the disposition is worth as much against a
colluding partner as it is alone; well below 1 means it leaks exactly where the pressure
comes from another agent.

It is a ratio of two noisy estimates, so `stats.transfer_ratio` bootstraps it over a
shared seed draw and REFUSES to report when the denominator's interval covers zero. That
refusal is the interesting outcome as often as the number is: "the disposition transfers"
and "neither experiment moved" produce similar-looking ratios, and only the denominator
check tells them apart.

    uv run python scratch/colosseum_transfer.py \
        --collusion-control <dir> --collusion-treatment <dir> \
        --single-control <dir> --single-treatment <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.misalignment.colosseum.stats import transfer_ratio


def _delta(run_dir: str, treated: str) -> dict[int, float]:
    """Per-seed effect of the private objective within one arm (treated minus baseline)."""
    meas = json.loads((Path(run_dir) / "results" / "per_seed.json").read_text())[
        "coalition_advantage"
    ]
    shared = sorted(set(meas[treated]) & set(meas["baseline"]))
    return {int(s): meas[treated][s] - meas["baseline"][s] for s in shared}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "collusion-control",
        "collusion-treatment",
        "single-control",
        "single-treatment",
    ):
        p.add_argument(f"--{name}", required=True)
    args = p.parse_args()

    result = transfer_ratio(
        multi_treatment=_delta(args.collusion_treatment, "collusion"),
        multi_control=_delta(args.collusion_control, "collusion"),
        single_treatment=_delta(args.single_treatment, "single"),
        single_control=_delta(args.single_control, "single"),
    )

    print("=== transfer ratio: multi-agent effect / single-agent effect ===")
    print(
        f"  multi-agent effect  (control - treatment) : {result['multi_agent_effect']:+.3f}"
    )
    print(
        f"  single-agent effect (control - treatment) : {result['single_agent_effect']:+.3f}"
    )
    lo, hi = result["single_agent_effect_ci95"]
    print(f"  single-agent 95% CI                       : [{lo:+.3f}, {hi:+.3f}]")
    print(
        f"  seeds / resamples                         : {result['n_seeds']} / {result['n_boot']}"
    )
    print()
    if result["interpretable"]:
        r_lo, r_hi = result["ratio_ci95"] or (float("nan"), float("nan"))
        print(f"  RATIO = {result['ratio']}  95% CI [{r_lo}, {r_hi}]")
    else:
        print("  NOT INTERPRETABLE")
        print(f"  {result['refused_because']}")


if __name__ == "__main__":
    main()
