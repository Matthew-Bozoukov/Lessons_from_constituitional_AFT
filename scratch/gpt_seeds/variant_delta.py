# ABOUTME: incentivized - mandated delta per ODCV arm, PAIRED bootstrap over the scenarios
# ABOUTME: that survive both variants' exclusions; reuses plot_seed_mean.ARMS for the sources.
# Run: uv run python scratch/gpt_seeds/variant_delta.py
#
# The two ODCV variants are the SAME scenarios under two pressure framings, not disjoint
# sets: the 34-vs-31 cell imbalance comes from an asymmetric `exclude_scenarios` list (10
# incentivized-only, 5 mandated-only, 3 of those naming scenarios dropped from both), so a
# scenario is the unit and carries one mandated and one incentivized rate. Resampling the
# two variants INDEPENDENTLY leaves between-scenario difficulty variance in the interval --
# with that mistake an ~8 pp effect carried a ~±15 pp CI and no arm separated from zero.
# Here each bootstrap draw takes the same scenario index on both sides, so the shared
# difficulty cancels and the contrast is measured on the 28 scenarios both variants kept.
#
# Note `summarise()` in src/eval/misalignment/odcv/odcv.py does NOT pair -- it pools every
# scenario-variant row and calls bootstrap_mean_ci -- and its `bootstrap_ci(paired=...)`
# helper is currently unused by it. The pairing is therefore reconstructed here from the
# published per-scenario medians, using the module's own scenario_violation_rate so a
# scenario that violated in 1 of 2 rollouts counts as 0.5 exactly as it does upstream.

import importlib.util
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    _rollouts,
    scenario_violation_rate,
)

_spec = importlib.util.spec_from_file_location(
    "p", Path(__file__).with_name("plot_seed_mean.py")
)
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

EXCL = set(
    OmegaConf.to_container(
        OmegaConf.load(ROOT / p.CFG).get("exclude_scenarios", []) or []
    )
)
N_BOOT = 20_000
REFS = ("base", "table2")


def scenario_rates(psm: dict, variant: str) -> dict[str, list[float]]:
    """{scenario: [severity per rollout]} for one variant, exclusions already applied."""
    out: dict[str, list[float]] = {}
    for cell, value in psm[variant].items():
        out.setdefault(cell.split("/")[0], []).extend(_rollouts(value))
    return out


def arm_pairs(arm: dict) -> tuple[np.ndarray, list[float]]:
    """Pool an arm's seeds per scenario -> ((n_shared, 2) rates, per-seed deltas).

    Seeds are pooled as extra rollouts of the same scenario, which is right for the
    WITHIN-arm variant contrast (the question holds the arm fixed); it deliberately does
    not carry training-seed variance -- the per-seed deltas returned alongside do.
    """
    mand: dict[str, list[float]] = {}
    inc: dict[str, list[float]] = {}
    per_seed = []
    for _, src in sorted(arm["seeds"].items()):
        psm = p._restrict(p._load(src)["per_scenario_medians"], EXCL)
        m, i = scenario_rates(psm, "mandated"), scenario_rates(psm, "incentivized")
        per_seed.append(
            100 * np.mean([scenario_violation_rate(v) for v in i.values()])
            - 100 * np.mean([scenario_violation_rate(v) for v in m.values()])
        )
        for s, v in m.items():
            mand.setdefault(s, []).extend(v)
        for s, v in i.items():
            inc.setdefault(s, []).extend(v)
    shared = sorted(set(mand) & set(inc))
    arr = np.array(
        [
            [scenario_violation_rate(mand[s]), scenario_violation_rate(inc[s])]
            for s in shared
        ]
    )
    return arr, per_seed


def paired_delta(arr: np.ndarray, seed: int = 0) -> tuple:
    """Paired bootstrap on 100*(mean inc rate - mean mand rate); scenario is the unit."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    draws = arr[rng.integers(0, n, size=(N_BOOT, n))]  # (N_BOOT, n, 2)
    d = 100.0 * (draws[:, :, 1].mean(axis=1) - draws[:, :, 0].mean(axis=1))
    obs = 100.0 * (arr[:, 1].mean() - arr[:, 0].mean())
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean())


def unpaired_delta(arr: np.ndarray, seed: int = 0) -> tuple:
    """The mistake this script corrects, kept so the difference stays visible."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    d = 100.0 * (
        arr[rng.integers(0, n, size=(N_BOOT, n)), 1].mean(axis=1)
        - arr[rng.integers(0, n, size=(N_BOOT, n)), 0].mean(axis=1)
    )
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main() -> None:
    print(
        f"{'arm':16s} {'shared':>6s} {'delta':>7s} {'paired 95% CI':>18s} {'P<=0':>7s}   "
        f"{'unpaired CI (wrong)':>21s}   per-seed deltas"
    )
    print("-" * 120)
    obs_by_arm = {}
    for key, arm in p.ARMS.items():
        arr, per_seed = arm_pairs(arm)
        obs, lo, hi, p0 = paired_delta(arr)
        ulo, uhi = unpaired_delta(arr)
        obs_by_arm[key] = obs
        print(
            f"{key:16s} {arr.shape[0]:6d} {obs:+6.1f} {f'[{lo:+.1f}, {hi:+.1f}]':>18s} "
            f"{p0:7.3f} {f'[{ulo:+.1f}, {uhi:+.1f}]':>21s}   "
            + ", ".join(f"{d:+.1f}" for d in per_seed)
            + ("  *" if (lo > 0 or hi < 0) else "")
        )
    trained = [v for k, v in obs_by_arm.items() if k not in REFS]
    print(
        f"\nARM-level sign test: {sum(v > 0 for v in trained)}/{len(trained)} trained arms "
        f"positive, two-sided p = {2 * 0.5 ** len(trained):.4f} (the ARM is the unit -- "
        f"seeds within an arm share a recipe and are not independent draws). "
        f"Mean delta {np.mean(trained):+.1f} pp. References: "
        + ", ".join(f"{k} {obs_by_arm[k]:+.1f}" for k in REFS)
        + "."
    )
    print("  * = paired 95% CI excludes zero")


if __name__ == "__main__":
    main()
