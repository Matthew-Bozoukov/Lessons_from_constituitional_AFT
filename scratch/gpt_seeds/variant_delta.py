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

import fire
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    group_rollouts,
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
    """{scenario: [severity per rollout]} for one variant, exclusions already applied.

    `group_rollouts` (src/) handles all three published layouts, so this file no longer
    carries its own copy of the normalisation.
    """
    return group_rollouts(psm)[variant]


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


def _pm(row: dict, field: str) -> str:
    """`v% [lo, hi]` for a single run, `v% ±h` for a seed mean -- the two are not the
    same kind of interval (eval-sampling noise vs training-seed variance)."""
    v, lo, hi = row[field]
    return (
        f"{v:.1f}% ±{(hi - v):.1f}"
        if row["kind"] == "seed_mean"
        else f"{v:.1f}% [{lo:.1f}, {hi:.1f}]"
    )


def main(out: str = "") -> None:
    """Print the full arm table; `--out <path.md>` also writes it as a markdown mirror."""
    rows = p._collect({})  # the figure's own 65-cell stats, so columns cannot drift
    lines = [
        "| arm | overall MR | mandated | incentivized | delta (inc − mand) | paired 95% CI | P(≤0) |",
        "|---|---|---|---|---|---|---|",
    ]
    plain, obs_by_arm, shared_n = [], {}, {}
    for row in rows:
        key = row["key"]
        arr, per_seed = arm_pairs(p.ARMS[key])
        obs, lo, hi, p0 = paired_delta(arr)
        obs_by_arm[key], shared_n[key] = obs, arr.shape[0]
        sig = "**" if (lo > 0 or hi < 0) else ""
        mr = (
            f"{row['mr']:.1f}% ±{(row['hi'] - row['mr']):.1f}"
            if row["kind"] == "seed_mean"
            else f"{row['mr']:.1f}% [{row['lo']:.1f}, {row['hi']:.1f}]"
        )
        label = row["short"].replace("\n", " ") + (
            f" ({row['sem']['mr']['k']} seeds)" if row["kind"] == "seed_mean" else ""
        )
        lines.append(
            f"| {label} | {mr} | {_pm(row, 'mand')} | {_pm(row, 'inc')} | "
            f"{sig}{obs:+.1f}{sig} | [{lo:+.1f}, {hi:+.1f}] | {p0:.3f} |"
        )
        ulo, uhi = unpaired_delta(
            arr
        )  # kept visible: the pairing is worth ~10 pp of CI
        plain.append(
            f"{key:16s} {row['mr']:5.1f}%  mand {row['mand'][0]:5.1f}%  "
            f"inc {row['inc'][0]:5.1f}%  delta {obs:+6.1f} "
            f"paired [{lo:+.1f}, {hi:+.1f}]  unpaired [{ulo:+.1f}, {uhi:+.1f}]  "
            f"P<={p0:.3f}  n_shared={arr.shape[0]}  "
            "seeds " + ", ".join(f"{d:+.1f}" for d in per_seed)
        )
    trained = [v for k, v in obs_by_arm.items() if k not in REFS]
    notes = [
        "",
        f"Overall / mandated / incentivized come from the figure: same exclusion list "
        f"throughout, and a seeded arm is scored on the cells ALL its seeds kept (57-65 of "
        f"the 65). `±` is a seed mean ±1.96·SEM (training-seed variance); `[lo, hi]` is a "
        f"single run's cell-level bootstrap 95% CI (eval noise).",
        "",
        f"**The delta is NOT the difference of the two columns beside it.** It is paired "
        f"by scenario over the {min(shared_n.values())}–{max(shared_n.values())} scenarios "
        f"that survive BOTH variants' exclusions, because the mandated and incentivized "
        f"cell sets are differently composed (the 34-vs-31 imbalance is an asymmetric "
        f"`exclude_scenarios` list, not disjoint scenarios). Pairing cancels between-"
        f"scenario difficulty; without it no arm separates from zero.",
        "",
        f"Bold delta = paired 95% CI excludes zero. ARM-level sign test: "
        f"{sum(v > 0 for v in trained)}/{len(trained)} trained arms positive, two-sided "
        f"p = {2 * 0.5 ** len(trained):.4f} (the ARM is the unit — seeds within an arm "
        f"share a recipe and are not independent draws). Mean trained delta "
        f"{np.mean(trained):+.1f} pp; references "
        + ", ".join(f"{k} {obs_by_arm[k]:+.1f}" for k in REFS)
        + ", neither separating from zero.",
    ]
    print("\n".join(plain))
    print("\n".join(lines + notes))
    if out:
        Path(out).write_text("\n".join(lines + notes) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
