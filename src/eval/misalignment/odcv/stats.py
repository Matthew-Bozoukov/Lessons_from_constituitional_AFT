# ABOUTME: ODCV comparison statistics on top of src/eval/stats.py: the paired arm difference
# ABOUTME: (MR and severity), exact McNemar on violation flips, and published-score agreement.
"""Statistics shared by the ODCV comparison and report pipelines.

`arm_difference` is the treatment-minus-control interval, paired on scenario (both arms ran
the same stories) with the 50/50 variant mixture as the estimand; it replaces the scenario
bootstrap this module used to carry (the difference of two means has a closed-form SE --
see src/eval/stats.py). `mcnemar_exact` is re-exported from there; `agreement` compares our
medians against the paper's row.
"""

from __future__ import annotations

import math

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD, _design_for, _rollouts
from src.eval.stats import Result, difference, mcnemar_exact, t_cdf

__all__ = ["arm_difference", "mcnemar_exact", "agreement"]


def _long(cells: dict[str, list | float], checkpoint: str) -> list[dict]:
    """{"variant/scenario": [median severity per rollout] | median} -> one row per rollout."""
    rows = []
    for key, value in cells.items():
        variant, scenario = key.split("/", 1)
        for k, sev in enumerate(_rollouts(value)):
            rows.append({"checkpoint": checkpoint, "scenario": scenario, "variant": variant, "pass": k,
                         "severity": float(sev), "violation": float(sev >= VIOLATION_THRESHOLD)})
    return rows


def _p_two_sided(r: Result) -> float:
    """z/t-test p-value from an interval result (0 when the SE is zero and the gap is not)."""
    if r.se == 0:
        return 1.0 if r.mean == 0 else 0.0
    return 2.0 * (1.0 - t_cdf(abs(r.mean) / r.se, r.df))


def arm_difference(treatment: dict[str, float], control: dict[str, float]) -> dict:
    """Treatment minus control on the cells both arms scored, paired on scenario.

    Args:
        treatment: {"variant/scenario": median severity} for the arm under test.
        control: The same for the control arm.

    Returns:
        MR difference (pp) and severity difference with 95% intervals and two-sided
        p-values, plus the full `src.eval.stats` results under "stats".
    """
    shared = sorted(set(treatment) & set(control))
    assert len(shared) >= 2, "arms share fewer than two scenario cells"
    # Both arms must carry the same variants, or one side's mixture is not the other's:
    # an incentivized-only arm is compared on the control's incentivized cells alone.
    variants = sorted({k.split("/", 1)[0] for k in shared})
    t_rows = _long({k: treatment[k] for k in shared}, "treatment")
    c_rows = _long({k: control[k] for k in shared}, "control")
    design = _design_for(variants)
    mr = difference([dict(r, value=100.0 * r["violation"]) for r in t_rows],
                    [dict(r, value=100.0 * r["violation"]) for r in c_rows], design)
    sev = difference([dict(r, value=r["severity"]) for r in t_rows],
                     [dict(r, value=r["severity"]) for r in c_rows], design)
    return {
        "mr_diff_pp": round(mr.mean, 1),
        "mr_diff_ci95": [round(mr.lo, 1), round(mr.hi, 1)],
        "mr_diff_p_two_sided": round(_p_two_sided(mr), 4),
        "sev_diff": round(sev.mean, 2),
        "sev_diff_ci95": [round(sev.lo, 2), round(sev.hi, 2)],
        "sev_diff_p_two_sided": round(_p_two_sided(sev), 4),
        "n_scenarios": mr.n_items,
        "method": mr.method,
        "stats": {"mr": mr.as_dict(), "severity": sev.as_dict()},
    }


def agreement(ours: dict, pub: dict) -> dict:
    """Per-scenario violation-flag agreement between our medians and the published row.

    Args:
        ours: {variant: {scenario: median}} from this run.
        pub: {variant: {scenario: median}} from the paper's CSV.

    Returns:
        Confusion counts, agreement percentage, and the list of disagreements.
    """
    both = neither = only_ours = only_pub = 0
    disagreements = []
    for variant in VARIANTS:
        for scenario, mine in ours.get(variant, {}).items():
            if scenario not in pub.get(variant, {}):
                continue
            theirs = pub[variant][scenario]
            a, b = mine >= VIOLATION_THRESHOLD, theirs >= VIOLATION_THRESHOLD
            if a and b:
                both += 1
            elif not a and not b:
                neither += 1
            elif a:
                only_ours += 1
                disagreements.append({"variant": variant, "scenario": scenario, "ours": mine, "published": theirs})
            else:
                only_pub += 1
                disagreements.append({"variant": variant, "scenario": scenario, "ours": mine, "published": theirs})
    n = both + neither + only_ours + only_pub
    return {
        "n_compared": n,
        "both_violation": both,
        "neither_violation": neither,
        "only_ours": only_ours,
        "only_published": only_pub,
        "agreement_pct": round(100.0 * (both + neither) / n, 1) if n else math.nan,
        "disagreements": disagreements,
    }
