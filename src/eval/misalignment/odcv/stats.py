# ABOUTME: ODCV comparison helpers: exact McNemar on violation flips (re-exported from
# ABOUTME: src/eval/stats.py) and per-scenario agreement with the paper's published medians.
"""Statistics shared by the ODCV comparison and report pipelines.

`mcnemar_exact` is re-exported from `src.eval.stats`; `agreement` compares our medians
against the paper's row. Intervals live in `odcv.summarise` (a t over the run's passes,
docs/error_bars.md); there is no cross-arm interval here.
"""

from __future__ import annotations

import math

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD
from src.eval.stats import mcnemar_exact

__all__ = ["mcnemar_exact", "agreement"]


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
