# ABOUTME: Reusable ODCV comparison statistics: scenario-level paired bootstrap,
# ABOUTME: exact McNemar on violation flips, and published-score agreement.
"""Statistics shared by the ODCV comparison and report pipelines.

Extracted verbatim from odcv_compare.py (`_paired_bootstrap`, the inline McNemar
exact test) and the odcv report generator (`agreement`) so the numeric cores are
importable and unit-tested independently of any report rendering.
"""

from __future__ import annotations

from math import comb

import numpy as np

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD


def paired_bootstrap(pairs: list[tuple[float, float]], n_boot: int, seed: int) -> dict:
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


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs under p=0.5.

    Args:
        b: Scenario cells where only the treatment arm flags a violation.
        c: Scenario cells where only the control arm flags a violation.

    Returns:
        The two-sided exact p-value (1.0 when there are no discordant pairs).
    """
    n_disc = b + c
    if not n_disc:
        return 1.0
    return min(1.0, 2 * sum(comb(n_disc, i) for i in range(0, min(b, c) + 1)) / 2**n_disc)


def agreement(ours: dict, pub: dict) -> dict:
    """Compare our per-scenario violation calls against the published ones.

    Args:
        ours: {variant: {scenario: median}} from this run.
        pub: {variant: {scenario: median}} from the paper.

    Returns:
        Confusion counts, agreement rate, and the disagreeing scenarios.
    """
    both, neither, only_ours, only_pub, disagreements = 0, 0, 0, 0, []
    for variant in VARIANTS:
        for scenario, our_score in sorted(ours[variant].items()):
            if scenario not in pub[variant]:
                continue
            pub_score = pub[variant][scenario]
            o, p = our_score >= VIOLATION_THRESHOLD, pub_score >= VIOLATION_THRESHOLD
            if o and p:
                both += 1
            elif not o and not p:
                neither += 1
            else:
                (only_ours, only_pub) = (only_ours + 1, only_pub) if o else (only_ours, only_pub + 1)
                disagreements.append(
                    {"variant": variant, "scenario": scenario,
                     "ours": our_score, "published": pub_score}
                )
    n = both + neither + only_ours + only_pub
    return {
        "n_compared": n,
        "both_violation": both,
        "neither_violation": neither,
        "only_ours": only_ours,
        "only_published": only_pub,
        "agreement_pct": round(100 * (both + neither) / n, 1) if n else 0.0,
        "disagreements": disagreements,
    }
