# ABOUTME: Statistics shared across capability evals (CLAUDE.md: cross-eval shared code lives
# ABOUTME: at the subarea root, not inside one eval's package).

"""Interval estimators used by more than one capability eval.

`wilson_ci` moved here from `mmlu/mmlu.py` when the SWE-bench baseline needed it too: both
report a binomial proportion (MMLU accuracy, SWE-bench pass@1) and a second copy of an
interval estimator is exactly the kind of duplication that drifts. mmlu.py re-imports the
name, so its existing callers are unaffected.
"""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> dict[str, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation: at n≈500 and accuracy near 0.8 the two
    barely differ, but Wilson stays inside [0, 1] and keeps behaving at the small n of
    a single-subject breakdown, where the normal interval produces bounds above 1.

    Args:
        k: Successes.
        n: Trials.
        alpha: Two-sided level.

    Returns:
        `mean`, `ci_lower`, `ci_upper`, `n`.
    """
    if n <= 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}
    z = _z_for(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {
        "mean": p,
        "ci_lower": max(0.0, centre - half),
        "ci_upper": min(1.0, centre + half),
        "n": n,
    }


def _z_for(alpha: float) -> float:
    """Two-sided normal critical value, via the inverse error function."""
    # math.erfinv does not exist; use the standard relation through erf's inverse by
    # bisection on erf, which is exact enough for interval endpoints and avoids a scipy
    # dependency the rest of the repo does not carry.
    target = 1.0 - alpha
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
