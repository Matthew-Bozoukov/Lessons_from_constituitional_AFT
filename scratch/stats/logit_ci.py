# ABOUTME: Delta-method interval on the log-odds scale, for rates near 0 or 1 where the
# ABOUTME: symmetric interval is the wrong shape. Candidate fix for src/eval/stats.interval.
"""A boundary-aware interval built from OUR standard error, not a binomial's.

The symmetric interval `mu_hat +/- t*SE` assumes the sampling distribution of mu_hat is
symmetric. Near 0 it is not: the mean cannot go below the floor but can go well above, so
the distribution is right-skewed (measured skew +0.72 at mu ~ 0.045 against +0.14 at 0.41),
and a symmetric interval sits too far left and covers ~90% instead of 95%.

Two ways to fix the SHAPE without touching the estimand or the SE:

  logit   do the +/- t*SE step on log-odds, where the boundary is at infinity, then map back.
          Delta method: Var(logit(p)) = Var(p) / (p(1-p))^2. Derived from our own SE, assumes
          nothing about the data-generating process, and produces an asymmetric interval that
          cannot leave (0, 1).

  Wilson  invert the binomial score test at an effective sample size n_eff = p(1-p)/SE^2.
          Also boundary-aware, but it borrows the shape of a BINOMIAL, which mu_hat is not --
          it is a hierarchical mean. Kept only for mu_hat exactly 0 or 1, where the logit
          transform is undefined.

Neither changes what is being estimated. They change the geometry of the interval around it.
"""

from __future__ import annotations

import math
from typing import Any

from src.eval.stats import Result, wilson


def logit_interval(r: Result, scale: float = 1.0) -> dict[str, Any]:
    """Re-derive `r`'s interval on the log-odds scale.

    Args:
        r: A Result whose `mean` is a rate on [0, scale].
        scale: 1.0 when values are proportions, 100.0 when they are percentages.

    Returns:
        `lo`, `hi` (same scale as `r.mean`) and `method`. Falls back to Wilson at an
        effective sample size when the point estimate sits exactly on a boundary, where
        the log-odds is undefined.
    """
    p, se = r.mean / scale, r.se / scale
    if not 0.0 < p < 1.0 or se <= 0.0:
        # Exactly 0 or 1 (or a degenerate SE): no log-odds. Use the binomial score interval
        # at the sample size that would have produced our SE, so the clustering is respected.
        n_eff = (p * (1 - p) / se ** 2) if se > 0 and 0 < p < 1 else _n_eff_at_boundary(r, scale)
        lo, hi = wilson(p * n_eff, max(int(round(n_eff)), 1), z=r.mult)
        return {"lo": scale * lo, "hi": scale * hi,
                "method": f"Wilson at n_eff={n_eff:.1f} (boundary; log-odds undefined)"}
    half = r.mult * se / (p * (1 - p))          # delta method on logit
    l = math.log(p / (1 - p))
    lo, hi = 1 / (1 + math.exp(-(l - half))), 1 / (1 + math.exp(-(l + half)))
    return {"lo": scale * lo, "hi": scale * hi,
            "method": f"logit delta method, {r.method}"}


def _n_eff_at_boundary(r: Result, scale: float) -> float:
    """Effective n when p is 0 or 1: fall back to the item count, which is the honest floor."""
    return float(max(r.n_items, 1))
