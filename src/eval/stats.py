# ABOUTME: The intervals every misalignment eval reports: a t-interval over repeated passes of
# ABOUTME: the WHOLE eval (log-odds scale for a bounded score), Wilson for independent draws, McNemar.

"""Error bars for eval results (docs/error_bars.md).

One run of an eval is one number: ODCV's misalignment rate over its 80 cells, MASK's honesty
score over its rows, the agentic-misalignment rate over its 18 conditions. The cells inside a
run are not independent draws of anything (the mandated and incentivized variants of one
scenario co-move; the 18 conditions share three scenarios), so nothing here puts a cell
count in a denominator. The unit that IS replicated is the run: `passes: 3` repeats the whole
eval three times, and those three numbers are the sample the interval is built from.

    one pass    -> no interval (nothing to measure the spread with)
    n passes    -> mean, and a t-interval on n-1 degrees of freedom over the n numbers

A bounded score (a rate in [0, 100], a 0-5 severity) gets the t-step on the log-odds scale
(`logit_t`) so the interval is asymmetric and cannot leave the range: a 5% rate with three
passes should not report "-3% to 13%". An unbounded value gets the plain step (`t_mean`).
Both are estimates of the same thing -- how far the eval's number would move on a fresh pass
of the same model -- and neither says anything about a fresh training seed; that question
needs several seeds, each with its own passes, and is answered over seed means, not here.

`wilson` is for the one shape where the draws really are independent repeats of one thing:
Colosseum's episodes, a corpus check's rows. `mcnemar_exact` is the paired flip test the
whistleblowing measures use.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

__all__ = ["logit_t", "t_mean", "wilson", "mcnemar_exact", "t_cdf", "t_quantile", "Z_95"]

Z_95 = NormalDist().inv_cdf(0.975)


def _finish(values: Sequence[float], mean: float, lo: float | None, hi: float | None,
            method: str) -> dict:
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values), "values": [float(v) for v in values],
            "method": method}


def t_mean(values: Sequence[float], level: float = 0.95) -> dict:
    """Mean of repeated passes and its t-interval over them (df n-1); no interval for one pass.

    Args:
        values: One number per pass of the whole eval.
        level: Two-sided coverage.

    Returns:
        `mean`, `lo`, `hi` (None when n < 2), `n`, `values`, `method`.
    """
    assert values, "no passes"
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return _finish(values, mean, None, None, "none: one pass (run passes >= 2 for an interval)")
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = t_quantile(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
    return _finish(values, mean, mean - half, mean + half,
                   f"t over {n} passes (df {n - 1})")


def logit_t(values: Sequence[float], n_items: int, scale: float = 100.0,
            level: float = 0.95) -> dict:
    """Mean of a bounded per-pass score and its t-interval taken on the log-odds scale.

    Each pass's value is a score in [0, scale] over `n_items` items (a rate over 80 cells, a
    0-5 mean over them). The interval: q = value/scale, z = ln((q + c) / (1 - q + c)) with the
    empirical-logit correction c = 0.5 / n_items (so 0% and 100% have a finite z), the mean
    and sd of the z's, half-width t_{n-1} * sd / sqrt(n), and the ends mapped back through the
    exact inverse of that transform (q = (e^z (1 + c) - c) / (1 + e^z), clamped to the range)
    so three identical passes give a zero-width interval at their own value. The POINT
    estimate is the plain mean of the values, so a one-pass run reports exactly its own
    number; with passes near the floor the interval sits around the mean of the log-odds,
    which is below the plain mean, so the two are reported side by side rather than forced
    to agree.

    Args:
        values: One score per pass, each in [0, scale].
        n_items: Items each pass's score is over; only sets the continuity correction.
        scale: The top of the range (100 for a percentage, 5 for a 0-5 severity, 1 for a fraction).
        level: Two-sided coverage.

    Returns:
        `mean`, `lo`, `hi` (None when n < 2), `n`, `values`, `method`.
    """
    assert values, "no passes"
    assert n_items > 0, "n_items must be positive"
    assert all(0.0 <= v <= scale for v in values), f"values must lie in [0, {scale}]: {values}"
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return _finish(values, mean, None, None, "none: one pass (run passes >= 2 for an interval)")
    c = 0.5 / n_items
    z = [math.log((v / scale + c) / (1.0 - v / scale + c)) for v in values]
    z_mean = sum(z) / n
    sd = math.sqrt(sum((x - z_mean) ** 2 for x in z) / (n - 1))
    half = t_quantile(0.5 + level / 2, n - 1) * sd / math.sqrt(n)

    def back(x: float) -> float:
        e = math.exp(x)
        return scale * min(1.0, max(0.0, (e * (1.0 + c) - c) / (1.0 + e)))

    return _finish(values, mean, back(z_mean - half), back(z_mean + half),
                   f"t over {n} passes (df {n - 1}) on the log-odds scale")


def wilson(k: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n of INDEPENDENT draws.

    Wilson rather than the naive `p +/- z*sqrt(p(1-p)/n)`: it stays inside [0, 1] and keeps a
    sensible width at the edges, where the naive interval has width zero at 0/n. Bounds are
    clamped: rounding can otherwise put them a floating-point hair outside [0, 1].
    """
    assert n > 0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------- t quantile (no scipy)

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t (df may be inf -> normal)."""
    if math.isinf(df):
        return NormalDist().cdf(t)
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t (df may be inf -> normal). Bisection on the CDF; no scipy."""
    assert 0 < p < 1
    if math.isinf(df):
        return NormalDist().inv_cdf(p)
    assert df > 0
    if p < 0.5:
        return -t_quantile(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)
