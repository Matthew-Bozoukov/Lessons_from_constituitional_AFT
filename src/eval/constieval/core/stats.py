# ABOUTME: Confidence intervals and judge-vs-human agreement. Every aggregate the
# ABOUTME: suite reports carries an interval, so no plot ever shows a bare point estimate.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .hashing import stream_rng

# 97.5th percentile of the standard normal, i.e. a two-sided 95% interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        mean: Point estimate.
        lo: Lower bound.
        hi: Upper bound.
        n: Number of observations behind the estimate.
    """

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def err(self) -> tuple[float, float]:
        """Return (lower, upper) distances from the mean, for matplotlib yerr."""
        return (max(0.0, self.mean - self.lo), max(0.0, self.hi - self.mean))

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable dict."""
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi, "n": self.n}


def wilson(successes: float, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a proportion.

    Wilson rather than normal-approximation because most clause-level cells are
    small (tens of items) and often near 0 or 1, exactly where the normal
    approximation produces bounds outside [0, 1].

    Args:
        successes: Number of successes (may be fractional for graded rubrics).
        n: Number of trials.
        z: Normal quantile; defaults to 95%.

    Returns:
        The interval. An empty cell returns mean 0 and bounds [0, 1] — unknown,
        not zero.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0, 0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin), n)


def bootstrap_mean(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0, z: float = Z_95
) -> Interval:
    """Percentile bootstrap CI for the mean of continuous scores.

    Used for graded (non-binary) axes, where Wilson does not apply. The RNG is
    seeded from the shared stream helper so a report is byte-reproducible.

    Args:
        values: Observations.
        n_boot: Bootstrap resamples.
        seed: RNG seed.
        z: Unused; kept so callers can swap wilson/bootstrap without a signature change.

    Returns:
        The interval; a single observation yields a degenerate interval at its value.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return Interval(0.0, 0.0, 1.0, 0)
    mean = sum(vals) / n
    if n == 1:
        return Interval(mean, mean, mean, 1)
    rng = stream_rng(seed, n, "bootstrap")
    means = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return Interval(mean, lo, hi, n)


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa between two label sequences.

    Args:
        a: First rater's labels.
        b: Second rater's labels, aligned with `a`.

    Returns:
        Kappa in [-1, 1]. Returns 1.0 when both raters are constant and agree
        (chance agreement is 1, so the usual formula is 0/0).

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"Rater sequences differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("Cannot compute kappa over zero items")

    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum(
        (sum(1 for x in a if x == lab) / n) * (sum(1 for y in b if y == lab) / n)
        for lab in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """Return raw agreement, Cohen's kappa, and n for two label sequences.

    Args:
        a: First rater's labels.
        b: Second rater's labels.

    Returns:
        Mapping with keys "raw", "kappa", "n".
    """
    n = len(a)
    raw = sum(1 for x, y in zip(a, b) if x == y) / n if n else 0.0
    return {"raw": raw, "kappa": cohens_kappa(a, b), "n": float(n)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for a paired binary comparison.

    Only DISCORDANT pairs carry information: pairs where both conditions agree tell you nothing
    about the difference. v1's real constraint was discordance, not n - 662 rows collapsed to 12
    discordant pairs in the best cell and 0-3 in most, which no amount of extra rows would fix if
    the effect is small.

    Args:
        b: Pairs where the first condition passed and the second failed.
        c: Pairs where the second passed and the first failed.

    Returns:
        Two-sided p-value from the exact binomial test on b vs b+c. Returns 1.0 when there are no
        discordant pairs (nothing to distinguish).
    """
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    # Two-sided: sum the probabilities of every outcome at least as extreme as min(b, c).
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[Any],
    n_boot: int = 4000,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for a mean, resampling CLUSTERS rather than rows.

    Items are not independent: a dozen scenarios written for one clause share that clause's
    difficulty and phrasing. Resampling rows treats them as independent and understates the
    interval - the effective sample size is closer to the number of clauses than the number of
    rows. v1's naive intervals were inflated by roughly sqrt(4) this way.

    Args:
        values: Per-row observations.
        clusters: Cluster label per row (the clause id).
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        The interval over the mean.

    Raises:
        ValueError: If values and clusters differ in length.
    """
    if len(values) != len(clusters):
        raise ValueError(f"values/clusters length mismatch: {len(values)} vs {len(clusters)}")
    if not values:
        return Interval(0.0, 0.0, 1.0, 0)

    grouped: dict[Any, list[float]] = {}
    for v, c in zip(values, clusters):
        grouped.setdefault(c, []).append(float(v))
    keys = sorted(grouped, key=str)
    mean = sum(values) / len(values)
    if len(keys) < 2:
        return Interval(mean, mean, mean, len(values))

    rng = stream_rng(seed, len(keys), "cluster_bootstrap")
    means = []
    for _ in range(n_boot):
        drawn = [grouped[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [x for g in drawn for x in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return Interval(
        mean, means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))], len(values)
    )


def paired_delta(
    clean: Sequence[float], stressed: Sequence[float], seed: int = 0, n_boot: int = 2000
) -> Interval:
    """Bootstrap CI for the mean paired difference (stressed - clean).

    Pairing is the entire point of the robustness and OOD transforms: every
    derived item shares a parent, so the difference is taken per pair rather than
    between two independent group means. That removes item difficulty from the
    comparison and tightens the interval substantially.

    Args:
        clean: Scores on parent items.
        stressed: Scores on the derived items, aligned with `clean`.
        seed: RNG seed.
        n_boot: Bootstrap resamples.

    Returns:
        The interval over the mean difference.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(clean) != len(stressed):
        raise ValueError(f"Unpaired inputs: {len(clean)} clean vs {len(stressed)} stressed")
    return bootstrap_mean([s - c for c, s in zip(clean, stressed)], n_boot=n_boot, seed=seed)
