# ABOUTME: Confidence intervals and judge-vs-human agreement. Every aggregate the
# ABOUTME: suite reports carries an interval, so no plot ever shows a bare point estimate.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

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
