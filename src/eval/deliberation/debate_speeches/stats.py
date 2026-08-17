# ABOUTME: Rank and agreement statistics for judge-vs-human ratings — Pearson, Spearman,
# ABOUTME: Kendall tau-b, quadratic weighted kappa — implemented here because scipy is not a dep.

"""Agreement statistics for the `debate_speeches` eval.

Written out rather than pulled from scipy (not a dependency of this repo, and adding a
compiled dependency to the darwin driver for four formulas is a bad trade). Every function
is pure and unit-tested offline against hand-computed values.

**Why more than one statistic.** A judge can fail three different ways and each statistic
sees only some of them:

- Pearson sees linear agreement but is dominated by a few extreme speeches.
- Spearman and Kendall see the *ranking* — the thing you actually want from a judge — and
  ignore whether it uses the same part of the scale as the humans.
- Quadratic weighted kappa sees calibration: a judge that ranks perfectly but rates
  everything 4 or 5 is penalised, where the rank statistics are not.

Report all of them. A model whose Spearman is fine and whose QWK is near zero has learned
an ordering and lost the scale, and that is a different finding from having learned nothing.
"""

from __future__ import annotations

import math
import statistics


def _ranks(values: list[float]) -> list[float]:
    """Fractional (average) ranks, so ties do not distort Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 when either series is constant (undefined, not an error —
    a judge that gives every speech the same score is a result worth reporting, not a crash)."""
    assert len(xs) == len(ys), "series must be the same length"
    if len(xs) < 2:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / den, 4) if den else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (Pearson on fractional ranks)."""
    if len(xs) < 2:
        return 0.0
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    """Kendall tau-b — the rank statistic *Debatable Intelligence* reports for this task.

    tau-b rather than tau-a because both series are heavily tied: the model emits integers
    1-5 while the human mean is near-continuous, and tau-a would charge those ties against
    the judge as if they were disagreements.

    O(n^2). At the dataset's 948 items that is ~450k pairs, which costs milliseconds and
    keeps the implementation obviously correct.
    """
    assert len(xs) == len(ys), "series must be the same length"
    n = len(xs)
    if n < 2:
        return 0.0
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            # A pair tied in BOTH series counts toward both tie totals — that is what makes
            # `tied_x` the standard n1 and `tied_y` the standard n2.
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            if dx and dy:
                concordant += dx * dy > 0
                discordant += dx * dy < 0

    # tau_b = (C - D) / sqrt((n0 - n1)(n0 - n2)), where n0 is ALL pairs and n1/n2 are the
    # tied ones. The ties are SUBTRACTED from the pair count, not added: an earlier version
    # added them, which understates agreement whenever the series is tie-heavy — and the
    # model's series always is, being integers 1-5 over hundreds of speeches. A monotone
    # relabelling scored 0.86 instead of 1.0 before this was fixed.
    total = n * (n - 1) // 2
    den = math.sqrt((total - tied_x) * (total - tied_y))
    return round((concordant - discordant) / den, 4) if den else 0.0


def quadratic_weighted_kappa(a: list[int], b: list[int], low: int = 1,
                             high: int = 5) -> float:
    """Cohen's kappa with quadratic weights over the integer scale `low..high`.

    Both series must already be integers on that scale — the caller rounds the human mean,
    deliberately and visibly, rather than having this function do it silently.
    """
    assert len(a) == len(b), "series must be the same length"
    size = high - low + 1
    n = len(a)
    if n == 0:
        return 0.0
    observed = [[0] * size for _ in range(size)]
    for x, y in zip(a, b):
        assert low <= x <= high and low <= y <= high, \
            f"ratings {x},{y} outside [{low},{high}]"
        observed[x - low][y - low] += 1
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(size)) for j in range(size)]

    num = den = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / ((size - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / n
            num += weight * observed[i][j]
            den += weight * expected
    return round(1 - num / den, 4) if den else 0.0
