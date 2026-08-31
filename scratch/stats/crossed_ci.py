# ABOUTME: Closed-form error bars for an n-models x J-scenarios table of ODCV outcomes under
# ABOUTME: different sampling assumptions (both random / scenarios fixed / model fixed / naive).
"""Error bars for a table V[i, j] of per-cell violation values (0/1, or a rate over rollouts).

Rows are trained models (independent draws from the pipeline), columns are scenarios
(independent draws from the scenario population). Everything is the two-way random-effects
algebra written up in the 2026-08-27 derivation note (scratchpad odcv_se.tex), which follows
Miller (arXiv:2411.00640) and adds the model axis. Nothing is bootstrapped.

    row means   Vbar_i.   column means   Vbar_.j   grand mean   mu_hat
    T_A = var(row means, ddof=1) / n          "clustered by model"
    T_B = var(col means, ddof=1) / J          "clustered by scenario"
    T_C = sum(double-centred residuals^2) / ((n-1)(J-1)) / (nJ)   interaction + rollout noise

    both random        SE^2 = T_A + T_B - T_C          +/-1.96 (Miller's normal approx.)
    scenarios fixed    SE^2 = T_A                       t, df n - 1     (Matthew's seed SEM)
    model i fixed      SE^2 = var(row i, ddof=1) / J    t, df J - 1     (Miller, one model)
    naive binomial     all nJ cells i.i.d. Bernoulli    Wilson          (lower bound: no structure)

Rollout noise never appears as a separate count: with one rollout per cell it sits inside
every spread above, so these intervals are the same whether or not a rollout is treated as
deterministic. What changes is only what the spread is *interpreted* as containing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Interval:
    mean: float
    se: float
    df: float
    mult: float
    lo: float
    hi: float
    method: str
    parts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["parts"] = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in self.parts.items()}
        return d


def _t(df: float) -> float:
    return float(stats.t.ppf(0.975, df))


def _interval(mean: float, se2: float, df: float, method: str, parts: dict) -> Interval:
    se = float(np.sqrt(max(se2, 0.0)))
    mult = _t(df)
    return Interval(float(mean), se, float(df), mult, float(mean - mult * se),
                    float(mean + mult * se), method, parts)


def crossed_terms(table) -> tuple[float, float, float, float, int, int]:
    """(mu_hat, T_A, T_B, T_C, n, J) for an n x J table."""
    tbl = np.asarray(table, dtype=float)
    assert tbl.ndim == 2 and tbl.shape[0] >= 2 and tbl.shape[1] >= 2, f"bad table {tbl.shape}"
    n, J = tbl.shape
    mu = tbl.mean()
    row, col = tbl.mean(axis=1), tbl.mean(axis=0)
    t_a = row.var(ddof=1) / n
    t_b = col.var(ddof=1) / J
    resid = tbl - row[:, None] - col[None, :] + mu
    t_c = (resid ** 2).sum() / ((n - 1) * (J - 1)) / (n * J)
    return float(mu), float(t_a), float(t_b), float(t_c), n, J


def both_random(table) -> Interval:
    """Models AND scenarios sampled: SE^2 = T_A + T_B - T_C, Satterthwaite df."""
    mu, t_a, t_b, t_c, n, J = crossed_terms(table)
    se2 = t_a + t_b - t_c
    fallback = bool(se2 <= 0)
    if fallback:  # small-sample fluke: interaction estimate exceeds both main terms
        se2 = max(t_a, t_b)
    # Composite SE: no exact df. Miller's normal approximation, +/- 1.96 (df = inf).
    return _interval(mu, se2, float("inf"), "both random: T_A + T_B - T_C, +/-1.96",
                     dict(T_A=t_a, T_B=t_b, T_C=t_c, n=n, J=J, negative_fallback=fallback))


def scenarios_fixed(table) -> Interval:
    """Scenarios are the fixed benchmark, models sampled: SE^2 = T_A, df = n - 1."""
    mu, t_a, _, _, n, J = crossed_terms(table)
    return _interval(mu, t_a, n - 1, "scenarios fixed: T_A, t_{n-1}", dict(T_A=t_a, n=n, J=J))


def model_fixed(row) -> Interval:
    """One model held fixed, scenarios sampled: SE^2 = s^2 / J, df = J - 1."""
    r = np.asarray(row, dtype=float)
    assert r.ndim == 1 and r.size >= 2
    J = r.size
    return _interval(r.mean(), r.var(ddof=1) / J, J - 1,
                     "model fixed, scenarios random: s^2/J, t_{J-1}", dict(J=J))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    assert n > 0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def naive_binomial(table) -> Interval:
    """All nJ cells as i.i.d. Bernoulli draws (no model or scenario structure): Wilson.

    This is the smallest interval any attribution of the spread can give: it is what the
    error bar would be if every bit of cell-to-cell variation were rollout luck.
    """
    tbl = np.asarray(table, dtype=float)
    N = tbl.size
    k = float(tbl.sum())
    lo, hi = wilson(k, N)
    p = k / N
    return Interval(p, float(np.sqrt(p * (1 - p) / N)), float("inf"), 1.96, lo, hi,
                    "naive binomial: nJ i.i.d. rollouts, Wilson", dict(k=k, N=N))


def both_random_diff(table_a, table_b) -> Interval:
    """Arm A minus arm B on the SAME scenarios (columns aligned), different models.

    Model terms add (unpaired); the scenario term is computed from per-scenario differences
    of column means (paired); the residual terms subtract as in the one-arm case.
    """
    mu_a, ta_a, _, tc_a, n_a, J = crossed_terms(table_a)
    mu_b, ta_b, _, tc_b, n_b, J2 = crossed_terms(table_b)
    assert J == J2, "arms must share the scenario set"
    d = np.asarray(table_a, float).mean(axis=0) - np.asarray(table_b, float).mean(axis=0)
    tb_d = d.var(ddof=1) / J
    se2 = ta_a + ta_b + tb_d - tc_a - tc_b
    fallback = bool(se2 <= 0)
    if fallback:
        se2 = ta_a + ta_b + tb_d
    return _interval(mu_a - mu_b, se2, float("inf"),
                     "diff, both random: T_A^A + T_A^B + T_B^(d) - T_C^A - T_C^B, +/-1.96",
                     dict(T_A_a=ta_a, T_A_b=ta_b, T_B_d=tb_d, T_C_a=tc_a, T_C_b=tc_b,
                          negative_fallback=fallback))


def scenarios_fixed_diff(table_a, table_b) -> Interval:
    """Arm A minus arm B with scenarios fixed: Welch t on the per-model rates."""
    ra = np.asarray(table_a, float).mean(axis=1)
    rb = np.asarray(table_b, float).mean(axis=1)
    va, vb = ra.var(ddof=1) / ra.size, rb.var(ddof=1) / rb.size
    se2 = va + vb
    df = se2 ** 2 / (va ** 2 / (ra.size - 1) + vb ** 2 / (rb.size - 1))
    return _interval(ra.mean() - rb.mean(), se2, df, "diff, scenarios fixed: Welch t on per-model rates",
                     dict(var_a=va, var_b=vb))
