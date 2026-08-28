# ABOUTME: Eval-agnostic error bars: closed-form intervals for a models x units table under a
# ABOUTME: declared Design, paired differences, and a cluster bootstrap for non-mean statistics.

"""Error bars for any eval that produces a table of per-cell outcomes.

The picture (derived in docs/error_bars.md; Miller, arXiv:2411.00640, plus a model axis):
every cell score is a true rate plus rollout noise; the true rate splits into a model level, a
unit level and their interaction; the four pieces are uncorrelated so variances add, each over
the number of independent draws of it:

    Var(mu_hat) = s_A^2/n + s_B^2/J + s_C^2/(nJ) + s_eps^2/(nJR)

None of the four is observable. Three spreads of the table are, and combine to it exactly:

    T_A  spread of the n row means / n          -> s_A^2/n + beta        (Miller's clustered SE, by model)
    T_B  spread of the J column means / J       -> s_B^2/J + beta        (Miller's clustered SE, by unit)
    T_C  spread of the double-centred residuals -> beta                  (interaction + rollout noise)
    Var(mu_hat) is estimated by T_A + T_B - T_C,  CI = mu_hat +/- t_nu SE.

The multiplier is a t quantile, never a flat 1.96: the variance is ESTIMATED, and a noisy
estimate needs a fatter multiplier to keep 95% coverage. df is how many independent numbers
went into it -- J-1 for a per-unit spread, n-1 for a per-model one. It matters at small
counts: at df 39, +/-1.96 covers 94.3%; at df 2 (three seeds) it covers 81%. Sums of
estimates with different dfs (T_A + T_B - T_C) have no exact df, so `satterthwaite` gives an
effective one -- roughly the df of whichever part dominates the sum.

A `Design` names the factors of the long table and what each one is:

    unit           the crossed-random draw from the benchmark population (scenario, prompt, subject)
    crossed_fixed  factors whose levels are ALL present in every unit with fixed weights (ODCV's
                   two variants at 1/2 each; Arena-Hard's two orderings). Enumerated, not sampled:
                   collapsed into the cell, contribute no variance term, only define the estimand.
    nested         draws inside a cell with no identity across cells (rollouts, questions within
                   a subject, judge calls). Averaged in; live only inside beta.
    units          "random" (generalise to the population) or "fixed" (this benchmark).

The model factor's kind is NOT declared: it is inferred at call time -- `models="random"` needs
n >= 2 checkpoints from one pipeline and adds T_A and T_C; n == 1 is `models="fixed"`, a claim
about that checkpoint. A config cannot assert a sample size it does not have.

Worked Designs. There is always exactly one `unit`; everything below it is `nested` and
everything enumerated beside it is `crossed_fixed`:

    ODCV, the 50/50 mixture      Design(unit="scenario", crossed_fixed={"variant": "equal"},
                                        nested=("pass",))
    ODCV, one variant            Design(unit="scenario", nested=("pass",))
    MMLU, Miller's framing       Design(unit="question")
        -- a question drawn from the MMLU-like question population; SE = sqrt(p(1-p)/n).
    MMLU, stratified by subject  Design(unit="subject", units="fixed",
                                        unit_weights="count", nested=("question",))
        -- MMLU's own subject mix, between-subject variance removed; narrower.
    Arena-Hard                   Design(unit="prompt", crossed_fixed={"ordering": "equal"},
                                        nested=("judge_call",))

The two MMLU rows are different claims, not different spellings of one: pick deliberately.

Rollouts. With R > 1 the within-cell spread estimates s_eps^2, so the rollout share of the error
bar is reported. With R == 1 the interval is still valid -- rollout noise sits inside every
spread and is measured with it -- but it cannot be separated, and the cell value has to be read
as the model's behaviour on that unit. The one thing R == 1 cannot support is the both-fixed
question ("these checkpoints on these units", where rollouts are the only randomness): that
raises `NotEstimable` instead of returning a zero-width bar.

Nothing here is bootstrapped except `cluster_bootstrap`, for statistics with no closed form
(Bradley-Terry ratings, medians). For a mean it agrees with `interval` to Monte Carlo error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054
__all__ = [
    "Design", "Table", "Result", "NotEstimable", "collapse", "crossed_terms", "satterthwaite",
    "interval", "difference", "cluster_bootstrap", "wilson", "mcnemar_exact", "t_quantile",
]


class NotEstimable(ValueError):
    """The requested interval cannot be computed from this data, and why."""


# --------------------------------------------------------------------------- design + table

@dataclass(frozen=True)
class Design:
    """What each column of the long table is. See the module docstring.

    Attributes:
        unit: Column holding the crossed-random unit (scenario, prompt, subject).
        units: "random" (a sample from the benchmark population) or "fixed" (the benchmark).
        crossed_fixed: `{column: "equal" | {level: weight}}` -- enumerated factors, all levels
            required in every cell, collapsed with these weights.
        nested: Columns of within-cell draws (rollout/pass, question, judge call), averaged in.
        model: Column naming the checkpoint.
        value: Column holding the outcome (0/1 or a score).
        unit_weights: "equal" or "count" (weight a fixed unit by its number of observations,
            e.g. MMLU's question-count weighting). Ignored when units are random.
        incomplete: "drop" units that are missing a model or a fixed level (recorded in
            `Table.dropped_units`), or "error".
    """
    unit: str
    units: str = "random"
    crossed_fixed: Mapping[str, Any] = field(default_factory=dict)
    nested: tuple[str, ...] = ()
    model: str = "model"
    value: str = "value"
    unit_weights: str = "equal"
    incomplete: str = "drop"

    def __post_init__(self):
        assert self.units in ("random", "fixed"), f"units must be random|fixed, got {self.units!r}"
        assert self.unit_weights in ("equal", "count"), f"unit_weights must be equal|count"
        assert self.incomplete in ("drop", "error"), f"incomplete must be drop|error"
        assert not (self.unit_weights == "count" and self.units == "random"), (
            "unit_weights='count' with units='random' is incoherent, so it is refused rather "
            "than silently ignored: weighting a SAMPLED unit by however many observations it "
            "happened to receive implies the population you are generalising to is over those "
            "observations, not over the units -- in which case make the observation the unit. "
            "Count-weighting is for fixed strata (MMLU's 57 subjects by question count).")
        object.__setattr__(self, "nested", tuple(self.nested))
        object.__setattr__(self, "crossed_fixed", dict(self.crossed_fixed))
        for col, spec in self.crossed_fixed.items():
            assert spec == "equal" or (isinstance(spec, Mapping) and spec), \
                f"crossed_fixed[{col!r}] must be 'equal' or a {{level: weight}} mapping"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Design":
        """Build from a config block (dict / OmegaConf); unknown keys are an error."""
        d = {k: (dict(v) if isinstance(v, Mapping) else (list(v) if isinstance(v, (list, tuple)) else v))
             for k, v in dict(cfg).items()}
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - allowed
        assert not unknown, f"unknown Design keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        if "nested" in d:
            d["nested"] = tuple(d["nested"])
        return cls(**d)

    def fixed_weights(self, col: str, levels: Sequence[Any]) -> dict[Any, float]:
        spec = self.crossed_fixed[col]
        if spec == "equal":
            return {lv: 1.0 / len(levels) for lv in levels}
        missing = set(levels) - set(spec)
        assert not missing, f"crossed_fixed[{col!r}] has no weight for levels {sorted(map(str, missing))}"
        total = float(sum(spec[lv] for lv in levels))
        return {lv: float(spec[lv]) / total for lv in levels}

    def describe_units(self) -> str:
        return ("units sampled from the benchmark population" if self.units == "random"
                else "these units as a fixed benchmark")


@dataclass
class Table:
    """The n x J table of cell scores a Design collapses a long table to.

    Attributes:
        values: (n, J) cell means (over fixed levels with their weights, then nested draws).
        models, units: Row and column labels.
        counts: (n, J) number of nested observations behind each cell, all levels together.
        reps: (n, J) rollouts per fixed level in the cell (the smallest level's count) --
            the R of the derivation; `counts` is R times the number of fixed-level combos.
        noise_var: (n, J) variance of the cell mean due to nested draws, estimated from the
            within-cell spread: sum over fixed levels of w^2 s^2 / R. NaN where any level has
            fewer than two draws.
        unit_weights: (J,) weights summing to 1 (equal unless the Design says otherwise).
        dropped_units: Units removed for being absent, or incomplete, under some model.
        design: The Design that produced it.
    """
    values: np.ndarray
    models: list[str]
    units: list[str]
    counts: np.ndarray
    reps: np.ndarray
    noise_var: np.ndarray
    unit_weights: np.ndarray
    dropped_units: list[str]
    design: Design

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_units(self) -> int:
        return len(self.units)

    def rollouts(self) -> dict[str, float]:
        return {"min": int(self.reps.min()), "max": int(self.reps.max()),
                "mean": float(self.reps.mean())}

    def select_units(self, units: Sequence[str]) -> "Table":
        idx = [self.units.index(u) for u in units]
        w = self.unit_weights[idx]
        return Table(self.values[:, idx], list(self.models), list(units), self.counts[:, idx],
                     self.reps[:, idx], self.noise_var[:, idx], w / w.sum(),
                     list(self.dropped_units), self.design)


def _frame(obs: Any) -> pd.DataFrame:
    if isinstance(obs, pd.DataFrame):
        return obs.copy()
    return pd.DataFrame(list(obs))


def collapse(obs: Any, design: Design) -> Table:
    """Long table -> n x J table of cell scores, honouring the Design's collapse rules.

    Args:
        obs: Rows with at least the columns the Design names (DataFrame or iterable of dicts).
        design: The Design.

    Returns:
        A balanced Table: every kept unit is present and complete for every model.
    """
    df = _frame(obs)
    fixed_cols = list(design.crossed_fixed)
    need = [design.model, design.unit, design.value, *fixed_cols]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"long table lacks columns {missing}; has {list(df.columns)}"
    stray = [c for c in design.nested if c not in df.columns]
    assert not stray, f"nested columns {stray} not in the long table"
    assert len(df), "empty long table"
    df = df[[design.model, design.unit, design.value, *fixed_cols]].copy()
    df[design.value] = df[design.value].astype(float)

    keys = [design.model, design.unit, *fixed_cols]
    g = df.groupby(keys, sort=True)[design.value]
    per_level = pd.DataFrame({"mean": g.mean(), "count": g.count(), "var": g.var(ddof=1)})
    models = sorted(df[design.model].unique(), key=str)
    units = sorted(df[design.unit].unique(), key=str)
    fixed_levels = {c: sorted(df[c].unique(), key=str) for c in fixed_cols}
    weights = {c: design.fixed_weights(c, fixed_levels[c]) for c in fixed_cols}
    combos = [()] if not fixed_cols else list(_product([fixed_levels[c] for c in fixed_cols]))
    combo_w = {combo: float(np.prod([weights[c][lv] for c, lv in zip(fixed_cols, combo)])) for combo in combos}

    idx = per_level.index
    values = np.full((len(models), len(units)), np.nan)
    counts = np.zeros((len(models), len(units)), dtype=int)
    reps = np.zeros((len(models), len(units)), dtype=int)
    noise = np.full((len(models), len(units)), np.nan)
    complete = np.ones((len(models), len(units)), dtype=bool)
    for i, m in enumerate(models):
        for j, u in enumerate(units):
            v = c = nv = 0.0
            r_min = None
            for combo in combos:
                key = (m, u, *combo) if fixed_cols else (m, u)
                if key not in idx:
                    complete[i, j] = False
                    break
                row = per_level.loc[key]
                w, cnt = combo_w[combo], int(row["count"])
                v += w * float(row["mean"])
                c += cnt
                r_min = cnt if r_min is None else min(r_min, cnt)
                nv += w * w * (float(row["var"]) / cnt if cnt >= 2 else np.nan)
            if complete[i, j]:
                values[i, j], counts[i, j], reps[i, j], noise[i, j] = v, int(c), int(r_min), nv

    keep = complete.all(axis=0)
    dropped = [u for u, k in zip(units, keep) if not k]
    if dropped and design.incomplete == "error":
        raise NotEstimable(f"{len(dropped)} unit(s) missing a model or a fixed level: {dropped[:10]}"
                           + (" ..." if len(dropped) > 10 else ""))
    kept = [j for j, k in enumerate(keep) if k]
    assert len(kept) >= 2, f"fewer than two complete units (dropped {dropped})"
    values, counts, reps, noise = values[:, kept], counts[:, kept], reps[:, kept], noise[:, kept]
    units = [units[j] for j in kept]
    if design.units == "fixed" and design.unit_weights == "count":
        uw = counts.sum(axis=0).astype(float)
    else:
        uw = np.ones(len(units))
    return Table(values, [str(m) for m in models], [str(u) for u in units], counts, reps, noise,
                 uw / uw.sum(), [str(u) for u in dropped], design)


def _product(levels: list[list[Any]]):
    if not levels:
        yield ()
        return
    for first in levels[0]:
        for rest in _product(levels[1:]):
            yield (first, *rest)


# --------------------------------------------------------------------------- the three spreads

def satterthwaite(parts: Sequence[tuple[float, float]]) -> float:
    """Effective df for a variance built as a sum of estimated parts.

    Each part is `(value, df)`. A t-interval assumes ONE variance estimate with a known df;
    `T_A + T_B - T_C` mixes estimates whose dfs differ (n-1, J-1, (n-1)(J-1)) and has no
    exact df. Satterthwaite matches the first two moments of the sum to a single scaled
    chi-square, giving

        nu = (sum of parts)^2 / sum over parts of (part^2 / df_part)

    which behaves as it should at the ends: if one part dominates, nu is that part's df; if
    several contribute equally, nu is larger than any single one, because averaging several
    noisy variance estimates gives a less noisy total. Signs are irrelevant in the
    denominator, so a subtracted term still costs df rather than adding it.

    Args:
        parts: `(value, df)` for each estimated component; parts with df <= 0 or value 0
            contribute nothing.

    Returns:
        The effective degrees of freedom (>= 1), or inf when nothing is estimated.
    """
    total = sum(v for v, _ in parts)
    denom = sum(v * v / d for v, d in parts if d and d > 0)
    if denom <= 0 or total <= 0:
        return float("inf")
    return max(1.0, total * total / denom)


def crossed_terms(values: np.ndarray, unit_weights: np.ndarray | None = None) -> dict[str, float]:
    """mu_hat and T_A, T_B, T_C for an n x J table (n >= 2 for T_A and T_C; NaN otherwise)."""
    v = np.asarray(values, dtype=float)
    assert v.ndim == 2 and v.shape[1] >= 2, f"need an n x J table with J >= 2, got {v.shape}"
    n_models, n_units = v.shape
    w = (np.ones(n_units) / n_units if unit_weights is None
         else np.asarray(unit_weights, float))
    # (v * w).sum rather than `v @ w`: numpy's BLAS matmul path raises spurious
    # divide-by-zero/overflow RuntimeWarnings on some shapes (reproducible with a plain
    # `@` on the same arrays, numpy 2.2), which would pollute stderr for every caller.
    row = (v * w[None, :]).sum(axis=1)   # per-model rates (unit-weighted)
    col = v.mean(axis=0)              # per-unit rates
    mu = float(row.mean())
    out = {"mu": mu, "T_A": float("nan"), "T_B": float(col.var(ddof=1) / n_units),
           "T_C": float("nan")}
    if n_models >= 2:
        resid = v - row[:, None] - col[None, :] + mu
        out["T_A"] = float(row.var(ddof=1) / n_models)
        out["T_C"] = float((resid ** 2).sum() / ((n_models - 1) * (n_units - 1))
                           / (n_models * n_units))
    return out


# --------------------------------------------------------------------------- results

@dataclass
class Result:
    """One interval and everything needed to read it honestly."""
    estimand: str
    method: str
    mean: float
    se: float
    lo: float
    hi: float
    mult: float
    df: float
    n_models: int
    n_units: int
    models: str
    units: str
    terms: dict[str, float] = field(default_factory=dict)
    rollouts: dict[str, float] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    claims: list[str] = field(default_factory=list)
    dropped_units: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["terms"] = {k: _jsonable(v) for k, v in self.terms.items()}
        d["noise"] = {k: _jsonable(v) for k, v in self.noise.items()}
        d["ci95"] = [self.lo, self.hi]
        return d


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _noise_parts(table: Table) -> list[tuple[float, float]]:
    """Each cell's contribution to Var(mu_hat) when rollouts are the only randomness.

    `(w_j^2 * noise_var_ij / n^2, R_ij - 1)` per cell, ready for `satterthwaite`. Pooling
    these as one estimate with df = sum(R-1) is exact only when every cell has the same R
    AND the same noise; Satterthwaite reduces to that when they do (K equal parts of df d
    give nu = K*d) and correctly loses df when one noisy cell dominates.
    """
    w = table.unit_weights
    return [(float(w[j] ** 2 * table.noise_var[i, j] / table.n_models ** 2), float(table.reps[i, j] - 1))
            for i in range(table.n_models) for j in range(table.n_units)]


def _noise_block(table: Table) -> dict[str, Any]:
    """Rollout-noise contribution to Var(mu_hat), where the data can support it."""
    nv = table.noise_var
    estimable = bool(np.isfinite(nv).all())
    if not estimable:
        return {"estimable": False, "sigma_eps2": None, "term": None, "share": None,
                "reason": "every cell needs >= 2 draws of every level to estimate rollout noise"}
    w = table.unit_weights
    term = float(((nv * w[None, :] ** 2).sum(axis=1)).sum() / table.n_models ** 2)
    per_draw = float(np.nanmean(nv * table.counts))   # s^2 per draw, roughly
    return {"estimable": True, "sigma_eps2": per_draw, "term": term, "share": None}


def _claims(table: Table, models: str) -> list[str]:
    d = table.design
    out = []
    out.append(f"{'models sampled' if models == 'random' else 'model(s) fixed'}: "
               + ("generalises to checkpoints from the same training pipeline "
                  f"(n={table.n_models} seeds; seed-to-seed variance estimated)" if models == "random"
                  else f"about {'this checkpoint' if table.n_models == 1 else f'these {table.n_models} checkpoints'} only; "
                       "pipeline (seed-to-seed) variance is not estimated"))
    out.append(("units sampled: generalises to units drawn like these "
                f"({table.n_units} {d.unit}s; unit-to-unit variance estimated)") if d.units == "random"
               else f"units fixed: about these {table.n_units} {d.unit}s only; no unit-to-unit term")
    r = table.rollouts()
    if r["max"] == 1:
        out.append("one rollout per cell: rollout noise is inside every spread and is measured "
                   "with it, but cannot be separated; a cell's value is read as the model's "
                   f"behaviour on that {d.unit}")
    else:
        out.append(f"{r['min']}-{r['max']} draws per cell: rollout noise estimated from "
                   "within-cell spread (see `noise`)")
    if d.crossed_fixed:
        out.append("fixed factors " + ", ".join(f"{c} ({'equal' if s == 'equal' else 'weighted'})"
                                                 for c, s in d.crossed_fixed.items())
                   + " are enumerated in every cell: in the estimand, no variance term")
    if table.dropped_units:
        out.append(f"{len(table.dropped_units)} incomplete unit(s) dropped to keep the table balanced")
    return out


def _estimand(table: Table, models: str) -> str:
    d = table.design
    who = ("a checkpoint from the pipeline" if models == "random"
           else ("this checkpoint" if table.n_models == 1 else f"the mean of these {table.n_models} checkpoints"))
    where = (f"a {d.unit} drawn like these" if d.units == "random" else f"these {table.n_units} {d.unit}s")
    mix = ("" if not d.crossed_fixed else
           " under the fixed mix of " + " x ".join(d.crossed_fixed))
    return f"mean outcome of {who} on {where}{mix}"


def _finish(table: Table, models: str, method: str, mean: float, se2: float, df: float,
            terms: dict[str, float], alpha: float) -> Result:
    se = math.sqrt(max(se2, 0.0))
    mult = t_quantile(1 - alpha / 2, df)
    noise = _noise_block(table)
    if noise["estimable"] and se2 > 0:
        noise["share"] = float(noise["term"] / se2)
    return Result(_estimand(table, models), method, float(mean), se, float(mean - mult * se),
                  float(mean + mult * se), mult, df, table.n_models, table.n_units, models, table.design.units,
                  terms, table.rollouts(), noise, _claims(table, models), list(table.dropped_units))


def _as_table(x: Any, design: Design | None) -> Table:
    if isinstance(x, Table):
        return x
    assert design is not None, "a Design is required to collapse a long table"
    return collapse(x, design)


def _models_kind(table: Table, models: str | None) -> str:
    if models is None:
        return "random" if table.n_models >= 2 else "fixed"
    assert models in ("random", "fixed")
    if models == "random" and table.n_models < 2:
        raise NotEstimable("models='random' needs >= 2 checkpoints; with one checkpoint the "
                           "seed-to-seed variance is not estimable -- use models='fixed' and "
                           "claim only about this checkpoint")
    return models


# --------------------------------------------------------------------------- intervals

def interval(obs: Any, design: Design | None = None, *, models: str | None = None,
             alpha: float = 0.05) -> Result:
    """The 95% interval for the table's mean under its Design.

    Args:
        obs: A long table (with `design`) or an already-collapsed Table.
        design: Required when `obs` is a long table.
        models: "random" (>= 2 checkpoints from one pipeline) or "fixed"; default random iff
            n >= 2. Random adds the model and interaction terms.
        alpha: Two-sided level.

    Returns:
        A Result. Which spreads are combined, and the df of the multiplier:

            models random, units random   T_A + T_B - T_C            Satterthwaite df
            models random, units fixed    T_A                        n - 1
            models fixed,  units random   spread of column means / J  J - 1
            models fixed,  units fixed    within-cell noise (R >= 2)  sum over cells of R-1
    """
    table = _as_table(obs, design)
    kind = _models_kind(table, models)
    t = crossed_terms(table.values, table.unit_weights)
    mu = t["mu"]
    n_models, n_units = table.n_models, table.n_units
    if kind == "random" and table.design.units == "random":
        se2 = t["T_A"] + t["T_B"] - t["T_C"]
        fallback = se2 <= 0
        if fallback:
            se2 = max(t["T_A"], t["T_B"])
        df = satterthwaite([(t["T_A"], n_models - 1), (t["T_B"], n_units - 1),
                            (t["T_C"], (n_models - 1) * (n_units - 1))])
        terms = {"T_A": t["T_A"], "T_B": t["T_B"], "T_C": t["T_C"], "negative_fallback": fallback,
                 "df_source": "Satterthwaite over T_A, T_B, T_C"}
        return _finish(table, kind, "T_A + T_B - T_C, t_nu (Satterthwaite)", mu, se2, df, terms, alpha)
    if kind == "random":  # units fixed
        return _finish(table, kind, "T_A (per-model rates), t_{n-1}", mu, t["T_A"], table.n_models - 1,
                       {"T_A": t["T_A"]}, alpha)
    if table.design.units == "random":  # models fixed
        return _finish(table, kind, "spread of per-unit rates over J (T_B), t_{J-1}", mu, t["T_B"],
                       table.n_units - 1, {"T_B": t["T_B"]}, alpha)
    # both fixed: rollouts are the only randomness
    if not np.isfinite(table.noise_var).all():
        raise NotEstimable(
            "models and units both fixed: the only randomness is the rollouts, and with one "
            "rollout in some cell their variance cannot be estimated. Either run >= 2 rollouts "
            "per cell, or treat units as sampled (units='random'), which is a different claim.")
    parts = _noise_parts(table)
    se2 = sum(v for v, _ in parts)
    return _finish(table, kind, "within-cell rollout noise only, t_nu (Satterthwaite over cells)",
                   mu, se2, satterthwaite(parts), {"noise_term": se2}, alpha)


def difference(obs_a: Any, obs_b: Any, design: Design | None = None, *, models: str | None = None,
               paired_models: bool = False, alpha: float = 0.05) -> Result:
    """Interval for mean(A) - mean(B), paired on every axis the two arms share.

    Units are always paired (both arms must cover the same units; the intersection is used
    and the rest recorded). Models are paired only when `paired_models` -- the same
    checkpoints under two conditions (e.g. mandated vs incentivized), in which case the
    per-cell difference table goes through `interval` directly. Otherwise the arms have
    different checkpoints and the model terms add.

    Args:
        obs_a, obs_b: Long tables or Tables for the two arms.
        design: Required for long tables.
        models: As in `interval`, applied to both arms.
        paired_models: Both arms contain the same checkpoint labels and are paired on them.
        alpha: Two-sided level.
    """
    A, B = _as_table(obs_a, design), _as_table(obs_b, design)
    shared = [u for u in A.units if u in set(B.units)]
    assert len(shared) >= 2, "arms share fewer than two units"
    A, B = A.select_units(shared), B.select_units(shared)
    dropped = sorted(set(A.dropped_units) | set(B.dropped_units)
                     | ({*obs_a.units, *obs_b.units} - set(shared) if isinstance(obs_a, Table) else set()))

    if paired_models:
        assert A.models == B.models, "paired_models needs identical checkpoint labels in both arms"
        D = Table(A.values - B.values, A.models, shared, A.counts + B.counts,
                  np.minimum(A.reps, B.reps), A.noise_var + B.noise_var, A.unit_weights,
                  dropped, A.design)
        r = interval(D, models=models, alpha=alpha)
        r.estimand = "difference (A - B), paired on units and checkpoints: " + r.estimand
        return r

    kind = _models_kind(A, models) if models else ("random" if min(A.n_models, B.n_models) >= 2 else "fixed")
    if kind == "random":
        assert A.n_models >= 2 and B.n_models >= 2, "models='random' needs >= 2 checkpoints in each arm"
    ta, tb = crossed_terms(A.values, A.unit_weights), crossed_terms(B.values, B.unit_weights)
    d_cols = A.values.mean(axis=0) - B.values.mean(axis=0)     # per-unit difference of column means
    mean = ta["mu"] - tb["mu"]
    n_units = A.n_units
    merged = Table(np.vstack([A.values, -B.values]), A.models + B.models, shared,
                   np.vstack([A.counts, B.counts]), np.vstack([A.reps, B.reps]),
                   np.vstack([A.noise_var, B.noise_var]), A.unit_weights, dropped,
                   A.design)   # only for rollout/claims bookkeeping
    if kind == "random" and A.design.units == "random":
        t_bd = float(d_cols.var(ddof=1) / n_units)
        se2 = ta["T_A"] + tb["T_A"] + t_bd - ta["T_C"] - tb["T_C"]
        fallback = se2 <= 0
        if fallback:
            se2 = ta["T_A"] + tb["T_A"] + t_bd
        terms = {"T_A_a": ta["T_A"], "T_A_b": tb["T_A"], "T_B_d": t_bd, "T_C_a": ta["T_C"],
                 "T_C_b": tb["T_C"], "negative_fallback": fallback,
                 "df_source": "Satterthwaite over the five terms"}
        df = satterthwaite([(ta["T_A"], A.n_models - 1), (tb["T_A"], B.n_models - 1),
                            (t_bd, n_units - 1),
                            (ta["T_C"], (A.n_models - 1) * (n_units - 1)),
                            (tb["T_C"], (B.n_models - 1) * (n_units - 1))])
        r = _finish(merged, kind, "T_A^A + T_A^B + T_B^(d) - T_C^A - T_C^B, t_nu (Satterthwaite)",
                    mean, se2, df, terms, alpha)
    elif kind == "random":  # units fixed: Welch's two-sample t on the per-model rates
        ra = (A.values * A.unit_weights[None, :]).sum(axis=1)   # see crossed_terms on `@`
        rb = (B.values * B.unit_weights[None, :]).sum(axis=1)
        va, vb = float(ra.var(ddof=1) / A.n_models), float(rb.var(ddof=1) / B.n_models)
        df = satterthwaite([(va, A.n_models - 1), (vb, B.n_models - 1)])   # Welch-Satterthwaite
        r = _finish(merged, kind, "Welch two-sample t on per-model rates, t_nu (Welch-Satterthwaite)",
                    mean, va + vb, df, {"var_a": va, "var_b": vb}, alpha)
    elif A.design.units == "random":  # models fixed, units random: paired on units
        se2 = float(d_cols.var(ddof=1) / n_units)
        r = _finish(merged, kind, "spread of per-unit differences over J, t_{J-1}", mean, se2,
                    n_units - 1,
                    {"T_B_d": se2}, alpha)
    else:  # both fixed
        if not (np.isfinite(A.noise_var).all() and np.isfinite(B.noise_var).all()):
            raise NotEstimable("both fixed: rollout noise not estimable with one rollout per cell")
        parts = _noise_parts(A) + _noise_parts(B)
        se2 = sum(v for v, _ in parts)
        r = _finish(merged, kind, "within-cell rollout noise only, t_nu (Satterthwaite over cells)",
                    mean, se2, satterthwaite(parts), {"noise_term": se2}, alpha)
    def who(t: Table) -> str:
        return ("a checkpoint from its pipeline" if kind == "random"
                else ("its checkpoint" if t.n_models == 1 else f"the mean of its {t.n_models} checkpoints"))
    where = (f"a {A.design.unit} drawn like these" if A.design.units == "random"
             else f"these {n_units} {A.design.unit}s")
    r.estimand = (f"difference (A - B), paired on units: A's {who(A)} minus B's {who(B)}, "
                  f"on {where}")
    r.n_models = A.n_models + B.n_models
    return r


# --------------------------------------------------------------------------- bootstrap + binomial

def cluster_bootstrap(obs: Any, statistic: Callable[[Any], float], design: Design | None = None,
                      *, models: str | None = None, on: str = "table", n_boot: int = 10_000,
                      seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap of `statistic(...)` resampling the Design's random axes.

    Units (columns) are resampled with replacement, each carrying its whole cell -- never
    rollouts or fixed levels. When models are random and n >= 2, rows are resampled too.
    Use only for statistics without a closed-form SE; for a mean, `interval` is exact, and a
    test asserts the two agree there.

    Args:
        obs: A long table, or a collapsed Table (only with `on="table"`).
        statistic: What to bootstrap. With `on="table"` it receives the resampled
            `(n_models, n_units)` matrix of cell means. With `on="rows"` it receives the
            resampled LONG rows as a DataFrame -- needed by any statistic that cannot be
            computed from cell means: a Bradley-Terry fit needs the individual battles and
            their style covariates, a median needs the raw values.
        design: Required when `obs` is a long table.
        models: "random"/"fixed" as in `interval`; default random iff n >= 2.
        on: "table" (default) or "rows". See `statistic`.
        n_boot, seed, alpha: Resamples, RNG seed, two-sided level.

    Returns:
        `mean` (the statistic on the observed data), `lo`, `hi`, `se`, `method`, and the
        axes that were resampled. A unit drawn twice contributes its rows twice, which is
        what "this unit was sampled twice" means for a downstream fit.
    """
    assert on in ("table", "rows"), f"on must be table|rows, got {on!r}"
    table = _as_table(obs, design)
    kind = _models_kind(table, models)
    rng = np.random.default_rng(seed)
    n_models, n_units = table.n_models, table.n_units

    if on == "rows":
        assert not isinstance(obs, Table), "on='rows' needs the long table, not a collapsed Table"
        df = _frame(obs)
        d = table.design
        # Row positions per (model, unit), so a resample is one concatenate + one .iloc
        # rather than a DataFrame slice per cell.
        pos = {k: np.asarray(v) for k, v in df.groupby([d.model, d.unit]).indices.items()}
        keep = {(m, u) for m in table.models for u in table.units if (m, u) in pos}
        take_all = np.concatenate([pos[k] for k in sorted(keep, key=str)]) if keep else np.array([], int)
        point = float(statistic(df.iloc[np.sort(take_all)]))
    else:
        point = float(statistic(table.values))

    draws = np.empty(n_boot)
    for b in range(n_boot):
        cols = (rng.integers(0, n_units, n_units) if table.design.units == "random"
                else np.arange(n_units))
        rows = (rng.integers(0, n_models, n_models) if kind == "random"
                else np.arange(n_models))
        if on == "table":
            draws[b] = statistic(table.values[np.ix_(rows, cols)])
        else:
            take = [pos[(table.models[i], table.units[j])] for i in rows for j in cols
                    if (table.models[i], table.units[j]) in pos]
            draws[b] = statistic(df.iloc[np.concatenate(take)] if take else df.iloc[:0])
    return {"mean": point, "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)), "se": float(draws.std(ddof=1)),
            "method": f"cluster bootstrap over {'units' if kind == 'fixed' else 'models and units'}, "
                      f"{n_boot} resamples, statistic on the {'cell table' if on == 'table' else 'long rows'}",
            "n_boot": n_boot, "models": kind, "units": table.design.units, "on": on}


def wilson(k: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Wilson rather than the naive `p +/- z*sqrt(p(1-p)/n)`: it stays inside [0, 1] and keeps
    a sensible width at the edges, where the naive interval has width zero at 0/n and can
    run below zero nearby. Use it for a rate near 0 or 1, where the symmetric intervals
    above stop making sense. Bounds are clamped: rounding can otherwise put them a
    floating-point hair outside [0, 1], which reads as 100.000000001% in a report.
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
