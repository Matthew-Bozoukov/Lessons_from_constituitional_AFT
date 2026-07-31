# ABOUTME: Six views over the results store, each feeding a specific element of the three plots.
# ABOUTME: Every metric is a binary rate, so every view returns rates with intervals - nothing else.

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .control import loader
from .core.stats import cluster_bootstrap, mcnemar_exact, wilson

# The four metrics, in the order they appear in tables and legends.
METRICS: tuple[str, ...] = ("knows", "notices", "acts", "discriminates")

# Rows that belong to no single clause (ingested capability numbers) carry this id.
GLOBAL = "_global"


class AnalysisError(ValueError):
    """Raised when the store cannot support the requested view."""


@lru_cache(maxsize=16)
def axis_title(axis: str) -> str:
    """Return an axis's display title, falling back to its name."""
    try:
        return str(loader.rubric(axis)["title"])
    except loader.PromptError:
        return axis.replace("_", " ")


def usable(df, axes: Sequence[str] | None = None, condition: str | None = None):
    """Drop errored rows, and optionally filter to axes and one condition.

    Errored rows are removed rather than scored 0: a model that timed out did not fail the metric,
    and counting it as a failure biases every rate that includes it.

    Args:
        df: A results frame.
        axes: Axis names to keep; None keeps all.
        condition: Condition to keep ("clean" or "pressure:<name>"); None keeps all.

    Returns:
        The filtered frame.
    """
    out = df[df["error"].fillna("") == ""]
    if axes is not None:
        out = out[out["axis"].isin(list(axes))]
    if condition is not None:
        out = out[out["condition"] == condition]
    return out


def recipes(df) -> list[str]:
    """Return the recipes present, sorted so colour assignment is stable."""
    return sorted(str(r) for r in df["recipe"].dropna().unique())


def check_comparable(df) -> list[str]:
    """Return the distinct item-set fingerprints present.

    More than one means the rows were measured on different items and must not be compared.
    """
    return sorted(str(x) for x in df["itemset_id"].dropna().unique())


def rates(df, condition: str | None = "clean"):
    """Pooled pass rate per (recipe, axis), with a clause-clustered interval.

    The interval resamples clauses rather than rows. A dozen scenarios written for one clause share
    its difficulty and phrasing, so treating them as independent understates the interval.

    Args:
        df: A results frame.
        condition: Condition to restrict to; None pools every condition.

    Returns:
        A frame of (recipe, axis, rate, lo, hi, n, n_clauses).
    """
    import pandas as pd

    frame = usable(df, condition=condition)
    frame = frame[frame["clause_id"] != GLOBAL]
    if frame.empty:
        return pd.DataFrame(columns=["recipe", "axis", "rate", "lo", "hi", "n", "n_clauses"])

    rows = []
    for (recipe, axis), group in frame.groupby(["recipe", "axis"], sort=True):
        interval = cluster_bootstrap(
            group["passed"].astype(float).tolist(), group["clause_id"].tolist()
        )
        rows.append(
            {
                "recipe": recipe,
                "axis": axis,
                "rate": interval.mean,
                "lo": interval.lo,
                "hi": interval.hi,
                "n": interval.n,
                "n_clauses": group["clause_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def per_clause_rates(df, condition: str | None = "clean"):
    """Pass rate per (recipe, axis, clause) - the faint background dots on the scatters.

    Args:
        df: A results frame.
        condition: Condition to restrict to.

    Returns:
        A frame of (recipe, axis, clause_id, clause_title, rate, n).
    """
    import pandas as pd

    frame = usable(df, condition=condition)
    frame = frame[frame["clause_id"] != GLOBAL]
    if frame.empty:
        return pd.DataFrame(columns=["recipe", "axis", "clause_id", "clause_title", "rate", "n"])

    return frame.groupby(
        ["recipe", "axis", "clause_id", "clause_title"], as_index=False
    ).agg(rate=("passed", "mean"), n=("passed", "size"))


def _widen(frame, x_axis: str, y_axis: str, value_cols: Sequence[str]):
    """Pivot a long (recipe, axis, ...) frame into x_/y_ prefixed columns."""
    out = None
    for axis, prefix in ((x_axis, "x"), (y_axis, "y")):
        part = frame[frame["axis"] == axis].drop(columns=["axis"])
        part = part.rename(columns={c: f"{prefix}_{c}" for c in value_cols})
        keys = [c for c in part.columns if not c.startswith(("x_", "y_"))]
        out = part if out is None else out.merge(part, on=keys, how="inner")
    return out


def scatter_pairs(df, x_axis: str, y_axis: str):
    """Join two metrics into the shape the knowing/noticing scatters need.

    Args:
        df: A results frame.
        x_axis: Metric on x.
        y_axis: Metric on y.

    Returns:
        Tuple of (pooled, per_clause). `pooled` is one row per recipe with x/y rates and bounds
        (the big dots); `per_clause` is one row per (recipe, clause) with x/y rates (faint dots).

    Raises:
        AnalysisError: If either metric is absent from the frame.
    """
    pooled_all = rates(df)
    present = set(pooled_all["axis"])
    missing = [a for a in (x_axis, y_axis) if a not in present]
    if missing:
        raise AnalysisError(f"metrics {missing} absent from results; present: {sorted(present)}")

    pooled = _widen(pooled_all, x_axis, y_axis, ["rate", "lo", "hi", "n", "n_clauses"])
    clause = _widen(per_clause_rates(df), x_axis, y_axis, ["rate", "n"])
    return pooled, clause


def paired_pressure(df, axis: str = "acts"):
    """Clean vs under-pressure pass rate for one metric, paired on the same scenario.

    Pairing is the whole point: a stressed item and its clean parent are the same scenario, so the
    difference removes item difficulty. Reports the exact McNemar p on discordant pairs, because
    discordance - not row count - is what limits this comparison.

    Args:
        df: A results frame.
        axis: Metric to compare.

    Returns:
        A frame of (recipe, clean_rate, pressure_rate, delta, n_pairs, n_broke, n_fixed, p).
    """
    import pandas as pd

    frame = usable(df, axes=[axis])
    parents = frame[frame["parent_item_id"].fillna("") == ""][
        ["run_id", "item_id", "passed"]
    ].rename(columns={"item_id": "parent_item_id", "passed": "clean_passed"})
    stressed = frame[frame["parent_item_id"].fillna("") != ""]
    joined = stressed.merge(parents, on=["run_id", "parent_item_id"], how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=["recipe", "clean_rate", "pressure_rate", "delta", "n_pairs",
                     "n_broke", "n_fixed", "p"]
        )

    rows = []
    for recipe, group in joined.groupby("recipe", sort=True):
        clean = group["clean_passed"].astype(bool)
        under = group["passed"].astype(bool)
        broke = int((clean & ~under).sum())   # held clean, broke under pressure
        fixed = int((~clean & under).sum())   # failed clean, passed under pressure
        rows.append(
            {
                "recipe": recipe,
                "clean_rate": float(clean.mean()),
                "pressure_rate": float(under.mean()),
                "delta": float(under.mean() - clean.mean()),
                "n_pairs": int(len(group)),
                "n_broke": broke,
                "n_fixed": fixed,
                "p": mcnemar_exact(broke, fixed),
            }
        )
    return pd.DataFrame(rows)


def headline_table(df):
    """One row per (recipe, condition, axis) with a Wilson interval, for the markdown mirror.

    Wilson here so the table reports the conventional interval a reader expects; the plots use the
    clause-clustered bootstrap, which is wider and more honest about item dependence.

    Args:
        df: A results frame.

    Returns:
        A frame of (recipe, condition, axis, rate, lo, hi, n).
    """
    import pandas as pd

    frame = usable(df)
    frame = frame[frame["clause_id"] != GLOBAL]
    if frame.empty:
        return pd.DataFrame(columns=["recipe", "condition", "axis", "rate", "lo", "hi", "n"])

    rows = []
    for (recipe, condition, axis), group in frame.groupby(
        ["recipe", "condition", "axis"], sort=True
    ):
        interval = wilson(float(group["passed"].sum()), len(group))
        rows.append(
            {
                "recipe": recipe,
                "condition": condition,
                "axis": axis,
                "rate": interval.mean,
                "lo": interval.lo,
                "hi": interval.hi,
                "n": interval.n,
            }
        )
    return pd.DataFrame(rows)


def health_warnings(df, min_n: int = 90, ceiling: float = 0.95, floor: float = 0.05) -> list[str]:
    """Flag the two failure modes that silently ruined v1: saturation and thin cells.

    v1 shipped with `notices` at 1.000 in both arms and `knows` at n=21. Neither was surfaced
    anywhere, so both plots looked fine and meant nothing. These checks make that loud.

    Args:
        df: A results frame.
        min_n: Minimum rows per (recipe, metric) cell.
        ceiling: Rate at or above which a metric is treated as saturated.
        floor: Rate at or below which a metric is treated as floored.

    Returns:
        Human-readable warnings; empty when the data can support the plots.
    """
    out: list[str] = []
    for row in rates(df).itertuples(index=False):
        if row.n < min_n:
            out.append(
                f"THIN CELL: {row.recipe}/{row.axis} has n={row.n} (want >={min_n}); its interval "
                f"is too wide to compare."
            )
        if row.rate >= ceiling:
            out.append(
                f"SATURATED: {row.recipe}/{row.axis} = {row.rate:.3f} at ceiling; this metric can "
                f"only detect degradation, not improvement."
            )
        if row.rate <= floor:
            out.append(
                f"FLOORED: {row.recipe}/{row.axis} = {row.rate:.3f}; likely an item or rubric "
                f"mismatch rather than a model result."
            )
    fingerprints = check_comparable(df)
    if len(fingerprints) > 1:
        out.append(
            f"NOT COMPARABLE: rows span {len(fingerprints)} item sets ({', '.join(fingerprints)})."
        )
    return out
