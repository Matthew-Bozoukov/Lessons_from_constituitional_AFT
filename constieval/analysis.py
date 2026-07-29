# ABOUTME: Views over the results store. Every plot and table derives from a function here,
# ABOUTME: so a metric is defined exactly once and a figure can never disagree with a table.

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from .control import loader
from .core.stats import Interval, bootstrap_mean, paired_delta, wilson

# Clause id used by rows that belong to no single clause; excluded from clause-level views.
GLOBAL = "_global"

# The axes reported as headline numbers on the trajectory plot, in display order.
HEADLINE_AXES: tuple[str, ...] = (
    "compliance",
    "tension_recognition",
    "justification_quality",
    "conflict_priority",
    "retrieval",
    "fake_discrimination",
    "over_refusal",
    "persona_drift",
)


class AnalysisError(ValueError):
    """Raised when the store cannot support the requested view."""


@lru_cache(maxsize=64)
def direction(axis: str) -> str:
    """Return "higher_better" or "lower_better" for an axis.

    Args:
        axis: Axis name.

    Returns:
        The declared direction; axes with no rubric (derived and capability rows) are
        higher-better by construction.
    """
    try:
        return str(loader.rubric(axis)["direction"])
    except loader.PromptError:
        return "higher_better"


@lru_cache(maxsize=64)
def axis_title(axis: str) -> str:
    """Return an axis's display title, falling back to its name."""
    try:
        return str(loader.rubric(axis)["title"])
    except loader.PromptError:
        return axis.replace("_", " ")


def orient(df):
    """Add a `score_oriented` column where higher always means more internalised.

    Over-refusal and persona drift count failures, so their raw scores run the other
    way. Flipping them once here means no plot has to remember which axes are inverted -
    and no plot can forget.

    Args:
        df: A results frame.

    Returns:
        A copy with `score_oriented` added.
    """
    out = df.copy()
    lower = out["axis"].map(lambda a: direction(a) == "lower_better")
    out["score_oriented"] = out["score"].where(~lower, 1.0 - out["score"])
    out["passed_oriented"] = out["passed"]
    return out


def usable(df, axes: Sequence[str] | None = None):
    """Drop errored rows, and optionally restrict to a set of axes.

    Errored rows are removed rather than scored as zero: a model that timed out did not
    fail the axis, and counting it as a failure biases every aggregate that includes it.

    Args:
        df: A results frame.
        axes: Axis names to keep; None keeps all.

    Returns:
        The filtered frame.
    """
    out = df[df["error"].fillna("") == ""]
    if axes is not None:
        out = out[out["axis"].isin(list(axes))]
    return out


def check_comparable(df) -> list[str]:
    """Return the distinct item-set fingerprints present in a frame.

    More than one means the rows were produced against different item sets and must not
    be compared. Callers surface this rather than silently plotting it.

    Args:
        df: A results frame.

    Returns:
        Sorted itemset ids.
    """
    return sorted(str(x) for x in df["itemset_id"].dropna().unique())


def aggregate(df, by: Sequence[str], value: str = "score_oriented", binary: bool = False):
    """Group and summarise with a confidence interval on every cell.

    Args:
        df: A results frame, already oriented and filtered.
        by: Grouping columns.
        value: Column to summarise.
        binary: Use a Wilson interval on the pass proportion instead of a bootstrap on
            the mean. Correct for pass/fail axes, where cells are small and often near
            0 or 1 - exactly where a normal approximation leaves [0, 1].

    Returns:
        A frame with the grouping columns plus mean, lo, hi, n.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    keys = list(by)
    if df.empty:
        return pd.DataFrame(columns=[*keys, "mean", "lo", "hi", "n"])

    for key, group in df.groupby(keys, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if binary:
            interval: Interval = wilson(float(group["passed_oriented"].sum()), len(group))
        else:
            interval = bootstrap_mean(group[value].tolist())
        rows.append(
            {
                **dict(zip(keys, key_tuple)),
                "mean": interval.mean,
                "lo": interval.lo,
                "hi": interval.hi,
                "n": interval.n,
            }
        )
    return pd.DataFrame(rows)


def clause_axis_matrix(df, recipe: str | None = None, clean_only: bool = True):
    """Clause x axis mean score, the view behind the heatmap.

    Args:
        df: A results frame.
        recipe: Restrict to one recipe; None pools every recipe.
        clean_only: Exclude stressed and OOD items, so the heatmap reports the baseline
            picture and robustness gets its own figure.

    Returns:
        A frame of (clause_id, clause_title, held_out, axis, mean, lo, hi, n).
    """
    frame = orient(usable(df))
    frame = frame[frame["clause_id"] != GLOBAL]
    if clean_only:
        frame = frame[frame["condition"] == "clean"]
    if recipe is not None:
        frame = frame[frame["recipe"] == recipe]
    return aggregate(frame, ["clause_id", "clause_title", "held_out", "axis"])


def retrieval_vs_application(df):
    """Per-clause retrieval score against per-clause action compliance.

    The off-diagonal mass is the thesis: clauses a model can name but does not act on.

    Args:
        df: A results frame.

    Returns:
        A frame of (recipe, clause_id, clause_title, held_out, retrieval, compliance, n_*).
    """
    return _paired_axis_view(df, "retrieval", "compliance")


def compliance_vs_tension(df):
    """Per-clause action compliance against per-clause tension recognition.

    Complying without recognising is the memorised-behaviour signature; the two are
    scored by separate judge calls precisely so they can be plotted apart.

    Args:
        df: A results frame.

    Returns:
        A frame of (recipe, clause_id, clause_title, held_out, compliance, tension_recognition, n_*).
    """
    return _paired_axis_view(df, "compliance", "tension_recognition")


def _paired_axis_view(df, x_axis: str, y_axis: str):
    """Join two axes' per-clause means into one row per (recipe, clause).

    Args:
        df: A results frame.
        x_axis: Axis plotted on x.
        y_axis: Axis plotted on y.

    Returns:
        A frame with one column per axis, plus per-axis n and CI bounds.
    """
    import pandas as pd

    frame = orient(usable(df, [x_axis, y_axis]))
    frame = frame[(frame["clause_id"] != GLOBAL) & (frame["condition"] == "clean")]
    agg = aggregate(frame, ["recipe", "clause_id", "clause_title", "held_out", "axis"])
    if agg.empty:
        return pd.DataFrame(
            columns=["recipe", "clause_id", "clause_title", "held_out", x_axis, y_axis]
        )

    keys = ["recipe", "clause_id", "clause_title", "held_out"]
    wide = None
    for axis in (x_axis, y_axis):
        part = agg[agg["axis"] == axis][[*keys, "mean", "lo", "hi", "n"]].rename(
            columns={
                "mean": axis,
                "lo": f"{axis}_lo",
                "hi": f"{axis}_hi",
                "n": f"n_{axis}",
            }
        )
        wide = part if wide is None else wide.merge(part, on=keys, how="inner")
    return wide.reset_index(drop=True)


def _join_parents(df):
    """Attach each derived row's parent score, on the same run and axis.

    The join is what makes the robustness and OOD numbers paired. A derived row whose
    parent is missing from the frame is dropped rather than compared against a group
    mean, which would reintroduce the item-difficulty confound the pairing removes.

    Args:
        df: A results frame.

    Returns:
        A frame of derived rows with a `parent_score` column added.
    """
    frame = orient(usable(df))
    parents = frame[frame["parent_item_id"].fillna("") == ""][
        ["run_id", "axis", "item_id", "score_oriented"]
    ].rename(columns={"item_id": "parent_item_id", "score_oriented": "parent_score"})
    derived = frame[frame["parent_item_id"].fillna("") != ""]
    return derived.merge(parents, on=["run_id", "axis", "parent_item_id"], how="inner")


def robustness_delta(df, axes: Sequence[str] = ("compliance", "tension_recognition")):
    """Paired stressed-minus-clean delta per (recipe, clause, wrapper).

    Args:
        df: A results frame.
        axes: Axes to include.

    Returns:
        A frame of (recipe, pressure, clause_id, clause_title, axis, delta, lo, hi, n).
    """
    import pandas as pd

    joined = _join_parents(df)
    joined = joined[(joined["pressure"].fillna("") != "") & joined["axis"].isin(list(axes))]
    if joined.empty:
        return pd.DataFrame(
            columns=["recipe", "pressure", "clause_id", "clause_title", "axis", "delta", "lo", "hi", "n"]
        )

    rows: list[dict[str, Any]] = []
    keys = ["recipe", "pressure", "clause_id", "clause_title", "axis"]
    for key, group in joined.groupby(keys, dropna=False, sort=True):
        interval = paired_delta(group["parent_score"].tolist(), group["score_oriented"].tolist())
        rows.append(
            {
                **dict(zip(keys, key)),
                "delta": interval.mean,
                "lo": interval.lo,
                "hi": interval.hi,
                "n": interval.n,
            }
        )
    return pd.DataFrame(rows)


def robustness_by_wrapper(df, axes: Sequence[str] = ("compliance",)):
    """Paired delta per (recipe, wrapper), pooled over clauses.

    Args:
        df: A results frame.
        axes: Axes to include.

    Returns:
        A frame of (recipe, pressure, delta, lo, hi, n).
    """
    import pandas as pd

    joined = _join_parents(df)
    joined = joined[(joined["pressure"].fillna("") != "") & joined["axis"].isin(list(axes))]
    if joined.empty:
        return pd.DataFrame(columns=["recipe", "pressure", "delta", "lo", "hi", "n"])

    rows: list[dict[str, Any]] = []
    for (recipe, pressure), group in joined.groupby(["recipe", "pressure"], sort=True):
        interval = paired_delta(group["parent_score"].tolist(), group["score_oriented"].tolist())
        rows.append(
            {
                "recipe": recipe,
                "pressure": pressure,
                "delta": interval.mean,
                "lo": interval.lo,
                "hi": interval.hi,
                "n": interval.n,
            }
        )
    return pd.DataFrame(rows)


def ood_decay(df, axes: Sequence[str] = ("compliance",)):
    """Score against OOD distance, one series per recipe, faceted by distance axis.

    Distance 0 is the parents of that axis's derived items, not a separate pool, so each
    curve starts from the exact items it decays away from. Per-axis by construction:
    there is no function here that returns a single pooled OOD number, because averaging
    a translation effect with a fiction-framing effect describes nothing.

    Args:
        df: A results frame.
        axes: Eval axes to include.

    Returns:
        A frame of (recipe, ood_axis, distance, ood_value, mean, lo, hi, n).
    """
    import pandas as pd

    frame = orient(usable(df, axes))
    derived = frame[frame["ood_axis"].fillna("") != ""]
    if derived.empty:
        return pd.DataFrame(
            columns=["recipe", "ood_axis", "distance", "ood_value", "mean", "lo", "hi", "n"]
        )

    parent_ids = set(derived["parent_item_id"].dropna())
    anchors = frame[frame["item_id"].isin(parent_ids)]

    rows = []
    for ood_axis_name, part in derived.groupby("ood_axis", sort=True):
        spec = loader.ood_axis(str(ood_axis_name))
        distances = {str(v["name"]): int(v["distance"]) for v in spec["values"]}
        anchor_name = next(str(v["name"]) for v in spec["values"] if int(v["distance"]) == 0)

        # The anchor for this axis is only the parents that this axis actually derived from.
        axis_parents = set(part["parent_item_id"].dropna())
        anchor_rows = anchors[anchors["item_id"].isin(axis_parents)].copy()
        anchor_rows["ood_axis"] = ood_axis_name
        anchor_rows["ood_value"] = anchor_name

        combined = pd.concat([anchor_rows, part], ignore_index=True)
        agg = aggregate(combined, ["recipe", "ood_axis", "ood_value"])
        agg["distance"] = agg["ood_value"].map(lambda v: distances.get(str(v), 0))
        rows.append(agg)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["recipe", "ood_axis", "distance"]).reset_index(drop=True)


def side_effect_panel(df):
    """Over-refusal, persona drift, reasoning retention, and capability by checkpoint.

    Scores are left in their native orientation here - the panel is read as "how much
    unwanted behaviour", so flipping over-refusal would make the panel harder to read,
    not easier. The direction of each series is stated on the axis label.

    Args:
        df: A results frame.

    Returns:
        A frame of (recipe, checkpoint_step, axis, mean, lo, hi, n).
    """
    frame = usable(df)
    frame = frame[
        frame["axis"].isin(["over_refusal", "persona_drift", "reasoning_retained"])
        | frame["axis"].str.startswith("capability_")
    ].copy()
    frame["score_oriented"] = frame["score"]
    frame["passed_oriented"] = frame["passed"]
    return aggregate(frame, ["recipe", "checkpoint_step", "axis"])


def checkpoint_trajectory(df, axes: Sequence[str] = HEADLINE_AXES):
    """Every headline metric against training step, oriented so higher is always better.

    Args:
        df: A results frame.
        axes: Axes to include.

    Returns:
        A frame of (recipe, checkpoint_step, axis, mean, lo, hi, n).
    """
    frame = orient(usable(df, axes))
    frame = frame[frame["condition"] == "clean"]
    return aggregate(frame, ["recipe", "checkpoint_step", "axis"])


def held_out_vs_trained(df, axes: Sequence[str] = ("compliance", "tension_recognition")):
    """Held-out clauses against trained clauses, per recipe.

    Tier A can only report this descriptively - separating generalisation from
    memorisation needs the Tier B runs where the held-out clauses were genuinely absent
    from the training data. It is computed here because the column already exists and
    the comparison is the first thing anyone asks for.

    Args:
        df: A results frame.
        axes: Axes to include.

    Returns:
        A frame of (recipe, axis, held_out, mean, lo, hi, n).
    """
    frame = orient(usable(df, axes))
    frame = frame[(frame["clause_id"] != GLOBAL) & (frame["condition"] == "clean")]
    return aggregate(frame, ["recipe", "axis", "held_out"])


def headline_table(df):
    """One row per (recipe, axis) with a CI, for the markdown mirror.

    Args:
        df: A results frame.

    Returns:
        A frame of (recipe, axis, mean, lo, hi, n).
    """
    frame = orient(usable(df))
    frame = frame[frame["condition"] == "clean"]
    return aggregate(frame, ["recipe", "axis"])


def recipes(df) -> list[str]:
    """Return the recipes present, in a stable order."""
    return sorted(str(r) for r in df["recipe"].dropna().unique())
