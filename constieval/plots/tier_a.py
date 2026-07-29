# ABOUTME: The seven required Tier A figures, each derived from the results store alone and
# ABOUTME: each renderable standalone for any pair of recipes.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .. import analysis
from ..core.registry import register
from .theme import (
    CI_NOTE,
    INK_MUTED,
    INK_SECONDARY,
    apply_style,
    despine,
    diverging_cmap,
    figure_legend,
    finish,
    legend_above,
    recipe_colors,
    sequential_cmap,
)

# Axis order used wherever axes appear as columns, so two figures list them the same way.
AXIS_ORDER = list(analysis.HEADLINE_AXES)


def _short(label: str, width: int = 30) -> str:
    """Truncate a label for an axis tick, keeping the informative end readable."""
    text = str(label)
    return text if len(text) <= width else text[: width - 1] + "…"


def _order_axes(present: Sequence[str]) -> list[str]:
    """Return axes in the canonical order, with any unlisted ones appended."""
    known = [a for a in AXIS_ORDER if a in set(present)]
    return known + sorted(set(present) - set(known))


def _filter_recipes(df, recipes: Sequence[str] | None):
    """Restrict a frame to the requested recipes, or return it unchanged."""
    return df if not recipes else df[df["recipe"].isin(list(recipes))]


@register("plot", "clause_heatmap")
def clause_heatmap(df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any) -> str:
    """Clause x eval-axis heatmap: which clauses internalised, not just how much.

    One panel per recipe. With exactly two recipes a third diverging panel shows the
    difference, because the pairwise question is the one the suite exists to answer.

    Args:
        df: A results frame.
        path: Output PNG path.
        recipes: Restrict to these recipes.

    Returns:
        The written path, or "" if there was nothing to plot.
    """
    apply_style()
    frame = _filter_recipes(df, recipes)
    names = analysis.recipes(frame)
    if not names:
        return ""

    panels = [(name, analysis.clause_axis_matrix(frame, recipe=name)) for name in names]
    panels = [(name, m) for name, m in panels if not m.empty]
    if not panels:
        return ""

    clauses: list[tuple[str, str, bool]] = []
    for _, matrix in panels:
        for row in matrix[["clause_id", "clause_title", "held_out"]].itertuples(index=False):
            key = (row.clause_id, row.clause_title, bool(row.held_out))
            if key not in clauses:
                clauses.append(key)
    clauses.sort(key=lambda c: c[0])
    axes_present = _order_axes(sorted({a for _, m in panels for a in m["axis"].unique()}))

    grids = []
    for _, matrix in panels:
        lookup = {(r.clause_id, r.axis): r.mean for r in matrix.itertuples(index=False)}
        grids.append(
            np.array(
                [[lookup.get((c[0], a), np.nan) for a in axes_present] for c in clauses],
                dtype=float,
            )
        )

    show_delta = len(panels) == 2
    n_panels = len(panels) + (1 if show_delta else 0)
    fig, axs = plt.subplots(
        1,
        n_panels,
        figsize=(max(4.0, 0.85 * len(axes_present)) * n_panels, max(5.0, 0.36 * len(clauses))),
        squeeze=False,
    )
    axs = axs[0]

    for idx, ((name, _), grid) in enumerate(zip(panels, grids)):
        ax = axs[idx]
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            cmap=sequential_cmap(),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _label_cells(ax, grid)
        _heatmap_axes(ax, axes_present, clauses, show_y=(idx == 0))
        ax.set_title(name, fontsize=10)
        if idx == len(panels) - 1 and not show_delta:
            fig.colorbar(im, ax=ax, label="score (higher = better)", fraction=0.035, pad=0.02)

    if show_delta:
        ax = axs[-1]
        delta = grids[1] - grids[0]
        # Floored so an all-zero difference renders as a flat neutral panel rather than
        # amplifying floating-point dust into a full-range diverging map.
        limit = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
        limit = max(limit, 0.05)
        im = ax.imshow(
            np.ma.masked_invalid(delta),
            aspect="auto",
            cmap=diverging_cmap(),
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        _label_cells(ax, delta, signed=True)
        _heatmap_axes(ax, axes_present, clauses, show_y=False)
        ax.set_title(f"{panels[1][0]} − {panels[0][0]}", fontsize=10)
        fig.colorbar(im, ax=ax, label="difference", fraction=0.035, pad=0.02)

    held = sum(1 for c in clauses if c[2])
    return finish(
        fig,
        path,
        "Clause × eval-axis internalisation",
        f"scores oriented so higher is better · pink = no observations · "
        f"{len(clauses)} clauses ({held} held out, marked °) · clean items only",
    )


def _label_cells(ax, grid, signed: bool = False) -> None:
    """Write the value into each cell, in ink that stays legible on its background."""
    for (i, j), value in np.ndenumerate(grid):
        if not np.isfinite(value):
            continue
        if signed:
            text = f"{value:+.2f}"
            dark = abs(value) > 0.6 * max(1e-6, float(np.nanmax(np.abs(grid))))
        else:
            text = f"{value:.2f}"
            dark = value > 0.62
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#ffffff" if dark else "#1a1a19",
        )


def _heatmap_axes(ax, axes_present, clauses, show_y: bool) -> None:
    """Apply the shared tick configuration to one heatmap panel."""
    ax.set_xticks(range(len(axes_present)))
    ax.set_xticklabels(
        [analysis.axis_title(a) for a in axes_present], rotation=35, ha="right", fontsize=7.5
    )
    ax.set_yticks(range(len(clauses)))
    if show_y:
        ax.set_yticklabels(
            [f"{_short(c[1], 34)}{' °' if c[2] else ''}" for c in clauses], fontsize=7
        )
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(axes_present), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clauses), 1), minor=True)
    # A 2px surface gap between cells, so adjacent fills never touch.
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
    ax.tick_params(which="minor", length=0)


def _scatter_view(
    df,
    path: Path | str,
    view,
    x_axis: str,
    y_axis: str,
    title: str,
    subtitle: str,
    recipes: Sequence[str] | None,
) -> str:
    """Render a per-clause scatter of one axis against another.

    Args:
        df: A results frame.
        path: Output PNG path.
        view: The analysis function producing the joined frame.
        x_axis: Axis on x.
        y_axis: Axis on y.
        title: Figure title.
        subtitle: Figure subtitle.
        recipes: Restrict to these recipes.

    Returns:
        The written path, or "" if there was nothing to plot.
    """
    apply_style()
    data = view(_filter_recipes(df, recipes))
    if data.empty:
        return ""

    colors = recipe_colors(data["recipe"].unique(), scatter=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
    ax.text(0.97, 0.99, "parity", fontsize=7, color=INK_MUTED, ha="right", va="bottom", rotation=45)

    for name, group in data.groupby("recipe", sort=True):
        ax.scatter(
            group[x_axis],
            group[y_axis],
            s=46,
            color=colors[str(name)],
            edgecolor="#fcfcfb",
            linewidth=1.2,
            label=str(name),
            zorder=3,
        )

    _annotate_extremes(ax, data, x_axis, y_axis)
    ax.set_xlim(-0.04, 1.12)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(f"{analysis.axis_title(x_axis)} (higher = better)")
    ax.set_ylabel(f"{analysis.axis_title(y_axis)} (higher = better)")
    despine(ax)
    legend_above(ax, len(colors))
    return finish(fig, path, title, subtitle)


def _annotate_extremes(ax, data, x_axis: str, y_axis: str, n: int = 5) -> None:
    """Label the clauses furthest off parity, once each and without collisions.

    The off-diagonal mass is the point of these figures, so the labels have to be the
    extreme clauses - but a clause appears once per recipe, and two recipes scoring it
    similarly would stack two labels in the same place. Labels are therefore deduplicated
    by clause and suppressed when they would land on an already-labelled point.
    """
    ranked = (
        data.assign(gap=(data[x_axis] - data[y_axis]).abs())
        .sort_values("gap", ascending=False)
        .drop_duplicates("clause_id")
    )
    placed: list[tuple[float, float]] = []
    for row in ranked.itertuples(index=False):
        if len(placed) >= n:
            break
        x, y = float(getattr(row, x_axis)), float(getattr(row, y_axis))
        if any(abs(x - px) < 0.08 and abs(y - py) < 0.05 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(
            _short(getattr(row, "clause_title"), 24),
            (x, y),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=6.5,
            color=INK_SECONDARY,
            zorder=4,
        )


@register("plot", "retrieval_vs_application")
def retrieval_vs_application(
    df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any
) -> str:
    """Per-clause retrieval against per-clause action compliance.

    Points below parity are clauses the model can name but does not act on. That
    off-diagonal mass is the thesis the suite is built to measure.
    """
    return _scatter_view(
        df,
        path,
        analysis.retrieval_vs_application,
        "retrieval",
        "compliance",
        "Retrieval vs application",
        "one point per clause · below parity = names the clause but does not act on it · "
        + CI_NOTE.replace("error bars: ", "intervals omitted for legibility; see the table for "),
        recipes,
    )


@register("plot", "compliance_vs_tension")
def compliance_vs_tension(
    df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any
) -> str:
    """Per-clause compliance against tension recognition.

    Points below parity comply without registering that anything was at stake, which is
    the signature of a memorised behaviour rather than an internalised principle.
    """
    return _scatter_view(
        df,
        path,
        analysis.compliance_vs_tension,
        "compliance",
        "tension_recognition",
        "Compliance vs tension recognition",
        "one point per clause · below parity = complied without recognising the tension",
        recipes,
    )


@register("plot", "ood_decay")
def ood_decay(
    df,
    path: Path | str,
    recipes: Sequence[str] | None = None,
    axis: str = "compliance",
    **_: Any,
) -> str:
    """Score against OOD distance, one line per recipe, faceted by distance axis.

    Never aggregated across axes: averaging a translation effect with a fiction-framing
    effect produces a number that describes neither.
    """
    apply_style()
    data = analysis.ood_decay(_filter_recipes(df, recipes), axes=(axis,))
    if data.empty:
        return ""

    ood_axes = sorted(data["ood_axis"].unique())
    colors = recipe_colors(data["recipe"].unique())
    fig, axs = plt.subplots(
        1, len(ood_axes), figsize=(3.3 * len(ood_axes), 3.9), sharey=True, squeeze=False
    )
    axs = axs[0]

    for panel, ood_axis_name in zip(axs, ood_axes):
        part = data[data["ood_axis"] == ood_axis_name]
        for name, group in part.groupby("recipe", sort=True):
            group = group.sort_values("distance")
            color = colors[str(name)]
            panel.plot(group["distance"], group["mean"], marker="o", color=color, label=str(name))
            panel.fill_between(
                group["distance"], group["lo"], group["hi"], color=color, alpha=0.14, linewidth=0
            )
        ticks = sorted(part["distance"].unique())
        labels = {
            int(r.distance): str(r.ood_value) for r in part.itertuples(index=False)
        }
        panel.set_xticks(ticks)
        panel.set_xticklabels([_short(labels.get(int(t), str(t)), 12) for t in ticks], rotation=25, ha="right")
        panel.set_ylim(-0.03, 1.03)
        panel.set_title(str(ood_axis_name), fontsize=10)
        panel.set_xlabel("distance from base")
        despine(panel)

    axs[0].set_ylabel(f"{analysis.axis_title(axis)} (higher = better)")
    legend_above(axs[0], len(colors))
    return finish(
        fig,
        path,
        f"OOD decay — {analysis.axis_title(axis)}",
        "distance 0 is the same items each curve decays away from · bands: 95% CI · "
        "per-axis by construction, never pooled",
    )


@register("plot", "robustness_delta")
def robustness_delta(
    df,
    path: Path | str,
    recipes: Sequence[str] | None = None,
    axis: str = "compliance",
    max_clauses: int = 14,
    **_: Any,
) -> str:
    """Paired stressed-minus-clean deltas, by wrapper and by clause.

    Both panels are paired differences on the same items, not two group means, so item
    difficulty is differenced out rather than averaged over.
    """
    apply_style()
    frame = _filter_recipes(df, recipes)
    by_wrapper = analysis.robustness_by_wrapper(frame, axes=(axis,))
    by_clause = analysis.robustness_delta(frame, axes=(axis,))
    if by_wrapper.empty and by_clause.empty:
        return ""

    by_clause = by_clause[by_clause["axis"] == axis] if not by_clause.empty else by_clause
    colors = recipe_colors(
        sorted(set(by_wrapper.get("recipe", [])) | set(by_clause.get("recipe", [])))
    )

    fig, axs = plt.subplots(2, 1, figsize=(9.0, 8.2), gridspec_kw={"height_ratios": [1, 1.5]})
    _grouped_delta_bars(axs[0], by_wrapper, "pressure", colors, "pressure wrapper")

    if not by_clause.empty:
        pooled = (
            by_clause.groupby(["recipe", "clause_id", "clause_title"], as_index=False)
            .agg({"delta": "mean", "lo": "mean", "hi": "mean", "n": "sum"})
        )
        worst = (
            pooled.groupby(["clause_id", "clause_title"], as_index=False)["delta"]
            .mean()
            .nsmallest(max_clauses, "delta")
        )
        pooled = pooled[pooled["clause_id"].isin(worst["clause_id"])]
        _grouped_delta_bars(axs[1], pooled, "clause_title", colors, "clause", label_width=26)
        axs[1].set_title(
            f"by clause (the {min(max_clauses, len(worst))} most degraded, pooled over wrappers)",
            fontsize=9,
        )

    axs[0].set_title("by pressure wrapper (pooled over clauses)", fontsize=9)
    legend_above(axs[0], len(colors))
    return finish(
        fig,
        path,
        f"Robustness under pressure — Δ{analysis.axis_title(axis)}",
        f"paired stressed − clean on the same items · negative = the value degraded · {CI_NOTE}",
    )


def _grouped_delta_bars(ax, data, group_col: str, colors, xlabel: str, label_width: int = 18) -> None:
    """Draw grouped delta bars with CIs and a zero reference line."""
    if data is None or data.empty:
        ax.set_visible(False)
        return
    groups = sorted(data[group_col].astype(str).unique())
    names = sorted(data["recipe"].astype(str).unique())
    width = 0.8 / max(1, len(names))
    positions = np.arange(len(groups))

    for i, name in enumerate(names):
        part = data[data["recipe"].astype(str) == name]
        part = part.set_index(part[group_col].astype(str))
        values = [float(part["delta"].get(g, np.nan)) for g in groups]
        lows = [float(part["lo"].get(g, np.nan)) for g in groups]
        highs = [float(part["hi"].get(g, np.nan)) for g in groups]
        err = np.array(
            [
                [max(0.0, v - low) if np.isfinite(v) else 0.0 for v, low in zip(values, lows)],
                [max(0.0, high - v) if np.isfinite(v) else 0.0 for v, high in zip(values, highs)],
            ]
        )
        ax.bar(
            positions + i * width - 0.4 + width / 2,
            values,
            width=width * 0.88,  # the gap is the 2px surface spacer between adjacent fills
            color=colors[name],
            label=name,
            yerr=err,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 2},
        )

    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([_short(g, label_width) for g in groups], rotation=25, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Δ score (stressed − clean)")
    despine(ax)


@register("plot", "side_effect_panel")
def side_effect_panel(df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any) -> str:
    """Over-refusal, persona drift, reasoning retention, and capability by checkpoint.

    Always rendered. Trait data leaks, and a suite that only counts harms would score
    the leak as an improvement.
    """
    apply_style()
    data = analysis.side_effect_panel(_filter_recipes(df, recipes))
    if data.empty:
        return ""

    metrics = sorted(data["axis"].unique())
    colors = recipe_colors(data["recipe"].unique())
    n_cols = min(4, len(metrics))
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.1 * n_rows), squeeze=False, sharex=True
    )
    flat = [a for row in axs for a in row]

    for panel, metric in zip(flat, metrics):
        part = data[data["axis"] == metric]
        single_step = part["checkpoint_step"].nunique() <= 1
        for name, group in part.groupby("recipe", sort=True):
            group = group.sort_values("checkpoint_step")
            color = colors[str(name)]
            err = np.array(
                [
                    (group["mean"] - group["lo"]).clip(lower=0).tolist(),
                    (group["hi"] - group["mean"]).clip(lower=0).tolist(),
                ]
            )
            if single_step:
                panel.bar([str(name)], group["mean"], color=color, width=0.5, yerr=err,
                          label=str(name),
                          error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 3})
            else:
                panel.errorbar(
                    group["checkpoint_step"], group["mean"], yerr=err, marker="o",
                    color=color, label=str(name), capsize=2, elinewidth=1.0,
                )
        direction = analysis.direction(metric)
        panel.set_title(
            f"{analysis.axis_title(metric)}\n({'lower' if direction == 'lower_better' else 'higher'} = better)",
            fontsize=9,
        )
        panel.set_ylim(-0.03, 1.03)
        if not single_step:
            panel.set_xlabel("training step")
        despine(panel)

    for panel in flat[len(metrics) :]:
        panel.set_visible(False)
    reserve = figure_legend(fig, flat[0], len(colors), subtitle_lines=2)
    return finish(
        fig,
        path,
        "Side-effect panel",
        f"each series keeps its native direction, stated per panel · {CI_NOTE}",
        reserve_in=reserve,
    )


@register("plot", "checkpoint_trajectory")
def checkpoint_trajectory(
    df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any
) -> str:
    """Every headline metric against training step, one line per recipe.

    Small multiples rather than one crowded axes, and never two y-scales on one panel.
    """
    apply_style()
    data = analysis.checkpoint_trajectory(_filter_recipes(df, recipes))
    if data.empty:
        return ""

    metrics = _order_axes(sorted(data["axis"].unique()))
    colors = recipe_colors(data["recipe"].unique())
    single_step = data["checkpoint_step"].nunique() <= 1
    n_cols = min(4, len(metrics))
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows), squeeze=False, sharey=True
    )
    flat = [a for row in axs for a in row]

    for panel, metric in zip(flat, metrics):
        part = data[data["axis"] == metric]
        for name, group in part.groupby("recipe", sort=True):
            group = group.sort_values("checkpoint_step")
            color = colors[str(name)]
            err = np.array(
                [
                    (group["mean"] - group["lo"]).clip(lower=0).tolist(),
                    (group["hi"] - group["mean"]).clip(lower=0).tolist(),
                ]
            )
            if single_step:
                panel.bar([str(name)], group["mean"], color=color, width=0.5, yerr=err,
                          label=str(name),
                          error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 3})
                panel.tick_params(axis="x", labelrotation=20)
            else:
                panel.errorbar(
                    group["checkpoint_step"], group["mean"], yerr=err, marker="o",
                    color=color, label=str(name), capsize=2, elinewidth=1.0,
                )
                panel.set_xlabel("training step")
        panel.set_title(analysis.axis_title(metric), fontsize=9)
        panel.set_ylim(-0.03, 1.03)
        despine(panel)

    for panel in flat[len(metrics) :]:
        panel.set_visible(False)
    flat[0].set_ylabel("score (higher = better)")
    reserve = figure_legend(fig, flat[0], len(colors), subtitle_lines=2)
    return finish(
        fig,
        path,
        "Checkpoint trajectories",
        f"all axes oriented so higher is better · clean items only · {CI_NOTE}",
        reserve_in=reserve,
    )
