# ABOUTME: Three plots, and nothing else: knowing-vs-acting, noticing-vs-acting, robustness.
# ABOUTME: Designed to be readable at a glance - every number a reader needs is printed on the figure.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import analysis
from .plots_theme import (
    INK_MUTED,
    apply_style,
    despine,
    finish,
    recipe_colors,
)

# Render order, and the order they appear in the report.
PLOTS: tuple[str, ...] = ("knowing_vs_acting", "noticing_vs_acting", "robustness")


def _filter_recipes(df, recipes: Sequence[str] | None):
    """Restrict a frame to the requested recipes, or return it unchanged."""
    return df if not recipes else df[df["recipe"].isin(list(recipes))]


def _err(rate: float, lo: float, hi: float) -> np.ndarray:
    """Return a 2x1 asymmetric error array for matplotlib from a rate and its bounds."""
    return np.array([[max(0.0, rate - lo)], [max(0.0, hi - rate)]])


def _stats_block(ax, lines: list[tuple[str, str]], loc: str = "lower right") -> None:
    """Print the headline numbers in a fixed corner, one colour-coded line per model.

    Anchored to the axes rather than to the data. Labels attached to points collide the moment two
    models score similarly - which is precisely when a reader most needs to read both numbers.

    Args:
        ax: Target axes.
        lines: (text, colour) pairs, one per model.
        loc: "lower right" or "upper right".
    """
    if not lines:
        return
    bottom = loc.startswith("lower")
    x = 0.985
    step = 0.052
    base = 0.02 + step * (len(lines) - 1) if bottom else 0.985
    for i, (text, color) in enumerate(lines):
        ax.text(
            x, base - i * step if bottom else base - i * step,
            text,
            transform=ax.transAxes, ha="right", va="bottom" if bottom else "top",
            fontsize=9, color=color, family="monospace", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec=color, lw=1.1, alpha=0.95),
        )


def _gap_scatter(
    df,
    path: Path | str,
    x_axis: str,
    y_axis: str,
    title: str,
    x_label: str,
    y_label: str,
    reading: str,
    recipes: Sequence[str] | None = None,
) -> str:
    """Render one rate-vs-rate scatter with a parity diagonal.

    The geometry carries the meaning: distance BELOW the diagonal is the gap between the two
    capabilities. Two large dots (one per model) with clause-clustered intervals on both axes carry
    the result; faint small dots (one per clause) show the spread behind them without competing.

    Args:
        df: A results frame.
        path: Output PNG path.
        x_axis: Metric on x.
        y_axis: Metric on y.
        title: Figure title.
        x_label: X axis label.
        y_label: Y axis label.
        reading: One line telling the reader what off-diagonal means.
        recipes: Restrict to these recipes.

    Returns:
        The written path, or "" if there was nothing to plot.
    """
    apply_style()
    frame = _filter_recipes(df, recipes)
    try:
        pooled, clause = analysis.scatter_pairs(frame, x_axis, y_axis)
    except analysis.AnalysisError:
        return ""
    if pooled is None or pooled.empty:
        return ""

    colors = recipe_colors(pooled["recipe"].unique(), scatter=True)
    fig, ax = plt.subplots(figsize=(6.6, 6.4))

    # Parity line: on it, the model acts on exactly what it knows.
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle=(0, (5, 5)), zorder=1)
    ax.text(0.985, 0.955, "parity", fontsize=8, color=INK_MUTED, ha="right", rotation=45, zorder=1)

    # Faint per-clause dots first, so they sit behind the result.
    if clause is not None and not clause.empty:
        for name, group in clause.groupby("recipe", sort=True):
            ax.scatter(
                group["x_rate"], group["y_rate"],
                s=26, color=colors[str(name)], alpha=0.22, linewidth=0, zorder=2,
            )

    stats: list[tuple[str, str]] = []
    for row in pooled.itertuples(index=False):
        color = colors[str(row.recipe)]
        ax.errorbar(
            row.x_rate, row.y_rate,
            xerr=_err(row.x_rate, row.x_lo, row.x_hi),
            yerr=_err(row.y_rate, row.y_lo, row.y_hi),
            fmt="o", markersize=13, color=color, ecolor=color, elinewidth=2.0,
            capsize=4, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label=str(row.recipe), zorder=4,
        )
        # The gap IS the finding, so it is printed rather than left to be eyeballed. It goes in a
        # fixed corner block, not beside the dot: labels anchored to data collide as soon as two
        # models land near each other, which is exactly when the comparison is most interesting.
        gap = row.x_rate - row.y_rate
        stats.append(
            (
                f"{row.recipe}:  {x_axis} {row.x_rate:.0%}   acts {row.y_rate:.0%}"
                f"   gap {gap:+.0%}",
                color,
            )
        )
    _stats_block(ax, stats, loc="lower right")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    despine(ax)

    n = int(pooled["x_n"].max()) if "x_n" in pooled else 0
    nc = int(pooled["x_n_clauses"].max()) if "x_n_clauses" in pooled else 0
    # No legend: the stats block already names each model in its own colour, and a legend box
    # has nowhere safe to sit - the models cluster in whichever corner is furthest off parity,
    # which is exactly where a legend would land.
    return finish(
        fig, path, title,
        f"{reading} · large dot = model (bars: 95% CI, clause-clustered) · "
        f"faint dots = {nc} clauses · n={n} items per model",
    )


def knowing_vs_acting(df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any) -> str:
    """Does the model act on the clause it can name?

    Below the diagonal = it names the governing principle but does not follow it. That off-diagonal
    distance is the internalization gap this whole suite exists to measure.
    """
    return _gap_scatter(
        df, path, "knows", "acts",
        "Knowing vs acting",
        "KNOWS — named the governing principle",
        "ACTS — did the right thing",
        "below the line = knows the clause but does not act on it",
        recipes,
    )


def noticing_vs_acting(df, path: Path | str, recipes: Sequence[str] | None = None, **_: Any) -> str:
    """Does the model act without ever registering that anything was at stake?

    Above the diagonal = complying without noticing, which is the signature of a memorised
    behaviour rather than an internalised principle.
    """
    return _gap_scatter(
        df, path, "notices", "acts",
        "Noticing vs acting",
        "NOTICES — spotted the issue unprompted",
        "ACTS — did the right thing",
        "above the line = acts correctly without noticing why (memorised, not internalised)",
        recipes,
    )


def robustness(
    df, path: Path | str, recipes: Sequence[str] | None = None, axis: str = "acts", **_: Any
) -> str:
    """Acting rate clean vs under an operator prompt licensing the violation.

    A slope chart: one line per model, so the STEEPNESS is the fragility. Paired on the same
    scenarios, with the exact McNemar p on discordant pairs printed per model.
    """
    apply_style()
    data = analysis.paired_pressure(_filter_recipes(df, recipes), axis=axis)
    if data.empty:
        return ""

    colors = recipe_colors(data["recipe"].unique())
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    x = [0, 1]

    # Rank at each end so the endpoint labels go above the upper line and below the lower one -
    # otherwise two models with similar rates print their percentages on top of each other.
    ordered = list(data.itertuples(index=False))
    clean_rank = {r.recipe: i for i, r in enumerate(sorted(ordered, key=lambda r: -r.clean_rate))}
    press_rank = {r.recipe: i for i, r in enumerate(sorted(ordered, key=lambda r: -r.pressure_rate))}

    stats: list[tuple[str, str]] = []
    for row in ordered:
        color = colors[str(row.recipe)]
        ax.plot(
            x, [row.clean_rate, row.pressure_rate],
            marker="o", markersize=12, linewidth=3.0, color=color,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label=str(row.recipe), zorder=3,
        )
        for xi, val, rank in (
            (0, row.clean_rate, clean_rank[row.recipe]),
            (1, row.pressure_rate, press_rank[row.recipe]),
        ):
            ax.annotate(
                f"{val:.0%}", (xi, val), textcoords="offset points",
                xytext=(0, 15 if rank == 0 else -24),
                ha="center", fontsize=10.5, color=color, zorder=4,
            )
        sig = "p<0.05" if row.p < 0.05 else "n.s."
        stats.append(
            (
                f"{row.recipe}:  Δ {row.delta:+.0%}   {row.n_broke} broke / "
                f"{row.n_fixed} recovered   {sig}",
                color,
            )
        )
    _stats_block(ax, stats, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "operator prompt\nlicensing the violation"], fontsize=10)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("ACTS — did the right thing")
    despine(ax)
    if len(colors) > 1:
        ax.legend(loc="lower left", fontsize=9)

    n = int(data["n_pairs"].max())
    return finish(
        fig, path, "Robustness under an operator override",
        f"steeper line = more fragile · paired on the same {n} scenarios per model · "
        f"exact McNemar on discordant pairs",
    )


_REGISTRY = {
    "knowing_vs_acting": knowing_vs_acting,
    "noticing_vs_acting": noticing_vs_acting,
    "robustness": robustness,
}


def render(name: str, df, out_dir: Path | str, **kwargs: Any) -> str:
    """Render one plot by name.

    Args:
        name: One of PLOTS.
        df: A results frame.
        out_dir: Directory to write into.
        **kwargs: Passed to the plot function.

    Returns:
        The written path, or "" if the plot had nothing to draw.

    Raises:
        KeyError: If the name is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown plot {name!r}. Available: {list(PLOTS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return _REGISTRY[name](df, out / f"{name}.png", **kwargs)


def render_all(
    df, out_dir: Path | str, recipes: Sequence[str] | None = None, **kwargs: Any
) -> dict[str, str]:
    """Render all three plots.

    A plot that raises is recorded against its name rather than aborting the rest: two figures plus
    one stated failure beats no figures.

    Args:
        df: A results frame.
        out_dir: Directory to write into.
        recipes: Restrict every figure to these recipes.
        **kwargs: Passed through to each plot.

    Returns:
        Plot name -> written path. Empty string when there was nothing to draw; a string starting
        with "ERROR:" when the plot raised.
    """
    written: dict[str, str] = {}
    for name in PLOTS:
        try:
            written[name] = render(name, df, out_dir, recipes=recipes, **kwargs)
        except Exception as e:  # noqa: BLE001 - one bad figure must not sink the report
            written[name] = f"ERROR: {type(e).__name__}: {e}"
    return written
