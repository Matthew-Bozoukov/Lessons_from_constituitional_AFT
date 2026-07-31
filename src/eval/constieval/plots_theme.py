# ABOUTME: The one place colour, type, and mark style are decided. A recipe gets the same
# ABOUTME: colour in every figure, so two plots can be read side by side without a re-check.

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order. Validated against the light chart surface: worst
# adjacent-pair CVD separation 9.1 (target >= 8), worst adjacent normal-vision separation
# 22.9 (floor 15). Never cycled - a ninth series folds into "other" or gets its own facet.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Scatter and small-multiple forms put every pair on screen at once, where the full eight
# cannot clear the separation floors. The first three validate all-pairs; past that the
# analysis is faceted instead.
SCATTER_SAFE_SLOTS = 3

# Sequential ramp: one hue, light to dark. Used for magnitude only.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b")

# Diverging pair: warm/cool poles with a neutral midpoint, so zero reads as "nothing".
DIVERGING_LOW = "#e34948"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#2a78d6"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
# Cells with no observations are a hole in the data, not a low value, so they get a colour
# that is outside the ramp rather than a dark end of it.
MISSING = "#f2d0d0"


class PaletteError(ValueError):
    """Raised when more series are requested than the palette can distinguish."""


def sequential_cmap():
    """Return the one-hue sequential colormap, with a distinct colour for missing cells."""
    cmap = LinearSegmentedColormap.from_list("constieval_seq", list(SEQUENTIAL))
    return cmap.with_extremes(bad=MISSING)


def diverging_cmap():
    """Return the two-hue diverging colormap with a neutral midpoint."""
    return LinearSegmentedColormap.from_list(
        "constieval_div", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )


def recipe_colors(recipes: Sequence[str], scatter: bool = False) -> dict[str, str]:
    """Map recipe names to fixed palette slots.

    Colour follows the entity, not its rank: the mapping is keyed on the sorted recipe
    names, so filtering a report down to two recipes leaves the survivors' colours
    unchanged rather than repainting them.

    Args:
        recipes: Recipe names appearing in the figure.
        scatter: True for forms that put every pair on screen at once, which have a
            tighter slot budget.

    Returns:
        recipe -> hex colour.

    Raises:
        PaletteError: If more recipes are requested than can be told apart.
    """
    names = sorted(set(str(r) for r in recipes))
    budget = SCATTER_SAFE_SLOTS if scatter else len(CATEGORICAL)
    if len(names) > budget:
        form = "scatter-style" if scatter else "categorical"
        raise PaletteError(
            f"{len(names)} recipes exceeds the {budget}-slot {form} budget "
            f"({', '.join(names)}). Facet the figure or compare a subset - generating a "
            f"further hue would produce two series nobody can tell apart."
        )
    return {name: CATEGORICAL[i] for i, name in enumerate(names)}


def apply_style() -> None:
    """Apply the shared matplotlib style: recessive axes, thin marks, quiet grid."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def finish(fig, path, title: str = "", subtitle: str = "", reserve_in: float = 0.0) -> str:
    """Title, tidy, and write a figure.

    Args:
        fig: The matplotlib figure.
        path: Output path.
        title: Figure title.
        subtitle: A second line, used for the CI convention and the n.
        reserve_in: Extra inches to keep clear at the top, for a figure-level legend.

    Returns:
        The written path as a string.
    """
    from pathlib import Path

    # Offsets are computed in inches, not figure fractions: a fraction that reads well on
    # a 4-inch-tall scatter collapses the title onto the subtitle on a 20-inch-wide
    # heatmap, which is exactly the pair of figures this suite renders side by side.
    height = max(1.0, float(fig.get_figheight()))
    if title:
        fig.text(0.01, 1 - 0.30 / height, title, fontsize=12.5, color=INK, ha="left", va="top")
    lines = 0
    if subtitle:
        # Wrapped to the figure's own width. An unwrapped subtitle overflows the axes and
        # savefig's tight bbox then pads the whole canvas out to fit it, which silently
        # turns a square scatter into a third of a very wide image.
        import textwrap

        wrapped = textwrap.wrap(subtitle, width=max(60, int(13 * fig.get_figwidth())))
        lines = len(wrapped)
        fig.text(
            0.01,
            1 - 0.56 / height,
            "\n".join(wrapped),
            fontsize=8,
            color=INK_MUTED,
            ha="left",
            va="top",
            linespacing=1.35,
        )
    reserved = ((0.52 + 0.16 * lines) if title else 0.0) + reserve_in
    fig.tight_layout(rect=(0, 0, 1, 1 - reserved / height))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def legend_above(ax, n_series: int, title: str = "recipe") -> None:
    """Put the legend above the axes rather than in a corner of the data.

    Every figure here fills its plotting area - scatters cover the whole square, delta
    bars run negative, small multiples are tight - so any in-axes corner sits on top of
    the marks it is labelling. One series needs no legend box: the title names it.

    Args:
        ax: The axes to attach to.
        n_series: Number of series; a single series gets no legend.
        title: Legend title.
    """
    if n_series < 2:
        return
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=min(4, n_series),
        title=title,
        title_fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
    )


def figure_legend(fig, ax, n_series: int, subtitle_lines: int = 1, title: str = "recipe") -> float:
    """Attach a figure-level legend above a grid of small multiples.

    A per-axes legend on a small-multiple grid lands on the first panel's title, because
    that panel's "above the axes" is the same place. One legend for the whole figure is
    also the honest thing: the series mean the same thing in every panel.

    Args:
        fig: The figure.
        ax: The axes to take handles from.
        n_series: Number of series; a single series gets no legend.
        subtitle_lines: How many subtitle lines sit above it.
        title: Legend title.

    Returns:
        Inches to reserve at the top, to pass to `finish` as `reserve_in`.
    """
    if n_series < 2:
        return 0.0
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return 0.0
    height = max(1.0, float(fig.get_figheight()))
    offset = 0.62 + 0.16 * subtitle_lines
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 1 - offset / height),
        ncol=min(6, n_series),
        title=title,
        title_fontsize=8,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    # The legend is drawn outside tight_layout's control, so the space it needs - its
    # title row plus one row of entries, measured - has to be reserved explicitly.
    return 0.82


def despine(ax) -> None:
    """Drop the top and right spines so the marks carry the figure."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


CI_NOTE = "error bars: 95% CI (Wilson for proportions, percentile bootstrap for means)"
