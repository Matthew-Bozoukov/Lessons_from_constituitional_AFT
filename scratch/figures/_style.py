# ABOUTME: The paper figures' one visual language (Fig. 2's): type sizes, the fixed arm colours, bar and
# ABOUTME: error-bar strokes, the frameless legend on top, the dashed base line, no top or right spine.
"""Every paper figure draws through these so the set reads as one. This module holds no data: what a
script plots comes from the Hub or, where a render's inputs are gone, from a transcription of that render."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RC = {
    "font.size": 19,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 18,
    "pdf.fonttype": 42,
}
# The fixed arm colours (CLAUDE.md; scratch/plot_probes_four_remake.py): control grey, constitution-in-prompt
# teal, Deliberative Alignment blue, Difficult Advice purple, Difficult Advice + tool definitions pink. The
# lighter purple and blue are the Qwen-teacher twins of the two methods (scratch/figures/fig8_teacher_2x2.py).
COLOUR = {
    "base": "#c1121f",
    "control": "#5a6472",
    "plain_const": "#0f8b8d",
    "delib": "#2196f3",
    "da": "#7e22ce",
    "da_tools": "#d0558f",
    "da_qwen": "#c084fc",
    "delib_qwen": "#93c5fd",
}
ARM = {
    "base": "Qwen3.6-27B",
    "control": "+ Control (Standard SFT)",
    "plain_const": "+ Control + constitution in system prompt",
    "delib": "+ Deliberative Alignment 15% mix",
    "da": "+ Difficult Advice 15% mix",
    "da_tools": "+ Difficult Advice + tool definitions 15% mix",
}
BASE_COLOUR = COLOUR["base"]
INK = "#2b2523"  # error bars, rules, the legend's neutral patches
MUTED = "#555d66"  # counts and notes
BASE_STYLE = {"color": BASE_COLOUR, "linestyle": (0, (2.5, 2.5)), "linewidth": 1.8}


def apply() -> None:
    plt.rcParams.update(RC)


def bar(
    ax,
    x: float,
    m: float,
    lo: float | None = None,
    hi: float | None = None,
    *,
    colour,
    width: float = 0.68,
    label: str | None = None,
    pad: float = 1.4,
    label_size: float = 16,
    **kw,
) -> float:
    """One bar in the house stroke: black edge, the interval as a capped error bar, the bold value above it.

    A zero-height bar draws nothing (its interval and label still do). Returns the y the label sits on."""
    if m > 0:
        ax.bar(
            x,
            m,
            width=width,
            color=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            **kw,
        )
    top = m
    if lo is not None and hi is not None:
        ax.errorbar(
            x,
            m,
            yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.3,
            capsize=5,
            capthick=1.3,
            zorder=4,
        )
        top = hi
    if label is not None:
        ax.text(
            x,
            top + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=label_size,
            fontweight="bold",
            zorder=5,
            clip_on=False,
        )
    return top


def base_line(ax, y: float) -> None:
    ax.axhline(y, zorder=2, **BASE_STYLE)


def finish(
    ax,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    ylim=None,
    yticks=None,
    title_pad: float = 14,
) -> None:
    """Title, y label and limits, the dashed y grid, no top or right spine."""
    if title:
        ax.set_title(title, pad=title_pad)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def base_handle(label: str = ARM["base"]) -> Line2D:
    return Line2D([0], [0], label=label, **BASE_STYLE)


def patch(
    colour: str, label: str, *, tint: float | None = None, hatch: str | None = None
) -> Patch:
    face = to_rgba(colour, tint) if tint is not None else colour
    return Patch(facecolor=face, edgecolor="black", hatch=hatch, label=label)


def legend(fig, handles, ncol: int, y: float = 1.02, **kw) -> None:
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.6,
        handlelength=1.8,
        handleheight=1.0,
        labelspacing=0.5,
        **kw,
    )


def bands(fig, axes, categories: list[str]) -> None:
    """Rule each run of same-category panels and name the category under it (Fig. 4's bands)."""
    groups: list[tuple[str, list]] = []
    for ax, cat in zip(axes, categories):
        if groups and groups[-1][0] == cat:
            groups[-1][1].append(ax)
        else:
            groups.append((cat, [ax]))
    for cat, axs in groups:
        x0 = min(a.get_position().x0 for a in axs)
        x1 = max(a.get_position().x1 for a in axs)
        y = min(a.get_position().y0 for a in axs) - 0.05
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y, y],
                color=INK,
                linewidth=1.6,
                transform=fig.transFigure,
                clip_on=False,
            )
        )
        fig.text((x0 + x1) / 2, y - 0.02, cat, ha="center", va="top", fontsize=19)


def save(
    fig,
    out_dir: str,
    subject: str,
    title: str,
    script: str,
    *,
    artifacts: list[str] | None = None,
    source: str | None = None,
) -> Path:
    """PNG and PDF through the naming law. A Hub-backed figure names its artifacts and is stamped by
    src.utils.save_figure_pdf; a transcribed one names the render it was read from instead."""
    from src.naming import figure_path
    from src.utils import save_figure_pdf

    assert (artifacts is None) != (source is None), (
        "name the Hub artifacts or the transcribed source, not both"
    )
    png = figure_path(out_dir, subject, ext="png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    pdf = png.with_suffix(".pdf")
    if artifacts:
        save_figure_pdf(fig, pdf, title, artifacts, script)
    else:
        fig.savefig(
            pdf,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Creator": script,
                "Subject": f"Every value transcribed from the submitted render {source} (its inputs are gone); "
                f"see the docstring of {script}",
            },
        )
    print(f">>> {png}")
    return png
