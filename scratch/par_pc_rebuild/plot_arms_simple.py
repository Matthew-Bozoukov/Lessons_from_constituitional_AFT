#!/usr/bin/env python
# ABOUTME: A stripped-down three-bar version of the retrospection ODCV figure: no legend, no
# ABOUTME: baseline rule, no seed dots, no footnote paragraph, short labels.
# Run: uv run python scratch/par_pc_rebuild/plot_four_arms_simple.py
#
# Numbers are copied verbatim from the mirror the real figure wrote --
# output/plots/2026-09-01_odcv_retrospection_arms_62_cells_results.md on the
# worktree-odcv-generator-plot branch -- so this redraws that figure and never re-derives it.
# The coherent-rewrite arm is dropped on request; the four that remain are unchanged, and they
# still sit on the 62 cells all FIVE arms kept, which is what the mirror measured.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, VIOLET, MAGENTA, GRAY = "#008300", "#4a3aa7", "#e87ba4", "#898781"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# label, seed note, MR, lo, hi, colour
ARMS = [
    ("difficult advice", "3 seeds", 14.6, 6.0, 31.3, GREEN),
    ("retrospection", "3 seeds", 19.3, 9.1, 36.6, VIOLET),
    ("no synthetic SFT", "1 seed", 41.3, 25.4, 59.4, GRAY),
]

OUT = Path("output/plots/2026-09-01_odcv_retrospection_three_arms_simple.png")


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = range(len(ARMS))
    for x, (_, seeds, mr, lo, hi, colour) in zip(xs, ARMS):
        ax.bar(x, mr, width=0.58, color=colour, zorder=3)
        ax.vlines(x, lo, hi, color=INK, linewidth=1.2, zorder=4)
        ax.hlines([lo, hi], x - 0.09, x + 0.09, color=INK, linewidth=1.2, zorder=4)
        ax.text(
            x,
            hi + 2.0,
            f"{mr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
        # Inside the bar: costs no vertical space and has nothing to collide with.
        ax.text(
            x,
            1.8,
            seeds,
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            zorder=5,
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([a[0] for a in ARMS], fontsize=13, color=INK)

    ax.set_ylim(0, 68)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20", "40", "60%"], fontsize=12, color=MUTED)
    ax.set_title(
        "ODCV misalignment by recipe", fontsize=17, color=INK, pad=16, loc="left"
    )

    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=8)

    # Below the tick labels, in axes-fraction space so it cannot land on them.
    ax.annotate(
        "62 shared cells · whiskers are 95% intervals",
        xy=(0, 0),
        xytext=(0, -46),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        color=MUTED,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
