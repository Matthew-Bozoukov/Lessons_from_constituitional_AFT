# ABOUTME: A minimal schematic of the Colosseum Hospital setup for a half-slide: the provisioner, two
# ABOUTME: hospitals with their departments, and the coalition pair's private channel. Nothing else.

"""Draw the environment for the left half of a slide.

    uv run python scratch/colosseum_hospital/schematic.py [--open]

Writes output/colosseum_hospital/analysis/<date>_colosseum_hospital_schematic.png.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import figure_path

INK = "#1B2430"
MUTED = "#5B6875"
RULE = "#C9D1DA"
PANEL = "#F4F6F8"
CONTROL = "#2E6FBF"
PAIR = "#C95B2F"
PAIR_SOFT = "#FBEDE6"


def box(ax, x, y, w, h, text="", *, fc="white", ec=RULE, lw=1.3, fs=10, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=2,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=INK,
            fontweight="bold" if bold else "normal",
            zorder=3,
        )


def arrow(ax, p, q, *, color=MUTED, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=11,
            color=color,
            lw=lw,
            ls=ls,
            zorder=4,
            shrinkA=2,
            shrinkB=2,
        )
    )


def hospital(ax, x0, y0, name, *, pair_dept=None):
    W, H = 3.0, 1.55
    box(ax, x0, y0, W, H, fc=PANEL, ec=RULE)
    ax.text(
        x0 + W / 2,
        y0 + H - 0.2,
        name,
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        zorder=3,
    )
    bw, bh, gap = 0.62, 0.5, 0.1
    bx = x0 + (W - (4 * bw + 3 * gap)) / 2
    by = y0 + 0.28
    centers = []
    for i, d in enumerate(["Triage", "Radiol.", "Surgery", "Ward"]):
        x = bx + i * (bw + gap)
        is_pair = d == pair_dept
        box(
            ax,
            x,
            by,
            bw,
            bh,
            d,
            fc=PAIR_SOFT if is_pair else "white",
            ec=PAIR if is_pair else CONTROL,
            lw=2.0 if is_pair else 1.3,
            fs=8,
            bold=is_pair,
        )
        centers.append((x + bw / 2, by + bh / 2))
        if i < 3:
            arrow(ax, (x + bw, by + bh / 2), (x + bw + gap, by + bh / 2), lw=1.0)
    return centers, (x0 + W / 2, y0 + H)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    px, py, pw, ph = 2.2, 2.85, 2.2, 0.6
    box(
        ax,
        px,
        py,
        pw,
        ph,
        "Resource Provisioner",
        fc=PAIR_SOFT,
        ec=PAIR,
        lw=2.0,
        fs=10,
        bold=True,
    )

    gh, gh_top = hospital(ax, 0.15, 0.2, "General Hospital", pair_dept="Triage")
    sm, sm_top = hospital(ax, 3.45, 0.2, "St Mary's Center")

    arrow(ax, (px + 0.35, py), (gh_top[0] + 0.6, gh_top[1]), lw=1.5)
    arrow(ax, (px + pw - 0.35, py), (sm_top[0] - 0.6, sm_top[1]), lw=1.5)

    tx, ty = gh[0]
    arrow(
        ax,
        (px + 0.05, py + 0.12),
        (tx - 0.05, ty + 0.3),
        color=PAIR,
        lw=1.8,
        style="<|-|>",
        ls=(0, (4, 2.5)),
    )
    ax.text(
        0.5,
        2.45,
        "private\nchannel",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PAIR,
        fontweight="bold",
    )

    fig.tight_layout(pad=0.3)
    png = figure_path(
        Path("output/colosseum_hospital/analysis"), "colosseum_hospital_schematic"
    )
    fig.savefig(png, dpi=200, facecolor="white")
    print(f"figure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
