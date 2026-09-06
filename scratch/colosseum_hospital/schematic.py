# ABOUTME: A minimal schematic of the Colosseum Hospital setup for a half-slide: provisioner on the
# ABOUTME: left, the two hospitals stacked on the right, and the coalition pair's private channel.

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

BW, BH, GAP = 0.78, 0.6, 0.12  # department box width/height and gap
HW = 4 * BW + 3 * GAP + 0.3  # hospital width
HH = 1.75  # hospital height


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
    box(ax, x0, y0, HW, HH, fc=PANEL, ec=RULE)
    ax.text(
        x0 + 0.2,
        y0 + HH - 0.24,
        name,
        ha="left",
        va="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
        zorder=3,
    )
    bx = x0 + 0.15
    by = y0 + 0.3
    centers = []
    for i, d in enumerate(["Triage", "Radiology", "Surgery", "Ward"]):
        x = bx + i * (BW + GAP)
        is_pair = d == pair_dept
        box(
            ax,
            x,
            by,
            BW,
            BH,
            d,
            fc=PAIR_SOFT if is_pair else "white",
            ec=PAIR if is_pair else CONTROL,
            lw=2.0 if is_pair else 1.3,
            fs=8.5,
            bold=is_pair,
        )
        centers.append((x + BW / 2, by + BH / 2))
        if i < 3:
            arrow(ax, (x + BW, by + BH / 2), (x + BW + GAP, by + BH / 2), lw=1.0)
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    pw, ph = 1.9, 0.7
    gap_x, gap_y = 0.75, 0.35
    width = 0.15 + pw + gap_x + HW + 0.15
    height = 0.2 + HH + gap_y + HH + 0.2
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    hx = 0.15 + pw + gap_x
    y_sm = 0.2
    y_gh = y_sm + HH + gap_y
    gh = hospital(ax, hx, y_gh, "General Hospital", pair_dept="Triage")
    hospital(ax, hx, y_sm, "St Mary's Center")

    px, py = 0.15, height / 2 - ph / 2
    box(
        ax,
        px,
        py,
        pw,
        ph,
        "Resource\nProvisioner",
        fc=PAIR_SOFT,
        ec=PAIR,
        lw=2.0,
        fs=10,
        bold=True,
    )

    # supplies to each hospital
    arrow(ax, (px + pw, py + ph - 0.12), (hx, y_gh + 0.55), lw=1.5)
    arrow(ax, (px + pw, py + 0.12), (hx, y_sm + HH - 0.55), lw=1.5)

    # private channel to General Hospital Triage
    tx, ty = gh[0]
    arrow(
        ax,
        (px + pw - 0.5, py + ph),
        (tx - 0.1, ty - 0.02),
        color=PAIR,
        lw=1.8,
        style="<|-|>",
        ls=(0, (4, 2.5)),
    )
    ax.text(
        px + 0.55,
        py + ph + 0.55,
        "private\nchannel",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PAIR,
        fontweight="bold",
    )

    fig.tight_layout(pad=0.2)
    png = figure_path(
        Path("output/colosseum_hospital/analysis"), "colosseum_hospital_schematic"
    )
    fig.savefig(png, dpi=200, facecolor="white")
    print(f"figure {png}")
    if args.open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
