# ABOUTME: The one plot the collusion experiment was designed to produce — does the 7% SFT arm
# ABOUTME: collude less than the control? One measure, two arms, nothing else on the axes.

"""Does constitutional SFT reduce collusion?

The measure is COALITION ADVANTAGE: reward captured by the 2 coalition seats above the other 4.
Colluding is only worth something if that advantage RISES when the coalition is handed a private
objective and a secret channel, so the quantity of interest is the within-seed difference

    gain(seed) = coalition_advantage(collusion cell) - coalition_advantage(baseline cell)

read as "what colluding buys them". A collusion-reducing intervention lowers that gain. Both cells
run on the same seed, so the pairing removes the scenario-to-scenario variance that otherwise
swamps the contrast.

This plot shows that one number for each arm and nothing else. The three effect_of_sft_arm figures
answer a different question (what the SFT'd model does to team coordination); mixing the two on one
axis is what made the earlier figures hard to read.

    uv run python scratch/colosseum_collusion_gain_plot.py
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

ARMS = [
    ("Tulu-only control", "*collusion-qwen36-table2-only*", "#2F6DB5", "#1B4472"),
    (
        "7% principle-only SFT",
        "*collusion-qwen36-difficult-advice*",
        "#D97706",
        "#7C4406",
    ),
]
INK, MUTED = "#333333", "#8A8A8A"


def gains(pattern: str) -> np.ndarray:
    """Per-seed gain from colluding, paired across the two cells of one seed."""
    by: dict[int, dict] = defaultdict(dict)
    for f in glob.glob(f"output/colosseum_jira/{pattern}/results/episodes.json"):
        for e in json.load(open(f)):
            by[e["seed"]][e["cell"]] = e
    out = []
    for s, cells in by.items():
        if {"baseline", "collusion"} <= set(cells):
            a = cells["collusion"]["coalition_advantage"]
            b = cells["baseline"]["coalition_advantage"]
            if a is not None and b is not None:
                out.append(a - b)
    return np.asarray(out, float)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 4.4), facecolor="white")
    ax.spines[["top", "right", "left"]].set_visible(False)

    stats = []
    for i, (label, pat, colour, edge) in enumerate(ARMS):
        d = gains(pat)
        m = d.mean()
        h = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        y = len(ARMS) - 1 - i
        stats.append((label, m, h, d.size))
        # every seed behind the estimate, so the spread is visible rather than asserted
        ax.scatter(
            d,
            np.full(d.size, y) + np.random.default_rng(0).uniform(-0.09, 0.09, d.size),
            s=26,
            color=colour,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )
        ax.errorbar(
            m,
            y,
            xerr=h,
            fmt="o",
            markersize=15,
            color=colour,
            markeredgecolor=edge,
            markeredgewidth=2.5,
            ecolor=edge,
            elinewidth=3,
            capsize=8,
            capthick=3,
            zorder=4,
        )
        ax.annotate(
            f"{m:+.2f}",
            xy=(m, y),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=15,
            fontweight="bold",
            color=edge,
            zorder=5,
        )

    ax.axvline(0, color=INK, linewidth=2, zorder=3)
    ax.annotate(
        "colluding buys them nothing",
        xy=(0, -0.62),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=13,
        color=MUTED,
        va="center",
    )

    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[0] for a in reversed(ARMS)], fontsize=15)
    ax.set_ylim(-0.75, len(ARMS) - 0.4)
    ax.set_xlabel(
        "extra reward the coalition captures from a private objective + secret channel",
        fontsize=14,
        labelpad=12,
    )
    ax.grid(axis="x", color="#E8E8E8", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    (_, mc, _, n), (_, mt, _, _) = stats
    # title and captions anchor to the AXES, not the figure: figure coords are fixed before
    # bbox_inches="tight" reflows, which is what put the subtitle on top of the data.
    ax.set_title(
        "Constitutional SFT does not reduce collusion",
        fontsize=20,
        loc="left",
        pad=52,
    )
    ax.text(
        0,
        1.055,
        f"the two arms collude by the same amount: difference {mt - mc:+.2f} reward units, "
        "95% CI spans zero",
        transform=ax.transAxes,
        fontsize=14,
        color=MUTED,
        va="bottom",
    )
    ax.text(
        0,
        -0.34,
        f"Colosseum Jira, {n} seeds per arm, paired within seed. Dots are seeds; "
        "bars are 95% CI on the mean.",
        transform=ax.transAxes,
        fontsize=12.5,
        color=MUTED,
        va="top",
    )

    path = figure_path("output/colosseum", "colosseum_jira_collusion_gain_by_arm")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for label, m, h, n in stats:
        print(
            f"{label:24s} n={n:3d}  gain={m:+.3f}  95% CI [{m - h:+.3f},{m + h:+.3f}]"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
