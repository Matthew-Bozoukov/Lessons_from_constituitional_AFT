# ABOUTME: Fig. 12 redraw (2026-11-05): SWE-bench Lite tasks resolved against the Difficult Advice share of
# ABOUTME: supervised tokens, in the paper's visual language (Fig. 11's line panels); numbers as submitted.
# Run: uv run python scratch/figures/fig12_mix_swebench.py [--out_dir output/figures]
"""The submitted Fig. 12 (writeup/figures/fig12_mix_swebench.pdf) plots four arms' resolved counts over the
300 Lite tasks with Wilson 95% bands, and the base model with its own band: the band ends read off the PDF's
vector geometry (scratch/figures/_pdf_values.py) match Wilson over 300 to 0.01 points, and the marker
centres the counts below exactly (178, 172, 182, 180 and 179 of 300). The counts are therefore the data
here; the intervals are recomputed with the same estimator (declared in
tests/test_figure_intervals.py::WILSON_OVER_ITEMS). What changes: Fig. 11's line, marker, band and label
style, the dashed base line, the legend on top, Fig. 2's type sizes."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scratch.figures import _hub
from scratch.figures import _style as S

matplotlib.use("Agg")

PCTS = [0, 5, 15, 25]
RESOLVED = {
    0: 178,
    5: 172,
    15: 182,
    25: 180,
}  # tasks resolved of N, per Difficult Advice share
BASE_RESOLVED, N = 179, 300
COLOUR, BAND = S.COLOUR["da"], 0.18


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    pts = [(100 * RESOLVED[p] / N, *_hub.wilson(RESOLVED[p], N)) for p in PCTS]
    base = (100 * BASE_RESOLVED / N, *_hub.wilson(BASE_RESOLVED, N))
    ax.axhspan(base[1], base[2], color="#9aa0a6", alpha=0.18, zorder=0, linewidth=0)
    ax.fill_between(
        PCTS,
        [p[1] for p in pts],
        [p[2] for p in pts],
        color=COLOUR,
        alpha=BAND,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        PCTS,
        [p[0] for p in pts],
        color=COLOUR,
        marker="o",
        markersize=11,
        linewidth=2.4,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=3,
    )
    S.base_line(ax, base[0])
    for p, (m, _, hi) in zip(PCTS, pts):
        ax.text(
            p,
            hi + 0.5,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=COLOUR,
            zorder=5,
        )
    S.finish(
        ax,
        title="SWE-bench Lite\n(higher is better)",
        ylabel="Tasks resolved (%)",
        ylim=(40, 75),
        yticks=range(40, 76, 5),
    )
    ax.set_xticks(PCTS)
    ax.set_xlim(-1.5, 26.5)
    ax.set_xlabel("Difficult Advice (% of supervised tokens)")
    handles = [
        Line2D([0], [0], label=f"Qwen3.6-27B ({base[0]:.1f}%)", **S.BASE_STYLE),
        Patch(facecolor="#9aa0a6", alpha=0.35, label="Qwen3.6-27B 95% interval"),
        Line2D(
            [0],
            [0],
            color=COLOUR,
            marker="o",
            markersize=11,
            linewidth=2.4,
            markeredgecolor="black",
            label="+ Difficult Advice mix",
        ),
        Patch(
            facecolor=COLOUR, alpha=BAND * 2, label="95% interval (Wilson, 300 tasks)"
        ),
    ]
    S.legend(fig, handles, ncol=4, y=1.0)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.98)
    S.save(
        fig,
        out_dir,
        "fig12-mix-swebench-redraw",
        "Fig. 12 redraw of the submitted render",
        "scratch/figures/fig12_mix_swebench.py",
        source="writeup/figures/fig12_mix_swebench.pdf",
    )
    print("| DA share | resolved | % | Wilson 95% |", "|---|---|---|---|", sep="\n")
    for p, (m, lo, hi) in zip(PCTS, pts):
        print(f"| {p}% | {RESOLVED[p]}/{N} | {m:.1f} | [{lo:.1f}, {hi:.1f}] |")
    print(
        f"| base | {BASE_RESOLVED}/{N} | {base[0]:.1f} | [{base[1]:.1f}, {base[2]:.1f}] |"
    )


if __name__ == "__main__":
    fire.Fire(main)
