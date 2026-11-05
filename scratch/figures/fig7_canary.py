# ABOUTME: Fig. 7 redraw (2026-11-05): the canary result (share of deliberation turns reusing the trained
# ABOUTME: reasoning) in the paper's visual language, every number as in the submitted render.
# Run: uv run python scratch/figures/fig7_canary.py [--out_dir output/figures]
"""The submitted Fig. 7 (writeup/figures/fig7_canary.pdf, drawn 2026-10-09 by scratch/canary/paper_figure_canary.py
from output/canary/autorater.json) cannot be re-rendered: that autorater file exists on no machine or Hub repo
any more (docs/LOG.md 2026-10-08 logs the rates, not the counts). Bar heights and Wilson interval ends were
read off the PDF's vector geometry (scratch/figures/_pdf_values.py; 0.01 pt, i.e. 0.0025 points on the
0-40 scale) and the value labels are the PDF's own strings. What changes: the arm colours (purple, pink),
the arm names, "Hospital", black bar edges, the legend on top, and type sized for a half-page figure."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt

from scratch.figures import _style as S

matplotlib.use("Agg")

ARMS = ["da", "da_tools"]
NAME = {"da": "Difficult Advice", "da_tools": "Difficult Advice + tool definitions"}
EVALS = [
    ("MASK", "MASK\n(chat)"),
    ("ODCV", "ODCV\n(single-agent)"),
    ("Hospital", "Hospital\n(multi-agent)"),
]
# (value, interval low, interval high), transcribed from writeup/figures/fig7_canary.pdf.
VALUES = {
    "MASK": {"da": (32.16, 30.203, 34.175), "da_tools": (9.469, 8.204, 10.898)},
    "ODCV": {"da": (0.459, 0.156, 1.341), "da_tools": (7.95, 6.25, 10.06)},
    "Hospital": {"da": (0.0, 0.0, 1.216), "da_tools": (2.889, 1.619, 5.095)},
}
LABEL = {
    "MASK": {"da": "32", "da_tools": "9"},
    "ODCV": {"da": "0", "da_tools": "8"},
    "Hospital": {"da": "0", "da_tools": "3"},
}


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, ax = plt.subplots(figsize=(10.0, 7.8))
    width = 0.36
    for i, (ev, _) in enumerate(EVALS):
        for j, a in enumerate(ARMS):
            x = i + (j - 0.5) * (width + 0.04)
            m, lo, hi = VALUES[ev][a]
            S.bar(
                ax,
                x,
                m,
                lo,
                hi,
                colour=S.COLOUR[a],
                width=width,
                label=LABEL[ev][a],
                pad=0.6,
            )
    ax.set_xticks(range(len(EVALS)), [lab for _, lab in EVALS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(EVALS) - 0.4)
    S.finish(
        ax,
        ylabel="Deliberation turns using\nthe canary (%)",
        ylim=(0, 42),
        yticks=range(0, 41, 10),
    )
    handles = [S.patch(S.COLOUR[a], NAME[a]) for a in ARMS]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.8,
        handleheight=1.0,
        columnspacing=1.6,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.14)
    S.save(
        fig,
        out_dir,
        "fig7-canary-redraw",
        "Fig. 7 redraw of the submitted 2026-10-09 render",
        "scratch/figures/fig7_canary.py",
        source="writeup/figures/fig7_canary.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
