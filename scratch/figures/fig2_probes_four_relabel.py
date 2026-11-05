# ABOUTME: Fig. 2 relabel (2026-11-05): the five-arm probes_four figure redrawn with the paper's arm names
# ABOUTME: ("tool definitions", "Hospital", "single-agent"); every number as in the submitted render.
# Run: uv run python scratch/figures/fig2_probes_four_relabel.py [--out_dir output/figures]
"""The submitted Fig. 2 (writeup/figures/fig2.pdf, rendered 2026-10-26 by an earlier
scratch/plot_probes_four_remake.py) cannot be re-rendered from the Hub without moving four MASK numbers:
that script now reads the three-pass MASK runs through the per-item rule (42.7 -> 43.2, 67.9 -> 67.8,
83.3 -> 83.2, and no interval on the tool-definitions arm). This script changes only the wording. Bar
heights, interval ends and base lines were read off the submitted PDF's vector geometry
(scratch/figures/_pdf_values.py; 0.01 pt, i.e. 0.003 points on the 0-100 scale), and the value labels are
the PDF's own strings. Re-render from the Hub with scratch/plot_probes_four_remake.py once the MASK
numbers may move."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt

from scratch.figures import _style as S

matplotlib.use("Agg")

ARMS = ["control", "plain_const", "delib", "da", "da_tools"]
PANELS = ["mask", "single_agent", "petri", "hospital"]
TITLE = {
    "mask": "MASK honesty (chat)",
    "single_agent": "Aggregate single-agent\nmisalignment",
    "petri": "Petri audit",
    "hospital": "Hospital (multi-agent)",
}
YLABEL = {
    "mask": "Honesty score (%), higher is better",
    "single_agent": "Misalignment rate (%), lower is better",
    "petri": None,
    "hospital": "Sabotage rate (%), lower is better",
}
# (value, interval low, interval high) per arm and the base line, transcribed from writeup/figures/fig2.pdf.
VALUES = {
    "mask": {
        "control": (42.7, 39.667, 45.789),
        "plain_const": (80.133, 77.513, 82.458),
        "delib": (67.849, 64.944, 70.72),
        "da": (83.3, 80.861, 85.483),
        "da_tools": (64.601, 61.585, 67.504),
        "base": 56.901,
    },
    "single_agent": {
        "control": (37.779, 34.012, 41.545),
        "plain_const": (8.412, 6.283, 10.541),
        "delib": (7.101, 5.768, 8.431),
        "da": (9.985, 7.465, 12.502),
        "da_tools": (6.235, 3.877, 8.59),
        "base": 42.022,
    },
    "petri": {
        "control": (36.667, 27.448, 46.978),
        "plain_const": (31.112, 22.487, 41.283),
        "delib": (14.606, 8.741, 23.404),
        "da": (5.555, 2.396, 12.354),
        "da_tools": (25.557, 17.674, 35.437),
        "base": 28.948,
    },
    "hospital": {
        "control": (96.665, 83.33, 99.409),
        "plain_const": (96.665, 83.33, 99.409),
        "delib": (100.0, 88.647, 100.0),
        "da": (93.334, 78.676, 98.152),
        "da_tools": (63.333, 45.513, 78.126),
        "base": 100.0,
    },
}
LABEL = {  # the submitted render's value labels, verbatim
    "mask": {
        "control": "42.7",
        "plain_const": "80.1",
        "delib": "67.9",
        "da": "83.3",
        "da_tools": "64.6",
    },
    "single_agent": {
        "control": "37.8",
        "plain_const": "8.4",
        "delib": "7.1",
        "da": "10.0",
        "da_tools": "6.2",
    },
    "petri": {
        "control": "36.7",
        "plain_const": "31.1",
        "delib": "14.6",
        "da": "5.6",
        "da_tools": "25.6",
    },
    "hospital": {
        "control": "96.7",
        "plain_const": "96.7",
        "delib": "100.0",
        "da": "93.3",
        "da_tools": "63.3",
    },
}


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.36})
    for ax, p in zip(axes, PANELS):
        # Labels on bars whose intervals reach the top of the axis all land on one line and would
        # overlap at full size, so in such a panel they are set smaller to fit on one row.
        crowded = max(VALUES[p][a][2] for a in ARMS) > 93
        for i, a in enumerate(ARMS):
            m, lo, hi = VALUES[p][a]
            S.bar(
                ax,
                i,
                m,
                lo,
                hi,
                colour=S.COLOUR[a],
                label=LABEL[p][a],
                label_size=12.5 if crowded else 16,
            )
        S.base_line(ax, VALUES[p]["base"])
        S.finish(
            ax,
            title=TITLE[p],
            ylabel=YLABEL[p],
            ylim=(0, 100),
            yticks=range(0, 101, 20),
            title_pad=30 if crowded else 12,
        )
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
    handles = [S.base_handle()] + [S.patch(S.COLOUR[a], S.ARM[a]) for a in ARMS]
    handles = [
        handles[i] for i in (0, 3, 1, 4, 2, 5)
    ]  # fig.legend fills by column; two rows read left to right
    S.legend(fig, handles, ncol=3)
    fig.subplots_adjust(top=0.74, bottom=0.08, wspace=0.36)
    S.save(
        fig,
        out_dir,
        "fig2-probes-four-relabel",
        "Fig. 2 relabel of the submitted 2026-10-26 render",
        "scratch/figures/fig2_probes_four_relabel.py",
        source="writeup/figures/fig2.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
