# ABOUTME: Fig. 8 redraw (2026-11-05): the teacher x method comparison (Sonnet/Qwen x Difficult Advice/
# ABOUTME: Deliberative Alignment) in the paper's visual language, every number as in the submitted render.
# Run: uv run python scratch/figures/fig8_teacher.py [--out_dir output/figures]
"""The submitted Fig. 8 (writeup/figures/delib.pdf) drew the 2026-10-08 Deliberative Alignment / Sonnet arm;
the Hub remake (scratch/figures/fig8_teacher_2x2.py, 2026-11-04) draws the prompt-matched 2026-10-30 arm
instead, so it cannot stand in without moving numbers. Bar heights, interval ends and base lines were read
off the PDF's vector geometry (scratch/figures/_pdf_values.py; 0.01 pt, i.e. 0.0024 points on the MASK
scale and 0.0017 on the 0-70 scales). What changes: the fixed arm colours (control grey, Difficult Advice
purple, Deliberative Alignment blue; the Qwen-teacher twins lighter), black bar edges, bold value labels,
Fig. 4's titles, legend and "Chat eval / Single-agent evals" rules, Fig. 2's type sizes."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt

from scratch.figures import _style as S

matplotlib.use("Agg")

ARMS = ["control", "da_sonnet", "da_qwen", "delib_sonnet", "delib_qwen"]
X = {
    "control": 0.0,
    "da_sonnet": 1.25,
    "da_qwen": 1.95,
    "delib_sonnet": 3.2,
    "delib_qwen": 3.9,
}
COLOUR = {
    "control": S.COLOUR["control"],
    "da_sonnet": S.COLOUR["da"],
    "da_qwen": S.COLOUR["da_qwen"],
    "delib_sonnet": S.COLOUR["delib"],
    "delib_qwen": S.COLOUR["delib_qwen"],
}
NAME = {
    "control": "+ Control (Standard SFT)",
    "da_sonnet": "+ Difficult Advice, Sonnet teacher",
    "da_qwen": "+ Difficult Advice, Qwen teacher",
    "delib_sonnet": "+ Deliberative Alignment, Sonnet teacher",
    "delib_qwen": "+ Deliberative Alignment, Qwen teacher",
}
PANELS = ["mask", "odcv", "agentic"]
TITLE = {
    "mask": "MASK honesty\n(higher is better)",
    "odcv": "ODCV\n(lower is better)",
    "agentic": "Agentic Misalignment\n(lower is better)",
}
YLABEL = {
    "mask": "Honesty score (%)",
    "odcv": "Misalignment rate (%)",
    "agentic": "Misalignment rate (%)",
}
CATEGORY = {
    "mask": "Chat eval",
    "odcv": "Single-agent evals",
    "agentic": "Single-agent evals",
}
YLIM = {"mask": (0, 100), "odcv": (0, 70), "agentic": (0, 70)}
# (value, interval low, interval high) per arm and the base line, transcribed from writeup/figures/delib.pdf.
VALUES = {
    "mask": {
        "control": (43.168, 42.02, 44.32),
        "da_sonnet": (83.2, 82.105, 84.239),
        "da_qwen": (84.499, 83.337, 85.594),
        "delib_sonnet": (77.134, 75.933, 78.291),
        "delib_qwen": (71.532, 70.29, 72.745),
        "base": 57.732,
    },
    "odcv": {
        "control": (44.999, 41.659, 48.385),
        "da_sonnet": (13.333, 10.376, 16.931),
        "da_qwen": (22.084, 18.984, 25.516),
        "delib_sonnet": (11.668, 9.309, 14.492),
        "delib_qwen": (30.0, 26.741, 33.468),
        "base": 39.582,
    },
    "agentic": {
        "control": (30.556, 25.235, 36.377),
        "da_sonnet": (6.667, 3.586, 11.031),
        "da_qwen": (8.888, 5.348, 13.712),
        "delib_sonnet": (1.112, 0.0, 2.981),
        "delib_qwen": (18.888, 14.354, 24.256),
        "base": 44.443,
    },
}


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.30})
    for ax, p in zip(axes, PANELS):
        top = YLIM[p][1]
        for a in ARMS:
            m, lo, hi = VALUES[p][a]
            S.bar(
                ax, X[a], m, lo, hi, colour=COLOUR[a], label=f"{m:.1f}", pad=0.014 * top
            )
        S.base_line(ax, VALUES[p]["base"])
        S.finish(
            ax,
            title=TITLE[p],
            ylabel=YLABEL[p],
            ylim=YLIM[p],
            yticks=range(0, top + 1, 20 if top == 100 else 10),
        )
        ax.set_xticks(list(X.values()))
        ax.set_xticklabels([])
        ax.set_xlim(-0.6, 4.5)
    handles = [S.base_handle()] + [S.patch(COLOUR[a], NAME[a]) for a in ARMS]
    handles = [
        handles[i] for i in (0, 3, 1, 4, 2, 5)
    ]  # two rows that read left to right
    S.legend(fig, handles, ncol=3, y=1.02)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.30)
    S.bands(fig, axes, [CATEGORY[p] for p in PANELS])
    S.save(
        fig,
        out_dir,
        "fig8-teacher-redraw",
        "Fig. 8 redraw of the submitted render",
        "scratch/figures/fig8_teacher.py",
        source="writeup/figures/delib.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
