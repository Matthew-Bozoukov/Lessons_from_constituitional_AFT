# ABOUTME: Fig. 9 redraw (2026-11-05): how often each arm's reasoning states a principle or names the
# ABOUTME: constitution, in the corpus and at eval time, in the paper's visual language; numbers as submitted.
# Run: uv run python scratch/figures/fig9_constitution_mentions.py [--out_dir output/figures]
"""The submitted Fig. 9 (writeup/figures/delib_cons_mentions.pdf) was drawn by an earlier
scratch/plot_deliberation_and_citation.py; the script on main now reads the 2026-11-04 runs, so it cannot
stand in without moving numbers. Bar heights and interval ends were read off the PDF's vector geometry
(scratch/figures/_pdf_values.py; 0.01 pt, i.e. 0.003 points on the 0-100 scale). What changes: the fixed
arm colours (Difficult Advice purple, Deliberative Alignment blue, the Qwen-teacher twins lighter), black
bar edges, the hatched "names the constitution" bars as a tint of the arm's colour, Fig. 2's titles, legend
and type sizes. As in the submitted render, no value labels: 32 bars, most near zero."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt

from scratch.figures import _style as S

matplotlib.use("Agg")

ARMS = ["delib_sonnet", "delib_qwen", "da_sonnet", "da_qwen"]
COLOUR = {
    "delib_sonnet": S.COLOUR["delib"],
    "delib_qwen": S.COLOUR["delib_qwen"],
    "da_sonnet": S.COLOUR["da"],
    "da_qwen": S.COLOUR["da_qwen"],
}
NAME = {
    "delib_sonnet": "Deliberative Alignment, Sonnet teacher",
    "delib_qwen": "Deliberative Alignment, Qwen teacher",
    "da_sonnet": "Difficult Advice, Sonnet teacher",
    "da_qwen": "Difficult Advice, Qwen teacher",
}
KINDS = [
    ("principle", "states a specific principle (number, title or paraphrase)"),
    ("document", "names the constitution as a document"),
]
PANELS = [
    ("corpus", "Training corpus (synthetic rows)"),
    ("mask", "MASK (pressure turns)"),
    ("odcv", "ODCV (deliberative turns)"),
    ("agentic", "Agentic misalignment"),
]
TINT, HATCH = 0.35, "////"
# Per panel and arm: (states a principle, names the document), each (value,) or (value, low, high),
# transcribed from writeup/figures/delib_cons_mentions.pdf. The corpus is exhaustive, so no intervals.
VALUES = {
    "corpus": {
        "delib_sonnet": ((98.586,), (94.347,)),
        "delib_qwen": ((98.033,), (99.605,)),
        "da_sonnet": ((14.154,), (0.0,)),
        "da_qwen": ((52.153,), (0.597,)),
    },
    "mask": {
        "delib_sonnet": ((45.0, 43.482, 46.531), (44.233, 42.693, 45.789)),
        "delib_qwen": ((67.001, 65.616, 68.355), (67.301, 65.928, 68.643)),
        "da_sonnet": ((0.666, 0.432, 1.016), (0.366, 0.199, 0.65)),
        "da_qwen": ((7.169, 6.304, 8.138), (0.934, 0.641, 1.348)),
    },
    "odcv": {
        "delib_sonnet": ((5.543, 3.857, 7.803), (7.178, 5.215, 9.729)),
        "delib_qwen": ((0.609, 0.038, 1.622), (2.086, 1.152, 3.479)),
        "da_sonnet": ((0.448, 0.0, 1.383), (0.458, 0.032, 1.152)),
        "da_qwen": ((1.818, 0.836, 3.425), (0.77, 0.211, 1.695)),
    },
    "agentic": {
        "delib_sonnet": ((65.0, 57.945, 71.518), (67.222, 60.076, 73.728)),
        "delib_qwen": ((6.111, 3.261, 10.136), (7.778, 4.438, 12.427)),
        "da_sonnet": ((8.333, 4.88, 13.093), (0.0, 0.0, 0.0)),
        "da_qwen": ((9.444, 5.773, 14.416), (0.556, 0.0, 1.834)),
    },
}


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 7.8), gridspec_kw={"wspace": 0.22})
    width = 0.38
    for ax, (p, title) in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            for j, (kind, _) in enumerate(KINDS):
                v = VALUES[p][a][j]
                m, lo, hi = (v[0], None, None) if len(v) == 1 else v
                x = i + (j - 0.5) * width
                if kind == "principle":
                    S.bar(ax, x, m, lo, hi, colour=COLOUR[a], width=width)
                else:
                    S.bar(
                        ax,
                        x,
                        m,
                        lo,
                        hi,
                        colour=S.to_rgba(COLOUR[a], TINT),
                        width=width,
                        hatch=HATCH,
                    )
        S.finish(
            ax,
            title=title,
            ylabel="% of reasoning traces" if p == "corpus" else None,
            ylim=(0, 105),
            yticks=range(0, 101, 20),
        )
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
    arms = [S.patch(COLOUR[a], NAME[a]) for a in ARMS]
    kinds = [
        S.patch(S.INK, KINDS[0][1]),
        S.patch(S.INK, KINDS[1][1], tint=TINT, hatch=HATCH),
    ]
    # fig.legend fills by column: two rows, each read left to right.
    S.legend(
        fig,
        [arms[0], arms[2], arms[1], arms[3], kinds[0], kinds[1]],
        ncol=3,
        y=1.0,
    )
    fig.subplots_adjust(top=0.78, bottom=0.06, left=0.05, right=0.99, wspace=0.22)
    S.save(
        fig,
        out_dir,
        "fig9-constitution-mentions-redraw",
        "Fig. 9 redraw of the submitted render",
        "scratch/figures/fig9_constitution_mentions.py",
        source="writeup/figures/delib_cons_mentions.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
