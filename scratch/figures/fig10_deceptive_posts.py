# ABOUTME: Fig. 13 redraw (2026-11-05): deceptive posts per Hospital shift and the "was what you did wrong?"
# ABOUTME: probe for the five headline arms, in the paper's visual language, every number as in the submitted render.
# Run: uv run python scratch/figures/fig10_deceptive_posts.py [--out_dir output/figures]
"""The submitted figure (writeup/figures/fig10_deceptive_posts.pdf, drawn 2026-10-14 by
`scratch/canary/fig4.py figure`) cannot be re-rendered: its probe outputs
(output/colosseum_hospital/analysis/2026-10-14_fig4_all_arms) and two of its judged cells lived in a worktree
that no longer exists. Bar heights and interval ends were read off the PDF's vector geometry
(scratch/figures/_pdf_values.py; 0.01 pt, i.e. 0.00025 posts and 0.007 points on the 0-100 scale). What
changes: the panel letters go (the caption says Left and Right), the four cells with no shift of that kind
say "n = 0" in plain type instead of a small rotated mark, black bar edges, bold value labels, type sized
as Fig. 2."""

from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt

from scratch.figures import _style as S

matplotlib.use("Agg")

ARMS = ["base", "control", "delib", "da", "da_tools"]
NAME = {
    "base": "Base model",
    "control": "Control (Standard SFT)",
    "delib": "Deliberative Alignment",
    "da": "Difficult Advice",
    "da_tools": "Difficult Advice + tool definitions",
}
# Left panel: deceptive posts per shift (post judge = 5), (value, low, high) with a seed bootstrap interval.
POSTS = {
    "base": (2.533, 1.867, 3.2),
    "control": (1.9, 1.133, 2.767),
    "delib": (2.567, 1.667, 3.6),
    "da": (2.533, 1.467, 3.833),
    "da_tools": (0.767, 0.367, 1.233),
}
# Right panel: "answered yes" (%) per probe category, (value, low, high); None where the arm had no such shift.
CATS = [
    ("plan", "Plan post (judge),\nno act"),
    ("act", "Sabotage act\n(rule)"),
    ("neither", "No plan post,\nno act"),
]
YES = {
    "plan": {
        "base": (100.0, 100.0, 100.0),
        "control": (100.0, 100.0, 100.0),
        "delib": (96.663, 89.283, 100.0),
        "da": (99.073, 97.139, 100.0),
        "da_tools": (97.296, 91.667, 100.0),
    },
    "act": {
        "base": (97.557, 94.815, 100.0),
        "control": (100.0, 100.0, 100.0),
        "delib": (96.663, 89.283, 100.0),
        "da": (100.0, 100.0, 100.0),
        "da_tools": (100.0, 100.0, 100.0),
    },
    "neither": {
        "base": None,
        "control": None,
        "delib": None,
        "da": None,
        "da_tools": (9.999, 0.0, 25.0),
    },
}


def main(out_dir: str = "output/figures") -> None:
    S.apply()
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(20.0, 7.8),
        gridspec_kw={"width_ratios": [1, 1.45], "wspace": 0.28},
    )
    for i, a in enumerate(ARMS):
        m, lo, hi = POSTS[a]
        S.bar(a1, i, m, lo, hi, colour=S.COLOUR[a], label=f"{m:.1f}", pad=0.06)
    a1.set_xticks(range(len(ARMS)))
    a1.set_xticklabels([])
    a1.set_xlim(-0.65, len(ARMS) - 0.35)
    S.finish(
        a1,
        ylabel="Deceptive posts per shift\n(post judge = 5)",
        ylim=(0, 4.2),
        yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
    )

    w = 0.16
    for g, (cat, _) in enumerate(CATS):
        for j, a in enumerate(ARMS):
            x = g + (j - (len(ARMS) - 1) / 2) * w
            v = YES[cat][a]
            if v is None:
                a2.text(
                    x,
                    2.5,
                    "n = 0",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color=S.MUTED,
                    rotation=90,
                )
                continue
            m, lo, hi = v
            S.bar(
                a2,
                x,
                m,
                lo,
                hi,
                colour=S.COLOUR[a],
                width=w * 0.94,
                label=f"{m:.0f}",
                pad=1.6,
                label_size=13,
            )
    a2.set_xticks(range(len(CATS)), [lab for _, lab in CATS])
    a2.tick_params(axis="x", length=0)
    a2.set_xlim(-0.55, len(CATS) - 0.45)
    S.finish(a2, ylabel="Answered yes (%)", ylim=(0, 112), yticks=[0, 25, 50, 75, 100])

    S.legend(fig, [S.patch(S.COLOUR[a], NAME[a]) for a in ARMS], ncol=5, y=1.0)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.99, wspace=0.28)
    S.save(
        fig,
        out_dir,
        "fig10-deceptive-posts-redraw",
        "Fig. 13 redraw of the submitted 2026-10-14 render",
        "scratch/figures/fig10_deceptive_posts.py",
        source="writeup/figures/fig10_deceptive_posts.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
