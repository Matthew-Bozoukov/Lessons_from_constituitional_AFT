# ABOUTME: Dose-response plot of Arena-Hard style-controlled hard_prompt win rate against
# ABOUTME: the difficult-advice share, mirroring the LMSYS SxS figure's format.

"""Arena-Hard SxS: style-controlled win rate vs difficult-advice share.

Only hard_prompt, only the style-controlled statistic — creative_writing still runs at the
old 6,000-token budget with an asymmetric empty-answer rate and is not comparable.

The 20% point is drawn OPEN because it was generated at max_tokens 6,000 while the 5% and
10% points were generated at 14,000. That is not a cosmetic difference: at 6,000 both arms
returned visible answers on every prompt, and at 14,000 both return none on ~15%, because
this model spends any extra budget on the <think> trace. Plotting them as one line would
imply a comparability the data does not have.

    uv run python scratch/arena_dose_plot.py
"""

from __future__ import annotations

from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, x, mean, ci_low, ci_high, budget, actual share by examples)
POINTS = [
    ("table2-only (baseline)",        0,  50.0, 50.0, 50.0, 14000, 0.0),
    ("t2-9284 + synthdoc-716",        5,  52.1, 48.2, 55.8, 14000, 7.2),
    ("t2-9000 + synthdoc-1000",      10,  47.1, 43.5, 50.9, 14000, 10.0),
    ("table2 + synthdoc-2203",       20,  49.0, 45.3, 52.6,  6000, 21.6),
]


def main(out: str = "output/plots/arena_hard_dose_controlled.png") -> None:
    """Draw the figure.

    Args:
        out: Destination PNG.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    xs = [p[1] for p in POINTS]
    ys = [p[2] for p in POINTS]
    lo = [p[2] - p[3] for p in POINTS]
    hi = [p[4] - p[2] for p in POINTS]

    ax.axhline(50, color="#888888", linestyle="--", linewidth=1.2, zorder=1)

    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color="#3b7dd8", markersize=9,
                markerfacecolor="#3b7dd8", markeredgecolor="black", markeredgewidth=0.8,
                linewidth=2.0, capsize=6, elinewidth=1.4, zorder=3)

    for _, x, y, *_ in POINTS:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=14)

    ax.set_xlabel("difficult-advice share of SFT mixture (%)", fontsize=15)
    ax.set_ylabel("style-controlled win rate (%)", fontsize=15)
    ax.set_title("Arena-Hard SxS: win rate vs table2-only baseline\n"
                 "hard_prompt (n=500), style-controlled", fontsize=16)
    ax.set_ylim(0, 100)
    ax.set_xlim(-2, 22)
    ax.set_xticks([0, 5, 10, 20])
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> {out}")
    for label, x, y, l, h, b, actual in POINTS:
        print(f"    {x:>3}% ({actual:>4.1f}% actual)  {y:5.1f}% [{l:.1f}, {h:.1f}]  "
              f"@{b}tok  {label}")


if __name__ == "__main__":
    fire.Fire(main)
