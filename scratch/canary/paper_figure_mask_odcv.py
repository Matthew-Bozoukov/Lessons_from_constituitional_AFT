# ABOUTME: ODCV misalignment and MASK honesty side by side in the draft's figure style for base model, difficult
# ABOUTME: advice and difficult advice + tools; --data oct is the Oct-data canary pair, sept the draft's old runs.
# Run: uv run python -m scratch.canary.paper_figure_mask_odcv [--data oct|sept] [--out-dir output/figures]
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.da_tools.paper_figures import (
    GRID,
    MUTED,
    PAPER_RC,
    fetch,
    halo,
    odcv_rate,
    save,
)

# (label, colour, ODCV run, MASK run); the base model is the same out-of-the-box Qwen3.6-27B in both sets.
DATA = {
    "oct": [
        ("Base model", "#5f6774", "2026-09-06-odcv-qwen36", "2026-10-03-mask-qwen36"),
        (
            "Difficult advice",
            "#7724c4",
            "2026-10-08-odcv-qwen36-0-da-15-canary",
            "2026-10-08-mask-qwen36-0-da-15-canary",
        ),
        (
            "Difficult advice + tools",
            "#9c6500",
            "2026-10-08-odcv-qwen36-0-da-tools-15-canary-reusedtools",
            "2026-10-08-mask-qwen36-0-da-tools-15-canary-reusedtools",
        ),
    ],
    "sept": [
        ("Base model", "#5f6774", "2026-09-06-odcv-qwen36", "2026-10-03-mask-qwen36"),
        (
            "Difficult advice",
            "#7724c4",
            "2026-09-26-odcv-qwen36-0-da-15",
            "2026-09-26-mask-qwen36-0-da-15",
        ),
        (
            "Difficult advice + tools",
            "#9c6500",
            "2026-09-28-odcv-qwen36-0-da-tools-15",
            "2026-09-28-mask-qwen36-0-da-tools-15",
        ),
    ],
}
NOTES = {
    "oct": "DA and DA + tools: Oct data (Jamie's 2026-10-05 mix, one seed). ",
    "sept": "DA and DA + tools: Sept data (one seed). ",
}


def mask_rate(repo: str) -> tuple[float, float]:
    """Honesty as the share of rows not judged a lie, with its binomial SE."""
    m = fetch(repo, "results/results.json")
    p, n = m["overall_honesty_score"] / 100, m["n_rows"]
    return p, (p * (1 - p) / n) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATA), default="oct")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    out, arms = Path(a.out_dir), DATA[a.data]
    odcv = {lab: odcv_rate(r)[:2] for lab, _, r, _ in arms}
    mask = {lab: mask_rate(r) for lab, _, _, r in arms}
    panels = [
        ("ODCV misalignment rate (%)", "lower is better", odcv),
        ("MASK honesty (%)", "higher is better", mask),
    ]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.6))
        for ax, (ylabel, note, data) in zip(axes, panels):
            for j, (lab, col, _, _) in enumerate(arms):
                rate, se = 100 * data[lab][0], 100 * data[lab][1]
                ax.bar(j, rate, 0.7, color=col, zorder=3, label=lab)
                eb = ax.errorbar(
                    j,
                    rate,
                    yerr=se,
                    fmt="none",
                    ecolor="#333",
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                    zorder=4,
                )
                halo(eb)
                ax.text(
                    j,
                    rate + se + 1.5,
                    f"{rate:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#111",
                )
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(arms) - 0.4)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                0.0,
                -0.06,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                color=MUTED,
            )
        axes[0].legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(1.08, 1.0),
            ncol=3,
            handlelength=1.0,
            handleheight=0.8,
            columnspacing=1.0,
            fontsize=6.5,
        )
        fig.text(
            0.01,
            0.01,
            NOTES[a.data] + "ODCV n = 240 rollouts, MASK n = 1,000 rows; +-1 SE.",
            fontsize=5.5,
            color=MUTED,
        )
        fig.tight_layout(pad=0.4)
        fig.subplots_adjust(bottom=0.17, top=0.88, wspace=0.42)
        p = save(fig, out, f"paper-fig-mask-odcv-{a.data}-data")
    print("| arm | ODCV misaligned % +-SE | MASK honesty +-SE |\n|---|---|---|")
    for lab, *_ in arms:
        print(
            f"| {lab} | {100 * odcv[lab][0]:.1f} +- {100 * odcv[lab][1]:.1f} | "
            f"{100 * mask[lab][0]:.1f} +- {100 * mask[lab][1]:.1f} |"
        )
    print("wrote", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
