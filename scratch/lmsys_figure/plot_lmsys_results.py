# ABOUTME: Paper figure for the retired LMSYS pairwise chat-quality eval: judge verdicts per
# ABOUTME: comparison (diverging win/tie/loss bars) beside win rate with a 95% Wilson interval.
# Run: uv run python scratch/lmsys_figure/plot_lmsys_results.py [--out output/lmsys_figure]
"""Every LMSYS result this project produced, on one figure.

The eval was removed on 2026-09-02 (commit 4e425744; Arena-Hard replaced it), so the numbers
below are a closed set and are recorded here with where each one came from. Counts are the
judge's verdicts over a fixed subset of `lmsys/lmsys-chat-1m` prompts; the judge was
`google/gemini-3-flash-preview`, one position-randomised verdict per prompt, thinking mode on
both sides.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import binomtest

from src.naming import figure_path
from src.utils import write_run_meta

SUBJECT = "lmsys pairwise results"

# Diverging pair with a neutral midpoint (validated: blue/red pass every categorical check on
# a white surface, all pairs; the gray is the deliberate "no preference" neutral and carries
# a visible count label because it sits under 3:1 contrast).
BLUE, GRAY, RED = "#2a78d6", "#c3c2b7", "#e34948"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


@dataclass(frozen=True)
class Comparison:
    arm: str  # the fine-tuned arm, as the paper should call it
    reference: str  # what it was judged against
    model: str
    date: str  # the day the eval ran
    n_prompts: int  # prompts generated for
    wins: int  # judge preferred the fine-tuned arm
    ties: int
    losses: int  # judge preferred the reference
    source: str

    @property
    def judged(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def decisive(self) -> int:
        return self.wins + self.losses


COMPARISONS = [
    Comparison(
        arm="DA-only SFT",
        reference="base model",
        model="Qwen3-32B",
        date="2026-07-27",
        n_prompts=60,
        wins=24,
        ties=4,
        losses=32,
        source="docs/LOG.md 2026-07-27 (pm-5); adapter matboz/qwen3-32b-difficult-advice-lora "
        "(2,119 difficult-advice examples)",
    ),
    Comparison(
        arm="80:20 Tulu:DA SFT",
        reference="base model",
        model="Qwen3.6-27B",
        date="2026-07-29",
        n_prompts=40,
        wins=8,
        ties=11,
        losses=21,
        source="docs/LOG.md 2026-07-29 (late-2); adapter "
        "matboz/qwen3.6-27b-difficult-advice-tulu-lora",
    ),
    Comparison(
        arm="90:10 general:DA SFT",
        reference="general-only SFT",
        model="Qwen3.6-27B",
        date="2026-08-08",
        n_prompts=60,
        wins=30,
        ties=5,
        losses=17,
        source="HF dougalldeepmind/2026-08-08-lmsys-qwen3-6-27b-lora-t2-9000-synthdoc-1000-r64 "
        "results.json (8 judge failures; reference = table2-only-9284)",
    ),
]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


def stats(c: Comparison) -> dict:
    lo, hi = wilson(c.wins, c.decisive)
    return {
        "win_rate": c.wins / c.decisive,
        "ci_lo": lo,
        "ci_hi": hi,
        "p": binomtest(c.wins, c.decisive, 0.5, alternative="two-sided").pvalue,
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "xtick.color": MUTED,
            "xtick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # embed TrueType: journals reject Type 3
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def bare(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)


def draw(out_dir: Path) -> list[Path]:
    style()
    fig, (ax_v, ax_w) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.6),
        sharey=True,
        gridspec_kw={
            "width_ratios": [1.55, 1.0],
            "wspace": 0.12,
            "left": 0.215,
            "right": 0.865,
            "top": 0.82,
            "bottom": 0.17,
        },
    )
    rows = list(range(len(COMPARISONS)))[::-1]  # first comparison on top
    bar_h = 0.40

    # --- (a) verdicts, as a share of judged prompts, centred on the ties -----------------
    for y, c in zip(rows, COMPARISONS):
        share = {
            k: 100 * v / c.judged
            for k, v in (("l", c.losses), ("t", c.ties), ("w", c.wins))
        }
        segs = [
            (-share["t"] / 2 - share["l"], share["l"], RED, c.losses, "white"),
            (-share["t"] / 2, share["t"], GRAY, c.ties, INK),
            (share["t"] / 2, share["w"], BLUE, c.wins, "white"),
        ]
        for left, width, colour, count, ink in segs:
            # the white edge IS the surface gap between touching segments, not a border
            ax_v.barh(
                y,
                width,
                left=left,
                height=bar_h,
                color=colour,
                edgecolor="white",
                linewidth=1.4,
            )
            if width >= 7:  # label only where the count fits
                ax_v.text(
                    left + width / 2,
                    y,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=ink,
                )
        # three lines per row: the arm, what it was judged against, then model and n
        for dy, text, size, colour in (
            (0.25, c.arm, 8, INK),
            (0.0, f"vs. {c.reference}", 7.5, INK_2),
            (-0.24, f"{c.model} · n = {c.judged}", 7, MUTED),
        ):
            ax_v.text(
                -0.035,
                y + dy,
                text,
                ha="right",
                va="center",
                transform=ax_v.get_yaxis_transform(),
                fontsize=size,
                color=colour,
            )
    # behind the bars: it marks the centre between rows without striking the tie counts
    ax_v.axvline(0, color=INK_2, linewidth=0.7, zorder=0.5)
    ax_v.set_xlim(-75, 75)
    ax_v.set_xticks([-60, -40, -20, 0, 20, 40, 60])
    ax_v.set_xticklabels(["60", "40", "20", "0", "20", "40", "60"])
    ax_v.set_xlabel("Share of judged prompts (%)")
    ax_v.set_ylim(-0.6, len(rows) - 0.4)
    bare(ax_v)
    ax_v.legend(
        handles=[
            Patch(color=RED, label="Reference preferred"),
            Patch(color=GRAY, label="Tie"),
            Patch(color=BLUE, label="Fine-tuned preferred"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.text(0.008, 0.90, "a", fontsize=10, fontweight="bold", va="center")

    # --- (b) win rate over decisive verdicts, 95% Wilson interval --------------------------
    for y, c in zip(rows, COMPARISONS):
        s = stats(c)
        ax_w.plot(
            [100 * s["ci_lo"], 100 * s["ci_hi"]],
            [y, y],
            color=INK_2,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3,
        )
        ax_w.plot(
            100 * s["win_rate"],
            y,
            "o",
            markersize=6.5,
            color=BLUE,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=4,
        )
        p = "p < 0.001" if s["p"] < 0.001 else f"p = {s['p']:.2f}"
        ax_w.text(
            1.03,
            y + 0.13,
            f"{100 * s['win_rate']:.0f}%",
            ha="left",
            va="center",
            transform=ax_w.get_yaxis_transform(),
            fontsize=8,
            color=INK,
        )
        ax_w.text(
            1.03,
            y - 0.16,
            f"{c.wins}/{c.decisive}, {p}",
            ha="left",
            va="center",
            transform=ax_w.get_yaxis_transform(),
            fontsize=7,
            color=MUTED,
        )
    ax_w.axvline(50, color=INK_2, linewidth=0.7, zorder=2)
    ax_w.text(
        50,
        len(rows) - 0.42,
        "no preference",
        ha="center",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    ax_w.set_xlim(0, 100)
    ax_w.set_xticks([0, 25, 50, 75, 100])
    ax_w.set_xlabel("Win rate, ties excluded (%)")
    bare(ax_w)
    fig.text(
        ax_w.get_position().x0 + 0.004,
        0.90,
        "b",
        fontsize=10,
        fontweight="bold",
        va="center",
    )

    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        path = figure_path(out_dir, SUBJECT, ext=ext)
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths


CAPTION = (
    "Pairwise chat quality on LMSYS-Chat-1M prompts. Each row compares a fine-tuned arm with "
    "its reference on a fixed random subset of English single-turn prompts; a Gemini 3 Flash "
    "judge saw both answers in random order and returned one verdict per prompt. (a) Verdicts "
    "as a share of judged prompts, centred on ties; numbers are prompt counts. (b) Win rate of "
    "the fine-tuned arm over decisive verdicts with a 95% Wilson interval; p-values are exact "
    "two-sided binomial tests against 50%. DA: difficult-advice data. General: the "
    "spec-filtered Table-2 instruction blend. The three comparisons differ in base model, "
    "reference and prompt count, so rows are read separately, not against each other."
)


def write_results(out_dir: Path, paths: list[Path]) -> Path:
    lines = [
        f"# {SUBJECT}",
        "",
        "| comparison | model | date | prompts | judged | W / T / L | win rate "
        "(ties excl.) | 95% Wilson | exact p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in COMPARISONS:
        s = stats(c)
        lines.append(
            f"| {c.arm} vs. {c.reference} | {c.model} | {c.date} | {c.n_prompts} | {c.judged} "
            f"| {c.wins} / {c.ties} / {c.losses} | {100 * s['win_rate']:.1f}% "
            f"| [{100 * s['ci_lo']:.1f}, {100 * s['ci_hi']:.1f}] | {s['p']:.3f} |"
        )
    lines += ["", "## Sources", ""] + [f"- {c.date}: {c.source}" for c in COMPARISONS]
    lines += [
        "",
        "## Not on the figure",
        "",
        "- Heuristic refusal counts: 2026-07-27 fine-tune 15/60 vs base 5/60; 2026-07-29 "
        "fine-tune 3/40 vs base 1/40; none recorded for 2026-08-08.",
        "- 2026-07-29 mean answer length: fine-tune 4,995 vs base 6,164 characters.",
        "- 2026-08-08 truncation rate: target 11.7%, reference 8.3%; 8 of 60 verdicts "
        "failed to parse and are excluded.",
        "",
        "## Suggested caption",
        "",
        CAPTION,
        "",
        "## Files",
        "",
    ] + [f"- `{p.name}`" for p in paths]
    out = paths[0].with_name(paths[0].stem + "_results.md")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/lmsys_figure", type=Path)
    args = ap.parse_args()
    paths = draw(args.out)
    md = write_results(args.out, paths)
    write_run_meta(
        args.out, {"subject": SUBJECT, "comparisons": [c.__dict__ for c in COMPARISONS]}
    )
    for p in [*paths, md]:
        print(p)


if __name__ == "__main__":
    main()
