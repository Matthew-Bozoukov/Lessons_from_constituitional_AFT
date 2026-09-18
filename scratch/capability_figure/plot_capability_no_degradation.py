# ABOUTME: Simple paper figure: the difficult-advice arm vs its matched general-only control on
# ABOUTME: MMLU and SWE-bench Verified, with 95% Wilson intervals and the paired test per benchmark.
# Run: uv run python scratch/capability_figure/plot_capability_no_degradation.py [--out output/capability_figure]
"""Does adding difficult-advice data cost capability? One matched pair, two benchmarks.

Both adapters are Qwen3.6-27B rank-64 LoRAs on the same recipe. The control saw 9,284
spec-filtered Table-2 instruction rows; the DA arm saw the same 9,284 rows plus 716
difficult-advice rows (7.2% of the mixture). Every number below is from a published run, and
each benchmark used the identical question set for both arms, so the tests are paired.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.naming import figure_path
from src.utils import write_run_meta

SUBJECT = "capability da vs control"

# Emphasis form: the arm under test in the accent hue, its control in the de-emphasis gray.
BLUE, GRAY = "#2a78d6", "#a9a79f"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


@dataclass(frozen=True)
class Benchmark:
    name: str
    metric: str
    n: int
    control_correct: int
    da_correct: int
    p_paired: float  # exact McNemar over the shared question set
    source: str


BENCHMARKS = [
    Benchmark(
        name="MMLU",
        metric="accuracy",
        n=1140,
        control_correct=980,
        da_correct=978,
        p_paired=0.933,
        source="HF dougalldeepmind/2026-08-05-mmlu-qwen3-6-27b-lora-table2-{only-9284,synthdoc}"
        "-r64 (thinking mode, 1,140 shared questions; discordant pairs 72 vs 70, exact "
        "McNemar computed from the two records.jsonl files on 2026-09-18)",
    ),
    Benchmark(
        name="SWE-bench Verified",
        metric="pass@1",
        n=250,
        control_correct=107,
        da_correct=116,
        p_paired=0.289,
        source="HF dougalldeepmind/2026-08-07-swebench-verified-qwen36-lora-comparison "
        "(repo-stratified 250 of 500, identical instances; exact McNemar p from "
        "docs/LOG.md 2026-08-07)",
    ),
]

# Context that is deliberately NOT on the figure; it goes in the results file.
BASE_MMLU = (957, 1140)  # Qwen/Qwen3.6-27B, same 1,140 questions, 2026-08-05


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "ytick.color": MUTED,
            "ytick.major.width": 0.6,
            "ytick.major.size": 2.5,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # embed TrueType: journals reject Type 3
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw(out_dir: Path) -> list[Path]:
    style()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.17)
    width, gap = 0.24, 0.03

    for i, b in enumerate(BENCHMARKS):
        top = 0.0
        for offset, correct, colour in (
            (-(width + gap) / 2, b.control_correct, GRAY),
            ((width + gap) / 2, b.da_correct, BLUE),
        ):
            score = 100 * correct / b.n
            lo, hi = (100 * v for v in wilson(correct, b.n))
            ax.bar(i + offset, score, width=width, color=colour, zorder=2)
            ax.plot(
                [i + offset] * 2,
                [lo, hi],
                color=INK_2,
                linewidth=0.9,
                zorder=3,
                solid_capstyle="butt",
            )
            ax.text(
                i + offset,
                hi + 1.5,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
            top = max(top, hi)
        delta = 100 * (b.da_correct - b.control_correct) / b.n
        # a true minus sign, and a surface-coloured box so no gridline strikes the text
        ax.text(
            i,
            top + 9.5,
            f"Δ = {delta:+.1f} pp, p = {b.p_paired:.2f}".replace("-", "−"),
            ha="center",
            va="bottom",
            fontsize=7,
            color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            zorder=4,
        )
        ax.text(i, -5.5, b.name, ha="center", va="top", fontsize=8, color=INK)
        ax.text(
            i,
            -12.5,
            f"{b.metric}, n = {b.n:,}",
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED,
        )

    ax.set_xlim(-0.55, len(BENCHMARKS) - 0.45)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Score (%)")
    ax.set_xticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    fig.legend(
        handles=[
            Patch(color=GRAY, label="General data only"),
            Patch(color=BLUE, label="+ 7% difficult-advice data"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.565, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        path = figure_path(out_dir, SUBJECT, ext=ext)
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths


CAPTION = (
    "Adding difficult-advice (DA) data does not reduce measured capability. Two Qwen3.6-27B "
    "LoRA fine-tunes on the same recipe: one on 9,284 general instruction rows, one on the "
    "same rows plus 716 DA rows (7% of the mixture). Both arms answered identical questions "
    "on each benchmark. Bars are MMLU accuracy (thinking mode) and SWE-bench Verified pass@1; "
    "whiskers are 95% Wilson intervals; p-values are exact McNemar tests on the paired "
    "outcomes. Neither difference is significant."
)


def write_results(paths: list[Path]) -> Path:
    lines = [
        f"# {SUBJECT}",
        "",
        "| benchmark | metric | n | general only | 95% Wilson | general + 7% DA | "
        "95% Wilson | delta | paired p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in BENCHMARKS:
        c_lo, c_hi = wilson(b.control_correct, b.n)
        d_lo, d_hi = wilson(b.da_correct, b.n)
        lines.append(
            f"| {b.name} | {b.metric} | {b.n:,} | {100 * b.control_correct / b.n:.1f}% "
            f"| [{100 * c_lo:.1f}, {100 * c_hi:.1f}] | {100 * b.da_correct / b.n:.1f}% "
            f"| [{100 * d_lo:.1f}, {100 * d_hi:.1f}] "
            f"| {100 * (b.da_correct - b.control_correct) / b.n:+.1f} pp | {b.p_paired:.3f} |"
        )
    k, n = BASE_MMLU
    lo, hi = wilson(k, n)
    lines += ["", "## Sources", ""] + [f"- {b.name}: {b.source}" for b in BENCHMARKS]
    lines += [
        "",
        "## Context not on the figure",
        "",
        f"- Base Qwen3.6-27B on the same MMLU questions: {100 * k / n:.1f}% "
        f"[{100 * lo:.1f}, {100 * hi:.1f}]. Both fine-tunes sit about 2 pp above it "
        "(paired p 0.13 and 0.09), so neither is below the base model either.",
        "- 2026-07-31 MMLU dose ladder (570 questions, thinking mode): base 91.8%, 0% "
        "synthetic 90.9%, 10% 90.9%, 20% 91.9%, 40% 90.7%. Flat at every share.",
        "- The limit of the claim: Arena-Hard (2026-07-31) found a 40% synthetic share "
        "DOES cost chat quality (39.4% controlled win rate, interval below the 45% gate) "
        "while 20% is flat, and the 80:20 arm lost to base Qwen on LMSYS (27.6%, "
        "p = 0.02). No degradation holds at low DA shares, not at any share.",
        "- These are August arms on the Table-2 base blend. The current paper arms "
        "(2026-09-15 da-7, delib-7, nosynth) have no capability scores yet.",
        "- The DA adapter is `table2-synthdoc-r64`; its 9,284 + 716 composition is read "
        "from the log's naming of that arm, not from a training manifest.",
        "",
        "## Suggested caption",
        "",
        CAPTION,
        "",
        "## Files",
        "",
    ]
    lines += [f"- `{p.name}`" for p in paths]
    out = paths[0].with_name(paths[0].stem + "_results.md")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/capability_figure", type=Path)
    args = ap.parse_args()
    paths = draw(args.out)
    md = write_results(paths)
    write_run_meta(
        args.out, {"subject": SUBJECT, "benchmarks": [b.__dict__ for b in BENCHMARKS]}
    )
    for p in [*paths, md]:
        print(p)


if __name__ == "__main__":
    main()
