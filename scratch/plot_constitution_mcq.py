# ABOUTME: Figure for the ConstitutionEval result: three arms x two splits, both template modes.
# ABOUTME: Per-experiment plotting code, so scratch/ -> output/ per CLAUDE.md. Run: uv run python <this>
"""ConstitutionEval accuracy by arm, split and template mode.

A connected dot plot, not bars. The reference the reader needs is chance (25%) and the
effect lives at 85-98%, so a bar chart would either crush the differences against a
zero baseline or truncate its own axis to show them — and a truncated bar lies, because
a bar's length IS the encoding. A dot encodes position, so it can share one honest axis
with a chance line 60 points away and still show an 8.7-point gap.

Two panels because the result must survive both renderings: `chat` puts the item through
the model's own chat template, `raw` sends bare text. They share no wording beyond the
item itself, so an artifact of the template would not reproduce across both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import figure_path  # noqa: E402

RUNS = Path("output/constitution_mcq")
OUT = Path("output/constitution_mcq")

# Validated 2-slot categorical palette (dataviz reference instance, light mode):
# adjacent-pair CVD dE 24.7 protan / 32.7 tritan, normal-vision 33.6, contrast >= 3:1.
FULL, HARD = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
SURFACE = "#fcfcfb"

# Bottom to top: worst hard-band score first, so the reading runs upward to the best.
ARMS = [
    (
        "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64",
        "table2-only\n(0% synthetic SFT)",
    ),
    ("Qwen/Qwen3.6-27B", "Qwen3.6-27B base\n(no SFT)"),
    (
        "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702"
        "-rank-64-dynbatch",
        "+ difficult advice\n(7% of mixture)",
    ),
]


def load() -> dict[str, dict]:
    out = {}
    for d in RUNS.iterdir():
        res = d / "results" / "results.json"
        if not res.exists():
            continue
        data = json.loads(res.read_text())
        if data.get("n_items") == 678:
            out[data["target"]] = data
    return out


def main() -> None:
    runs = load()
    missing = [t for t, _ in ARMS if t not in runs]
    if missing:
        raise SystemExit(f"!!! no 678-item run for: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True, facecolor=SURFACE)
    for ax, mode in zip(axes, ("chat", "raw")):
        ax.set_facecolor(SURFACE)
        for y, (target, _) in enumerate(ARMS):
            d = runs[target]
            full = d[f"{mode}_accuracy_debiased"] * 100
            hard = d[f"{mode}_hard_accuracy_debiased"] * 100
            # The connector carries the within-arm cost of difficulty; recessive on purpose.
            ax.plot(
                [hard, full],
                [y, y],
                color=MUTED,
                lw=2,
                zorder=1,
                solid_capstyle="round",
            )
            ax.scatter(
                [full], [y], s=110, color=FULL, zorder=3, edgecolor=SURFACE, linewidth=2
            )
            ax.scatter(
                [hard], [y], s=110, color=HARD, zorder=3, edgecolor=SURFACE, linewidth=2
            )
            # Six marks per panel, so every one is labelled: the numbers ARE the finding.
            ax.annotate(
                f"{hard:.1f}",
                (hard, y),
                xytext=(0, -17),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_2,
            )
            ax.annotate(
                f"{full:.1f}",
                (full, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_2,
            )

        # Chance (25%) sits 60 points below everything plotted. Drawing it spends two
        # thirds of the canvas on empty space and squeezes the 8.7-point effect into a
        # sliver, so it is stated in the subtitle instead. A DOT encodes position, not
        # length, which is what makes a non-zero baseline honest here where it would not
        # be for a bar -- and every point carries its own value besides.
        ax.set_xlim(82.5, 100.6)
        ax.set_ylim(-0.55, 2.62)
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([label for _, label in ARMS], fontsize=9.5, color=INK)
        ax.set_xlabel("swap-debiased accuracy, %", fontsize=9.5, color=INK_2)
        ax.set_title(f"{mode} template", fontsize=10.5, color=INK, loc="left", pad=8)
        ax.xaxis.grid(True, color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d1")
        ax.tick_params(colors=INK_2, length=0)

    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", ms=9, color=FULL, label="full set (678 items)"
        ),
        plt.Line2D(
            [], [], marker="o", ls="", ms=9, color=HARD, label="hard band (217 items)"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.03),
        labelcolor=INK_2,
    )
    fig.suptitle(
        "ConstitutionEval: general SFT costs constitution-following on a spec we did not write;\n"
        "difficult-advice data pays it back",
        fontsize=12.5,
        color=INK,
        x=0.012,
        ha="left",
        y=1.22,
    )
    fig.text(
        0.012,
        1.015,
        "678 four-way items from SPP's constitution, never shown to the model. Chance is 25%: "
        "the axis starts at 82, so every arm is far above it.",
        fontsize=9.5,
        color=INK_2,
        ha="left",
    )
    fig.tight_layout()
    # THE figure filename (src/utils.figure_path): dated + subject, validated on the way
    # out. A hand-built f-string would be the same string today and unattributable later.
    path = figure_path(OUT, "constitution_mcq_arms")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
