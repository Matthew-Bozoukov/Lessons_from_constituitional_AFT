# ABOUTME: Methodology figure for the paper draft — the six-stage difficult-advice data generation
# ABOUTME: pipeline, one sentence per stage, each stage labelled with the model that runs it.

"""The data generation pipeline as a vertical stack of six rows, read top to bottom.

Stage order, names and models are read off `configs/data/synth/da.yaml` (the project's DA
baseline), not off the draft: the draft credits stage 1 to Haiku, but `chunk_constitution` is
`kind: segment` — a deterministic heading split with no model call — so the row says so.

Colour encodes the MODEL here, never an arm, so none of the fixed arm colours are used as a
model colour; the model is also written on every row, so colour is a redundant cue.

The first cut of this figure was one horizontal row of cards (commit 503f620e).

    uv run python scratch/datagen_pipeline_diagram.py [--out_dir output/figures]
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.naming import figure_path

HAIKU, SONNET, NO_MODEL = "#1F8A7A", "#B23A6B", "#7A7F87"
INK, MUTED = "#1C1F24", "#5B616B"

# (title, the one sentence, model label, model colour)
STAGES = [
    (
        "Chunk the constitution",
        "Split the constitution by heading into its 9 traits, one chunk each.",
        "No model",
        NO_MODEL,
    ),
    (
        "Generate scenarios",
        "Invent diverse situations where the easy path would violate that trait.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Draft prompts",
        "Write the user's message and an ordinary system prompt for each scenario.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Refine prompts",
        "Sharpen both prompts so they press on the trait: realistic, tempting, no hints.",
        "Claude Sonnet 5",
        SONNET,
    ),
    (
        "Draft response",
        "Answer as the assistant: private reasoning first, then the reply.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Refine response",
        "Rewrite reasoning and reply so the deliberation openly weighs the trait.",
        "Claude Sonnet 5",
        SONNET,
    ),
]

W = 3.6  # inches; data coordinates ARE inches
MARGIN = 0.04
ROW_W, ROW_H, GAP = W - 2 * MARGIN, 0.58, 0.16
MODEL_W = 1.12  # the coloured block at the left of each row
PILL_H = 0.3
H = 2 * MARGIN + 2 * PILL_H + len(STAGES) * ROW_H + (len(STAGES) + 1) * GAP
ROUND = "round,pad=0,rounding_size=0.07"


def tint(hex_colour: str, keep: float = 0.09) -> tuple[float, float, float]:
    """The colour mixed toward white — the row body behind dark text."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - keep * (1 - c) for c in (r, g, b))


def balanced(text: str, width: int) -> str:
    """Wrap to as few lines as `width` allows, then even them out — no one-word last line."""
    n_lines = len(textwrap.wrap(text, width))
    for w in range(10, width + 1):
        if len(textwrap.wrap(text, w)) == n_lines:
            return textwrap.fill(text, w)
    return textwrap.fill(text, width)


def arrow_down(ax, y_from: float, y_to: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (W / 2, y_from),
            (W / 2, y_to),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color=MUTED,
            shrinkA=0,
            shrinkB=0,
        )
    )


def pill(ax, y: float, w: float, text: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            ((W - w) / 2, y),
            w,
            PILL_H,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc="white",
            ec=INK,
            lw=0.9,
        )
    )
    ax.text(
        W / 2, y + PILL_H / 2, text, ha="center", va="center", fontsize=7.2, color=INK
    )


def row(
    ax, y: float, n: int, title: str, sentence: str, model: str, colour: str
) -> None:
    """One stage: model block on the left, stage name + its sentence on the right."""
    x = MARGIN
    ax.add_patch(
        FancyBboxPatch(
            (x, y), ROW_W, ROW_H, boxstyle=ROUND, fc=tint(colour), ec=colour, lw=1.0
        )
    )
    # Model block: clipped to the rounded row so its left corners follow the outline.
    block = Rectangle((x, y), MODEL_W, ROW_H, fc=colour, ec="none")
    ax.add_patch(block)
    block.set_clip_path(
        FancyBboxPatch((x, y), ROW_W, ROW_H, boxstyle=ROUND, transform=ax.transData)
    )
    mid = y + ROW_H / 2
    ax.text(
        x + MODEL_W / 2,
        mid + 0.1,
        f"STAGE {n}",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color="white",
        alpha=0.85,
    )
    ax.text(
        x + MODEL_W / 2,
        mid - 0.06,
        model,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="white",
    )
    body_x = x + MODEL_W + 0.1
    ax.text(
        body_x,
        y + ROW_H - 0.08,
        title,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        body_x,
        y + ROW_H - 0.25,
        balanced(sentence, 46),
        ha="left",
        va="top",
        fontsize=6.7,
        color=INK,
        linespacing=1.25,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="output/figures")
    args = ap.parse_args()

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Top to bottom: input, the six stages, output — an arrow across every gap.
    y = H - MARGIN - PILL_H
    pill(ax, y, 1.9, "Input: constitution (9 traits)")
    for n, stage in enumerate(STAGES, start=1):
        arrow_down(ax, y - 0.02, y - GAP + 0.02)
        y -= GAP + ROW_H
        row(ax, y, n, *stage)
    arrow_down(ax, y - 0.02, y - GAP + 0.02)
    pill(
        ax,
        y - GAP - PILL_H,
        3.45,
        "Output: difficult-advice SFT example (system, user, reasoning, reply)",
    )

    for ext in ("png", "pdf", "svg"):
        path = figure_path(args.out_dir, "datagen_pipeline_diagram_vertical", ext=ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(path)


if __name__ == "__main__":
    main()
