# ABOUTME: Methodology figure for the paper draft — the six-stage difficult-advice data generation
# ABOUTME: pipeline, one sentence per stage, each stage labelled with the model that runs it.

"""The data generation pipeline as one row of six cards.

Stage order, names and models are read off `configs/data/synth/da.yaml` (the project's DA
baseline), not off the draft: the draft credits stage 1 to Haiku, but `chunk_constitution` is
`kind: segment` — a deterministic heading split with no model call — so the card says so.

Colour encodes the MODEL here, never an arm, so none of the fixed arm colours are used as a
model colour; the model is also written on every card, so colour is a redundant cue.

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
        "Chunk the\nconstitution",
        "Split the constitution by heading into its 9 traits, one chunk each.",
        "No model",
        NO_MODEL,
    ),
    (
        "Generate\nscenarios",
        "Invent diverse situations where the easy path would violate that trait.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Draft\nprompts",
        "Write the user's message and an ordinary system prompt for each scenario.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Refine\nprompts",
        "Sharpen both prompts so they press on the trait: realistic, tempting, no hints.",
        "Claude Sonnet 5",
        SONNET,
    ),
    (
        "Draft\nresponse",
        "Answer as the assistant: private reasoning first, then the reply.",
        "Claude Haiku 4.5",
        HAIKU,
    ),
    (
        "Refine\nresponse",
        "Rewrite reasoning and reply so the deliberation openly weighs the trait.",
        "Claude Sonnet 5",
        SONNET,
    ),
]

W, H = 7.2, 2.95  # inches; data coordinates ARE inches
CARD_W, CARD_H, STRIP_H = 1.03, 1.72, 0.26
MARGIN = 0.04
GAP = (W - 2 * MARGIN - len(STAGES) * CARD_W) / (len(STAGES) - 1)
CARD_Y = 0.62  # bottom edge of the card row
PILL_H = 0.3


def tint(hex_colour: str, keep: float = 0.09) -> tuple[float, float, float]:
    """The colour mixed toward white — the card body behind dark text."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - keep * (1 - c) for c in (r, g, b))


def arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color=MUTED,
            shrinkA=0,
            shrinkB=0,
        )
    )


def pill(ax, x, y, w, text) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            PILL_H,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc="white",
            ec=INK,
            lw=0.9,
        )
    )
    ax.text(
        x + w / 2,
        y + PILL_H / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
    )


def card(ax, x, n, title, sentence, model, colour) -> None:
    top = CARD_Y + CARD_H
    ax.add_patch(
        FancyBboxPatch(
            (x, CARD_Y),
            CARD_W,
            CARD_H,
            boxstyle="round,pad=0,rounding_size=0.07",
            fc=tint(colour),
            ec=colour,
            lw=1.0,
        )
    )
    # Header strip: clipped to the rounded card so its top corners follow the outline.
    strip = Rectangle((x, top - STRIP_H), CARD_W, STRIP_H, fc=colour, ec="none")
    ax.add_patch(strip)
    strip.set_clip_path(
        FancyBboxPatch(
            (x, CARD_Y),
            CARD_W,
            CARD_H,
            boxstyle="round,pad=0,rounding_size=0.07",
            transform=ax.transData,
        )
    )
    ax.text(
        x + CARD_W / 2,
        top - STRIP_H / 2,
        model,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + 0.08,
        top - STRIP_H - 0.09,
        f"STAGE {n}",
        ha="left",
        va="top",
        fontsize=5.6,
        color=colour,
        fontweight="bold",
    )
    ax.text(
        x + 0.08,
        top - STRIP_H - 0.22,
        title,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
        linespacing=1.05,
    )
    ax.text(
        x + 0.08,
        top - STRIP_H - 0.62,
        textwrap.fill(sentence, 19),
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

    xs = [MARGIN + i * (CARD_W + GAP) for i in range(len(STAGES))]
    for i, (x, stage) in enumerate(zip(xs, STAGES)):
        card(ax, x, i + 1, *stage)
        if i:
            arrow(
                ax,
                (xs[i - 1] + CARD_W + 0.02, CARD_Y + CARD_H / 2),
                (x - 0.02, CARD_Y + CARD_H / 2),
            )

    # Input drops into stage 1; output drops out of stage 6.
    top = CARD_Y + CARD_H
    pill(ax, xs[0], top + 0.2, 1.75, "Input: constitution (9 traits)")
    arrow(ax, (xs[0] + CARD_W / 2, top + 0.2), (xs[0] + CARD_W / 2, top + 0.02))
    out_w = 3.35
    pill(
        ax,
        xs[-1] + CARD_W - out_w,
        CARD_Y - 0.2 - PILL_H,
        out_w,
        "Output: difficult-advice SFT example (system, user, reasoning, reply)",
    )
    arrow(ax, (xs[-1] + CARD_W / 2, CARD_Y - 0.02), (xs[-1] + CARD_W / 2, CARD_Y - 0.2))

    for ext in ("png", "pdf", "svg"):
        path = figure_path(args.out_dir, "datagen_pipeline_diagram", ext=ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(path)


if __name__ == "__main__":
    main()
