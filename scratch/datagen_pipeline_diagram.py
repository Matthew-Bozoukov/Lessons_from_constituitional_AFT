# ABOUTME: Methodology figure for the paper draft — the six-stage difficult-advice data generation
# ABOUTME: pipeline, one sentence per stage, each stage labelled with the model that runs it.

"""The data generation pipeline as six boxed stages, read top to bottom.

Each stage is one box — number badge, stage name over its one sentence, and the model that
runs it as a chip on the right — with an arrow across every gap. The width is set by the
longest sentence, so there is no empty band; `check_fit` fails the run if a sentence would
reach its model chip after a wording change.

Stage order, names and models are read off `configs/data/synth/da.yaml` (the project's DA
baseline), not off the draft: the draft credits stage 1 to Haiku, but `chunk_constitution` is
`kind: segment` — a deterministic heading split with no model call — so the box says so.

Colour encodes the MODEL here, never an arm, so none of the fixed arm colours are used as a
model colour; the model is also written in every box, so colour is a redundant cue.

Earlier cuts: one horizontal row of cards (commit 503f620e), tall boxed rows (0f76f733),
an unboxed numbered spine (738bda8b), one-line boxes at 6.5in wide (eaed2586).

    uv run python scratch/datagen_pipeline_diagram.py [--out_dir output/figures]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from src.naming import figure_path

HAIKU, SONNET, NO_MODEL = "#147D73", "#A83265", "#7F858E"
INK, BODY, MUTED, ARROW = "#15181D", "#3F454E", "#8A9099", "#6B727C"

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

W = 4.95  # inches; data coordinates ARE inches, so circles stay round
PAD = 0.03  # figure edge to the boxes
BOX_H, END_H, GAP = 0.4, 0.24, 0.11
H = 2 * PAD + 2 * END_H + len(STAGES) * BOX_H + (len(STAGES) + 1) * GAP
LEFT, RIGHT = PAD, W - PAD
# Columns inside a box: number badge, stage name over its sentence, model chip.
BADGE_X, TEXT_X = LEFT + 0.2, LEFT + 0.41
BADGE_R = 0.1
CHIP_W, CHIP_H = 1.0, 0.19
CHIP_X = RIGHT - 0.07 - CHIP_W


def tint(hex_colour: str, keep: float) -> tuple[float, float, float]:
    """The colour mixed toward white; `keep` is how much of it survives."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - keep * (1 - c) for c in (r, g, b))


def caps(ax, x: float, y: float, text: str, ha: str = "left") -> None:
    """The small letterspaced label used for INPUT / OUTPUT / MODEL."""
    ax.text(
        x,
        y,
        " ".join(text.upper()),
        ha=ha,
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color=MUTED,
    )


def arrow(ax, y_from: float, y_to: float) -> None:
    """Box to box, in the badge column so the numbers and arrows read as one line."""
    ax.add_patch(
        FancyArrowPatch(
            (BADGE_X, y_from),
            (BADGE_X, y_to),
            arrowstyle="-|>",
            mutation_scale=6,
            lw=0.9,
            color=ARROW,
            shrinkA=0,
            shrinkB=0,
        )
    )


def endpoint(ax, y: float, label: str, text: str) -> None:
    """The input / output box: slimmer than a stage, neutral, sized to its text."""
    body = ax.text(
        LEFT + 0.7,
        y + END_H / 2,
        text,
        ha="left",
        va="center",
        fontsize=7.2,
        color=INK,
    )
    caps(ax, LEFT + 0.12, y + END_H / 2, label)
    ax.figure.canvas.draw()
    end = body.get_window_extent().x1 / ax.figure.dpi
    ax.add_patch(
        FancyBboxPatch(
            (LEFT, y),
            end + 0.12 - LEFT,
            END_H,
            boxstyle="round,pad=0,rounding_size=0.06",
            fc="white",
            ec=INK,
            lw=0.8,
        )
    )


def stage(ax, y: float, n: int, title: str, sentence: str, model: str, colour: str):
    """One boxed stage; returns the sentence text so `check_fit` can measure it."""
    mid = y + BOX_H / 2
    ax.add_patch(
        FancyBboxPatch(
            (LEFT, y),
            RIGHT - LEFT,
            BOX_H,
            boxstyle="round,pad=0,rounding_size=0.07",
            fc=tint(colour, 0.05),
            ec=tint(colour, 0.6),
            lw=0.9,
        )
    )
    ax.add_patch(Circle((BADGE_X, mid), BADGE_R, fc=colour, ec="none"))
    ax.text(
        BADGE_X,
        mid - 0.004,
        str(n),
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="white",
    )
    ax.text(
        TEXT_X,
        mid + 0.078,
        title,
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    body = ax.text(
        TEXT_X, mid - 0.088, sentence, ha="left", va="center", fontsize=7.0, color=BODY
    )
    ax.add_patch(
        FancyBboxPatch(
            (CHIP_X, mid - CHIP_H / 2),
            CHIP_W,
            CHIP_H,
            boxstyle=f"round,pad=0,rounding_size={CHIP_H / 2}",
            fc=tint(colour, 0.16),
            ec="none",
        )
    )
    ax.text(
        CHIP_X + CHIP_W / 2,
        mid - 0.003,
        model,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=colour,
    )
    return body


def check_fit(fig, sentences) -> None:
    """Fail rather than ship a sentence that runs into its model chip."""
    fig.canvas.draw()
    for body in sentences:
        end = body.get_window_extent().x1 / fig.dpi
        if end > CHIP_X - 0.06:
            raise ValueError(
                f"sentence reaches the model chip ({end:.2f}in > {CHIP_X - 0.06:.2f}in): "
                f"{body.get_text()!r} — shorten it or widen W"
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

    # Top to bottom: input, the six stages, output — `y` is each box's bottom edge.
    y = H - PAD - END_H
    endpoint(ax, y, "Input", "Constitution (9 traits)")
    caps(ax, CHIP_X + CHIP_W / 2, y + END_H / 2, "Model", ha="center")
    sentences = []
    for n, spec in enumerate(STAGES, start=1):
        arrow(ax, y - 0.01, y - GAP + 0.015)
        y -= GAP + BOX_H
        sentences.append(stage(ax, y, n, *spec))
    arrow(ax, y - 0.01, y - GAP + 0.015)
    endpoint(
        ax,
        y - GAP - END_H,
        "Output",
        "Difficult-advice SFT example: system, user, reasoning, reply",
    )
    check_fit(fig, sentences)

    for ext in ("png", "pdf", "svg"):
        path = figure_path(args.out_dir, "datagen_pipeline_diagram_vertical", ext=ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(path)


if __name__ == "__main__":
    main()
