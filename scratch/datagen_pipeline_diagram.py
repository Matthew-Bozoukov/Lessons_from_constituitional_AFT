# ABOUTME: Methodology figure for the paper draft — the six-stage difficult-advice data generation
# ABOUTME: pipeline, one sentence per stage, each stage labelled with the model that runs it.

"""The data generation pipeline as a numbered spine, read top to bottom, at full text width.

Stage order, names and models are read off `configs/data/synth/da.yaml` (the project's DA
baseline), not off the draft: the draft credits stage 1 to Haiku, but `chunk_constitution` is
`kind: segment` — a deterministic heading split with no model call — so the row says so.

Colour encodes the MODEL here, never an arm, so none of the fixed arm colours are used as a
model colour; the model is also written on every row, so colour is a redundant cue.

Earlier cuts: one horizontal row of cards (commit 503f620e), tall boxed rows (0f76f733).

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
INK, BODY, MUTED = "#15181D", "#4A505A", "#8A9099"
SPINE, HAIRLINE = "#AEB5BE", "#E6E8EC"

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

W = 6.5  # inches; data coordinates ARE inches, so circles stay round
PITCH = 0.44  # stage row to stage row
END_PITCH = 0.36  # input/output row to its neighbouring stage
EDGE = 0.17  # input/output row centre to the figure edge
H = 2 * EDGE + 2 * END_PITCH + (len(STAGES) - 1) * PITCH
SPINE_X, TEXT_X, RIGHT = 0.2, 0.47, W - 0.03
NODE_R, END_R = 0.105, 0.05
CHIP_W, CHIP_H = 1.08, 0.2


def tint(hex_colour: str, keep: float = 0.12) -> tuple[float, float, float]:
    """The colour mixed toward white — the chip fill behind its own coloured text."""
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


def link(ax, y_from: float, y_to: float) -> None:
    """One spine segment, arrowhead at its lower end."""
    ax.add_patch(
        FancyArrowPatch(
            (SPINE_X, y_from),
            (SPINE_X, y_to),
            arrowstyle="-|>",
            mutation_scale=6,
            lw=0.9,
            color=SPINE,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def endpoint(ax, y: float, label: str, text: str) -> None:
    """The input / output row: a hollow node, a caps label, and what flows there."""
    ax.add_patch(Circle((SPINE_X, y), END_R, fc="white", ec=INK, lw=0.9, zorder=3))
    caps(ax, TEXT_X, y, label)
    ax.text(TEXT_X + 0.52, y, text, ha="left", va="center", fontsize=7.4, color=INK)


def stage(
    ax, y: float, n: int, title: str, sentence: str, model: str, colour: str
) -> None:
    ax.add_patch(Circle((SPINE_X, y), NODE_R, fc=colour, ec="none", zorder=3))
    ax.text(
        SPINE_X,
        y - 0.004,
        str(n),
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    ax.text(
        TEXT_X,
        y + 0.075,
        title,
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        TEXT_X, y - 0.085, sentence, ha="left", va="center", fontsize=7.6, color=BODY
    )
    ax.add_patch(
        FancyBboxPatch(
            (RIGHT - CHIP_W, y - CHIP_H / 2),
            CHIP_W,
            CHIP_H,
            boxstyle=f"round,pad=0,rounding_size={CHIP_H / 2}",
            fc=tint(colour),
            ec="none",
        )
    )
    ax.text(
        RIGHT - CHIP_W / 2,
        y - 0.003,
        model,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=colour,
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

    # Row centres, top to bottom: input, the six stages, output.
    y_in = H - EDGE
    ys = [y_in - END_PITCH - i * PITCH for i in range(len(STAGES))]
    y_out = ys[-1] - END_PITCH

    endpoint(ax, y_in, "Input", "Constitution (9 traits)")
    caps(ax, RIGHT - CHIP_W / 2, y_in, "Model", ha="center")
    link(ax, y_in - END_R, ys[0] + NODE_R + 0.015)
    for n, (y, spec) in enumerate(zip(ys, STAGES), start=1):
        stage(ax, y, n, *spec)
        if n > 1:
            link(ax, ys[n - 2] - NODE_R, y + NODE_R + 0.015)
            mid = y + PITCH / 2
            ax.plot([TEXT_X, RIGHT], [mid, mid], color=HAIRLINE, lw=0.6, zorder=0)
    link(ax, ys[-1] - NODE_R, y_out + END_R + 0.015)
    endpoint(
        ax,
        y_out,
        "Output",
        "Difficult-advice SFT example: system, user, reasoning, reply",
    )

    for ext in ("png", "pdf", "svg"):
        path = figure_path(args.out_dir, "datagen_pipeline_diagram_vertical", ext=ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(path)


if __name__ == "__main__":
    main()
