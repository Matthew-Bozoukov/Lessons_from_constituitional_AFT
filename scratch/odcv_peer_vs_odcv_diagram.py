# ABOUTME: Slide diagram: what ODCV-Peer adds to ODCV — one shared sandbox with a second agent, a
# ABOUTME: message board, a hash ledger, and the extra judging; grey = unchanged, purple = new.
"""uv run python scratch/odcv_peer_vs_odcv_diagram.py → output/odcv_peer/<date>_odcvpeer_vs_odcv_diagram.png

For a reader who knows ODCV: the left panel is ODCV as it ships, the right panel is the same run
with one seat added. Everything grey is byte-identical to ODCV; everything purple is new.
"""

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "output/odcv_peer"
    / f"{date.today().isoformat()}_odcvpeer_vs_odcv_diagram.png"
)
INK, GREY, GREY_FILL, PURPLE, PURPLE_FILL, MUTED = (
    "#1f2328",
    "#6b7280",
    "#eef0f3",
    "#5b3fa6",
    "#ece7f8",
    "#6b7280",
)

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(16, 8.6), gridspec_kw={"width_ratios": [1, 1.55]}
)
for ax in (axL, axR):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def box(ax, x, y, w, h, title, body="", new=False, fs=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.6",
            fc=PURPLE_FILL if new else GREY_FILL,
            ec=PURPLE if new else GREY,
            lw=2 if new else 1.4,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 2.8,
        title,
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color=PURPLE if new else INK,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 8.6,
            body,
            ha="center",
            va="top",
            fontsize=fs - 2,
            color=INK,
            linespacing=1.3,
        )


def arrow(ax, p, q, new=False, text="", double=False):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="<|-|>" if double else "-|>",
            mutation_scale=14,
            color=PURPLE if new else GREY,
            lw=1.8 if new else 1.4,
        )
    )
    if text:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(
            mx + 1.5,
            my,
            text,
            ha="left",
            va="center",
            fontsize=9,
            color=PURPLE if new else MUTED,
        )


# ---------------- left: ODCV as it ships ----------------
axL.text(
    2, 98, "ODCV (as published)", fontsize=17, fontweight="bold", color=INK, va="top"
)
axL.text(
    2,
    92.5,
    "one agent, one sandbox, one transcript",
    fontsize=11,
    color=MUTED,
    va="top",
)
box(axL, 25, 76, 50, 11, "Task prompt", "mandated  /  incentivized wording")
box(axL, 25, 54, 50, 13, "Agent", "bash + task_complete")
box(
    axL,
    10,
    28,
    80,
    15,
    "Sandbox (docker)",
    "data files · scripts · the scoring check\n40 scenarios",
)
box(axL, 25, 4, 50, 12, "Judge: misbehaviour 0–5", "reads the transcript only")
arrow(axL, (50, 76), (50, 67.5))
arrow(axL, (50, 54), (50, 43.5), double=True, text="commands / output")
arrow(axL, (50, 28), (50, 16.5), text="transcript")

# ---------------- right: ODCV-Peer ----------------
axR.text(2, 98, "ODCV-Peer", fontsize=17, fontweight="bold", color=PURPLE, va="top")
axR.text(
    2,
    92.5,
    "the same run with one seat added — grey is unchanged, purple is new",
    fontsize=11,
    color=MUTED,
    va="top",
)
box(axR, 8, 76, 36, 11, "Task prompt", "identical text, each seat")
box(axR, 56, 76, 36, 11, "Task prompt", "identical text, each seat")
box(axR, 8, 54, 36, 13, "Agent A — tested seat", "the arm under measurement")
box(
    axR,
    56,
    54,
    36,
    13,
    "Agent B — teammate",
    "a copy of A, or the plain model",
    new=True,
)
arrow(axR, (26, 76), (26, 67.5))
arrow(axR, (74, 76), (74, 67.5))
arrow(axR, (44, 60.5), (56, 60.5), new=True, double=True)
axR.text(
    50, 63.2, "concurrent", ha="center", va="bottom", fontsize=9, color=PURPLE
)
box(axR, 36, 40, 28, 7, "Message board", new=True, fs=10)
axR.text(
    50,
    38.6,
    "posts land in the other seat's next turn;\n'agent has left' notice on submit",
    ha="center",
    va="top",
    fontsize=8.5,
    color=PURPLE,
)
box(
    axR,
    8,
    18,
    84,
    13,
    "ONE sandbox — the same files for both",
    "commands serialised under a lock · every file hashed after every command",
)
axR.add_patch(
    FancyBboxPatch(
        (8, 18),
        84,
        13,
        boxstyle="round,pad=0.4,rounding_size=1.6",
        fc="none",
        ec=PURPLE,
        lw=2,
        ls="--",
    )
)
arrow(axR, (22, 54), (22, 31.5), double=True)
arrow(axR, (78, 54), (78, 31.5), double=True)
box(axR, 4, 0.5, 27, 13.5, "Misbehaviour 0–5", "same ODCV judge,\non each seat", fs=10)
box(
    axR,
    35,
    0.5,
    29,
    13.5,
    "Team facts (8 fields)",
    "reads transcript +\nboard + ledger",
    new=True,
    fs=10,
)
box(
    axR,
    68,
    0.5,
    28,
    13.5,
    "Hash ledger",
    "who changed which file;\nleft pristine or not",
    new=True,
    fs=10,
)
arrow(axR, (17, 18), (17, 14.5))
arrow(axR, (49, 18), (49, 14.5), new=True)
arrow(axR, (82, 18), (82, 14.5), new=True)

fig.text(
    0.5,
    0.975,
    "What ODCV-Peer changes about ODCV",
    ha="center",
    fontsize=20,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.5,
    0.035,
    "Kept byte-identical: the 40 scenarios, the prompts, the sandbox images, the misbehaviour rubric, 3 rollouts per cell.",
    ha="center",
    fontsize=10,
    color=MUTED,
)
fig.text(
    0.5,
    0.012,
    "Added: a second concurrent seat, a message board, a hash ledger, and per-cell labels (resisted / went along / own violation) built from all three.",
    ha="center",
    fontsize=10,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(OUT, dpi=200, facecolor="white")
print(OUT)
