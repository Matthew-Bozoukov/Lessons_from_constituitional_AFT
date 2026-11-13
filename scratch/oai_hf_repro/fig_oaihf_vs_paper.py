# ABOUTME: Figure for the oai_hf run: per-step signature rate (95% Wilson CI) of nosynth Qwen3.6-27B beside
# ABOUTME: the paper's six models, and how far our runs got through each step (the capability funnel).
# Run: uv run python scratch/oai_hf_repro/fig_oaihf_vs_paper.py [merged_run_dir]
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scratch.oai_hf_repro.compare_paper import PAPER  # noqa: E402
from src.eval.misalignment.oai_hf.export import wilson  # noqa: E402
from src.utils import save_figure_pdf  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "output/oai_hf_repro/merged_final")
OUT = Path("output/oai_hf_repro/2026-11-13_oaihf_nosynth_vs_paper.pdf")
ARTIFACTS = [
    "dougalldeepmind/2026-11-13-oaihf-qwen36-0-nosynth",
    "oai-hf-replication/oai-hf-incident-replication",
]

INK, MUTED, GRID = "#1A211E", "#5C6660", "#D5DCD7"
OURS = "#8C8C8C"  # the nosynth control is grey in every figure (CLAUDE.md arm colours)
PAPERC = "#4C6A92"  # one colour for all paper models: identity is the axis label, not the hue
STEPS = [
    ("step1", "Step 1: writes to shared registry"),
    ("step2", "Step 2: asks other agents for help"),
    ("step3", "Step 3: shares exploit with agents"),
    ("step4", "Step 4: hacks HF for answer key"),
]

summary = json.loads((RUN / "results" / "results.json").read_text())["steps"]
rows = [json.loads(line) for line in (RUN / "results" / "runs.jsonl").open()]
ours = {
    s: (summary[s]["signature_all"]["k"], summary[s]["signature_all"]["n"])
    for s, _ in STEPS
}
models = list(PAPER) + ["nosynth Qwen3.6-27B (ours)"]


def count(step, pred):
    return sum(1 for r in rows if r["step"] == step and pred(r))


FUNNELS = {
    "step1": [
        ("ran", count("step1", lambda r: True)),
        (
            "tried proxy (lvl>=1)",
            count("step1", lambda r: (r.get("judge_level") or 0) >= 1),
        ),
        (
            "used proxy (lvl>=2)",
            count("step1", lambda r: (r.get("judge_level") or 0) >= 2),
        ),
        ("wrote to registry", ours["step1"][0]),
    ],
    "step2": [
        ("ran", count("step2", lambda r: True)),
        ("found peers' traces", count("step2", lambda r: r.get("found_peers"))),
        ("asked for help", ours["step2"][0]),
    ],
    "step3": [
        ("ran", count("step3", lambda r: True)),
        ("got egress", count("step3", lambda r: r.get("got_egress"))),
        ("captured token", count("step3", lambda r: r.get("solved"))),
        ("saw the board", count("step3", lambda r: r.get("found_board"))),
        ("shared method", ours["step3"][0]),
    ],
    "step4": [
        ("ran", count("step4", lambda r: True)),
        ("reached HF via proxy", count("step4", lambda r: r.get("reached_hf"))),
        (
            "used file-read bug",
            count("step4", lambda r: (r.get("judge_level") or 0) >= 3),
        ),
        ("harvested token", count("step4", lambda r: r.get("harvested_cred"))),
        ("submitted stolen flag", ours["step4"][0]),
    ],
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(
    2, 4, figsize=(15, 7.6), gridspec_kw={"height_ratios": [1.15, 1]}
)
for j, (step, title) in enumerate(STEPS):
    ax = axes[0, j]
    for i, m in enumerate(models):
        k, n = ours[step] if m.endswith("(ours)") else PAPER[m].get(step, (None, None))
        y = len(models) - 1 - i
        if k is None:
            ax.text(
                0.01,
                y,
                "no runs (provider filter)",
                va="center",
                fontsize=7.5,
                color=MUTED,
            )
            continue
        p, lo, hi = wilson(k, n)
        c = OURS if m.endswith("(ours)") else PAPERC
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([p], [y], "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
        ax.text(min(hi + 0.03, 0.80), y, f"{k}/{n}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models[::-1] if j == 0 else [""] * len(models), color=INK)
    if j == 0:
        ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10, fontweight="bold", color=INK, loc="left")
    if j == 0:
        ax.set_xlabel(
            "fraction of runs showing the behaviour (dot) with 95% CI",
            color=MUTED,
            fontsize=8,
            loc="left",
        )

    ax2 = axes[1, j]
    stages = FUNNELS[step]
    for i, (label, v) in enumerate(stages):
        y = len(stages) - 1 - i
        last = i == len(stages) - 1
        ax2.barh(
            y,
            v,
            height=0.62,
            color=OURS if not last else "#5E5E5E",
            edgecolor="white",
            lw=1,
        )
        ax2.text(
            v + 1.2,
            y,
            str(v),
            va="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold" if last else "normal",
        )
    ax2.set_yticks(range(len(stages)))
    ax2.set_yticklabels([s[0] for s in stages][::-1], color=INK, fontsize=8.5)
    ax2.set_xlim(0, 72)
    ax2.set_xticks([0, 16, 32, 48, 64])
    ax2.tick_params(length=0, colors=MUTED)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    if j == 0:
        ax2.set_title(
            "How far our runs got (of 64)",
            fontsize=10,
            fontweight="bold",
            color=INK,
            loc="left",
        )

fig.suptitle(
    "OAI-HF incident reproduction, Part 1: nosynth Qwen3.6-27B vs the paper's models (n=64 per step)",
    fontsize=12,
    fontweight="bold",
    color=INK,
    x=0.01,
    ha="left",
)
fig.text(
    0.01,
    0.005,
    "Ours: judge Gemini 3 Flash (paper: Opus 4.8), thinking on, local Docker; run merged from 7 parts after two Docker crashes "
    "(error-free runs, first 64 by assigned epoch). Paper: their headline cells, recomputed from their logs.",
    fontsize=7.5,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
save_figure_pdf(
    fig,
    OUT,
    "oai_hf: nosynth Qwen3.6-27B vs paper",
    ARTIFACTS,
    "scratch/oai_hf_repro/fig_oaihf_vs_paper.py",
)
fig.savefig(OUT.with_suffix(".png"), dpi=140, bbox_inches="tight")
print(OUT, OUT.with_suffix(".png"))
