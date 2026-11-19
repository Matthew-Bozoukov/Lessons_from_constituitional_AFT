# ABOUTME: Figure for oai_hf: per-step signature rate (95% Wilson CI) of our arms beside the paper's models,
# ABOUTME: one 2x2 panel per step, one row per model; our arms are read from their Hub eval repos' runs.jsonl.
# Run: uv run python scratch/oai_hf_repro/fig_oaihf_vs_paper.py
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

from scratch.oai_hf_repro.compare_paper import PAPER  # noqa: E402
from src.eval.misalignment.oai_hf.export import wilson  # noqa: E402
from src.utils import save_figure_pdf  # noqa: E402

OUT = Path("output/oai_hf_repro/2026-11-19_oaihf_qwen38_nosynth_da_datools_vs_paper.pdf")
INK, MUTED, GRID = "#1A211E", "#5C6660", "#D5DCD7"
PAPERC = "#4C6A92"  # one colour for all paper models: identity is the axis label, not the hue
STEPS = [
    ("step1", "Step 1: writes to shared registry"),
    ("step2", "Step 2: asks other agents for help"),
    ("step3", "Step 3: shares exploit with agents"),
    ("step4", "Step 4: hacks HF for answer key"),
]
# Our arms: (label, Hub eval repo, pinned revision, colour). Every row is read from the Hub (the run
# dirs are scratch); the revisions are the newest pushes on 2026-11-18/19.
ARMS = [
    ("Qwen3.8-27B DA (ours)", "dougalldeepmind/2026-11-19-oaihf-qwen38-0-da-15", "f04f2548", "#7E22CE"),
    ("Qwen3.8-27B DA+tools (ours)", "dougalldeepmind/2026-11-18-oaihf-qwen38-0-da-tools-15", "d014d4bb", "#d0558f"),
    ("Qwen3.8-27B nosynth (ours)", "dougalldeepmind/2026-11-18-oaihf-qwen38-0-plain", "785c9ca2", "#5F5F5F"),
    ("Qwen3.8-27B base, API (ours)", "dougalldeepmind/2026-11-18-oaihf-openrouter-qwen3-8-27b", "5a326442", "#2A8C82"),
    ("Qwen3.6-27B nosynth (ours)", "dougalldeepmind/2026-11-13-oaihf-qwen36-0-nosynth", "a2f6622a", "#B0B0B0"),
]


def arm_counts(repo: str, revision: str | None) -> dict[str, tuple[int, int]]:
    """{step: (k, n)} from one published eval repo's results/runs.jsonl."""
    f = hf_hub_download(repo, "results/runs.jsonl", repo_type="dataset", revision=revision)
    k, n = {}, {}
    for r in (json.loads(x) for x in open(f)):
        k[r["step"]] = k.get(r["step"], 0) + int(bool(r["signature"]))
        n[r["step"]] = n.get(r["step"], 0) + 1
    return {s: (k[s], n[s]) for s in n}


counts_of = {label: arm_counts(repo, rev) for label, repo, rev, _ in ARMS}
colour_of = {label: c for label, _, _, c in ARMS}
models = list(PAPER) + [label for label, _, _, _ in ARMS]


def cell(model: str, step: str):
    if model in counts_of:
        return counts_of[model].get(step), colour_of[model]
    return PAPER[model].get(step), PAPERC


def draw_rates(ax, step, title, labels: bool, xlabel: bool):
    """One step's panel: a row per model, dot = signature rate, line = 95% Wilson CI."""
    for i, m in enumerate(models):
        kn, c = cell(m, step)
        y = len(models) - 1 - i
        if kn is None:
            note = (
                "not run (Step 3 only)"
                if m in counts_of
                else "no runs (provider filter)"
            )
            ax.text(0.01, y, note, va="center", fontsize=7.5, color=MUTED)
            continue
        k, n = kn
        p, lo, hi = wilson(k, n)
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([p], [y], "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
        ax.text(min(hi + 0.03, 0.80), y, f"{k}/{n}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models[::-1] if labels else [""] * len(models), color=INK)
    if labels:
        for t in ax.get_yticklabels()[: len(ARMS)]:
            t.set_fontweight("bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10, fontweight="bold", color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(
            "fraction of runs showing the behaviour (dot) with 95% CI",
            color=MUTED,
            fontsize=8,
            loc="left",
        )


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
for j, (step, title) in enumerate(STEPS):
    draw_rates(axes[j // 2, j % 2], step, title, labels=j % 2 == 0, xlabel=j >= 2)
fig.suptitle(
    "OAI-HF incident reproduction, Part 1: our Qwen models vs the paper's models",
    fontsize=12,
    fontweight="bold",
    color=INK,
    x=0.01,
    ha="left",
)
fig.text(
    0.01,
    0.005,
    "Ours: judge Gemini 3 Flash (paper: Opus 4.8), thinking on, local Docker. Qwen3.8-27B DA, DA+tools and nosynth: our LoRA arms "
    "(seed 0), served by our vLLM, Step 3 only, n=64.\nQwen3.8-27B base: untrained, via OpenRouter (DeepInfra, bf16), n=32 "
    "per step. Qwen3.6 nosynth: our control, n=64. Paper rows: their headline cells (Qwen 3.8 Max is the closed API model).",
    fontsize=7.5,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
save_figure_pdf(
    fig,
    OUT,
    "oai_hf: Qwen3.8-27B DA, DA+tools and nosynth, Qwen3.8 base, Qwen3.6 nosynth vs paper",
    [repo for _, repo, _, _ in ARMS] + ["oai-hf-replication/oai-hf-incident-replication"],
    "scratch/oai_hf_repro/fig_oaihf_vs_paper.py",
)
fig.savefig(OUT.with_suffix(".png"), dpi=140, bbox_inches="tight")
print(OUT)
for label in counts_of:
    print(label, counts_of[label])
