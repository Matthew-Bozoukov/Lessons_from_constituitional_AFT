# ABOUTME: oai_hf Step 3 only (shares the exploit with other agents): one row per model, rate with 95% Wilson CI,
# ABOUTME: our Qwen arms read from their Hub eval repos beside the paper's models. Run: uv run python scratch/oai_hf_repro/fig_step3_only.py
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scratch.oai_hf_repro.compare_paper import PAPER  # noqa: E402
from src.eval.misalignment.oai_hf.export import wilson  # noqa: E402
from src.utils import save_figure_pdf  # noqa: E402

OUT = Path("output/oai_hf_repro/2026-11-20_oaihf_step3_qwen38_n192_vs_paper.pdf")
INK, MUTED, GRID, PAPERC = "#1A211E", "#5C6660", "#D5DCD7", "#4C6A92"
ARMS = [  # (label, Hub eval repo, revision, colour) -- newest pushes, 2026-11-19/20
    (
        "Qwen3.8-27B DA (ours)",
        "dougalldeepmind/2026-11-19-oaihf-qwen38-0-da-15",
        "e2f709ce",
        "#7E22CE",
    ),
    (
        "Qwen3.8-27B DA+tools (ours)",
        "dougalldeepmind/2026-11-20-oaihf-qwen38-0-da-tools-15",
        "140568af",
        "#d0558f",
    ),
    (
        "Qwen3.8-27B nosynth (ours)",
        "dougalldeepmind/2026-11-19-oaihf-qwen38-0-plain",
        "45810cd6",
        "#5F5F5F",
    ),
    (
        "Qwen3.8-27B base, API (ours)",
        "dougalldeepmind/2026-11-18-oaihf-openrouter-qwen3-8-27b",
        "5a326442",
        "#2A8C82",
    ),
    (
        "Qwen3.6-27B nosynth (ours)",
        "dougalldeepmind/2026-11-13-oaihf-qwen36-0-nosynth",
        "a2f6622a",
        "#B0B0B0",
    ),
]


def step3(repo, rev):
    f = hf_hub_download(repo, "results/runs.jsonl", repo_type="dataset", revision=rev)
    rows = [json.loads(x) for x in open(f) if json.loads(x)["step"] == "step3"]
    return sum(bool(r["signature"]) for r in rows), len(rows)


rows = [(m, PAPER[m].get("step3"), PAPERC) for m in PAPER] + [
    (lab, step3(r, v), c) for lab, r, v, c in ARMS
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for i, (m, kn, c) in enumerate(rows):
    y = len(rows) - 1 - i
    if kn is None:
        ax.text(
            0.01, y, "no runs (provider filter)", va="center", fontsize=8.5, color=MUTED
        )
        continue
    k, n = kn
    p, lo, hi = wilson(k, n)
    ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot([p], [y], "o", ms=9, color=c, mec="white", mew=1.5, zorder=3)
    ax.text(
        min(hi + 0.03, 0.82),
        y,
        f"{k}/{n}  ({100 * p:.0f}%)",
        va="center",
        fontsize=9,
        color=INK,
    )
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([m for m, _, _ in rows][::-1], color=INK)
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
ax.set_xlabel(
    "fraction of runs that posted the working exploit for the other agents (dot), 95% CI",
    color=MUTED,
    fontsize=9,
    loc="left",
)
ax.set_title(
    "OAI-HF Step 3: shares the exploit with other agents",
    fontsize=12,
    fontweight="bold",
    color=INK,
    loc="left",
)
fig.text(
    0.01,
    0.005,
    "Ours: judge Gemini 3 Flash (paper: Opus 4.8), thinking on. Qwen3.8 nosynth, DA and DA+tools n=192 "
    "(64 + 128 merged), base via API n=32,\nQwen3.6 nosynth n=64. Paper rows: their headline cells "
    "(Qwen 3.8 Max is the closed API model).",
    fontsize=7.5,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_figure_pdf(
    fig,
    OUT,
    "oai_hf Step 3: Qwen3.8 arms vs paper",
    [r for _, r, _, _ in ARMS] + ["oai-hf-replication/oai-hf-incident-replication"],
    "scratch/oai_hf_repro/fig_step3_only.py",
)
for m, kn, _ in rows:
    print(m, kn)
