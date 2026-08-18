# ABOUTME: Plots the 31-prompt fabrication sweep: overall rates with CIs and the paired
# ABOUTME: per-prompt comparison between the table2-only and table2+synth arms.

"""Two panels, because the pooled number alone is misleading.

Prompt-to-prompt variance (6%-97%) dwarfs the arm difference, so a single pair of bars hides
what actually justifies the conclusion: the effect holds *within* nearly every prompt. The
right-hand panel is the evidence; the left is the headline.
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "src/eval/audits/petri")

from funnel_stats import clopper_pearson as cp  # noqa: E402
from src.utils import timestamp  # noqa: E402

ARMS = ["t2only", "t2synth"]
LABEL = {"t2only": "table2 only\n(no synth)", "t2synth": "table2 + synth"}
COLOR = {"t2only": "#c44e52", "t2synth": "#4c72b0"}

summary = {}
for f in glob.glob("output/fabrication_sweep/judged_*/summary.json"):
    summary.update(json.load(open(f))["summary"])

OUT = Path("output/plots")
OUT.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                               gridspec_kw={"width_ratios": [1, 2.6]})

# --- left: pooled rate with exact binomial CIs -----------------------------------
rates, los, his = [], [], []
for arm in ARMS:
    s = summary[arm]
    lo, hi = cp(s["fabricated"], s["n"])
    rates.append(s["rate"] * 100)
    los.append((s["rate"] - lo) * 100)
    his.append((hi - s["rate"]) * 100)

bars = ax1.bar([LABEL[a] for a in ARMS], rates, color=[COLOR[a] for a in ARMS],
               edgecolor="black", linewidth=0.8, width=0.6,
               yerr=[los, his], capsize=6, error_kw={"linewidth": 1.4, "ecolor": "black"})
for b, r, arm in zip(bars, rates, ARMS):
    s = summary[arm]
    ax1.text(b.get_x() + b.get_width() / 2, r + 4.5, f"{r:.1f}%",
             ha="center", fontsize=17, fontweight="bold")
    ax1.text(b.get_x() + b.get_width() / 2, r / 2, f"{s['fabricated']}/{s['n']}",
             ha="center", fontsize=14, color="white", fontweight="bold")

ax1.set_ylabel("Responses fabricating benchmark data (%)", fontsize=15)
ax1.set_title("Overall rate\n(31 prompts × 32 samples)", fontsize=16)
ax1.set_ylim(0, 100)
ax1.tick_params(labelsize=14)
ax1.grid(True, linestyle="--", alpha=0.2, axis="y")
ax1.set_axisbelow(True)

# --- right: paired per-prompt, sorted by the synth arm ---------------------------
pp = {a: summary[a]["per_prompt"] for a in ARMS}
ids = sorted(set(pp["t2only"]) & set(pp["t2synth"]),
             key=lambda i: pp["t2synth"][i]["rate"])
x = np.arange(len(ids))
w = 0.4
for k, arm in enumerate(ARMS):
    ax2.bar(x + (k - 0.5) * w, [pp[arm][i]["rate"] * 100 for i in ids], w,
            label=LABEL[arm].replace("\n", " "), color=COLOR[arm],
            edgecolor="black", linewidth=0.6)

ax2.set_xticks(x)
ax2.set_xticklabels(ids, rotation=90, fontsize=14)
ax2.set_ylabel("Fabrication rate (%)", fontsize=15)
ax2.set_xlabel("Prompt (sorted by table2+synth rate)", fontsize=15)
ax2.set_title("Per-prompt: synth arm lower on 28 of 31 prompts (sign test p < 0.00001)",
              fontsize=16)
ax2.set_ylim(0, 100)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, loc="upper left", framealpha=0.95)
ax2.grid(True, linestyle="--", alpha=0.2, axis="y")
ax2.set_axisbelow(True)

# The identical-prompt pair bounds what any single-prompt difference can mean.
ax2.text(0.985, 0.06,
         "noise floor ≈16 pts: identical prompts p03/p04 scored 81% vs 97%",
         transform=ax2.transAxes, ha="right", fontsize=14, style="italic",
         bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "grey"})

fig.suptitle("Fabricated benchmark data — Qwen3.6-27B LoRA arms, 1,984 samples",
             fontsize=18, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))

dest = OUT / f"fabrication_sweep_{timestamp()}.png"
fig.savefig(dest, dpi=200, bbox_inches="tight")
print(f"wrote {dest}")
