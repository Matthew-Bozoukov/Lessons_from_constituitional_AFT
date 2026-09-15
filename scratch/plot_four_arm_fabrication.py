# ABOUTME: Plots the four-arm ablation: overall fabrication rate and the "I ran this" claim
# ABOUTME: rate, which rank the arms differently and so are shown side by side.

"""Two metrics, two different winners.

On overall fabrication the synth arm is best; on falsely claiming to have executed something
the self-reflect arm is best, and synth is no better than the no-admixture baseline. Plotting
them together is the point — a single "which arm is safest" bar would hide that.
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
from src.naming import figure_path

ORDER = ["t2only", "memself", "selfreflect", "t2synth"]
LABEL = {"t2only": "table2 only\n(baseline)", "memself": "+20%\nmem-self",
         "selfreflect": "+20%\nself-reflect", "t2synth": "+20%\nsynth"}
# Red for the arm that made things worse, blue for the two that helped, grey for baseline.
COLOR = {"t2only": "#8c8c8c", "memself": "#c44e52",
         "selfreflect": "#55a868", "t2synth": "#4c72b0"}

summary = {}
for f in glob.glob("output/fabrication_sweep/judged_*/summary.json"):
    summary.update(json.load(open(f))["summary"])

OUT = Path("output/plots")
OUT.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7))


def draw(ax, key, title, note):
    """One panel: per-arm rate of `key` as a share of all samples, with exact CIs."""
    vals, los, his, labels, colors = [], [], [], [], []
    for arm in ORDER:
        s = summary[arm]
        cnt = s[key] if key != "fabricated" else s["fabricated"]
        lo, hi = cp(cnt, s["n"])
        vals.append(cnt / s["n"] * 100)
        los.append((cnt / s["n"] - lo) * 100)
        his.append((hi - cnt / s["n"]) * 100)
        labels.append(LABEL[arm])
        colors.append(COLOR[arm])

    bars = ax.bar(np.arange(len(ORDER)), vals, 0.62, color=colors,
                  edgecolor="black", linewidth=0.9,
                  yerr=[los, his], capsize=6,
                  error_kw={"linewidth": 1.4, "ecolor": "black"})
    for b, v, arm in zip(bars, vals, ORDER):
        s = summary[arm]
        cnt = s[key] if key != "fabricated" else s["fabricated"]
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.045, f"{v:.1f}%",
                ha="center", fontsize=16, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{cnt}/{s['n']}",
                ha="center", fontsize=13, color="white", fontweight="bold")

    # Baseline reference: the whole question is whether an admixture beats table2-only.
    ax.axhline(vals[0], color="black", linestyle=":", linewidth=1.6, alpha=0.75)
    ax.text(3.42, vals[0], " baseline", va="center", fontsize=13, style="italic")

    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_title(title, fontsize=17)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax.set_axisbelow(True)
    ax.text(0.5, 0.965, note, transform=ax.transAxes, ha="center", va="top", fontsize=13,
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "grey"})


draw(ax1, "fabricated",
     "Any fabricated benchmark data",
     "synth best (56.4%), but its 5.3-pt edge over\nself-reflect is inside the ~16-pt noise floor")
ax1.set_ylabel("% of 992 responses per arm", fontsize=15)

draw(ax2, "claims_own_execution",
     'Falsely claims "I ran / measured this"',
     "self-reflect halves it (3.4% vs 7.6%, p<0.0001);\nsynth is no better than baseline")
ax2.set_ylabel("% of 992 responses per arm", fontsize=15)

fig.suptitle("Which 20% admixture helps depends on which failure you measure\n"
             "Qwen3.6-27B LoRA arms · 31 prompts × 32 samples × 4 arms = 3,968 generations",
             fontsize=17, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))

dest = figure_path(OUT, "four_arm_fabrication")
fig.savefig(dest, dpi=200, bbox_inches="tight")
print(f"wrote {dest}")
