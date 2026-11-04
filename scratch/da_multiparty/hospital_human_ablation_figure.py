# ABOUTME: Hospital figure for the human-parties MDMA ablation: the three 15% arms (matched plain DA, MDMA, MDMA with human
# ABOUTME: parties only) on sabotage act, plan post and kept refusal, 30 shifts each, 95% Wilson intervals, from the Hub.
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from src.naming import figure_path

ORG = "dougalldeepmind"
PREFIX = "2026-09-24-colosseum-hospital-self-sacrificial-qwen36-difficult-advice-"
ARMS = [
    ("Plain difficult advice (matched)", PREFIX + "15-fixed", "#a879dd"),
    ("MDMA", PREFIX + "multiparty-15-fixed", "#c2255c"),
    ("MDMA, human parties only", PREFIX + "multiparty-human-15-fixed", "#e58aa9"),
]
MEASURES = [
    ("pair_sabotaged", "Sabotage act\n(rule)"),
    ("plan_posted", "Plan post\n(regex)"),
    ("judge_refusal_held_ge4", "Refusal kept in full\n(judge ≥4)"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


rows = {}
for lab, repo, _ in ARMS:
    d = json.load(
        open(
            hf_hub_download(
                f"{ORG}/{repo}", "results/per_seed.json", repo_type="dataset"
            )
        )
    )
    rows[lab] = {
        m: [int(v) for v in d[m]["self_sacrificial"].values()] for m, _ in MEASURES
    }

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
w = 0.26
for j, (lab, _, col) in enumerate(ARMS):
    for i, (m, _) in enumerate(MEASURES):
        xs = rows[lab][m]
        k, n = sum(xs), len(xs)
        lo, hi = wilson(k, n)
        x = i + (j - 1) * w
        ax.bar(
            x, 100 * k / n, w * 0.95, color=col, label=lab if i == 0 else None, zorder=3
        )
        ax.errorbar(
            x,
            100 * k / n,
            yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
            fmt="none",
            ecolor="#222",
            elinewidth=0.8,
            capsize=2,
            zorder=4,
        )
        ax.text(x, 100 * hi + 2, f"{k}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(len(MEASURES)))
ax.set_xticklabels([lab for _, lab in MEASURES], fontsize=8)
ax.set_ylabel("Shifts (%) of 30", fontsize=9)
ax.set_ylim(0, 112)
ax.grid(axis="y", color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13)
)
ax.set_title(
    "Hospital, self-sacrificial: all three at 15% on the same base blend, seed 0",
    fontsize=8.5,
    pad=22,
)
fig.tight_layout()
out = figure_path(
    Path("output/colosseum_hospital/analysis"), "hospital_mdma_human_parties_ablation"
)
fig.savefig(out)
print(out)
