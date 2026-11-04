# ABOUTME: The paper's Figure 2 ODCV panel, plus the human-parties MDMA arm, (no synthetic 2026-09-09, delib-7 and da-7 2026-09-17) with the MDMA arm
# ABOUTME: added, same colours, value labels and +-1 binomial SE bars over the 240 rollouts; reads the published runs.
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.patches import Patch

from src.naming import figure_path

ARMS = [
    ("No synthetic data", "2026-09-09-odcv-qwen36-0-nosynth", "#5f6774"),
    ("Deliberative alignment", "2026-09-17-odcv-qwen36-0-delib-7", "#2c78f1"),
    ("Difficult advice", "2026-09-17-odcv-qwen36-0-da-7", "#7724c4"),
    ("MDMA", "2026-09-24-odcv-qwen36-0-da-multiparty-15", "#c2255c"),
    ("MDMA (human parties)", "2026-09-24-odcv-qwen36-0-da-multiparty-human-15", "#e58aa9"),
]
rows = []
for lab, repo, col in ARMS:
    o = json.load(
        open(
            hf_hub_download(
                f"dougalldeepmind/{repo}", "results/results.json", repo_type="dataset"
            )
        )
    )["ours"]["overall"]
    p, n = o["mr_pct"] / 100, o["n_rollouts"]
    rows.append((lab, 100 * p, 100 * (p * (1 - p) / n) ** 0.5, col))

fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
for i, (lab, v, se, col) in enumerate(rows):
    ax.bar(i, v, 0.8, color=col, zorder=3)
    ax.errorbar(
        i, v, yerr=se, fmt="none", ecolor="#222", elinewidth=1, capsize=3, zorder=4
    )
    ax.text(i, v + se + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks([1.5])
ax.set_xticklabels(["ODCV"], fontsize=9)
ax.set_ylabel("Misalignment rate (%)", fontsize=9)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.tick_params(labelsize=8)
ax.grid(axis="y", color="#e1e0d9", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(
    handles=[Patch(color=c, label=l) for l, _, _, c in rows],
    frameon=False,
    fontsize=7,
    ncol=2,
    loc="upper right",
)
fig.tight_layout()
out = figure_path(Path("output/odcv"), "odcv_paper_panel_mdma_and_human_parties")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
