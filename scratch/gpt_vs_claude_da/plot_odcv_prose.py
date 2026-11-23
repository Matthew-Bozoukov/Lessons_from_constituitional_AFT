# ABOUTME: ODCV bar chart: control vs da-15 vs GPT DA vs markdown-removed GPT DA, in the paper's Figure 5 colours.
# ABOUTME: Reads published results from the Hub; writes a dated PDF + PNG under output/gpt_vs_claude_da/.
import json
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from src.utils import save_figure_pdf

ARMS = [  # label, HF eval repo, colour (paper Fig. 5), hatch
    ("Control\n(standard SFT)", "dougalldeepmind/2026-10-06-odcv-qwen36-0-plain", "#5a6472", None),
    ("Difficult Advice\n(Claude)", "dougalldeepmind/2026-10-06-odcv-qwen36-0-da-15", "#7e22ce", None),
    ("Difficult Advice\n(GPT)", "dougalldeepmind/2026-11-04-odcv-qwen36-0-da-gpt-15", "#2a9d8f", None),
    ("Difficult Advice\n(GPT, markdown removed)", "dougalldeepmind/2026-11-20-odcv-qwen36-0-da-gpt-prose-15", "#2a9d8f", "//"),
]
out = Path("output/gpt_vs_claude_da"); out.mkdir(parents=True, exist_ok=True)
rows = []
for label, repo, col, hatch in ARMS:
    o = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset"), encoding="utf-8"))["ours"]["overall"]
    rows.append((label, o["mr_pct"], o["mr_ci95"], col, hatch))
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for i, (label, v, (lo, hi), col, hatch) in enumerate(rows):
    ax.bar(i, v, color=col if not hatch else "#9fd6cf", edgecolor=col, hatch=hatch, linewidth=1.2, width=0.65)
    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=4, linewidth=1)
    ax.text(i, hi + 1, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)
ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
ax.set_ylabel("Misalignment rate (%)"); ax.set_ylim(0, 65)
ax.set_title("ODCV (lower is better)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout()
stem = out / "2026-11-23_odcv_gpt_da_markdown_removed"
fig.savefig(f"{stem}.png", dpi=200)
save_figure_pdf(fig, f"{stem}.pdf", title="ODCV: GPT DA with markdown removed", artifacts=[a[1] for a in ARMS], script=__file__)
print(stem, [(r[0].replace(chr(10), ' '), r[1], r[2]) for r in rows])
