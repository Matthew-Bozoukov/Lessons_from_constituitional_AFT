# ABOUTME: Plots the "I ran this" (own-execution) claim rate against the overall fabrication
# ABOUTME: rate, showing that training removes silent invented data but not false claims.

"""The claim rate is the number that does not move.

Overall fabrication drops 82.0% -> 56.4%, but explicitly claiming to have run/measured
something sits at ~8% in both arms. Plotting them together is the point: the left panel
shows what changed, the right shows what the residual failures are made of.
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

rows = {}
for arm in ARMS:
    for f in glob.glob(f"output/fabrication_sweep/judged_*/{arm}.json"):
        rows[arm] = json.load(open(f))

OUT = Path("output/plots")
OUT.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# --- left: overall fabrication vs "I ran this", as % of ALL samples ---------------
x = np.arange(2)
w = 0.36
for k, arm in enumerate(ARMS):
    s = summary[arm]
    vals, los, his = [], [], []
    for cnt in (s["fabricated"], s["claims_own_execution"]):
        lo, hi = cp(cnt, s["n"])
        vals.append(cnt / s["n"] * 100)
        los.append((cnt / s["n"] - lo) * 100)
        his.append((hi - cnt / s["n"]) * 100)
    b = ax1.bar(x + (k - 0.5) * w, vals, w, label=LABEL[arm].replace("\n", " "),
                color=COLOR[arm], edgecolor="black", linewidth=0.8,
                yerr=[los, his], capsize=5, error_kw={"linewidth": 1.3, "ecolor": "black"})
    for bb, v in zip(b, vals):
        ax1.text(bb.get_x() + bb.get_width() / 2, v + 2.5, f"{v:.1f}%",
                 ha="center", fontsize=15, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(["Any fabrication\n(invented data or claim)",
                     'Claims "I ran / measured this"'], fontsize=15)
ax1.set_ylabel("% of all 992 responses per arm", fontsize=15)
ax1.set_title("Training cuts fabrication by 26 points —\nbut not the false execution claim",
              fontsize=16)
ax1.set_ylim(0, 100)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=14, loc="upper right", framealpha=0.95)
ax1.grid(True, linestyle="--", alpha=0.2, axis="y")
ax1.set_axisbelow(True)
ax1.annotate("unchanged\n(CIs overlap)", xy=(1.0, 12), xytext=(1.0, 34),
             ha="center", fontsize=14, style="italic",
             arrowprops={"arrowstyle": "->", "linewidth": 1.4})

# --- right: composition of what each arm still gets wrong -------------------------
cats = ["Silent invented data\n(no execution claim)", "Both: invented data\n+ execution claim",
        "Execution claim only"]
for k, arm in enumerate(ARMS):
    h = [r for r in rows[arm] if r.get("fabricated")]
    both = sum(1 for r in h if r.get("claims_own_execution") and r.get("invented_data_as_real"))
    inv_only = sum(1 for r in h if r.get("invented_data_as_real")) - both
    exec_only = sum(1 for r in h if r.get("claims_own_execution")) - both
    vals = [inv_only, both, exec_only]
    b = ax2.bar(np.arange(3) + (k - 0.5) * w, vals, w,
                label=f"{LABEL[arm].replace(chr(10), ' ')} (n={len(h)})",
                color=COLOR[arm], edgecolor="black", linewidth=0.8)
    for bb, v in zip(b, vals):
        ax2.text(bb.get_x() + bb.get_width() / 2, v + 12, str(v),
                 ha="center", fontsize=14, fontweight="bold")

ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(cats, fontsize=14)
ax2.set_ylabel("Number of fabricating responses", fontsize=15)
ax2.set_title("What the failures are made of\n(87% of the synth arm's are silent)",
              fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=13, loc="upper right", framealpha=0.95)
ax2.grid(True, linestyle="--", alpha=0.2, axis="y")
ax2.set_axisbelow(True)
ax2.set_ylim(0, 800)

fig.suptitle('"I ran this" claims vs silent fabrication — Qwen3.6-27B LoRA arms, '
             "31 prompts × 32 samples", fontsize=17, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))

dest = OUT / f"own_execution_claims_{timestamp()}.png"
fig.savefig(dest, dpi=200, bbox_inches="tight")
print(f"wrote {dest}")
