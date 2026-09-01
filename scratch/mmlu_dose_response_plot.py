# ABOUTME: Restyle the MMLU capability-eval result as a dose-response figure matching the
# ABOUTME: ODCV-Bench mixture-sweep plot style. Run: python3 scratch/mmlu_dose_response_plot.py

"""Two panels in the ODCV-Bench figure style:

left  — absolute MMLU accuracy vs difficult-advice share, Wilson CI band, dotted
        line for the untuned base anchor (measured, same subset/mode);
right — paired accuracy difference vs base over the identical 570 questions, CI
        band, zero line and the pre-registered -3pp non-inferiority margin.

Reads the report JSON produced by mmlu_report.py; writes mmlu_dose_response.png
next to it. Style constants mirror output/odcv (steelblue series + orange
TULU-only marker + dotted reference), so the capstone figures read as one set.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from src.utils import figure_path

REPORT_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1 else "output/mmlu_eval/report/think_20260731_154631"
)

BLUE = "#4878A8"
BLUE_EDGE = "#22374E"
ORANGE = "#F5921B"
ORANGE_EDGE = "#8C4A05"
BAND = "#4878A8"
INK = "#333333"

ARMS = [  # (share of training tokens %, arm key)
    (0, "arm_a_synth00"),
    (10, "arm_b_synth10"),
    (20, "arm_c_synth20"),
    (40, "arm_d_synth40"),
]

data = json.loads((REPORT_DIR / "mmlu_scores.json").read_text())
scores, comps = data["scores"], data["comparisons"]
base = scores["arm_base"]

x = [s for s, _ in ARMS]
acc = [scores[k]["mean"] * 100 for _, k in ARMS]
acc_lo = [scores[k]["ci_lower"] * 100 for _, k in ARMS]
acc_hi = [scores[k]["ci_upper"] * 100 for _, k in ARMS]
diff = [comps[k]["diff_pp"] for _, k in ARMS]
diff_lo = [comps[k]["ci_lower_pp"] for _, k in ARMS]
diff_hi = [comps[k]["ci_upper_pp"] for _, k in ARMS]

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.6), facecolor="white")
fig.suptitle("MMLU: difficult-advice mixture sweep, Qwen3.6-27B (thinking mode)", fontsize=24, y=1.0)

def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle=(0, (4, 4)), color="#DDDDDD", linewidth=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(x)
    ax.set_xlabel("Difficult-advice share of training tokens (%)")

def series(ax, ys):
    """The shared line + markers: blue mixture series, orange TULU-only point at 0%."""
    ax.plot(x, ys, color=BLUE, linewidth=3.5, zorder=2)
    ax.scatter(x[1:], ys[1:], s=260, color=BLUE, edgecolor=BLUE_EDGE, linewidth=2.5, zorder=3)
    ax.scatter([x[0]], [ys[0]], s=300, color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=2.5, zorder=4)

# --- left: absolute accuracy -------------------------------------------------------
style(ax1)
ax1.set_title("Dose-response: MMLU accuracy")
ax1.set_ylabel("MMLU accuracy (%)")
ax1.fill_between(x, acc_lo, acc_hi, color=BAND, alpha=0.14, linewidth=0, zorder=1)
series(ax1, acc)
ax1.axhline(base["mean"] * 100, color="#444444", linestyle=(0, (1.5, 2.5)), linewidth=2.5, zorder=2)
# Points within ~1pp of the dotted base line get their label BELOW the marker, else the
# label lands on the line and becomes unreadable.
for xi, yi in zip(x, acc):
    below = yi < base["mean"] * 100 and (base["mean"] * 100 - yi) < 1.5
    ax1.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                 xytext=(0, -34) if below else (0, 16), ha="center", fontsize=19)
ax1.set_ylim(86, 96)
ax1.set_xlim(-3, 43)
handles = [
    plt.Line2D([], [], color=BLUE, linewidth=3.5, marker="o", markersize=13,
               markerfacecolor=BLUE, markeredgecolor=BLUE_EDGE, markeredgewidth=2,
               label="difficult-advice mixture LoRA"),
    plt.Line2D([], [], color="none", marker="o", markersize=14, markerfacecolor=ORANGE,
               markeredgecolor=ORANGE_EDGE, markeredgewidth=2,
               label="TULU-only LoRA (0% difficult advice)"),
    plt.Line2D([], [], color="#444444", linestyle=(0, (1.5, 2.5)), linewidth=2.5,
               label=f"untuned Qwen3.6-27B base  {base['mean'] * 100:.1f}%"),
]
ax1.legend(handles=handles, loc="lower right", frameon=False)

# --- right: paired difference vs base ----------------------------------------------
style(ax2)
ax2.set_title("Paired Δ vs base, 95% CI (shared questions)")
ax2.set_ylabel("Accuracy difference vs base (pp)")
ax2.fill_between(x, diff_lo, diff_hi, color=BAND, alpha=0.14, linewidth=0, zorder=1)
ax2.axhline(0, color="#555555", linewidth=1.8, zorder=2)
ax2.axhline(-3, color="#C0392B", linestyle=(0, (4, 3)), linewidth=2.5, zorder=2)
ax2.text(19, -3.35, "non-inferiority margin  −3pp", color="#C0392B", fontsize=17,
         ha="center", va="top")
series(ax2, diff)
for xi, yi in zip(x, diff):
    ax2.annotate(f"{yi:+.1f}", (xi, yi), textcoords="offset points", xytext=(0, 16),
                 ha="center", fontsize=19)
ax2.set_ylim(-4.6, 3.2)

fig.text(0.5, -0.035,
         "n = 570 questions/arm, stratified across 57 MMLU subjects · identical subset, "
         "prompts and decoding for every arm · temperature 0 · band = 95% CI",
         ha="center", fontsize=15, color="#666666")

fig.tight_layout(rect=(0, 0, 1, 0.97))
out = figure_path(REPORT_DIR, "mmlu_dose_response")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(out)

# --- companion: same style, split by MMLU subject group ------------------------------
CATS = [("STEM", "STEM"), ("humanities", "humanities"),
        ("social_sciences", "social sciences"), ("other", "other")]
fig2, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharey=True, facecolor="white")
fig2.suptitle("MMLU by subject group — dotted line and band = untuned Qwen3.6-27B base (thinking mode)",
              fontsize=23, y=1.04)
for ax, (key, label) in zip(axes, CATS):
    style(ax)
    b = base["by_category"][key]
    ax.set_title(f"{label}\nn={b['n']}", fontsize=19)
    ax.axhspan(b["ci_lower"] * 100, b["ci_upper"] * 100, color="#999999", alpha=0.15, linewidth=0)
    ax.axhline(b["mean"] * 100, color="#444444", linestyle=(0, (1.5, 2.5)), linewidth=2.2)
    ys = [scores[k]["by_category"][key]["mean"] * 100 for _, k in ARMS]
    lo = [scores[k]["by_category"][key]["ci_lower"] * 100 for _, k in ARMS]
    hi = [scores[k]["by_category"][key]["ci_upper"] * 100 for _, k in ARMS]
    ax.fill_between(x, lo, hi, color=BAND, alpha=0.14, linewidth=0, zorder=1)
    series(ax, ys)
    ax.set_xlim(-3, 43)
    ax.set_xlabel("")
fig2.supxlabel("Difficult-advice share of training tokens (%)", fontsize=19, y=-0.02)
axes[0].set_ylabel("MMLU accuracy (%)")
axes[0].set_ylim(78, 100)
fig2.text(0.5, -0.06,
          "blue band = 95% CI of the mixture arms · gray band = 95% CI of the base · "
          "orange = TULU-only LoRA (0% difficult advice)",
          ha="center", fontsize=15, color="#666666")
fig2.tight_layout(rect=(0, 0, 1, 0.96))
out2 = figure_path(REPORT_DIR, "mmlu_dose_response_by_category")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
print(out2)
