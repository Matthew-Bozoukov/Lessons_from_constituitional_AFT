# ABOUTME: Four-bar ODCV figure: the two difficult-advice arms, deliberation-without-morality,
# ABOUTME: and no synthetic SFT. Run: uv run python scratch/nonmoral/plot_four_arms.py

"""The one comparison this arm was built to make, and nothing else on the axes.

Numbers come from each arm's published `results.json` on HF — one consistent source — rather
than mixing a dashboard re-score with a published figure. That matters here: the dashboard
shows da716 at 16.3% because it re-scores on the restricted 65-cell set, while its published
overall is 17.8%. Neither is wrong; they are different cell sets, and the caption says so.

Two encodings are load-bearing rather than decorative:

* The pass count sits inside each bar, because it is what makes the figure honest — the arm
  under test has ONE pass against four and five for the others, so its interval is wide for a
  reason the reader should see without hunting for it.
* Low stakes shows its two SEEDS as points on a stem, not a confidence interval, because that
  is what was actually measured. Their 6.1 pp spread is the empirical argument for why a
  single-seed number cannot be ranked — including this one's.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import timestamp

DA, MINE, REF = "#eb6834", "#2a78d6", "#8b9199"

# (label, MR %, ci_lo, ci_hi, passes label, colour, seed points or None)
ARMS = [
    ("difficult advice\n(low stakes)",  13.8, None, None, "2 seeds",  DA,   [16.9, 10.8]),
    ("difficult advice\n(high stakes)", 17.8, 13.3, 25.0, "4 passes", DA,   None),
    ("non-moral\ndeliberation",         25.0, 13.4, 41.9, "1 pass",   MINE, None),
    ("no synthetic SFT",                43.6, 37.3, 51.7, "5 passes", REF,  None),
]

OUT = Path("output/nonmoral_deliberation")
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = list(range(len(ARMS)))
ax.bar(x, [a[1] for a in ARMS], width=0.54, color=[a[5] for a in ARMS], zorder=2)

# Confidence intervals where one was computed on this protocol.
ci_x = [i for i, a in enumerate(ARMS) if a[2] is not None]
if ci_x:
    ax.errorbar(
        ci_x, [ARMS[i][1] for i in ci_x],
        yerr=[[ARMS[i][1] - ARMS[i][2] for i in ci_x],
              [ARMS[i][3] - ARMS[i][1] for i in ci_x]],
        fmt="none", ecolor="#1a1a1a", elinewidth=1.3, capsize=9, capthick=1.3, zorder=3)

# Low stakes: the two seeds themselves, on a stem. Not a CI — these are the measurements.
for i, a in enumerate(ARMS):
    seeds = a[6]
    if not seeds:
        continue
    ax.plot([i, i], [min(seeds), max(seeds)], color="#1a1a1a", linewidth=1.3, zorder=3)
    ax.plot([i] * len(seeds), seeds, "o", markersize=8, markerfacecolor="white",
            markeredgecolor="#1a1a1a", markeredgewidth=1.3, zorder=4)
    # Annotate to the RIGHT of the stem: on the left it collides with the y-axis label.
    ax.annotate(
        "", xy=(i + 0.33, max(seeds)), xytext=(i + 0.33, min(seeds)),
        arrowprops=dict(arrowstyle="<->", color="#1a1a1a", linewidth=1.0), zorder=4)
    ax.text(i + 0.39, sum(seeds) / 2, f"{max(seeds) - min(seeds):.1f} pp\nbetween seeds",
            ha="left", va="center", fontsize=8.5, color="#333333", linespacing=1.4, zorder=4)

# Value above whatever the bar's tallest mark is; pass count inside the bar.
for i, (_, mr, _, ci_hi, passes, _, seeds) in enumerate(ARMS):
    top = ci_hi if ci_hi is not None else (max(seeds) if seeds else mr)
    ax.text(i, top + 2.2, f"{mr:.1f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color="#111111", zorder=5)
    ax.text(i, 1.6, passes, ha="center", va="bottom", fontsize=10, color="white", zorder=5)

ax.set_title("ODCV misalignment by recipe", fontsize=17, color="#111111", loc="left", pad=18)
ax.set_ylabel("misalignment rate (%)", fontsize=11, color="#444444")
ax.set_ylim(0, 62)
ax.set_yticks([0, 20, 40, 60])
ax.set_yticklabels(["0", "20", "40", "60%"], fontsize=11, color="#444444")
ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in ARMS], fontsize=12, color="#111111")

ax.yaxis.grid(True, color="#dddddd", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")
ax.tick_params(axis="both", length=0)

fig.text(
    0.005, 0.005,
    "Bars are the ODCV-Bench misalignment rate. Whiskers are the 95% CI; low stakes instead shows "
    "its two seeds (16.9%, 10.8%) as points,\nbecause that is what was measured. Each arm is its own "
    "published results.json. Cell sets are NOT identical: the non-moral arm scores 28 scenarios /\n56 cells "
    "after nine incomplete scenarios were dropped; the others are on their own published sets. One "
    "pass here means rollout\nnoise sits inside the interval and cannot be separated from it.",
    fontsize=7.6, color="#666666", va="bottom", linespacing=1.6)

fig.tight_layout(rect=(0, 0.10, 1, 1))
stamp = timestamp()
png = OUT / f"odcv_four_arms_{stamp}.png"
fig.savefig(png, facecolor="white")
print(f"wrote {png}")

md = OUT / f"odcv_four_arms_{stamp}_results.md"
md.write_text(
    "# ODCV misalignment by recipe — four arms\n\n"
    "| arm | MR % | 95% CI | passes | source |\n|---|---:|---|---|---|\n"
    "| difficult advice (low stakes) | 13.8 | seeds 16.9 / 10.8 (6.1 pp apart) | 2 seeds | "
    "team dashboard, low-stakes arm |\n"
    "| difficult advice (high stakes, da716) | 17.8 | [13.3, 25.0] | 4 | "
    "`2026-08-14-qwen36-lora-table2-9284-difficult-advice-716-rank-64-dynbatch"
    "/combined4x_20260814_230249` |\n"
    "| non-moral deliberation (craft tensions, 684) | 25.0 | [13.4, 41.9] | 1 | "
    "`2026-09-02-odcv-nonmoral-deliberation-684-eval/passes/laptop/20260902_104119` |\n"
    "| no synthetic SFT (table2-only 9,284) | 43.6 | [37.3, 51.7] | 5 | "
    "`2026-08-05-qwen36-table2-only-9284-rank-64/combined5x_20260805_132959` |\n\n"
    "Cell sets differ: the non-moral arm scores 28 scenarios / 56 cells (nine incomplete scenarios "
    "dropped because the inherited exclusion list names only one variant of each); the others "
    "are on their own published sets. The dashboard shows da716 at 16.3% because it re-scores "
    "on the restricted 65-cell set — 17.8% is its published overall.\n\n"
    "The low-stakes seed spread (6.1 pp) is the empirical argument for why the single-pass "
    "number for the non-moral arm cannot be ranked against either neighbour.\n",
    encoding="utf-8")
print(f"wrote {md}")
