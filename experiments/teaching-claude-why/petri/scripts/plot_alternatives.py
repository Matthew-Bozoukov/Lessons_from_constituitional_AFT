# ABOUTME: Three alternative renderings of the dose-sweep result, for picking a house style.
# ABOUTME: Each answers a different question; none replaces the reference-layout figure yet.
"""Chart options for "how is each arm doing on frequency of violations".

The reference-layout figure (`plot_violation_curve.py`) reproduces a published
chart built for a clean dose-response measured over hundreds of samples per point.
This run's shape is different - one arm separates, the trend is not monotonic, and
the judge's own error rate varies by arm - so the borrowed layout draws a
connecting line through four points whose intervals all overlap, which reads as a
trend the data does not support.

Three options, each answering a question the line chart cannot:

A  "How often, and can I believe it?"  Rate per arm with its interval, beside the
   judge's false-positive rate on the control seeds. The caveat sits next to the
   number it undermines instead of in a footnote.

B  "What actually changed?"  The paired flips. McNemar's p=0.029 is 22 scenarios
   fixed against 9 broken; that is a concrete, countable claim and it should look
   like one.

C  "Show me the raw frequency."  One dot per audit. No axis to decode, no interval
   to interpret - the proportion is the picture. Weakest for inference, strongest
   for a reader who wants to see the sample size honestly.

Usage:
    python scripts/plot_alternatives.py --results output/analysis-v2/results.json \
        --out output/analysis-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LABEL = {"base": "Base\n(0% difficult-advice)", "dose-10-90": "10% arm",
         "dose-20-80": "20% arm", "dose-40-60": "40% arm"}
# Option C lays labels out in figure space rather than on an axis, so it needs a
# one-line form; the two-line base label overflows the margin there.
LABEL_1L = {"base": "Base  (0%)", "dose-10-90": "10% arm",
            "dose-20-80": "20% arm", "dose-40-60": "40% arm"}
ORDER = ["base", "dose-10-90", "dose-20-80", "dose-40-60"]

RED = "#d1495b"      # violation
GREEN = "#0f7b6c"    # improvement / safe
BLUE = "#2b7bba"
GREY = "#8b929c"
LIGHT = "#e6e9ed"
GRID = "#dfe3e8"
INK = "#1f2328"


def _clean(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=GREY, labelsize=10, length=0)
    (ax.xaxis if xgrid else ax.yaxis).grid(True, color=GRID, linewidth=0.9)
    (ax.yaxis if xgrid else ax.xaxis).grid(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- option A ---
def option_a(res: dict, out: Path) -> Path:
    per = res["per_arm"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ys = list(range(len(ORDER)))[::-1]

    ax = axes[0]
    for y, arm in zip(ys, ORDER):
        r = per[arm]["any_violation"]
        lo, hi = r["ci95"]
        best = arm == "dose-40-60"
        col = GREEN if best else GREY
        ax.plot([lo, hi], [y, y], color=col, linewidth=3.2, alpha=.35,
                solid_capstyle="round", zorder=2)
        ax.plot(r["rate"], y, "o", markersize=13, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=1.8)
        ax.annotate(f"{r['rate']*100:.1f}%", (r["rate"], y), xytext=(0, 15),
                    textcoords="offset points", ha="center", fontsize=12,
                    fontweight="bold", color=INK if best else GREY)
        ax.annotate(f"{r['k']} of {r['n']} audits", (r["rate"], y), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=9, color=GREY)
    ax.set_yticks(ys, [LABEL[a] for a in ORDER], fontsize=11, color=INK)
    ax.set_xlim(0, .46)
    # Headroom above the top row and below the bottom one: both carry stacked
    # annotations, and without it the bottom caption lands on the tick labels.
    ax.set_ylim(-0.72, len(ORDER) - 0.42)
    ax.set_xticks([0, .1, .2, .3, .4], ["0%", "10%", "20%", "30%", "40%"])
    ax.set_xlabel("Share of audits with at least one violation", fontsize=10.5, color=GREY)
    ax.set_title("How often did each model violate the constitution?",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=26)
    _clean(ax)

    ax = axes[1]
    for y, arm in zip(ys, ORDER):
        c = per[arm]["control_false_positive"]
        bad = c["rate"] >= .3
        col = RED if bad else GREY
        ax.barh(y, c["rate"], height=.42, color=col, alpha=.85 if bad else .45)
        ax.annotate(f"{c['rate']*100:.0f}%   ({c['k']} of {c['n']})",
                    (c["rate"], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10.5,
                    fontweight="bold" if bad else "normal", color=INK if bad else GREY)
    ax.set_yticks(ys, ["", "", "", ""])
    ax.set_xlim(0, .82)
    ax.set_ylim(-0.72, len(ORDER) - 0.42)
    ax.set_xticks([0, .2, .4], ["0%", "20%", "40%"])
    ax.set_xlabel("Flags on seeds with nothing to violate", fontsize=10.5, color=GREY)
    ax.set_title("…and how often was the judge wrong?",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=26)
    _clean(ax)

    fig.text(.5, .015,
             "Left: lower is better. Right: every one of these is a false positive by construction — the control seeds contain nothing to violate.\n"
             "The 40% arm has the lowest violation rate AND the least reliable judge. That is why it is a lead, not a result.",
             ha="center", fontsize=9.2, color=GREY, linespacing=1.6)
    fig.subplots_adjust(top=.83, bottom=.24, left=.155, right=.985, wspace=.12)
    p = out / "option_a_rate_and_reliability.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- option B ---
def option_b(res: dict, out: Path) -> Path:
    paired = res["paired_vs_base"]
    arms = [a for a in ORDER if a in paired]
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ys = list(range(len(arms)))[::-1]

    for y, arm in zip(ys, arms):
        p = paired[arm]
        fixed, broke = p["base_violation_arm_safe"], p["base_safe_arm_violation"]
        ax.barh(y, -fixed, height=.5, color=GREEN, alpha=.9)
        ax.barh(y, broke, height=.5, color=RED, alpha=.9)
        ax.annotate(f"{fixed} fixed", (-fixed, y), xytext=(-9, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=11, fontweight="bold", color=GREEN)
        ax.annotate(f"{broke} broken", (broke, y), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=11, fontweight="bold", color=RED)
        sig = p["mcnemar_exact_p"] < .05
        ax.annotate(f"net {fixed-broke:+d}   ·   p = {p['mcnemar_exact_p']:.3g}"
                    + ("  ✓" if sig else ""),
                    (0, y), xytext=(0, -25), textcoords="offset points", ha="center",
                    fontsize=9.5, color=INK if sig else GREY,
                    fontweight="bold" if sig else "normal")

    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(ys, [LABEL[a] for a in arms], fontsize=11.5, color=INK)
    ax.set_xlim(-30, 30)
    # The net/p-value caption hangs 25pt below each bar; the bottom one needs
    # room or it lands on the axis tick labels.
    ax.set_ylim(-0.78, len(arms) - 0.45)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30], ["30", "20", "10", "0", "10", "20", "30"])
    ax.set_xlabel("Scenarios that changed verdict, out of ~135 run against both models",
                  fontsize=10.5, color=GREY)
    ax.set_title("Same scenario, two models: what actually changed?",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=30)
    ax.legend(handles=[Patch(color=GREEN, alpha=.9, label="Base violated  →  tuned model did not"),
                       Patch(color=RED, alpha=.9, label="Base was fine  →  tuned model violated")],
              frameon=False, fontsize=10, loc="upper center",
              bbox_to_anchor=(.5, 1.16), ncol=2)
    _clean(ax)

    fig.text(.5, .02,
             "Only scenarios where the two models disagreed are counted — the ~100 they both got right or both got wrong carry no information about the difference.\n"
             "This is the comparison McNemar's test runs, and where nearly all of this design's statistical power sits.",
             ha="center", fontsize=9.2, color=GREY, linespacing=1.6)
    fig.subplots_adjust(top=.72, bottom=.30, left=.20, right=.97)
    p = out / "option_b_paired_flips.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- option C ---
def option_c(res: dict, out: Path) -> Path:
    per = res["per_arm"]
    cols = 25
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(10.6, 6.6))

    for ax, arm in zip(axes, ORDER):
        r = per[arm]["any_violation"]
        n, k = r["n"], r["k"]
        for i in range(n):
            row, col = divmod(i, cols)
            ax.plot(col, -row, "o", markersize=8.5,
                    color=RED if i < k else LIGHT,
                    markeredgecolor="white", markeredgewidth=.8)
        rows = (n - 1) // cols + 1
        ax.set_xlim(-1, cols)
        ax.set_ylim(-rows + .4, .6)
        ax.axis("off")
        ax.text(-1.6, -(rows - 1) / 2, LABEL_1L[arm], ha="right",
                va="center", fontsize=12, color=INK)
        ax.text(cols + .8, -(rows - 1) / 2 - .30, f"{r['rate']*100:.1f}%",
                ha="left", va="center", fontsize=14, fontweight="bold",
                color=RED if arm == "base" else (GREEN if arm == "dose-40-60" else GREY))
        ax.text(cols + .8, -(rows - 1) / 2 + .62, f"{k} of {n}", ha="left",
                va="center", fontsize=9.5, color=GREY)

    fig.text(.5, .955, "Every audit, one dot", ha="center", fontsize=14.5,
             fontweight="bold", color=INK)
    fig.text(.5, .918, "Filled = the judge flagged at least one constitution violation",
             ha="center", fontsize=10.5, color=GREY)
    fig.text(.5, .028,
             "No axis to decode — the proportion is the picture, and the sample size is visible rather than stated.\n"
             "These are judge flags, not adjudicated findings; the 40% arm's own control seeds were misflagged 45% of the time.",
             ha="center", fontsize=9.2, color=GREY, linespacing=1.6)
    fig.subplots_adjust(top=.88, bottom=.13, left=.17, right=.83, hspace=.35)
    p = out / "option_c_dot_matrix.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (option_a, option_b, option_c):
        print("wrote", f(res, out))


if __name__ == "__main__":
    main()
