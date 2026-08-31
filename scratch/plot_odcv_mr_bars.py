# ABOUTME: Bar plot of ODCV misalignment rate (with 95% CI error bars) for the memself,
# ABOUTME: selfreflect and table2-synthdoc r64 arms, read straight from each run's results.json.
"""Compare ODCV misalignment rates across three r64 arms as a bar chart.

Values (mr_pct + mr_ci95) come from the `ours.overall` block of each arm's combined ODCV
results.json — all three judged by grok-4.20 + gemini-3.1-pro-preview, so comparable. Output:
a PNG + a markdown mirror of the numbers.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import timestamp

# (label, results.json)
ARMS = [
    ("memself",
     "output/odcv_bench/qwen3_6-27b-lora-table2-80-memself-20-r64/combined4x_20260808_142305/results.json"),
    ("selfreflect",
     "output/odcv_bench/qwen3_6-27b-lora-table2-80-selfreflect-20-r64/combined4x_20260809_134735/results.json"),
    ("difficult advice",
     "output/odcv_bench/qwen3.6-27b-table2-synthdoc-r64/combined5x_20260805_103343/results.json"),
]

COLORS = ["#d1495b", "#edae49", "#30638e"]


def main() -> None:
    """Draw the misalignment-rate bar chart and write PNG + markdown mirror under output/report/."""
    labels, mr, lo_err, hi_err, ns = [], [], [], [], []
    for label, path in ARMS:
        o = json.load(open(path))["ours"]["overall"]
        labels.append(label)
        mr.append(o["mr_pct"])
        lo_err.append(o["mr_pct"] - o["mr_ci95"][0])
        hi_err.append(o["mr_ci95"][1] - o["mr_pct"])
        ns.append(o["n"])

    plt.rcParams.update({"font.size": 11, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = range(len(labels))
    bars = ax.bar(x, mr, width=0.62, color=COLORS, edgecolor="black", linewidth=0.6, zorder=3)
    ax.errorbar(x, mr, yerr=[lo_err, hi_err], fmt="none", ecolor="#222", capsize=6,
                capthick=1.4, elinewidth=1.4, zorder=4)

    for xi, (m, hi) in enumerate(zip(mr, hi_err)):
        ax.text(xi, m + hi + 1.1, f"{m:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("misalignment %")
    ax.set_title("ODCV Misalignment Rate", fontsize=14, fontweight="bold", pad=16)
    ax.text(0.5, 1.015, "95% CI · lower is better",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, style="italic", color="#555")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(m + h for m, h in zip(mr, hi_err)) + 6)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    png = f"output/report/odcv_mr_bars_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f">>> {png}")

    md = ["# ODCV misalignment rate — memself vs selfreflect vs table2-synthdoc", "",
          "Judges: grok-4.20 + gemini-3.1-pro-preview (identical across arms).", "",
          "| arm | MR % | 95% CI | n |", "|---|---|---|---|"]
    for (label, path), m, le, he, n in zip(ARMS, mr, lo_err, hi_err, ns):
        md.append(f"| {label.replace(chr(10), ' ')} | {m:.1f} | [{m - le:.1f}, {m + he:.1f}] | {n} |")
    md += ["", f"Plot: `{png}`", ""]
    mdp = f"output/report/odcv_mr_bars_{ts}.md"
    open(mdp, "w").write("\n".join(md))
    print(f">>> {mdp}")


if __name__ == "__main__":
    main()
