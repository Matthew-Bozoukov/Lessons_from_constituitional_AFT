# ABOUTME: Bar chart of premature-suspicion rate on the 932 WildChat bait prompts across the
# ABOUTME: four SFT arms (difficult-advice, selfreflect, memself, standard), with Wilson 95% CIs.

"""Four-arm WildChat premature-suspicion bar chart.

Rates are 'either' (reasoning or response) over the judge-benign prompts, read from the
per-arm verdict JSONLs. Error bars are Wilson 95% intervals — appropriate for a proportion,
and honest at the small rates here where a normal approximation would dip below zero.
"""

from __future__ import annotations

import json
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import timestamp  # noqa: E402

# (label, verdict jsonl, colour) ordered high -> low. Per-arm colours match
# scratch/plot_odcv_mr_bars.py so the same arm is the same colour across both figures.
ARMS = [
    ("difficult advice", "output/wildchat/wildchat_suspicion_20260810_175203.jsonl", "#30638e"),
    ("selfreflect 20%", "output/wildchat/wildchat_suspicion_20260810_185251.jsonl", "#edae49"),
    ("memself 20%", "output/wildchat/wildchat_suspicion_20260810_185132.jsonl", "#d1495b"),
    ("standard SFT", "output/wildchat/wildchat_suspicion_20260810_175322.jsonl", "#8d99ae"),
]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Return (point, lo, hi) Wilson 95% interval for k successes in n, as percentages."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, max(0.0, (centre - half)) * 100, (centre + half) * 100


def arm_rate(path: str) -> tuple[float, float, float, int, int]:
    """(point%, lo%, hi%, flagged, benign_n) for the 'either' premature rate."""
    v = [json.loads(line) for line in open(path)]
    ben = [r for r in v if not r["error"] and not r["prompt_ambiguous"]]
    fe = sum(r["suspicion_in_reasoning"] or r["suspicion_in_response"] for r in ben)
    pt, lo, hi = wilson(fe, len(ben))
    return pt, lo, hi, fe, len(ben)


def main() -> None:
    """Draw the bar chart and write a markdown mirror under output/plots/."""
    stats = [(lbl, colour, *arm_rate(path)) for lbl, path, colour in ARMS]

    plt.rcParams.update({"font.size": 14, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    x = range(len(stats))
    pts = [s[2] for s in stats]
    lo_err = [s[2] - s[3] for s in stats]
    hi_err = [s[4] - s[2] for s in stats]

    ax.bar(x, pts, width=0.62, color=[s[1] for s in stats], edgecolor="black",
           linewidth=0.6, zorder=3)
    ax.errorbar(x, pts, yerr=[lo_err, hi_err], fmt="none", ecolor="#222", capsize=6,
                capthick=1.4, elinewidth=1.4, zorder=4)
    for xi, s in enumerate(stats):
        ax.text(xi, s[4] + 0.3, f"{s[2]:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("premature suspicion %")
    ax.set_title("Premature Suspicion on Benign WildChat Prompts",
                 fontsize=15, fontweight="bold", pad=16)
    ax.text(0.5, 1.015, "n=932 · 95% CI · lower is better", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, style="italic", color="#555")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in stats])
    ax.set_ylim(0, max(s[4] for s in stats) + 1.8)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    png = f"output/plots/wildchat_bait_arms_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")

    md = ["# Premature suspicion on 932 WildChat bait prompts — four arms", "",
          "Rate = reasoning OR response, over judge-benign prompts. Wilson 95% CI.", "",
          "| arm | premature | flagged / benign | 95% CI |", "|---|---|---|---|"]
    for lbl, _c, pt, lo, hi, fe, n in stats:
        md.append(f"| {lbl} | {pt:.1f}% | {fe}/{n} | [{lo:.1f}, {hi:.1f}] |")
    md += ["", f"Plot: `{png}`", ""]
    mdp = f"output/plots/wildchat_bait_arms_{ts}.md"
    open(mdp, "w").write("\n".join(md))
    print(f">>> {png}")
    print(f">>> {mdp}")


if __name__ == "__main__":
    main()
