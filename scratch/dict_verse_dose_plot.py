# ABOUTME: Dose-response line plot of dict-verse (L7) jailbreak RESISTANCE vs the difficult-advice
# ABOUTME: share of the SFT mixture, with selfreflect-20 marked separately (different data type).

"""Dict-verse resistance vs difficult-advice dose (higher is better)."""

from __future__ import annotations

import json
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import timestamp  # noqa: E402

# difficult-advice dose points: (share %, results.json)
DOSE = [
    (0, "output/dict_verse/qwen3_6-27b-lora-table2-only-9284-r64/rerun8192_20260811_112430/results.json"),
    (5, "output/dict_verse/qwen3_6-27b-lora-t2-9284-synthdoc-716-r64/20260811_115618/results.json"),
    (20, "output/dict_verse/qwen3_6-27b-lora-table2-synthdoc-r64/20260811_130435/results.json"),
]
# a different data type at 20% share, shown as a lone marker for contrast
SELFREFLECT = (20, "output/dict_verse/qwen3_6-27b-lora-table2-80-selfreflect-20-r64/20260811_123620/results.json")


def wilson(path: str) -> tuple[float, float, float]:
    """(point, lo, hi) Wilson 95% resistance interval, as percentages, from a results.json."""
    d = json.load(open(path))
    n = d["n_scored"]
    k = round(d["resistance_rate"] * n)
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, max(0.0, centre - half) * 100, (centre + half) * 100


def main() -> None:
    """Draw the dose-response line + markdown mirror under output/plots/."""
    xs, ys, los, his = [], [], [], []
    for share, path in DOSE:
        pt, lo, hi = wilson(path)
        xs.append(share)
        ys.append(pt)
        los.append(lo)
        his.append(hi)
    sr_pt, sr_lo, sr_hi = wilson(SELFREFLECT[1])

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    ax.fill_between(xs, los, his, color="#30638e", alpha=0.15, zorder=1)
    ax.plot(xs, ys, "-o", color="#30638e", markersize=10, markerfacecolor="#30638e",
            markeredgecolor="black", markeredgewidth=0.8, linewidth=2.2, zorder=3,
            label="difficult advice")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=14, fontweight="bold", color="#30638e")

    # selfreflect: lone amber marker at 20% share
    ax.errorbar([SELFREFLECT[0]], [sr_pt], yerr=[[sr_pt - sr_lo], [sr_hi - sr_pt]],
                fmt="D", color="#edae49", markersize=10, markeredgecolor="black",
                markeredgewidth=0.8, capsize=5, elinewidth=1.4, zorder=4,
                label="selfreflect (20% share)")
    ax.annotate(f"{sr_pt:.1f}%", (SELFREFLECT[0], sr_pt), textcoords="offset points",
                xytext=(16, -4), ha="left", fontsize=13, fontweight="bold", color="#b9822b")

    ax.set_xlabel("difficult-advice share of SFT mixture (%)", fontsize=15)
    ax.set_ylabel("jailbreak resistance %", fontsize=15)
    ax.set_title("Dictator-Bench (L7) Resistance vs Difficult-Advice Dose",
                 fontsize=15, fontweight="bold", pad=14)
    ax.text(0.5, 1.015, "27 verse scenarios x 5 samples · 95% CI · higher is better",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            style="italic", color="#555")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(0, max(his + [sr_hi]) + 4)
    ax.set_xlim(-1.5, 22)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    png = f"output/plots/dict_verse_dose_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    md = ["# Dict-verse (L7) resistance vs difficult-advice dose", "",
          "Resistance = judge score >= 4 over scorable answers. Wilson 95% CI.", "",
          "| difficult-advice share | resistance | 95% CI |", "|---|---|---|"]
    for (share, path), y, lo, hi in zip(DOSE, ys, los, his):
        md.append(f"| {share}% | {y:.1f}% | [{lo:.1f}, {hi:.1f}] |")
    md.append(f"| selfreflect (20% share) | {sr_pt:.1f}% | [{sr_lo:.1f}, {sr_hi:.1f}] |")
    md += ["", f"Plot: `{png}`", ""]
    open(f"output/plots/dict_verse_dose_{ts}.md", "w").write("\n".join(md))
    print(f">>> {png}")


if __name__ == "__main__":
    main()
