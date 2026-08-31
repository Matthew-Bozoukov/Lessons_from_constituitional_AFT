# ABOUTME: Bar chart of premature suspicion (mean of conditions) for the three 20%-share
# ABOUTME: arms — difficult advice vs memself vs selfreflect — at equal synthetic budget.

"""Premature suspicion across the three 20%-synthetic arms.

All three mix 20% synthetic data into the same table2 base, so this contrast isolates the
data *type* rather than the dose. Colours match `scratch/plot_odcv_mr_bars.py` so the same
arm is the same colour in both figures.

The statistic and its scenario-clustered bootstrap CI come from `suspicion_dose_plot.arm_stats`
— one implementation, so the bars and the dose figure cannot drift apart.

    uv run python scratch/suspicion_ablation_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "scratch")
from suspicion_dose_plot import arm_stats  # noqa: E402

from src.utils import timestamp  # noqa: E402

# (label, judged jsonl, colour) — colours per arm as in plot_odcv_mr_bars.py
ARMS = [
    ("difficult advice", "output/report/premature_suspicion_20260809_205845.jsonl", "#30638e"),
    ("memself", "output/report/premature_suspicion_20260810_074841.jsonl", "#d1495b"),
    ("selfreflect", "output/report/premature_suspicion_20260810_074911.jsonl", "#edae49"),
]


def main(out_dir: str = "output/plots") -> None:
    """Draw the bar chart and write a markdown mirror of the numbers.

    Args:
        out_dir: Directory for the PNG and its markdown mirror.
    """
    stats = [(label, colour, *arm_stats(path)) for label, path, colour in ARMS]

    plt.rcParams.update({"font.size": 14, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    x = range(len(stats))
    rates = [s[2] for s in stats]
    lo_err = [s[2] - s[3] for s in stats]
    hi_err = [s[4] - s[2] for s in stats]

    ax.bar(x, rates, width=0.62, color=[s[1] for s in stats],
           edgecolor="black", linewidth=0.8, zorder=3)
    ax.errorbar(x, rates, yerr=[lo_err, hi_err], fmt="none", ecolor="#222",
                capsize=6, capthick=1.4, elinewidth=1.4, zorder=4)

    for xi, (rate, hi) in enumerate(zip(rates, hi_err)):
        ax.text(xi, rate + hi + 1.4, f"{rate:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    ax.set_ylabel("premature suspicion (%)", fontsize=15)
    ax.set_title("Premature suspicion of the user\n20% synthetic share, three data types",
                 fontsize=16, fontweight="bold", pad=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in stats], fontsize=14)
    ax.set_ylim(0, max(r + h for r, h in zip(rates, hi_err)) + 8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"premature_suspicion_ablation_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")

    md = ["# Premature suspicion — 20% synthetic share, three data types", "",
          "Mean of the incentivized and mandated conditions. 95% CI from a bootstrap that "
          "resamples scenarios, since rollouts within a scenario repeat one task.", "",
          "| arm | premature suspicion | 95% CI | n transcripts |", "|---|---|---|---|"]
    for label, _, rate, lo, hi, n in stats:
        md.append(f"| {label} | {rate:.1f}% | [{lo:.1f}, {hi:.1f}] | {n} |")
    md += ["", f"Plot: `{png}`", ""]
    (dest / f"premature_suspicion_ablation_{ts}.md").write_text("\n".join(md))

    print(f">>> {png}")
    print(f">>> {dest / f'premature_suspicion_ablation_{ts}.md'}")
    for label, _, rate, lo, hi, n in stats:
        print(f"    {label:<20} {rate:>5.1f}%  [{lo:.1f}, {hi:.1f}]  n={n}")


if __name__ == "__main__":
    fire.Fire(main)
