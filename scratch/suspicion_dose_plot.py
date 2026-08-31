# ABOUTME: Dose-response plot of premature suspicion (mean of the incentivized and mandated
# ABOUTME: conditions) against the difficult-advice share of the SFT mixture.

"""Premature suspicion vs difficult-advice share, read from the per-arm judge JSONLs.

The plotted statistic is the unweighted mean of the two ODCV conditions, matching the
comparison table. Confidence intervals come from a bootstrap that resamples *scenarios*, not
transcripts: each scenario contributes 4-5 rollouts of the same task, so treating transcripts
as independent would understate the interval substantially.

x follows the same nominal ladder as the Arena-Hard dose figure (0 / 5 / 10 / 20); the exact
share by examples is 0 / 7.2 / 10.0 / 21.6.

    uv run python scratch/suspicion_dose_plot.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import timestamp  # noqa: E402

# (nominal share %, label, judged jsonl)
ARMS = [
    (0, "table2-only", "output/report/premature_suspicion_20260809_210319.jsonl"),
    (5, "t2-9284 + synthdoc-716", "output/report/premature_suspicion_20260809_210744.jsonl"),
    (10, "t2-9000 + synthdoc-1000", "output/report/premature_suspicion_20260809_210813.jsonl"),
    (20, "table2 + synthdoc-2203", "output/report/premature_suspicion_20260809_205845.jsonl"),
]

N_BOOT = 4000
SEED = 0


def condition_of(row: dict) -> str:
    """Return the ODCV condition ('incentivized' or 'mandated') for a judged row."""
    return "incentivized" if "incentivized" in row["condition"] else "mandated"


def macro_rate(clusters: dict[str, list[list[bool]]]) -> float:
    """Mean of the two per-condition flag rates, given scenario-grouped verdicts."""
    rates = []
    for verdicts in clusters.values():
        flat = [v for scenario in verdicts for v in scenario]
        rates.append(sum(flat) / len(flat) * 100)
    return sum(rates) / len(rates)


def arm_stats(jsonl: str) -> tuple[float, float, float, int]:
    """Mean-of-conditions rate and a scenario-clustered bootstrap 95% CI.

    Args:
        jsonl: A premature_suspicion_*.jsonl written by the judge.

    Returns:
        (rate, ci_low, ci_high, n_transcripts).
    """
    rows = [json.loads(line) for line in open(jsonl)]
    assert not any(r["error"] for r in rows), f"unresolved judge errors in {jsonl}"

    # condition -> scenario -> [verdicts]
    grouped: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[condition_of(r)][r["scenario"]].append(bool(r["premature"]))

    clusters = {cond: list(sc.values()) for cond, sc in grouped.items()}
    point = macro_rate(clusters)

    rng = random.Random(SEED)
    draws = []
    for _ in range(N_BOOT):
        resampled = {cond: [rng.choice(scen) for _ in scen] for cond, scen in clusters.items()}
        draws.append(macro_rate(resampled))
    draws.sort()
    return point, draws[int(0.025 * N_BOOT)], draws[int(0.975 * N_BOOT)], len(rows)


def main(out_dir: str = "output/plots") -> None:
    """Draw the dose-response figure and write a markdown mirror of its numbers.

    Args:
        out_dir: Directory for the PNG (the markdown mirror lands beside it).
    """
    stats = [(share, label, *arm_stats(path)) for share, label, path in ARMS]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    xs = [s[0] for s in stats]
    ys = [s[2] for s in stats]
    lo = [s[3] for s in stats]
    hi = [s[4] for s in stats]

    ax.fill_between(xs, lo, hi, color="#3b7dd8", alpha=0.18, zorder=2)
    ax.plot(xs, ys, "-o", color="#3b7dd8", markersize=10, markerfacecolor="#3b7dd8",
            markeredgecolor="black", markeredgewidth=0.8, linewidth=2.2, zorder=3)

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 16),
                    ha="center", fontsize=14, fontweight="bold")

    ax.set_xlabel("difficult-advice share of SFT mixture (%)", fontsize=15)
    ax.set_ylabel("premature suspicion (%)", fontsize=15)
    ax.set_title("Premature suspicion of the user vs difficult-advice share",
                 fontsize=16, fontweight="bold", pad=14)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.set_ylim(0, 60)
    ax.grid(True, linestyle="--", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"premature_suspicion_dose_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f">>> {png}")

    md = ["# Premature suspicion vs difficult-advice share", "",
          "Mean of the incentivized and mandated conditions. 95% CI from a "
          f"{N_BOOT}-draw bootstrap resampling scenarios (seed {SEED}), since rollouts "
          "within a scenario are repeats of one task and are not independent.", "",
          "| share | arm | premature suspicion | 95% CI | n transcripts |",
          "|---|---|---|---|---|"]
    for share, label, rate, l, h, n in stats:
        md.append(f"| {share}% | {label} | {rate:.1f}% | [{l:.1f}, {h:.1f}] | {n} |")
    md += ["", f"Plot: `{png}`", ""]
    mdp = dest / f"premature_suspicion_dose_{ts}.md"
    mdp.write_text("\n".join(md))
    print(f">>> {mdp}")
    for share, label, rate, l, h, n in stats:
        print(f"    {share:>3}%  {rate:>5.1f}%  [{l:.1f}, {h:.1f}]  n={n}  {label}")


if __name__ == "__main__":
    fire.Fire(main)
