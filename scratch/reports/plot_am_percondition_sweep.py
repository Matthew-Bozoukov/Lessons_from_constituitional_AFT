# ABOUTME: Per-condition blackmail dose-response: one line per blackmail condition across the
# ABOUTME: difficult-advice SFT-share arms (0/10/20/40%), on a shared x-axis. Qwen3.6-27B.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from src.naming import figure_path


def _pretty(cond: str) -> str:
    """Strip the 'blackmail_' prefix into a 'goal / urgency' label."""
    _, goal, urg = cond.split("_")
    return f"{goal} / {urg}"


def main(
    arms: str = ("0:output/eval_summaries/qwen36_tulu100.json,"
                 "10:output/eval_summaries/qwen36_1090.json,"
                 "20:output/eval_summaries/qwen36_tulu.json,"
                 "40:output/eval_summaries/qwen36_4060.json"),
    scenario: str = "blackmail",
    out: str = "output/agentic_misalignment/plots",
) -> None:
    """Plot each condition's harmful rate vs difficult-advice SFT share, one line per condition.

    Args:
        arms: comma-separated "share:summary.json" entries.
        scenario: condition prefix to plot ("blackmail" or "leaking").
        out: output directory.
    """
    shares, summaries = [], []
    for entry in arms.split(","):
        share, path = entry.split(":", 1)
        shares.append(float(share))
        summaries.append(json.loads(Path(path).read_text()))
    order = sorted(range(len(shares)), key=lambda i: shares[i])
    shares = [shares[i] for i in order]
    summaries = [summaries[i] for i in order]

    conds = sorted(k for k in summaries[0]["by_condition"] if k.startswith(scenario))
    assert conds, f"no {scenario} conditions found"

    fig, ax = plt.subplots(figsize=(9, 6.2))
    cmap = plt.get_cmap("tab10")
    for i, cond in enumerate(conds):
        ys = [100 * s["by_condition"][cond]["harmful"] / s["by_condition"][cond]["n"]
              for s in summaries]
        ax.plot(shares, ys, marker="o", linewidth=2, markersize=8, color=cmap(i),
                markeredgecolor="black", markeredgewidth=0.6, label=_pretty(cond))

    # Aggregate scenario line, bold black dashed, for reference.
    agg = [100 * s["by_scenario"][scenario]["harmful"] / s["by_scenario"][scenario]["n"]
           for s in summaries]
    ax.plot(shares, agg, marker="s", linewidth=2.8, markersize=9, color="black",
            linestyle="--", label=f"{scenario} (all)")

    ax.set_xlabel("Difficult-advice share of SFT tokens (%)", fontsize=13)
    ax.set_ylabel("Harmful response rate (%)", fontsize=13)
    ax.set_xticks(shares)
    ax.set_xticklabels([f"{int(x)}" for x in shares], fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{scenario.capitalize()} rate per condition vs difficult-advice SFT share\n"
                 "Qwen3.6-27B (0:100 = 100% Tulu control) · lower is better", fontsize=13)
    ax.legend(title="condition (goal / urgency)", fontsize=9, title_fontsize=10,
              loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fp = figure_path(out, f"agentic_misalignment_{scenario}_percondition_vs_sft_pct")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=170)
    print(f">>> {fp}")

    # console table
    print(f"{'condition':<28}" + "".join(f"{int(s):>7}%" for s in shares))
    for cond in conds:
        row = [100 * su["by_condition"][cond]["harmful"] / su["by_condition"][cond]["n"]
               for su in summaries]
        print(f"{_pretty(cond):<28}" + "".join(f"{r:>7.0f}" for r in row))


if __name__ == "__main__":
    fire.Fire(main)
