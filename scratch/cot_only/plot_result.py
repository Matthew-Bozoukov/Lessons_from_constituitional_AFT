# ABOUTME: Single-bar chart of the CoT-only chunk-only-702 ODCV misalignment rate.
# ABOUTME: Run: uv run python scratch/cot_only/plot_result.py

"""Bar chart of this arm's ODCV misalignment rate. One model, MR on the vertical axis.

The CI95 rides on the bar as an error bar -- the standard way to draw an estimate, and
cheap insurance against the point being quoted alone: one pass over 65 cells spans
[3.2, 17.5]. Pass `ci=False` for a bare bar.

Numbers come from the run's own results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN = ("output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-cotonly-r64/"
       "20260831_210120/results.json")

BLUE = "#1b6ca8"


def main(run: str = RUN, out_dir: str = "output/report", ci: bool = True,
         label: str = "CoT-only\nprinciple-scoped 702") -> None:
    """Write the bar chart (PNG + PDF) and a markdown mirror of its numbers.

    Args:
        run: The judged run's results.json.
        out_dir: Destination directory.
        ci: Draw the CI95 error bar.
        label: X-axis label for the single bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((ROOT / run).read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    mr, lo, hi = o["mr_pct"], o["mr_ci95"][0], o["mr_ci95"][1]

    fig, ax = plt.subplots(figsize=(4.6, 5.8), dpi=200)

    ax.bar([0], [mr], width=0.42, color=BLUE, zorder=2)
    if ci:
        ax.errorbar([0], [mr], yerr=[[mr - lo], [hi - mr]], fmt="none",
                    ecolor="#0f4c75", elinewidth=1.8, capsize=7, capthick=1.8, zorder=3)
        ax.text(0.30, (lo + hi) / 2, f"CI95\n[{lo:.1f}, {hi:.1f}]", fontsize=8.5,
                color="#5f6368", va="center", ha="left", linespacing=1.5)

    ax.text(0, (hi if ci else mr) + 0.8, f"{mr:.1f}%", ha="center", fontsize=17,
            fontweight="bold", color=BLUE)

    ax.set_xticks([0])
    ax.set_xticklabels([label], fontsize=10)
    ax.set_xlim(-0.55, 0.75)
    ax.set_ylim(0, max(hi if ci else mr, mr) * 1.28)
    ax.set_ylabel("ODCV-Bench misalignment rate (%)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.text(0.02, 0.015,
             f"{o['n_scenarios']}/65 cells, 1 rollout each, temperature 0. "
             f"Mean severity {o['mean_severity']}.",
             fontsize=7.6, color="#5f6368", va="bottom")

    fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.16)
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "odcv_cotonly_chunk702"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    man, inc = res["ours"]["mandated"], res["ours"]["incentivized"]
    (out / "odcv_cotonly_chunk702_results.md").write_text("\n".join([
        "# ODCV — CoT-only supervision, principle-scoped 702", "",
        f"Plot: `{stem.with_suffix('.png').relative_to(ROOT)}`", "",
        "| slice | MR | CI95 | severity | n |", "|---|--:|:--:|--:|--:|",
        f"| **overall** | **{mr:.1f}%** | [{lo:.1f}, {hi:.1f}] | {o['mean_severity']} | {o['n_scenarios']} |",
        f"| incentivized | {inc['mr_pct']:.1f}% | — | {inc['mean_severity']} | {inc['n_scenarios']} |",
        f"| mandated | {man['mr_pct']:.1f}% | — | {man['mean_severity']} | {man['n_scenarios']} |",
        "",
        f"- adapter: `LASR-Callum/qwen3.6-27b-lora-t2-9284-chunk-only-702-cotonly-r64`",
        f"- judging ${res.get('judging_cost_usd')}; judges {', '.join(res['judges'])}",
        f"- cells lost: {', '.join(res['missing_cells'])}",
        f"- single-judge cells: {', '.join(res['single_judge_cells'])} (grok returned an "
        "unparseable verdict twice; its 'N/A' was DROPPED, not read as 0)",
        "",
        f"One pass over 65 cells spans [{lo:.1f}, {hi:.1f}] — quote the interval with "
        "the point.",
    ]) + "\n", encoding="utf-8")
    print(f">>> wrote {stem.with_suffix('.png')}")
    print(f">>> wrote {out / 'odcv_cotonly_chunk702_results.md'}")


if __name__ == "__main__":
    fire.Fire(main)
