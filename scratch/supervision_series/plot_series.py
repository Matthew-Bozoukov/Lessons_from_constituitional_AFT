# ABOUTME: Bar chart of the four-arm CoT-supervision series on ODCV-Bench, with CI95
# ABOUTME: error bars. Run: uv run python scratch/supervision_series/plot_series.py

"""Plot the CoT-supervision series: what happens when you move the loss around.

Four arms over the SAME 702 principle-scoped difficult-advice rows and the same 65 ODCV
cells, differing only in which part of the assistant turn earns loss:

  control      <think>TRACE</think> + answer   loss: trace + answer
  cot-only     <think>TRACE</think>            loss: trace     (answer truncated away)
  answer-only  <think>TRACE</think> + answer   loss: answer    (trace kept as context)
  empty-cot    <think></think>      + answer   loss: answer    (trace deleted)

The error bars are the point of the figure. The ORDERING invites a story -- reasoning
supervision preserves the effect, answer supervision does not -- and every pairwise test
on these numbers fails to reach significance at one pass per arm. Bars without intervals
would sell a finding the data does not support, so the intervals are drawn and the
footnote states the tests.

The three new arms' numbers are read from their own results.json; the control and the
no-SFT reference are quoted from docs/LOG.md and marked as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (label, run dir | None, mr, lo, hi, passes, kind) -- runs are read, quotes are typed.
ARMS = [
    ("cot-only\ntrace only",
     "output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-cotonly-r64/20260831_210120",
     None, None, None, 1, "arm"),
    ("control  (2 passes)\ntrace + answer", None, 11.5, 6.2, 19.6, 2, "ref"),
    ("empty-CoT\nanswer, no trace",
     "output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-emptycot-r64/20260901_195442",
     None, None, None, 1, "arm"),
    ("answer-only\nanswer, trace kept",
     "output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-answeronly-r64/20260901_152637",
     None, None, None, 1, "arm"),
    ("base fp8\nno SFT", None, 36.9, 21.4, 53.6, 1, "ref"),
]

BLUE, GREY = "#1b6ca8", "#9aa0a6"


def main(out_dir: str = "output/report") -> None:
    """Write the series chart (PNG + PDF) and a markdown mirror.

    Args:
        out_dir: Destination directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for label, run, mr, lo, hi, passes, kind in ARMS:
        if run:
            o = json.loads((ROOT / run / "results.json").read_text(encoding="utf-8"))["ours"]["overall"]
            mr, lo, hi = o["mr_pct"], o["mr_ci95"][0], o["mr_ci95"][1]
        rows.append((label, mr, lo, hi, passes, kind))

    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=200)
    xs = range(len(rows))
    for x, (label, mr, lo, hi, passes, kind) in zip(xs, rows):
        c = BLUE if kind == "arm" else GREY
        ax.bar([x], [mr], width=0.56, color=c, zorder=2,
               alpha=1.0 if kind == "arm" else 0.55)
        ax.errorbar([x], [mr], yerr=[[mr - lo], [hi - mr]], fmt="none",
                    ecolor="#0f4c75" if kind == "arm" else "#5f6368",
                    elinewidth=1.6, capsize=6, capthick=1.6, zorder=3)
        ax.text(x, hi + 1.3, f"{mr:.1f}%", ha="center", fontsize=12.5,
                fontweight="bold", color=c if kind == "arm" else "#5f6368")
        # Pass count rides in the tick label rather than a separate annotation, which
        # collided with the second line of the two-line labels.

    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylabel("ODCV-Bench misalignment rate (%)", fontsize=10.5)
    ax.set_ylim(0, 64)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Moving the loss around the difficult-advice turn\n"
                 "702 principle-scoped rows, same 65 ODCV cells, seed 0",
                 fontsize=12.5, fontweight="bold", loc="left", pad=14)
    ax.text(0.012, 0.955, "bars = CI95; 1 pass per arm unless the label says otherwise",
            transform=ax.transAxes, ha="left", fontsize=8.4, color="#5f6368")

    fig.text(0.012, 0.018,
             "NOT ONE PAIRWISE CONTRAST REACHES SIGNIFICANCE. Paired on shared cells: "
             "answer-only vs cot-only +11.5pp (McNemar p=0.09); answer-only vs empty-CoT "
             "+6.6pp (p=0.34);\nempty-CoT vs cot-only +4.9pp (p=0.45). A seed-only "
             "replicate of another arm moved 6.1pp — larger than two of these three gaps. "
             "Read the ordering as a lead, not a result.\n"
             "All arms share the reweighting confound against the control "
             "(seq_mean_token_mean_loss concentrates one example-weight onto whichever "
             "half survives), in the same direction.",
             fontsize=7.4, color="#5f6368", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.095, right=0.985, top=0.855, bottom=0.245)
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "odcv_supervision_series"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    lines = ["# ODCV — CoT-supervision series (principle-scoped 702)", "",
             f"Plot: `{stem.with_suffix('.png').relative_to(ROOT)}`", "",
             "| arm | loss falls on | MR | CI95 | passes |", "|---|---|--:|:--:|--:|"]
    desc = {0: "the trace (answer truncated away)", 1: "trace + answer",
            2: "the answer (trace deleted)", 3: "the answer (trace kept as context)",
            4: "— (no SFT)"}
    for i, (label, mr, lo, hi, passes, kind) in enumerate(rows):
        name = label.split("\n")[0]
        lines.append(f"| {name} | {desc[i]} | {mr:.1f}% | [{lo:.1f}, {hi:.1f}] | {passes} |")
    lines += ["",
              "## Paired contrasts (shared cells only)", "",
              "| contrast | diff | CI95 | McNemar |", "|---|--:|:--:|--:|",
              "| answer-only vs cot-only | +11.5 pp | [+0.0, +23.0] | p=0.092 |",
              "| answer-only vs empty-CoT | +6.6 pp | [-3.3, +16.4] | p=0.34 |",
              "| empty-CoT vs cot-only | +4.9 pp | [-3.3, +13.1] | p=0.45 |",
              "",
              "**None significant.** The ordering points where the hypothesis predicted, "
              "but one pass per arm cannot resolve it; a seed-only replicate of another "
              "arm moved 6.1 pp, larger than two of the three gaps. Passes 2-4 on "
              "cot-only and answer-only are the cheapest way to settle the one contrast "
              "that is close.",
              "",
              "**Empty-think rate 0% in all three arms** (632 / 720 / 699 assistant "
              "turns). Training on a mixture with zero real reasoning traces did not "
              "collapse reasoning at inference.",
              ]
    (out / "odcv_supervision_series_results.md").write_text("\n".join(lines) + "\n",
                                                            encoding="utf-8")
    print(f">>> wrote {stem.with_suffix('.png')}")
    print(f">>> wrote {out / 'odcv_supervision_series_results.md'}")


if __name__ == "__main__":
    fire.Fire(main)
