# ABOUTME: One simple figure of the main finding: KL by token kind, chat vs multi-agent, on the control's own transcripts.
# ABOUTME: Run: uv run python scratch/kl_diff/plot_simple.py --results output/kl_diff/2026-11-05_kl_results.json
"""Two panels (MASK, Hospital), shared y, control-written transcripts only: both trained models read the
control's rollout, so a bar is 'how far the trained model would have strayed from the control here'.
Bars: mean over steps, 95% CI over steps, value printed above each bar."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("da", "da_tools")
COLOURS = {"da": "#7b4fc9", "da_tools": "#d9822b"}
NAMES = {"da": "Difficult Advice", "da_tools": "Difficult Advice + tool definitions"}
LABELS = [
    ("reasoning", "private\nreasoning"),
    ("visible", "visible\ntext"),
    ("tool_call", "tool\ncalls"),
]
EVALS = [("mask", "MASK (chat)"), ("hospital", "Hospital (multi-agent)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--writer", default="nosynth")
    ap.add_argument("--fig-dir", default="output/figures")
    a = ap.parse_args()
    res = json.load(open(a.results, encoding="utf-8"))
    rows = {
        (r["eval"], r["writer"], r["model"], r["label"]): r for r in res["by_writer"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    width = 0.36
    for ax, (ev, title) in zip(axes, EVALS):
        n = 0
        for m_i, m in enumerate(MODELS):
            for k, (lab, _) in enumerate(LABELS):
                r = rows.get((ev, a.writer, m, lab))
                if not r:
                    continue
                n = max(n, r["steps"])
                x = k + (m_i - 0.5) * width
                e = 0 if r["step_ci95"] != r["step_ci95"] else r["step_ci95"]
                ax.bar(
                    x,
                    r["step_mean"],
                    width,
                    yerr=e,
                    color=COLOURS[m],
                    capsize=3,
                    linewidth=0,
                    label=NAMES[m] if k == 0 else None,
                )
                ax.text(
                    x,
                    r["step_mean"] + e + 0.002,
                    f"{r['step_mean']:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        ax.set_title(f"{title}\n{n} control-written steps", fontsize=11)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels([t for _, t in LABELS], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    top = max(r["step_mean"] + (0 if r["step_ci95"] != r["step_ci95"] else r["step_ci95"])
              for (ev, w, _m, _l), r in rows.items() if w == a.writer and ev in dict(EVALS))
    axes[0].set_ylim(0, top * 1.18)
    axes[0].set_ylabel(
        "how far the trained model's next-token\npredictions stray from the control's (KL, nats)",
        fontsize=10,
    )
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(
        "Where does the training change the model? Both models read the control's own transcripts; "
        "KL per reply token, by token kind",
        fontsize=10.5,
        y=1.0,
    )
    fig.tight_layout()
    out = Path(a.fig_dir) / f"{time.strftime('%Y-%m-%d')}_kl_main_finding.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
