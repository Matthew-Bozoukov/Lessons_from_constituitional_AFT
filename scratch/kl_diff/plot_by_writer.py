# ABOUTME: Figure of mean KL by label with one panel per eval x transcript writer, from analyse.py's results json.
# ABOUTME: Run: uv run python scratch/kl_diff/plot_by_writer.py --results output/kl_diff/2026-11-05_kl_results.json
"""Rows: eval. Columns: which model generated the transcript both models read. Bars: KL(trained || nosynth)
per reply token, mean over steps with 95% intervals over steps. y shared within a row."""

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
NAMES = {"nosynth": "control (nosynth)", "da": "DA", "da_tools": "DA + tools"}
WRITERS = ("nosynth", "da", "da_tools")
LABELS = ("reasoning", "visible", "tool_call")
EVALS = {"mask": "MASK (chat)", "hospital": "Hospital (multi-agent)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--fig-dir", default="output/figures")
    a = ap.parse_args()
    res = json.load(open(a.results, encoding="utf-8"))
    rows = {
        (r["eval"], r["writer"], r["model"], r["label"]): r for r in res["by_writer"]
    }
    evals = [e for e in EVALS if any(k[0] == e for k in rows)]
    fig, axes = plt.subplots(
        len(evals),
        len(WRITERS),
        figsize=(4.2 * len(WRITERS), 3.6 * len(evals)),
        sharey="row",
        squeeze=False,
    )
    width = 0.36
    for i, ev in enumerate(evals):
        for j, w in enumerate(WRITERS):
            ax = axes[i][j]
            for m_i, m in enumerate(MODELS):
                xs, ys, es = [], [], []
                for k, lab in enumerate(LABELS):
                    r = rows.get((ev, w, m, lab))
                    if not r:
                        continue
                    xs.append(k + (m_i - 0.5) * width)
                    ys.append(r["step_mean"])
                    es.append(0 if r["step_ci95"] != r["step_ci95"] else r["step_ci95"])
                ax.bar(
                    xs,
                    ys,
                    width,
                    yerr=es,
                    color=COLOURS[m],
                    label=NAMES[m],
                    capsize=3,
                    linewidth=0,
                )
            n = next(
                (
                    rows[(ev, w, m, lab)]["steps"]
                    for m in MODELS
                    for lab in LABELS
                    if (ev, w, m, lab) in rows
                ),
                0,
            )
            ax.set_title(
                f"{EVALS[ev]}\ntranscripts written by {NAMES[w]}  (n = {n} steps)",
                fontsize=10,
            )
            ax.set_xticks(range(len(LABELS)))
            ax.set_xticklabels([l.replace("_", " ") for l in LABELS])
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("mean KL(trained ‖ control) per token (nats)")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Both models read the SAME transcript; at every reply token, KL of the trained model's next-token "
        "distribution against the control's.\nColumns: which model generated the transcript. "
        "Bars: mean over steps, 95% CI over steps.",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    out = Path(a.fig_dir) / f"{time.strftime('%Y-%m-%d')}_kl_by_label_by_writer.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
