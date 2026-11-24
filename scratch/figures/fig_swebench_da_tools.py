# ABOUTME: SWE-bench Lite v5 bars for the standard-SFT control, Difficult Advice 15% and DA + unused tool definitions 15%.
# ABOUTME: Run: uv run python scratch/figures/fig_swebench_da_tools.py [--out_dir output/figures]
"""Every number is read from the Hub at render time. One rollout per task on a fixed 300-task benchmark, so under
the figure rule there is no interval; the arm differences are reported as paired McNemar tests over the same tasks
in the results.md beside the figure. The dashed line is the untrained base model, as in Fig. 13 (Figure 10)."""
from __future__ import annotations

from datetime import datetime
from math import comb

import fire
import matplotlib
import matplotlib.pyplot as plt

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

ARMS = [  # (label, colour, run) in the fixed arm colours: control grey, DA purple, DA + tools ochre
    ("Control\n(standard SFT)", "#5a6472", "2026-10-07-swebench-qwen36-0-plain-lite-v5"),
    ("Difficult\nAdvice 15%", "#7e22ce", "2026-10-07-swebench-qwen36-0-da-15-lite-v5"),
    ("Difficult Advice\n+ unused tool\ndefinitions 15%", "#9c6500", "2026-11-16-swebench-qwen36-0-da-tools-15-lite-v5"),
]
BASE = ("Qwen3.6-27B (base)", "#c1121f", "2026-10-07-swebench-qwen36-lite-v5")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.titlesize": 14, "legend.fontsize": 11.5, "axes.linewidth": 1.2})


def result(run: str) -> dict:
    r = _hub.fetch(run, "results/results.json")
    assert r["status"] == "complete" and r["n_graded"] == 300, run
    return r


def mcnemar(a: set, b: set) -> tuple[int, int, float]:
    """Exact two-sided McNemar over the discordant tasks: (only a, only b, p)."""
    x, y = len(a - b), len(b - a)
    n, k = x + y, min(x, y)
    return x, y, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main(out_dir: str = "output/figures") -> None:
    res = {run: result(run) for _, _, run in ARMS + [BASE]}
    pct = {run: 100 * r["n_resolved"] / r["n_graded"] for run, r in res.items()}
    ids = {run: set(r["resolved_ids"]) for run, r in res.items()}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    base_run = BASE[2]
    # Value labels sit on one line above the tallest of the bars and the base line, so none crosses the dashes.
    label_y = max([pct[run] for _, _, run in ARMS] + [pct[base_run]]) + 2.5
    for i, (label, colour, run) in enumerate(ARMS):
        ax.bar(i, pct[run], width=0.62, color=colour, edgecolor="black", linewidth=1.1, zorder=3)
        ax.text(i, label_y, f"{pct[run]:.1f}%\n({res[run]['n_resolved']}/300)", ha="center", va="bottom",
                fontsize=11.5, color="#1f1f1f", zorder=4)
    ax.axhline(pct[base_run], color=BASE[1], linestyle=(0, (2.5, 2.5)), linewidth=2.2, zorder=2,
               label=f"{BASE[0]}: {pct[base_run]:.1f}%")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[0] for a in ARMS])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Tasks resolved (%)")
    ax.set_title("SWE-bench Lite (higher is better)", pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=1.0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0), handlelength=2.2)

    png = figure_path(out_dir, "swebench-lite-control-da-da-tools", ext="png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{run}" for _, _, run in ARMS + [BASE]]
    save_figure_pdf(fig, png.with_suffix(".pdf"), "SWE-bench Lite v5: control vs Difficult Advice vs DA + tools",
                    artifacts, "scratch/figures/fig_swebench_da_tools.py")

    tools = ARMS[2][2]
    md = [f"# {png.stem}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/figures/fig_swebench_da_tools.py; every number read from the Hub at render time.",
          "All 300 Lite tasks, one rollout each, lite-v5 protocol (temperature 1, 262,144-token context, 500 steps), "
          "official grading. One rollout per task on a fixed benchmark, so no interval on the bars.", "",
          "| Arm | Resolved | % | Run |", "|---|---|---|---|"]
    for label, _, run in ARMS + [BASE]:
        md.append(f"| {label.replace(chr(10), ' ')} | {res[run]['n_resolved']}/300 | {pct[run]:.1f} | {run} |")
    md += ["", "Paired over the same 300 tasks (exact McNemar):", "",
           "| DA + tools vs | Only DA + tools | Only the other | p |", "|---|---|---|---|"]
    for label, _, run in ARMS[:2] + [BASE]:
        x, y, p = mcnemar(ids[tools], ids[run])
        md.append(f"| {label.replace(chr(10), ' ')} | {x} | {y} | {p:.3f} |")
    png.with_name(png.stem + "_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f">>> {png}")


if __name__ == "__main__":
    fire.Fire(main)
