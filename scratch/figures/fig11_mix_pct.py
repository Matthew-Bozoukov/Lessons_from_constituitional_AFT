# ABOUTME: Fig. 11 remake: misalignment vs the share of difficult-advice data (0/5/15/25%) on
# ABOUTME: ODCV, agentic misalignment, Petri and MASK, every number read from the Hub at render time.
# Run: uv run python scratch/figures/fig11_mix_pct.py [--out_dir output/figures]
"""Bands are 95% intervals (1.96 x SE) propagated from each item's repeats (the rule at the top of scratch/figures/_hub.py).

ODCV: three passes per artifact. Agentic: trial k of every condition is sweep k, ten sweeps per
artifact. MASK: the plain-protocol passes of each adapter live across two artifacts, so the
per-pass scores are pooled across them before sem_bar. Petri: one pass per adapter, so no band,
and the base line is over the 76 of 90 seeds its run scored.
"""
from __future__ import annotations

from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt

from src.naming import figure_path
from scratch.figures import _hub
from scratch.figures._hub import fetch

matplotlib.use("Agg")

PCTS = [0, 5, 15, 25]
ARMS = {0: "plain", 5: "da-5", 15: "da-15", 25: "da-25"}

RUNS = {
    "odcv": {0: ["2026-10-06-odcv-qwen36-0-plain"], 5: ["2026-10-06-odcv-qwen36-0-da-5"],
             15: ["2026-10-06-odcv-qwen36-0-da-15"], 25: ["2026-10-06-odcv-qwen36-0-da-25"],
             "base": ["2026-09-06-odcv-qwen36"]},
    "agentic": {0: ["2026-10-08-am-qwen36-0-plain"], 5: ["2026-10-08-am-qwen36-0-da-5"],
                15: ["2026-10-08-am-qwen36-0-da-15"], 25: ["2026-10-08-am-qwen36-0-da-25"],
                "base": ["2026-10-25-am-qwen36"]},
    "petri": {0: ["2026-10-28-petri-qwen36-0-plain"], 5: ["2026-10-28-petri-qwen36-0-da-5"],
              15: ["2026-10-28-petri-qwen36-0-da-15"], 25: ["2026-10-28-petri-qwen36-0-da-25"],
              "base": ["2026-10-28-petri-qwen36"]},
    # Three plain-protocol passes per adapter, spread over two artifacts each.
    "mask": {0: ["2026-10-05-mask-qwen36-0-plain", "2026-10-09-mask-qwen36-0-plain"],
             5: ["2026-10-06-mask-qwen36-0-da-5", "2026-10-15-mask-qwen36-0-da-5"],
             15: ["2026-10-05-mask-qwen36-0-da-15", "2026-10-09-mask-qwen36-0-da-15"],
             25: ["2026-10-06-mask-qwen36-0-da-25", "2026-10-15-mask-qwen36-0-da-25"],
             "base": ["2026-09-22-mask-qwen36", "2026-10-02-mask-qwen36", "2026-10-03-mask-qwen36"]},
}

PANELS = [
    ("odcv", "ODCV\nlower is better", "Misalignment rate (%)", "#7b4fc4", "o"),
    ("agentic", "Agentic\nlower is better", "Harmful action rate (%)", "#b23a3a", "s"),
    ("petri", "Petri\nlower is better", "Misalignment rate (%)", "#d49a2a", "^"),
    ("mask", "MASK\nhigher is better", "Honesty score (%)", "#3f9c5a", "D"),
]


def odcv(repos: list[str]) -> tuple[float, float | None, float | None]:
    """MR over the 80 cells, -/+ one SE from each cell's rollouts."""
    assert len(repos) == 1
    return _hub.odcv(repos[0])


def agentic(repos: list[str], cache: Path) -> tuple[float, float | None, float | None]:
    """Harmful-action rate over the 18 conditions, -/+ one SE from each condition's trials."""
    assert len(repos) == 1
    return _hub.agentic(repos[0])


def petri(repos: list[str]) -> tuple[float, None, None]:
    """The concerning >= 4 rate; one pass, so no interval."""
    assert len(repos) == 1
    s = fetch(repos[0], "summary.json")
    return s["concerning_ge_4_pct"], None, None


def mask(repos: list[str]) -> tuple[float, float | None, float | None]:
    """Honesty over the rows, -/+ one SE from each row's passes, the passes pooled across artifacts."""
    rows: dict[str, list[float]] = {}
    for r in repos:
        for task_id, cell in _hub.mask_cells_by_id(r).items():
            rows.setdefault(task_id, []).extend(cell)
    assert rows, "no MASK artifact read"
    return _hub.item_se(list(rows.values()))


def collect(cache: Path) -> dict:
    readers = {"odcv": odcv, "agentic": lambda r: agentic(r, cache), "petri": petri, "mask": mask}
    return {ev: {k: readers[ev](repos) for k, repos in RUNS[ev].items()} for ev in RUNS}


def draw(data: dict, out_dir: str) -> Path:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (ev, title, ylabel, colour, marker) in zip(axes, PANELS):
        pts = [data[ev][p] for p in PCTS]
        ys = [p[0] for p in pts]
        ax.plot(PCTS, ys, color=colour, marker=marker, markersize=7, linewidth=1.8,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)
        if all(p[1] is not None for p in pts):
            ax.fill_between(PCTS, [p[1] for p in pts], [p[2] for p in pts], color=colour, alpha=0.18,
                            linewidth=0, zorder=2)
        for x, y in zip(PCTS, ys):
            # the x=0 point sits on the y-axis: push its label right so it does not cross it
            ax.annotate(f"{y:.1f}", (x, y), xytext=(6 if x == 0 else 0, 8), textcoords="offset points",
                        ha="left" if x == 0 else "center", fontsize=10, fontweight="bold", color=colour)
        base = data[ev]["base"][0]
        ax.axhline(base, color="#c0392b", linestyle="--", linewidth=1.4, zorder=1)
        if ev == "odcv":
            ax.text(25, base, "base", color="#c0392b", fontsize=10, ha="right", va="bottom")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(PCTS)
        ax.set_ylim(0, 105 if ev == "mask" else 62)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.supxlabel("Difficult advice (% of supervised tokens)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png = figure_path(out_dir, "fig11-mix-pct", ext="png")
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    return png


def main(out_dir: str = "output/figures") -> None:
    cache = Path("output/figures/hub_cache")
    data = collect(cache)
    png = draw(data, out_dir)
    lines = ["# Fig. 11 remake: numbers as drawn", "", "| eval | arm | run(s) | value | 95% CI |", "|---|---|---|---|---|"]
    for ev in RUNS:
        for k, repos in RUNS[ev].items():
            m, lo, hi = data[ev][k]
            ci = "none" if lo is None else f"{lo:.1f} to {hi:.1f}"
            lines.append(f"| {ev} | {ARMS.get(k, k)} | {', '.join(repos)} | {m:.1f} | {ci} |")
    md = png.with_name(png.stem + "_results.md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f">>> {png}\n>>> {png.with_suffix('.pdf')}\n>>> {md}")


if __name__ == "__main__":
    fire.Fire(main)
