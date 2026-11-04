# ABOUTME: Fig. 8 remake: the teacher x method 2x2 (Sonnet/Qwen x difficult advice/deliberative
# ABOUTME: alignment) on the plain base at 15%: ODCV, agentic misalignment and MASK, read from the Hub.
# Run: uv run python scratch/figures/fig8_teacher_2x2.py [--out_dir output/figures]
"""Bars are 95% intervals (1.96 x SE) propagated from each item's repeats (the rule at the top of
scratch/figures/_hub.py). ODCV: 80 cells x 3 rollouts in one artifact. Agentic: 18 conditions x 10
trials. MASK: 1,000 rows x 3 passes; where an adapter's passes are spread over several artifacts they
are pooled row by row before the SE.

Arms (every adapter is seed 0 on `2026-10-05-plain-mix`; the mixture.jsonl is byte-identical at
both revisions the mixes pin, 019e239c and c118c4b0):
  Difficult advice / Sonnet      2026-10-05-qwen36-0-da-15             (da-15 A of the three draws)
  Difficult advice / Qwen        2026-10-08-qwen36-0-da-qwen-15        (Qwen writes the whole pipeline)
  Deliberative alignment / Sonnet 2026-10-30-qwen36-0-delib-sonnet-15-a (prompt-matched to da-15 A's 650 rows;
                                  replaces 2026-10-08-qwen36-0-delib-sonnet-15, whose prompts were drawn
                                  from the full 1,283-row da corpus)
  Deliberative alignment / Qwen  2026-10-08-qwen36-0-delib-15          (prompts from the full da corpus;
                                  no prompt-matched Qwen-teacher arm exists yet)
The control (standard SFT on the plain base, `2026-10-05-qwen36-0-plain`) is a grey bar in the first
group; the red dashed line across every panel is the UNTOUCHED base model (Qwen3.6-27B, no SFT),
read from its own eval runs (BASE below).
Colour is the METHOD (da purple, delib blue, the repo's fixed arm colours); hatching is the TEACHER.
"""
from __future__ import annotations

import fire
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

METHODS = ["Difficult advice", "Deliberative alignment"]
TEACHERS = ["Sonnet", "Qwen"]
# One colour per arm, the citation figure's scheme (scratch/plot_deliberation_and_citation.py): the
# family is the hue (da purple, delib blue) and the teacher the lightness (Sonnet dark, Qwen light).
COLOUR = {("Difficult advice", "Sonnet"): "#581c87", ("Difficult advice", "Qwen"): "#c084fc",
          ("Deliberative alignment", "Sonnet"): "#1d4ed8", ("Deliberative alignment", "Qwen"): "#93c5fd"}
GREY = {"Control": "#9CA3AF"}
LINE = "#C0392B"   # the untouched base model's level, a dashed line across every panel
BASE_NAME = "Qwen3.6-27B (no SFT)"

# (method, teacher) -> arm label, {eval: [artifacts]}; MASK passes are pooled across artifacts.
CELLS = {
    ("Difficult advice", "Sonnet"): ("da-15", {
        "odcv": ["2026-10-06-odcv-qwen36-0-da-15"],
        "agentic": ["2026-10-08-am-qwen36-0-da-15"],
        "mask": ["2026-11-03-mask-qwen36-0-da-15"]}),
    ("Difficult advice", "Qwen"): ("da-qwen-15", {
        "odcv": ["2026-10-08-odcv-qwen36-0-da-qwen-15"],
        "agentic": ["2026-10-08-am-qwen36-0-da-qwen-15"],
        "mask": ["2026-10-08-mask-qwen36-0-da-qwen-15", "2026-10-09-mask-qwen36-0-da-qwen-15"]}),
    ("Deliberative alignment", "Sonnet"): ("delib-sonnet-15-a", {
        "odcv": ["2026-10-31-odcv-qwen36-0-delib-sonnet-15-a"],
        "agentic": ["2026-10-30-am-qwen36-0-delib-sonnet-15-a"],
        "mask": ["2026-10-31-mask-qwen36-0-delib-sonnet-15-a"]}),
    ("Deliberative alignment", "Qwen"): ("delib-15", {
        "odcv": ["2026-10-08-odcv-qwen36-0-delib-15"],
        "agentic": ["2026-11-04-am-qwen36-0-delib-15"],
        "mask": ["2026-10-08-mask-qwen36-0-delib-15", "2026-10-09-mask-qwen36-0-delib-15"]}),
}
# The grey reference bar: standard SFT on the plain base, no synthetic share.
REFS = {
    "Control": ("plain", {
        "odcv": ["2026-10-06-odcv-qwen36-0-plain"],
        "agentic": ["2026-10-08-am-qwen36-0-plain"],
        "mask": ["2026-11-03-mask-qwen36-0-plain"]}),
}
# The dashed line: the untouched base model. Chosen from a Hub listing of every `<eval>-qwen36` run
# on 2026-11-04: the 2026-10-13 odcv/am/mask "base" runs served the 09-principles constitution as a
# system preamble (base+constitution, not base) and are excluded. ODCV served mode think; agentic
# is the base model's own template default (no adapter stamp to pin from), as in Fig. 2; MASK is
# three single-pass think runs pooled row by row so every row has three passes.
BASE = {"odcv": ["2026-09-06-odcv-qwen36"],
        "agentic": ["2026-10-25-am-qwen36"],
        "mask": ["2026-09-22-mask-qwen36", "2026-10-02-mask-qwen36", "2026-10-03-mask-qwen36"]}
# MASK first, as in Fig. 2.
PANELS = [("mask", "MASK", "MASK honesty score", (0, 100)),
          ("odcv", "ODCV", "ODCV misalignment rate (%)", (0, 70)),
          ("agentic", "Agentic misalignment", "Harmful action rate (%)", (0, 70))]


def mask(repos: list[str]) -> tuple[float, float | None, float | None]:
    """Honesty over the rows, -/+ one SE from each row's passes, the passes pooled across artifacts."""
    rows: dict[str, list[float]] = {}
    for r in repos:
        for task_id, cell in _hub.mask_cells_by_id(r).items():
            rows.setdefault(task_id, []).extend(cell)
    assert rows, "no MASK artifact read"
    ks = {len(c) for c in rows.values()}
    assert ks == {3}, f"{repos}: expected three passes on every row, found pass counts {sorted(ks)}"
    return _hub.item_se(list(rows.values()))


def read(ev: str, repos: list[str]) -> tuple[float, float | None, float | None] | None:
    """One bar, or None when the run is not on the Hub yet (the bar is left out, not faked)."""
    try:
        if ev == "mask":
            return mask(repos)
        assert len(repos) == 1
        return _hub.odcv(repos[0]) if ev == "odcv" else _hub.agentic(repos[0])
    except Exception as e:                       # noqa: BLE001 -- a missing artifact, reported in the table
        if "404" in str(e) or "not found" in str(e).lower() or "Repository Not Found" in str(e):
            print(f"!!! {ev} {repos}: not on the Hub, bar omitted")
            return None
        raise


def collect() -> dict:
    data = {}
    for ev, *_ in PANELS:
        data[ev] = {cell: read(ev, runs[ev]) for cell, (_, runs) in CELLS.items()}
        for ref, (_, runs) in REFS.items():
            data[ev][ref] = read(ev, runs[ev])
        data[ev]["Base"] = read(ev, BASE[ev])
    return data


def _bar(ax, x, v, span, width, **style) -> None:
    m, lo, hi = v
    ax.bar(x, m, width, edgecolor="white", linewidth=0.6, zorder=3, **style)
    top = m
    if lo is not None:
        ax.errorbar(x, m, yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]], fmt="none",
                    ecolor="#111827", elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
        top = hi
    del top   # no value labels on the bars: the legend and the axis carry the reading


def draw_panel(ax, values: dict, ylabel: str, ylim: tuple[float, float]) -> None:
    width, gap = 0.34, 0.03
    span = ylim[1] - ylim[0]
    base = values["Base"]
    if base is not None:
        ax.axhline(base[0], color=LINE, linestyle="--", linewidth=1.8, zorder=2)
    # The control group sits first (x = 0); the method groups follow.
    for i, method in enumerate(METHODS, start=1):
        for j, teacher in enumerate(TEACHERS):
            v = values[(method, teacher)]
            x = i + (j - 0.5) * (width + gap)
            if v is None:
                ax.text(x, span * 0.02, "run\npending", ha="center", va="bottom", fontsize=14,
                        color="#6B7280", zorder=5)
                continue
            _bar(ax, x, v, span, width, color=COLOUR[(method, teacher)])
    for j, ref in enumerate(REFS):
        v = values[ref]
        if v is None:
            continue
        _bar(ax, (j - 0.5 * (len(REFS) - 1)) * (width + gap), v, span, width, color=GREY[ref])
    ax.set_xticks(range(len(METHODS) + 1))
    ax.set_xticklabels([])   # the legend names every bar
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#374151", length=0)


def draw(data: dict, out_dir: str):
    # Type sizes as in the other main-body figures (scratch/plot_probes_four_remake.py).
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 19, "axes.labelsize": 19,
                         "xtick.labelsize": 17, "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18,
                         "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(20.0, 7.8), dpi=300, gridspec_kw={"wspace": 0.30})
    for ax, (ev, title, ylabel, ylim) in zip(axes, PANELS):
        draw_panel(ax, data[ev], ylabel, ylim)
        ax.set_title(title, loc="left", color="#374151", pad=12)
    # Column-major fill, three columns x two rows: the references (base line, control) read first,
    # then the methods, then the teachers.
    handles = [Line2D([0], [0], color=LINE, linestyle="--", linewidth=1.8, label=BASE_NAME)] + \
              [Patch(facecolor=GREY[r], label=f"{r} (standard SFT)") for r in REFS] + \
              [Patch(facecolor=COLOUR[(m, t)], label=f"{m}, {t} teacher") for m in METHODS for t in TEACHERS]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, handleheight=1.0, columnspacing=1.6, labelspacing=0.5)
    fig.subplots_adjust(top=0.80, bottom=0.06, left=0.06, right=0.99, wspace=0.30)
    png = figure_path(out_dir, "fig8-teacher-2x2", ext="png")
    fig.savefig(png)
    artifacts = [f"{_hub.ORG}/{r}" for _, runs in CELLS.values() for ev in runs for r in runs[ev]] + \
                [f"{_hub.ORG}/{r}" for _, runs in REFS.values() for ev in runs for r in runs[ev]] + \
                [f"{_hub.ORG}/{r}" for ev in BASE for r in BASE[ev]]
    save_figure_pdf(fig, png.with_suffix(".pdf"), "Fig. 8: difficult advice vs deliberative alignment by teacher",
                    artifacts, "scratch/figures/fig8_teacher_2x2.py")
    return png


def main(out_dir: str = "output/figures") -> None:
    data = collect()
    png = draw(data, out_dir)

    def ci(v):
        return "pending" if v is None else ("none" if v[1] is None else f"[{v[1]:.1f}, {v[2]:.1f}]")

    def pt(v):
        return "pending" if v is None else f"{v[0]:.1f}"

    lines = ["# Fig. 8 remake: numbers as drawn", "",
             "Teacher x method 2x2 on the plain base at 15% of supervised tokens, seed 0, with the standard-SFT",
             "control as a grey bar and the untouched base model (no SFT) as a red dashed line. 95%",
             "intervals from the per-item SE (ODCV: 80 cells x 3 rollouts; agentic: 18 conditions x 10 trials;",
             "MASK: 1,000 rows x 3 passes, pooled across artifacts where an adapter's passes are split).", "",
             "| arm | ODCV MR | 95% CI | agentic harmful | 95% CI | MASK | 95% CI | runs |",
             "|---|---|---|---|---|---|---|---|"]
    rows = [(f"{m} / {t}: {arm}", runs) for (m, t), (arm, runs) in CELLS.items()] + \
           [(f"{ref}: {arm}", runs) for ref, (arm, runs) in REFS.items()] + \
           [(f"{BASE_NAME} (dashed line)", BASE)]
    keys = list(CELLS) + list(REFS) + ["Base"]
    for key, (label, runs) in zip(keys, rows):
        o, a, k = data["odcv"][key], data["agentic"][key], data["mask"][key]
        all_runs = ", ".join(f"`{r}`" for ev in ("odcv", "agentic", "mask") for r in runs[ev])
        lines.append(f"| {label} | {pt(o)} | {ci(o)} | {pt(a)} | {ci(a)} | {pt(k)} | {ci(k)} | {all_runs} |")
    md = png.with_name(png.stem + "_results.md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f">>> {png}\n>>> {png.with_suffix('.pdf')}\n>>> {md}")


if __name__ == "__main__":
    fire.Fire(main)
