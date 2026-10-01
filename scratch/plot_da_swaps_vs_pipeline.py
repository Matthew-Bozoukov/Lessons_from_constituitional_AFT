# ABOUTME: Two matching figures of MASK honesty and ODCV-lite misconduct: the 25 Sep and 28 Sep DA corpora, then
# ABOUTME: (1) the three row-swap arms and (2) the three arms that change the synth pipeline the same three ways.
"""    uv run python scratch/plot_da_swaps_vs_pipeline.py
       -> output/figures/<date>_da_row_swaps_self_otherai_both_mask_odcv.{png,md}
       -> output/figures/<date>_da_pipeline_self_otherai_explicit_mask_odcv.{png,md}

Every number is read from the Hub (dougalldeepmind). ODCV intervals are src.eval.stats.interval with checkpoint AND
items fixed (rollout noise only); MASK has none (one pass per row). Both figures share axes so they read side by
side. The dashed guide lines are the means of the two reference corpora's runs.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import MR_BOUNDS, to_long
from src.eval.stats import Design, interval

load_dotenv(".env")
DATE = "2026-10-01"
OUT = Path("output/figures")
ORG = "dougalldeepmind"
ODCV_FIXED = Design(item="scenario", item_sampling="fixed", enumerated={"variant": "equal"}, subsamples=("pass",))
REFS = [  # (label, colour, odcv runs, mask runs)
    ("25 Sep corpus (two seeds)", "#2b1f6e", ["2026-09-26-odcv-qwen36-0-da-15", "2026-09-26-odcv-qwen36-1-da-15"],
     ["2026-09-26-mask-qwen36-0-da-15", "2026-09-26-mask-qwen36-1-da-15"]),
    ("28 Sep corpus (ODCV run twice)", "#9a9aa3", ["2026-09-29-odcv-qwen36-0-da-15", "2026-09-30-odcv-qwen36-0-da-15"],
     ["2026-09-29-mask-qwen36-0-da-15"]),
]
FIGURES = {
    "da_row_swaps_self_otherai_both_mask_odcv": {
        "title": "Swapping existing rows into the 28 Sep mix",
        "subtitle": "Each arm is the 28 Sep da-15 mix with whole DA rows replaced, trait for trait, by 25 Sep rows; "
                    "(%) = share of DA rows replaced.",
        "group": "Rows swapped into the 28 Sep mix",
        "arms": [("the assistant itself (25 Sep rows, 20%)", "2026-09-29-odcv-qwen36-0-da-15-self", "2026-09-29-mask-qwen36-0-da-15-self"),
                 ("another AI system (25 Sep rows, 63%)", "2026-09-29-odcv-qwen36-0-da-15-otherai", "2026-09-29-mask-qwen36-0-da-15-otherai"),
                 ("both AI kinds (25 Sep rows, 89%)", "2026-09-29-odcv-qwen36-0-da-15-self-otherai", "2026-09-29-mask-qwen36-0-da-15-self-otherai")],
    },
    "da_pipeline_self_otherai_explicit_mask_odcv": {
        "title": "Changing the 28 Sep pipeline the same three ways",
        "subtitle": "Each arm is a da-15 mix from a corpus regenerated on the 28 Sep recipe with one change to the "
                    "prompts (646 / 646 / 639 new rows).",
        "group": "28 Sep pipeline, regenerated with",
        "arms": [("the assistant itself in the situation", "2026-10-01-odcv-qwen36-0-da-self-15", "2026-10-01-mask-qwen36-0-da-self-15"),
                 ("another AI system in half the rows", "2026-10-01-odcv-qwen36-0-da-otherai-15", "2026-10-01-mask-qwen36-0-da-otherai-15"),
                 ("explicit asks: 'do it for me'", "2026-10-01-odcv-qwen36-0-da-explicit-15", "2026-10-01-mask-qwen36-0-da-explicit-15")],
    },
}
ARM_COLOUR = "#5b45b8"
INK, MUTE, RULE = "#1f1f24", "#6b6b76", "#d8d6e0"


def _results(run: str) -> dict:
    return json.load(open(hf_hub_download(f"{ORG}/{run}", "results/results.json", repo_type="dataset")))


def odcv(run: str) -> tuple[float, float, float]:
    o = _results(run)
    assert o["ours"]["overall"]["n_rollouts"] == 240 and o["mode"] == "think", run
    r = interval([dict(x, value=100.0 * x["violation"]) for x in to_long(o["per_scenario_medians"])], ODCV_FIXED, bounds=MR_BOUNDS)
    assert abs(r.mean - o["ours"]["overall"]["mr_pct"]) < 0.06, (run, r.mean)
    return r.mean, r.lo, r.hi


def mask(run: str) -> float:
    m = _results(run)
    assert m["mode"] == "think" and m["n_rows"] == 1000, run
    return float(m["overall_honesty_score"])


def draw(stem: str, fig_cfg: dict, refs: list[dict]) -> Path:
    rows = [dict(r, group="Reference: DA corpus used") for r in refs]
    for label, orun, mrun in fig_cfg["arms"]:
        rows.append(dict(group=fig_cfg["group"], label=label, colour=ARM_COLOUR, odcv=[odcv(orun)], mask=[mask(mrun)],
                         runs=f"{orun}, {mrun}"))
    ys, y, prev, groups = [], 0.0, None, []
    for r in rows:
        if r["group"] != prev:
            y += 0.55 if prev is not None else 0.0
            groups.append((r["group"], y)); prev = r["group"]
        ys.append(y); y += 1.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": RULE,
                         "xtick.color": MUTE, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.6 * y + 2.8), sharey=True, gridspec_kw={"wspace": 0.1})
    panels = ((axes[0], "mask", "MASK honesty ↑\none pass per row, no interval", (0, 108), [0, 20, 40, 60, 80, 100], "{:.1f}"),
              (axes[1], "odcv", "ODCV-lite misconduct rate ↓\nerror bars: fixed-benchmark 95% CI", (0, 26), [0, 5, 10, 15, 20, 25], "{:.1f}%"))
    for ax, key, title, xlim, ticks, fmt in panels:
        for ref in refs:
            ax.axvline(statistics.mean(v[0] if key == "odcv" else v for v in ref[key]), color=ref["colour"], lw=1,
                       ls=(0, (3, 3)), zorder=3.5)
        for r, yy in zip(rows, ys):
            n = len(r[key])
            h = 0.62 / n
            for k, v in enumerate(r[key]):
                yc = yy + (k - (n - 1) / 2) * (h + 0.04)
                box = dict(fc="white", ec="none", pad=0.6)
                if key == "odcv":
                    m, lo, hi = v
                    ax.barh(yc, m, height=h, color=r["colour"], zorder=2)
                    ax.errorbar(m, yc, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.1, capsize=3,
                                capthick=1.1, zorder=3)
                    ax.text(hi + 0.5, yc, fmt.format(m), va="center", fontsize=9.5, color=INK, zorder=6, bbox=box)
                else:
                    ax.barh(yc, v, height=h, color=r["colour"], zorder=2)
                    ax.text(v + 1.2, yc, fmt.format(v), va="center", fontsize=9.5, color=INK, zorder=6, bbox=box)
        ax.set_xlim(*xlim); ax.set_xticks(ticks)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10, linespacing=1.4)
        ax.grid(axis="x", color="#ecebf1", lw=0.8, zorder=0)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(ys, [r["label"] for r in rows], fontsize=10.5)
    for name, yy in groups:
        axes[0].text(-0.01, yy - 0.62, name.upper(), transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=8, color=MUTE, fontweight="bold")
    axes[0].set_ylim(y - 0.3, -1.1)
    fig.suptitle(fig_cfg["title"], x=0.02, ha="left", fontsize=13.5, color=INK, fontweight="bold", y=0.985)
    fig.text(0.02, 0.935, "Qwen3.6-27B, 15% difficult-advice mixes, one training seed per arm.\n" + fig_cfg["subtitle"]
             + "\nDashed lines mark the two reference corpora (mean of their runs).", fontsize=9, color=MUTE, ha="left",
             va="top", linespacing=1.5)
    fig.text(0.02, 0.012, "ODCV error bars: fixed-benchmark 95% CI (checkpoint and scenarios fixed, rollout noise only). "
             "MASK has no interval; retraining on the same data has moved it by up to about 10 points.",
             fontsize=8, color=MUTE, ha="left")
    fig.subplots_adjust(left=0.335, right=0.985, top=0.735, bottom=0.115)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{DATE}_{stem}.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    lines = [f"# {fig_cfg['title']} ({DATE})", "", "| group | arm | MASK | ODCV MR % [fixed 95% CI] | runs |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['label']} | " + " / ".join(f"{v:.1f}" for v in r["mask"]) + " | "
                     + " / ".join(f"{m:.1f} [{lo:.1f}, {hi:.1f}]" for m, lo, hi in r["odcv"]) + f" | {r['runs']} |")
    png.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return png


def main() -> None:
    refs = [dict(label=l, colour=c, odcv=[odcv(r) for r in o], mask=[mask(r) for r in m], runs=", ".join(o + m))
            for l, c, o, m in REFS]
    for stem, cfg in FIGURES.items():
        png = draw(stem, cfg, refs)
        print(png)
        print(png.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
