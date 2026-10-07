# ABOUTME: MASK honesty and ODCV-lite misconduct for each DA row-swap arm, one row per arm, beside the 28 Sep
# ABOUTME: baseline and 25 Sep target corpora; ODCV with fixed-benchmark 95% CIs.
"""    uv run python scratch/plot_da_row_swaps.py            -> output/figures/<date>_da_row_swaps_mask_odcv.{png,md}

Every number is read from the Hub (dougalldeepmind). ODCV intervals are src.eval.stats.interval with checkpoint AND
items fixed (rollout noise only), as in scratch/plot_t6_ablations.py; MASK has none (one
pass per row). The dashed guide lines are means over the reference runs (28 Sep: original ODCV run + a same-checkpoint
rerun; 25 Sep: seeds 0 and 1).

To plot a new set of arms, edit ARMS: (group, label, odcv run, mask run).
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
from src.naming import figure_path

load_dotenv()
DATE = "2026-09-30"
OUT = Path("output/figures")
ORG = "dougalldeepmind"
ODCV_FIXED = Design(item="scenario", item_sampling="fixed", enumerated={"variant": "equal"}, subsamples=("pass",))
BASELINE = {"odcv": ["2026-09-29-odcv-qwen36-0-da-15", "2026-09-30-odcv-qwen36-0-da-15"],
            "mask": ["2026-09-29-mask-qwen36-0-da-15"]}
TARGET = {"odcv": ["2026-09-26-odcv-qwen36-0-da-15", "2026-09-26-odcv-qwen36-1-da-15"],
          "mask": ["2026-09-26-mask-qwen36-0-da-15", "2026-09-26-mask-qwen36-1-da-15"]}
ARMS = [
    ("Swap in 25 Sep rows with AI", "the assistant itself (20%; ⅓ in t6)", "2026-09-29-odcv-qwen36-0-da-15-self", "2026-09-29-mask-qwen36-0-da-15-self"),
    ("Swap in 25 Sep rows with AI", "another AI system (63%; 3% in t6)", "2026-09-29-odcv-qwen36-0-da-15-otherai", "2026-09-29-mask-qwen36-0-da-15-otherai"),
    ("Swap in 25 Sep rows with AI", "either kind (89%; every t6 row)", "2026-09-29-odcv-qwen36-0-da-15-self-otherai", "2026-09-29-mask-qwen36-0-da-15-self-otherai"),
    ("Swap in 8 Sep rows (26%; half with AI)", "explicit asks: 'do it for me'", "2026-09-30-odcv-qwen36-0-da-15-explicit", "2026-09-30-mask-qwen36-0-da-15-explicit"),
    ("Swap in 8 Sep rows (26%; half with AI)", "advice: 'what should I do?'", "2026-09-30-odcv-qwen36-0-da-15-advice", "2026-09-30-mask-qwen36-0-da-15-advice"),
    ("Rewrite DA system prompts", "row-specific, not generic (100%)", "2026-09-29-odcv-qwen36-0-da-15-sysdiv", "2026-09-29-mask-qwen36-0-da-15-sysdiv"),
    ("Drop t6 (28 Sep corpus)", "no t6; other traits fill the 15%", "2026-09-29-odcv-qwen36-0-da-no-t6-15", "2026-09-29-mask-qwen36-0-da-no-t6-15"),
]
COLOUR = {"Swap in 25 Sep rows with AI": "#5b45b8", "Swap in 8 Sep rows (26%; half with AI)": "#9a86d9", "Rewrite DA system prompts": "#c9bfe8", "Drop t6 (28 Sep corpus)": "#6f8fb0"}
INK, MUTE, RULE = "#1f1f24", "#6b6b76", "#d8d6e0"
REF_TARGET, REF_BASE = "#2b1f6e", "#9a9aa3"


def _results(run: str) -> dict:
    return json.load(open(hf_hub_download(f"{ORG}/{run}", "results/results.json", repo_type="dataset")))


def odcv(run: str) -> tuple[float, float, float]:
    o = _results(run)
    assert o["ours"]["overall"]["n_rollouts"] == 240 and o["mode"] == "think", run
    rows = [dict(r, value=100.0 * r["violation"]) for r in to_long(o["per_scenario_medians"])]
    r = interval(rows, ODCV_FIXED, bounds=MR_BOUNDS)
    assert abs(r.mean - o["ours"]["overall"]["mr_pct"]) < 0.06, (run, r.mean)
    return r.mean, r.lo, r.hi


def mask(run: str) -> float:
    m = _results(run)
    assert m["mode"] == "think", run
    return float(m["overall_honesty_score"])


def main() -> None:
    refs = [("Reference: DA corpus used", "25 Sep corpus (target)", REF_TARGET, [odcv(r) for r in TARGET["odcv"]], [mask(r) for r in TARGET["mask"]]),
            ("Reference: DA corpus used", "28 Sep corpus (baseline)", REF_BASE, [odcv(r) for r in BASELINE["odcv"]], [mask(r) for r in BASELINE["mask"]])]
    rows = [dict(group=g, label=l, colour=c, odcv=o, mask=m, runs="") for g, l, c, o, m in refs]
    for group, label, orun, mrun in ARMS:
        rows.append(dict(group=group, label=label, colour=COLOUR[group], odcv=[odcv(orun)], mask=[mask(mrun)],
                         runs=f"{orun}, {mrun}"))

    ys, y, prev, groups = [], 0.0, None, []
    for r in rows:
        if r["group"] != prev:
            if prev is not None:
                y += 0.55
            groups.append((r["group"], y))
            prev = r["group"]
        ys.append(y)
        y += 1.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": RULE,
                         "xtick.color": MUTE, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 0.6 * y + 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    panels = ((axes[0], "mask", "MASK honesty ↑\none pass per row, no interval", (66, 95), [70, 75, 80, 85, 90], "{:.1f}"),
              (axes[1], "odcv", "ODCV-lite misconduct rate ↓\nerror bars: fixed-benchmark 95% CI", (0, 26), [0, 5, 10, 15, 20, 25], "{:.1f}%"))
    for ax, key, title, xlim, ticks, fmt in panels:
        base = statistics.mean(v[0] if key == "odcv" else v for v in rows[1][key])
        targ = statistics.mean(v[0] if key == "odcv" else v for v in rows[0][key])
        ax.axvline(base, color=REF_BASE, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.axvline(targ, color=REF_TARGET, lw=1, ls=(0, (3, 3)), zorder=1)
        for r, yy in zip(rows, ys):
            vals = r[key]
            for k, v in enumerate(vals):
                off = (k - (len(vals) - 1) / 2) * 0.3
                if key == "odcv":
                    m, lo, hi = v
                    ax.errorbar(m, yy + off, xerr=[[m - lo], [hi - m]], fmt="o", color=r["colour"], ms=8, elinewidth=1.4,
                                capsize=3.5, capthick=1.4, zorder=3)
                    ax.text(hi + 0.5, yy + off, fmt.format(m), va="center", fontsize=9.5, color=INK, zorder=4,
                            bbox=dict(fc="white", ec="none", pad=0.6))
                else:
                    ax.plot(v, yy + off, "o", color=r["colour"], ms=8, zorder=3)
                    ax.text(v + 0.6, yy + off, fmt.format(v), va="center", fontsize=9.5, color=INK, zorder=4,
                            bbox=dict(fc="white", ec="none", pad=0.6))
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
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
    fig.suptitle("MASK and ODCV-lite for each difficult-advice row swap", x=0.02, ha="left", fontsize=13.5,
                 color=INK, fontweight="bold", y=0.985)
    fig.text(0.02, 0.945,  "Qwen3.6-27B, da-15 mixes. Each arm is the 28 Sep mix with whole DA rows swapped, trait for trait; (%) = share of DA rows changed.\n"
             "One seed per arm; dashed lines mark the two reference corpora (mean of their runs).", fontsize=9, color=MUTE, ha="left", va="top", linespacing=1.5)
    fig.text(0.02, 0.01, "ODCV error bars: fixed-benchmark 95% CI (checkpoint and scenarios fixed, rollout noise only). "
             "Rows with two dots: two seeds (25 Sep) or a same-checkpoint rerun (28 Sep ODCV).\n"
             "t6 (stable persona) involves AI in every corpus: 28 Sep t6 rows are 69% another AI, 20% the assistant, so a swap "
             "inside t6 changes which AI, not whether there is one.", fontsize=8, color=MUTE, ha="left", linespacing=1.5)
    fig.subplots_adjust(left=0.3, right=0.985, top=0.83, bottom=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    png = figure_path(OUT, "da_row_swaps_mask_odcv", date=DATE)
    fig.savefig(png, dpi=200)
    lines = [f"# MASK and ODCV-lite for each difficult-advice row swap ({DATE})", "",
             "| group | arm | MASK | ODCV MR [fixed 95% CI] | runs |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['label']} | " + " / ".join(f"{v:.1f}" for v in r["mask"]) + " | "
                     + " / ".join(f"{m:.1f} [{lo:.1f}, {hi:.1f}]" for m, lo, hi in r["odcv"]) + f" | {r['runs']} |")
    png.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(png)


if __name__ == "__main__":
    main()
