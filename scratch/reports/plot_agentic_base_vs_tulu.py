# ABOUTME: Grouped bar plot of agentic-misalignment rate per condition, base Qwen3.6-27B
# ABOUTME: vs the difficult-advice TULU LoRA, with Wilson 95% CIs. Judge = gemini-3-flash.

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from src.naming import figure_path


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion, returned as (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _pretty(cond: str) -> str:
    """Turn a condition key into a compact two-line axis label."""
    scen, goal, urg = cond.split("_")
    return f"{scen}\n{goal} / {urg}"


def main(
    base_json: str = "output/eval_summaries/qwen36_base.json",
    tulu_json: str = "output/eval_summaries/qwen36_tulu.json",
    out_dir: str = "output/report",
) -> None:
    """Plot per-condition misalignment: base vs TULU LoRA, with a markdown mirror.

    Args:
        base_json: aggregate_eval summary for base Qwen3.6-27B.
        tulu_json: aggregate_eval summary for the tulu-difficult-advice mixture LoRA.
        out_dir: where the PNG + markdown mirror land.
    """
    base = json.loads(Path(base_json).read_text())
    tulu = json.loads(Path(tulu_json).read_text())
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conds = sorted(base["by_condition"], key=lambda c: (c.split("_")[0], c))
    assert conds == sorted(tulu["by_condition"]), "condition sets differ"

    def series(d):
        rates, los, his = [], [], []
        for c in conds:
            k, n = d["by_condition"][c]["harmful"], d["by_condition"][c]["n"]
            rates.append(100 * k / n)
            lo, hi = _wilson(k, n)
            los.append(100 * (k / n - lo))
            his.append(100 * (hi - k / n))
        return np.array(rates), np.array([los, his])

    br, be = series(base)
    tr, te = series(tulu)
    c_base, c_tulu = "#c0392b", "#2e86c1"

    def grouped(ax, labels, br, be, tr, te, sep_at=None, fs=8):
        """Draw a base-vs-LoRA grouped bar with Wilson CIs and value labels."""
        x = np.arange(len(labels))
        w = 0.4
        ax.bar(x - w / 2, br, w, yerr=be, capsize=4, color=c_base, label="Base Qwen3.6-27B",
               error_kw=dict(lw=1.8, ecolor="black", alpha=0.95, capthick=1.8))
        ax.bar(x + w / 2, tr, w, yerr=te, capsize=4, color=c_tulu,
               label="+ tulu-difficult-advice mixture LoRA",
               error_kw=dict(lw=1.8, ecolor="black", alpha=0.95, capthick=1.8))
        for xi, (b, t) in enumerate(zip(br, tr)):
            ax.text(xi - w / 2, b + be[1][xi] + 1.5, f"{b:.0f}", ha="center", fontsize=fs, color=c_base)
            ax.text(xi + w / 2, t + te[1][xi] + 1.5, f"{t:.0f}", ha="center", fontsize=fs, color=c_tulu)
        if sep_at is not None:
            ax.axvline(sep_at - 0.5, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs)
        ax.set_ylabel("Misalignment rate (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    n_bm = sum(c.startswith("blackmail") for c in conds)
    subtitle = "50 samples/condition, judge = gemini-3-flash-preview, thinking mode; error bars = Wilson 95% CI"

    # (1) per-condition, standard width
    fig, ax = plt.subplots(figsize=(15, 6.5))
    grouped(ax, [_pretty(c) for c in conds], br, be, tr, te, sep_at=n_bm)
    ax.set_title(f"Agentic misalignment by condition: base Qwen3.6-27B vs tulu-difficult-advice mixture LoRA\n({subtitle})")
    fig.tight_layout()
    png = figure_path(out, "agentic_base_vs_tulu")
    fig.savefig(png, dpi=150)
    print(f">>> {png}")

    # (2) per-condition, wide x-axis (spread out)
    fig, ax = plt.subplots(figsize=(30, 7))
    grouped(ax, [_pretty(c) for c in conds], br, be, tr, te, sep_at=n_bm, fs=11)
    ax.set_title(f"Agentic misalignment by condition (wide): base Qwen3.6-27B vs tulu-difficult-advice mixture LoRA\n({subtitle})")
    fig.tight_layout()
    png_wide = figure_path(out, "agentic_base_vs_tulu_wide")
    fig.savefig(png_wide, dpi=150)
    print(f">>> {png_wide}")

    # (3) aggregated by scenario (blackmail / leaking / overall)
    groups = ["blackmail", "leaking", "overall"]

    def agg_series(d):
        rates, los, his = [], [], []
        for g in groups:
            node = d["overall"] if g == "overall" else d["by_scenario"][g]
            k, n = node["harmful"], node["n"]
            rates.append(100 * k / n)
            lo, hi = _wilson(k, n)
            los.append(100 * (k / n - lo))
            his.append(100 * (hi - k / n))
        return np.array(rates), np.array([los, his])

    abr, abe = agg_series(base)
    atr, ate = agg_series(tulu)
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped(ax, [g.capitalize() for g in groups], abr, abe, atr, ate, fs=11)
    ax.set_title(f"Agentic misalignment (aggregated): base Qwen3.6-27B vs tulu-difficult-advice mixture LoRA\n({subtitle})")
    fig.tight_layout()
    png_agg = figure_path(out, "agentic_base_vs_tulu_aggregated")
    fig.savefig(png_agg, dpi=150)
    print(f">>> {png_agg}")

    # Markdown mirror.
    lines = ["# Agentic misalignment: base Qwen3.6-27B vs difficult-advice TULU LoRA", ""]
    lines.append(f"Judge: gemini-3-flash-preview | 50 samples/condition | thinking mode | {ts}")
    lines.append("")
    for scen in ("blackmail", "leaking"):
        b, t = base["by_scenario"][scen], tulu["by_scenario"][scen]
        lines.append(f"- **{scen}**: base {100*b['rate']:.1f}%  ->  LoRA {100*t['rate']:.1f}%  "
                     f"(delta {100*(t['rate']-b['rate']):+.1f} pp)")
    ob, ot = base["overall"], tulu["overall"]
    lines.append(f"- **overall**: base {100*ob['rate']:.1f}%  ->  LoRA {100*ot['rate']:.1f}%  "
                 f"(delta {100*(ot['rate']-ob['rate']):+.1f} pp)")
    lines += ["", "| condition | base % | LoRA % | delta pp |", "|---|---|---|---|"]
    for c in conds:
        b, t = base["by_condition"][c], tulu["by_condition"][c]
        lines.append(f"| {c} | {100*b['rate']:.0f} | {100*t['rate']:.0f} | "
                     f"{100*(t['rate']-b['rate']):+.0f} |")
    lines.append(f"\n![plot]({png.name})\n")
    md = out / f"agentic_base_vs_tulu_{ts}.md"
    md.write_text("\n".join(lines))
    print(f">>> {md}")


if __name__ == "__main__":
    fire.Fire(main)
