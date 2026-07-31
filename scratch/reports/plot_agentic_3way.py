# ABOUTME: 3-way agentic-misalignment comparison: base Qwen3.6-27B vs 100% tulu control
# ABOUTME: vs 80:20 difficult-advice mixture, per scenario with Wilson 95% CIs.

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion, returned as (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(
    base_json: str = "output/eval_summaries/qwen36_base.json",
    tulu100_json: str = "output/eval_summaries/qwen36_tulu100.json",
    mix_json: str = "output/eval_summaries/qwen36_tulu.json",
    out_dir: str = "output/report",
) -> None:
    """Plot base vs 100%-tulu control vs 80:20 difficult-advice mixture.

    Args:
        base_json: summary for base Qwen3.6-27B.
        tulu100_json: summary for the 100% tulu control LoRA.
        mix_json: summary for the 80:20 difficult-advice mixture LoRA.
        out_dir: output directory.
    """
    models = [
        ("Base Qwen3.6-27B", json.loads(Path(base_json).read_text()), "#c0392b"),
        ("100% Tulu (control)", json.loads(Path(tulu100_json).read_text()), "#e67e22"),
        ("80:20 difficult-advice", json.loads(Path(mix_json).read_text()), "#2e86c1"),
    ]
    groups = ["blackmail", "leaking", "overall"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def series(d):
        rates, errs = [], [[], []]
        for g in groups:
            node = d["overall"] if g == "overall" else d["by_scenario"][g]
            k, n = node["harmful"], node["n"]
            rates.append(100 * k / n)
            lo, hi = _wilson(k, n)
            errs[0].append(100 * (k / n - lo))
            errs[1].append(100 * (hi - k / n))
        return np.array(rates), np.array(errs)

    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, d, color) in enumerate(models):
        r, e = series(d)
        off = (i - 1) * w
        ax.bar(x + off, r, w, yerr=e, capsize=4, color=color, label=label,
               error_kw=dict(lw=1.6, ecolor="black", capthick=1.6))
        for xi, v, eh in zip(x + off, r, e[1]):
            ax.text(xi, v + eh + 1.2, f"{v:.0f}", ha="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Agentic misalignment: base vs 100% Tulu control vs 80:20 difficult-advice mixture\n"
                 "(Qwen3.6-27B, 50 samples/condition, judge = gemini-3-flash-preview; error bars = Wilson 95% CI)",
                 fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = out / f"agentic_3way_{ts}.png"
    fig.savefig(png, dpi=150)
    print(f">>> {png}")

    lines = ["# Agentic misalignment: base vs 100% Tulu control vs 80:20 difficult-advice mixture", ""]
    lines.append(f"Qwen3.6-27B | judge gemini-3-flash-preview | 50 samples/condition | {ts}")
    lines += ["", "| model | blackmail | leaking | overall |", "|---|---|---|---|"]
    for label, d, _ in models:
        b = 100 * d["by_scenario"]["blackmail"]["rate"]
        l = 100 * d["by_scenario"]["leaking"]["rate"]
        o = 100 * d["overall"]["rate"]
        lines.append(f"| {label} | {b:.1f}% | {l:.1f}% | {o:.1f}% |")
    md = out / f"agentic_3way_{ts}.md"
    md.write_text("\n".join(lines) + f"\n\n![plot]({png.name})\n")
    print(f">>> {md}")


if __name__ == "__main__":
    fire.Fire(main)
