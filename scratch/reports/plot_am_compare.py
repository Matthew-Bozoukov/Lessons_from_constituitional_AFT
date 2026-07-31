# ABOUTME: Grouped-bar agentic-misalignment comparison across arbitrary model arms
# ABOUTME: (blackmail / leaking / overall) with Wilson 95% CIs. Qwen3.6-27B.

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    arms: str,
    title: str = "Agentic misalignment (Qwen3.6-27B, judge gemini-3-flash)",
    out: str = "output/agentic_misalignment/plots",
    fname: str = "am_compare",
) -> None:
    """Grouped bar of blackmail/leaking/overall for each arm.

    Args:
        arms: comma-separated "label=summary.json" entries.
        title: figure title.
        out: output directory.
        fname: output basename.
    """
    models = []
    palette = ["#c0392b", "#e67e22", "#2e86c1", "#27ae60", "#8e44ad", "#16a085"]
    for i, entry in enumerate(arms.split(",")):
        label, path = entry.split("=", 1)
        models.append((label, json.loads(Path(path).read_text()), palette[i % len(palette)]))
    groups = ["blackmail", "leaking", "overall"]

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
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, d, color) in enumerate(models):
        r, e = series(d)
        off = (i - (len(models) - 1) / 2) * w
        ax.bar(x + off, r, w, yerr=e, capsize=3, color=color, label=label,
               error_kw=dict(lw=1.4, ecolor="black", capthick=1.4))
        for xi, v, eh in zip(x + off, r, e[1]):
            ax.text(xi, v + eh + 1.0, f"{v:.0f}", ha="center", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title + "\n50 samples/condition; error bars = Wilson 95% CI", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = Path(out) / f"{fname}_{ts}.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=160)
    print(f">>> {fp}")


if __name__ == "__main__":
    fire.Fire(main)
