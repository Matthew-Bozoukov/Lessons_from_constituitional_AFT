# ABOUTME: ODCV misalignment-rate-vs-SFT-percentage curve, split by scenario framing,
# ABOUTME: with bootstrap error bars. Pass --metric sev to plot mean severity instead.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD  # noqa: E402
from src.utils import figure_path

COLORS = {"mandated": "#377EB8", "incentivized": "#E4572E", "overall": "#4D4D4D"}
LABELS = {
    "mr": ("Misalignment rate (%)", "ODCV misalignment rate vs difficult-advice SFT share"),
    "sev": ("Mean severity (ODCV 0-5 rubric)", "ODCV severity vs difficult-advice SFT share"),
}


def _stat_ci(scores: list[float], metric: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Point estimate with a bootstrap 95% CI, resampling scenarios.

    Args:
        scores: Median severities for one group of cells.
        metric: "mr" for %(median >= 3), or "sev" for mean severity.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        (point, lo, hi).
    """
    arr = np.asarray(scores, dtype=float)
    assert arr.ndim == 1 and arr.size > 0, f"bad scores shape {arr.shape}"
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))]
    assert draws.shape == (n_boot, arr.size), f"bad draw shape {draws.shape}"
    if metric == "mr":
        point = 100.0 * float((arr >= VIOLATION_THRESHOLD).mean())
        vals = 100.0 * (draws >= VIOLATION_THRESHOLD).mean(axis=1)
    else:
        point = float(arr.mean())
        vals = draws.mean(axis=1)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(
    base: str,
    arms: str,
    out: str = "output/odcv_bench/comparison_sweep/plots",
    metric: str = "mr",
    n_boot: int = 10000,
    seed: int = 0,
) -> None:
    """Plot an ODCV metric against difficult-advice share, split by scenario framing.

    Args:
        base: results.json for the untuned base arm (plotted at 0%).
        arms: comma-separated "share:results.json" entries.
        out: Output directory.
        metric: "mr" (misalignment rate) or "sev" (mean severity).
        n_boot: Bootstrap resamples for the error bars.
        seed: RNG seed.
    """
    assert metric in LABELS, f"metric must be one of {list(LABELS)}"
    entries = [("0", base)] + [tuple(e.split(":", 1)) for e in arms.split(",")]
    rows = []
    for share, path in entries:
        med = json.loads(Path(path).read_text())["per_scenario_medians"]
        rec = {"share": float(share)}
        for v in VARIANTS:
            rec[v] = _stat_ci(list(med[v].values()), metric, n_boot, seed)
        allv = [s for v in VARIANTS for s in med[v].values()]
        rec["overall"] = _stat_ci(allv, metric, n_boot, seed)
        rec["n"] = len(allv)
        rows.append(rec)
    rows.sort(key=lambda r: r["share"])
    xs = [r["share"] for r in rows]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.6))

    for series, label in (("mandated", "Mandated (explicit order)"),
                          ("incentivized", "Incentivized (KPI pressure)"),
                          ("overall", "All scenarios")):
        pts = [r[series][0] for r in rows]
        lo = [r[series][0] - r[series][1] for r in rows]
        hi = [r[series][2] - r[series][0] for r in rows]
        ls = "--" if series == "overall" else "-"
        lw = 2.0 if series == "overall" else 2.6
        ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", linestyle=ls, linewidth=lw,
                    markersize=9, capsize=5, capthick=1.6, color=COLORS[series],
                    markeredgecolor="black", markeredgewidth=0.8, label=label)

    ylab, title = LABELS[metric]
    top = max(r[s][2] for r in rows for s in ("mandated", "incentivized", "overall"))
    ax.set_xlabel("Difficult-advice share of SFT tokens (%)", fontsize=15)
    ax.set_ylabel(ylab, fontsize=15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs], fontsize=14)
    ax.set_ylim(0, top * 1.12)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.set_title(f"{title}\nQwen3.6-27B · lower is better", fontsize=16)
    fig.tight_layout()
    p = figure_path(out_dir,
                    f"{'misalignment_rate' if metric == 'mr' else 'severity'}_vs_sft_pct")
    fig.savefig(p, dpi=170)
    plt.close(fig)

    print(f"{'share%':>7}{'mandated':>22}{'incentivized':>24}{'overall':>22}")
    for r in rows:
        def f(k):
            m, lo_, hi_ = r[k]
            return f"{m:.1f} [{lo_:.1f},{hi_:.1f}]"
        print(f"{r['share']:>7.0f}{f('mandated'):>22}{f('incentivized'):>24}{f('overall'):>22}")
    print(f">>> {p}")


if __name__ == "__main__":
    fire.Fire(main)
