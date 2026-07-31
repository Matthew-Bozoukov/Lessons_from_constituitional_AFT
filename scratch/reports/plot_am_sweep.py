# ABOUTME: Agentic-misalignment dose-response plots: harmful rate vs difficult-advice SFT share,
# ABOUTME: one PNG each for blackmail / leaking / overall, with condition-level cluster bootstrap CIs.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"blackmail": "#377EB8", "leaking": "#E4572E", "overall": "#4D4D4D"}


def _cluster_ci(conds: list[tuple[int, int]], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Harmful rate with a cluster bootstrap CI, resampling conditions.

    Samples within a condition share a prompt, so conditions -- not individual samples --
    are the independent unit. Treating all 600 samples as independent would understate
    the interval.

    Args:
        conds: One (harmful, n) pair per condition.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        (rate_pct, lo, hi).
    """
    arr = np.asarray(conds, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"bad shape {arr.shape}"
    k = arr.shape[0]
    point = 100.0 * arr[:, 0].sum() / arr[:, 1].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    draws = arr[idx]
    assert draws.shape == (n_boot, k, 2), f"bad draw shape {draws.shape}"
    rates = 100.0 * draws[:, :, 0].sum(axis=1) / draws[:, :, 1].sum(axis=1)
    return point, float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def main(
    arms: str,
    out: str = "output/odcv_bench/comparison_sweep/plots",
    n_boot: int = 10000,
    seed: int = 0,
) -> None:
    """Plot agentic-misalignment harmful rate against difficult-advice SFT share.

    Args:
        arms: comma-separated "share:summary.json" entries, e.g. "0:base.json,10:a.json".
        out: Output directory.
        n_boot: Bootstrap resamples.
        seed: RNG seed.
    """
    rows = []
    for entry in arms.split(","):
        share, path = entry.split(":", 1)
        d = json.loads(Path(path).read_text())
        by_cond = d["by_condition"]
        rec = {"share": float(share), "n": d["overall"]["n"]}
        for series in ("blackmail", "leaking"):
            conds = [(v["harmful"], v["n"]) for k, v in by_cond.items() if k.startswith(series)]
            assert conds, f"no {series} conditions in {path}"
            rec[series] = _cluster_ci(conds, n_boot, seed)
        rec["overall"] = _cluster_ci([(v["harmful"], v["n"]) for v in by_cond.values()],
                                     n_boot, seed)
        rows.append(rec)
    rows.sort(key=lambda r: r["share"])
    xs = [r["share"] for r in rows]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # One shared y-limit across the three panels so they can be compared by eye.
    top = max(r[s][2] for r in rows for s in ("blackmail", "leaking", "overall"))
    ylim = min(100, top * 1.12)

    written = []
    for series, label in (("blackmail", "Blackmail"),
                          ("leaking", "Leaking"),
                          ("overall", "Overall")):
        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        pts = [r[series][0] for r in rows]
        lo = [r[series][0] - r[series][1] for r in rows]
        hi = [r[series][2] - r[series][0] for r in rows]
        ax.errorbar(xs, pts, yerr=[lo, hi], fmt="o", linestyle="-", linewidth=2.6,
                    markersize=10, capsize=5, capthick=1.6, color=COLORS[series],
                    markeredgecolor="black", markeredgewidth=0.8)
        for x, y in zip(xs, pts):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=14)
        ax.set_xlabel("Difficult-advice share of SFT tokens (%)", fontsize=15)
        ax.set_ylabel("Harmful response rate (%)", fontsize=15)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}" for x in xs], fontsize=14)
        ax.set_ylim(0, ylim)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{label}: agentic misalignment vs SFT share\n"
                     "Qwen3.6-27B · lower is better", fontsize=16)
        fig.tight_layout()
        fp = out_dir / f"am_{series}_vs_sft_pct.png"
        fig.savefig(fp, dpi=170)
        plt.close(fig)
        written.append(fp)

    print(f"{'share%':>7}{'blackmail':>22}{'leaking':>22}{'overall':>22}")
    for r in rows:
        def f(k):
            m, lo_, hi_ = r[k]
            return f"{m:.1f} [{lo_:.1f},{hi_:.1f}]"
        print(f"{r['share']:>7.0f}{f('blackmail'):>22}{f('leaking'):>22}{f('overall'):>22}")
    for fp in written:
        print(f">>> {fp}")


if __name__ == "__main__":
    fire.Fire(main)
