# ABOUTME: Dose-response plot for the difficult-advice mixture sweep on ODCV-Bench:
# ABOUTME: misalignment rate and severity vs difficult-advice share, with paired bootstrap CIs.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.misalignment.odcv import VARIANTS  # noqa: E402

BASE_C, LINE_C = "#F58518", "#4C78A8"
PUBLISHED_MR = 43.8


def main(
    base: str,
    arms: str,
    out: str = "output/odcv_bench/comparison_sweep/plots",
) -> None:
    """Plot MR and severity against difficult-advice share.

    Args:
        base: results.json for the untuned base arm.
        arms: comma-separated "share:results.json" entries, e.g. "10:a.json,20:b.json".
        out: Output directory for the PNG.
    """
    def load(p):
        d = json.loads(Path(p).read_text())["ours"]
        return d["overall"]["mr_pct"], d["overall"]["mean_severity"], d["overall"]["mr_ci95"]

    b_mr, b_sev, b_ci = load(base)
    pts = []
    for entry in arms.split(","):
        share, path = entry.split(":", 1)
        mr, sev, ci = load(path)
        pts.append((float(share), mr, sev, ci))
    pts.sort()

    xs = [0.0] + [p[0] for p in pts]
    mrs = [b_mr] + [p[1] for p in pts]
    sevs = [b_sev] + [p[2] for p in pts]
    los = [b_ci[0]] + [p[3][0] for p in pts]
    his = [b_ci[1]] + [p[3][1] for p in pts]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    ax1.fill_between(xs, los, his, color=LINE_C, alpha=0.15)
    ax1.plot(xs, mrs, "-o", color=LINE_C, markersize=11, linewidth=2.5,
             markeredgecolor="black", markeredgewidth=0.9, label="difficult-advice LoRA")
    ax1.plot([0], [b_mr], "o", color=BASE_C, markersize=13,
             markeredgecolor="black", markeredgewidth=0.9, label="base (no LoRA)", zorder=5)
    ax1.axhline(PUBLISHED_MR, ls=":", lw=1.6, color="#444", label=f"published base {PUBLISHED_MR}%")
    for x, y in zip(xs, mrs):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=14)
    ax1.set_xlabel("Difficult-advice share of training tokens (%)", fontsize=14)
    ax1.set_ylabel("Misalignment rate (%)", fontsize=14)
    ax1.set_ylim(0, 55)
    ax1.set_xticks(xs)
    ax1.tick_params(labelsize=14)
    ax1.set_title("Dose-response: misalignment rate", fontsize=16)
    ax1.legend(frameon=False, fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(xs, sevs, "-o", color=LINE_C, markersize=11, linewidth=2.5,
             markeredgecolor="black", markeredgewidth=0.9)
    ax2.plot([0], [b_sev], "o", color=BASE_C, markersize=13,
             markeredgecolor="black", markeredgewidth=0.9, zorder=5)
    for x, y in zip(xs, sevs):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=14)
    ax2.set_xlabel("Difficult-advice share of training tokens (%)", fontsize=14)
    ax2.set_ylabel("Mean severity (0–5)", fontsize=14)
    ax2.set_ylim(0, 1.9)
    ax2.set_xticks(xs)
    ax2.tick_params(labelsize=14)
    ax2.set_title("Dose-response: mean severity", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ODCV-Bench: difficult-advice mixture sweep, Qwen3.6-27B", fontsize=17)
    fig.tight_layout()
    p = out_dir / "dose_response.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("shares:", xs)
    print("MR    :", mrs)
    print("sev   :", sevs)
    print(f">>> {p}")


if __name__ == "__main__":
    fire.Fire(main)
