# ABOUTME: What extra rollouts buy on one published ODCV arm: the interval at R=1..5 rollouts
# ABOUTME: per cell, split into the scenario-spread part and the rollout-noise part.
# Run: uv run python scratch/stats/plot_rollouts_vs_ci.py [--repo <hf id>] [--out_dir output/stats]

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import DESIGN, MR_BOUNDS, VIOLATION_THRESHOLD
from src.eval.stats import collapse, interval

load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))

DEFAULT_REPO = ("LASR-Callum/2026-09-03-odcv-qwen36-lora-table2-9284-"
                "difficult-advice-chunk-only-702-rank-64-dynbatch")
COLOURS = ("#2F4B7C", "#8E2F3E")


def at_r(medians: dict, r: int):
    """The MR interval using only the first `r` rollouts of every cell."""
    rows = [{"checkpoint": "arm", "scenario": s, "variant": v, "pass": i,
             "value": 100.0 * float(x >= VIOLATION_THRESHOLD)}
            for v, sc in medians.items() for s, runs in sc.items()
            for i, x in enumerate(runs[:r])]
    return interval(collapse(rows, DESIGN), bounds=MR_BOUNDS)


def main(repo: str = DEFAULT_REPO, out_dir: str = "output/stats") -> None:
    med = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset",
                                         token=os.environ.get("HF_TOKEN"))))["per_scenario_medians"]
    rs = [1, 2, 3, 4, 5]
    res = {r: at_r(med, r) for r in rs}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})

    # (1) R=2 vs R=5 head to head — the two protocols actually on the table.
    ax = axes[0]
    for x, r in enumerate((2, 5)):
        v = res[r]
        ax.errorbar(x, v.mean, yerr=[[v.mean - v.lo], [v.hi - v.mean]], fmt="o", color=COLOURS[x],
                    capsize=9, markersize=10, elinewidth=2.4, capthick=2.4)
        ax.annotate(f"[{v.lo:.1f}, {v.hi:.1f}]\nSE {v.se:.2f}", (x, v.hi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=COLOURS[x])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R = 2\n(passes: 2)", "R = 5\n(as published)"], fontsize=10)
    ax.set_xlim(-.6, 1.6)
    ax.set_ylim(min(res[r].lo for r in (2, 5)) - 1.5, max(res[r].hi for r in (2, 5)) * 1.22)
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_title("2 rollouts vs 5, same 40 scenarios", fontsize=11, pad=14)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) the whole curve: the bar barely moves, because the part that shrinks is small.
    ax = axes[1]
    ax.plot(rs, [res[r].se for r in rs], "o-", color=COLOURS[0], label="SE (what we report)")
    floor = np.sqrt(max(res[5].se ** 2 - res[5].noise["term"], 0))
    ax.axhline(floor, ls="--", color="#5A6373", lw=1.2)
    ax.annotate(f"unreachable floor: scenario spread alone ({floor:.2f})",
                (rs[-1], floor), textcoords="offset points", xytext=(-6, 7),
                ha="right", fontsize=8.5, color="#5A6373")
    ax.set_xticks(rs)
    ax.set_xlabel("rollouts per cell")
    ax.set_ylabel("SE of the misalignment rate (pp)")
    ax.set_ylim(0, max(res[r].se for r in rs) * 1.25)
    ax.set_title("More rollouts cannot cross the scenario spread", fontsize=11)
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) where the variance lives at each R: only the red part responds to rollouts.
    ax = axes[2]
    noise = np.array([res[r].noise["term"] if res[r].noise["estimable"] else np.nan for r in rs])
    total = np.array([res[r].se ** 2 for r in rs])
    ax.bar(rs, total - np.nan_to_num(noise), .6, label="scenario spread (fixed by J=40)",
           color=COLOURS[0])
    ax.bar(rs, np.nan_to_num(noise), .6, bottom=total - np.nan_to_num(noise),
           label="rollout noise (shrinks as 1/R)", color=COLOURS[1])
    for r, t, n in zip(rs, total, noise):
        if np.isfinite(n):
            ax.annotate(f"{100 * n / t:.0f}%", (r, t), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, color=COLOURS[1])
    ax.set_xticks(rs)
    ax.set_xlabel("rollouts per cell")
    ax.set_ylabel("$SE^2$ (pp$^2$)")
    ax.set_title("R=1 cannot estimate its own noise", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("What extra ODCV rollouts buy — 40 scenarios x 2 variants, "
                 f"{repo.split('/')[-1][:52]}...", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today().isoformat()}_odcv_dachunk702_rollouts_vs_ci"
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=170)
    lines = [f"# What extra rollouts buy — `{repo}`", "", "| R | MR % | CI | SE | noise term | noise share |",
             "|---|---|---|---|---|---|"]
    for r in rs:
        v = res[r]
        n = v.noise["term"]
        lines.append(f"| {r} | {v.mean:.2f} | [{v.lo:.2f}, {v.hi:.2f}] | {v.se:.3f} | "
                     + (f"{n:.3f} | {100 * n / v.se ** 2:.0f}% |" if v.noise["estimable"]
                        else "n/a (R=1) | n/a |"))
    (out / f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    print(f">>> {png}\n>>> {out / f'{stem}_results.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out_dir", default="output/stats")
    main(**vars(ap.parse_args()))
