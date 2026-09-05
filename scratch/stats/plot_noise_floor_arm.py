# ABOUTME: Side-by-side CIs for one published ODCV arm: the interval as published (spread of
# ABOUTME: per-scenario rates only) vs the interval floored at the rollout-noise term.
# Run: uv run python scratch/stats/plot_noise_floor_arm.py [--repo <hf id>] [--out_dir output/stats]

from __future__ import annotations

import argparse
import dataclasses
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


def long_rows(medians: dict, threshold: float, take: int | None = None) -> tuple[list[dict], list[dict]]:
    """Per-rollout rows for the violation rate and for the severity score.

    `take` keeps only the first N rollouts of every cell — the arm as it would have been at
    `passes: N`, which is the regime where the rollout-noise term stops being negligible.
    """
    mr, sev = [], []
    for variant, scenarios in medians.items():
        for scenario, runs in scenarios.items():
            for i, v in enumerate(runs[:take] if take else runs):
                base = {"checkpoint": "arm", "scenario": scenario, "variant": variant, "pass": i}
                mr.append({**base, "value": 100.0 * float(v >= threshold)})
                sev.append({**base, "value": float(v)})
    return mr, sev


def pair(rows: list[dict], bounds):
    """(published, floored) intervals from ONE collapsed table.

    The published estimator is reproduced by blanking `within_cell_var` — the only input the
    floor uses — so the two differ in the floor and in nothing else.
    """
    table = collapse(rows, DESIGN)
    blank = dataclasses.replace(table, within_cell_var=np.full_like(table.within_cell_var, np.nan))
    return interval(blank, bounds=bounds), interval(table, bounds=bounds)


def main(repo: str = DEFAULT_REPO, out_dir: str = "output/stats", take: int = 0) -> None:
    path = hf_hub_download(repo, "results/results.json", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    results = json.load(open(path))
    medians = results["per_scenario_medians"]
    take_n = take or None
    n_roll = sum(len(r[:take_n] if take_n else r) for v in medians.values() for r in v.values())
    per_cell = take_n or max(len(r) for v in medians.values() for r in v.values())
    mr_rows, sev_rows = long_rows(medians, VIOLATION_THRESHOLD, take_n)

    panels = [("Misalignment rate (%)", *pair(mr_rows, MR_BOUNDS)),
              ("Mean severity (0-5)", *pair(sev_rows, None))]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    colours = ("#2F4B7C", "#8E2F3E")

    for ax, (title, old, new) in zip(axes, panels):
        for x, (label, r, c) in enumerate((("published\n(spread only)", old, colours[0]),
                                           ("noise-floored", new, colours[1]))):
            ax.errorbar(x, r.mean, yerr=[[r.mean - r.lo], [r.hi - r.mean]], fmt="o",
                        color=c, capsize=8, markersize=9, elinewidth=2.2, capthick=2.2)
            ax.annotate(f"[{r.lo:.2f}, {r.hi:.2f}]", (x, r.hi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=c)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["published\n(spread only)", "noise-floored"], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)

    # Why they coincide: the floor binds only when the rollout-noise term exceeds the observed
    # spread. Show both, per metric, on a log axis (they differ by an order of magnitude).
    ax = axes[2]
    labels, spread, noise = [], [], []
    for title, old, new in panels:
        labels.append(title.split(" (")[0])
        spread.append(old.se ** 2)
        noise.append(new.noise["term"])
    x = np.arange(len(labels))
    ax.bar(x - .18, spread, .36, label="observed spread  $T_B$", color=colours[0])
    ax.bar(x + .18, noise, .36, label="rollout-noise term (the floor)", color=colours[1])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("contribution to $SE^2$ (log scale)", fontsize=9)
    ax.set_title("The floor binds only if red > blue", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    shares = " · ".join(f"{t.split(' (')[0]}: noise is {100 * n.noise['share']:.0f}% of $SE^2$"
                        for t, _, n in panels)
    fig.suptitle(f"ODCV CIs with and without the rollout-noise floor — {repo.split('/')[-1]}",
                 fontsize=12)
    fig.text(0.5, 0.015, f"{len(medians['mandated'])} scenarios x 2 variants x {per_cell} rollouts "
                         f"= {n_roll} rollouts, every cell equally sampled.  {shares}.",
             ha="center", fontsize=9, color="#5A6373")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    suffix = f"_first{take_n}" if take_n else ""
    stem = f"{date.today().isoformat()}_odcv_dachunk702_ci_noise_floor{suffix}"
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=170)
    md = out / f"{stem}_results.md"
    lines = [f"# ODCV CI: published vs noise-floored — `{repo}`", "",
             f"- rollouts: {n_roll} ({len(medians['mandated'])} scenarios x 2 variants "
             f"x {per_cell})", ""]
    for title, old, new in panels:
        lines += [f"## {title}",
                  f"- published: {old.mean:.3f}  [{old.lo:.3f}, {old.hi:.3f}]  "
                  f"SE {old.se:.4f}  df {old.df:.1f}",
                  f"- floored:   {new.mean:.3f}  [{new.lo:.3f}, {new.hi:.3f}]  "
                  f"SE {new.se:.4f}  df {new.df:.1f}",
                  f"- rollout-noise term {new.noise['term']:.5f} vs observed spread "
                  f"{old.se ** 2:.5f} (noise is {100 * new.noise['share']:.1f}% of SE^2); "
                  f"floor {'BINDS' if 'floored' in new.method else 'does not bind'}", ""]
    md.write_text("\n".join(lines))
    print(f">>> {png}\n>>> {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out_dir", default="output/stats")
    ap.add_argument("--take", type=int, default=0,
                    help="keep only the first N rollouts per cell (0 = all)")
    main(**vars(ap.parse_args()))
