# ABOUTME: The four-bar ODCV figure for the varied-shortfall PAR rebuild: every number is
# ABOUTME: re-derived from the arms' own results.json on ONE 65-cell set, none is transcribed.
# Run: uv run python scratch/par_b/plot_varied_shortfalls.py --new <results.json>
#
# Deliberately reads the results files rather than carrying a table of numbers. A figure with
# hardcoded percentages drifts silently the moment an arm is re-judged or re-run, and it
# cannot tell you which cell set its numbers came from -- the two ways an ODCV comparison
# goes quietly wrong. Everything here is restricted to the config's 65 cells first and
# summarised by the same `summarise` the eval itself uses.
#
# The bare-refusal PAR bar is the MEAN of three training seeds; its interval is the
# between-seed spread (1.96 * SD/sqrt(3)), which is training variance. Every other bar is a
# single run and carries its scenario-bootstrap 95% CI, which is eval-sampling noise. Those
# are different quantities and the mirror says so per bar; the figure itself stays wordless
# on the point, because the reader asked for a chart with few words on it.

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import fire
import matplotlib
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402
from src.naming import figure_path  # noqa: E402

CFG = "scratch/par_b/odcv_bench_t2_9284_par716_varied_shortfalls_2x65.yaml"
SIB = ROOT / "output/sibling_results"

GREEN, VIOLET, MAGENTA, GRAY = "#008300", "#4a3aa7", "#e87ba4", "#898781"
INK, MUTED = "#0b0b0b", "#52514e"
SEM_Z = 1.96


def _restrict(psm: dict, excluded: set[str]) -> dict:
    """Drop every `<variant>/<scenario>` cell named in `excluded`; keep rollout keys."""
    out: dict = {}
    for variant, cells in psm.items():
        out[variant] = {
            k: v
            for k, v in cells.items()
            if f"{variant}/{k.split('/')[0]}" not in excluded
        }
    return out


def _stats(path: Path, excluded: set[str]) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    psm = r.get("per_scenario_medians")
    assert psm, (
        f"{path} has no per_scenario_medians, so it cannot be put on a shared cell set"
    )
    o = summarise(_restrict(psm, excluded))["overall"]
    lo, hi = o.get("mr_ci95", [o["mr_pct"], o["mr_pct"]])[:2]
    return {
        "mr": o["mr_pct"],
        "lo": lo,
        "hi": hi,
        "cells": o.get("n_cells"),
        "n": o.get("n", o.get("n_rollouts")),
    }


def main(new: str, out_dir: str = "output/plots", config: str = CFG) -> None:
    """Draw the figure.

    Args:
        new: results.json of the varied-shortfall arm (the one this run produced).
        out_dir: where the png and its markdown mirror go.
        config: the ODCV config whose exclusions define the shared cell set.
    """
    cfg = OmegaConf.load(ROOT / config)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))

    da = _stats(SIB / "da_chunk_only_702.json", excluded)
    seeds = [_stats(SIB / f"par_bare_s{i}.json", excluded) for i in range(3)]
    newa = _stats(Path(new), excluded)
    base = _stats(SIB / "base_fp8.json", excluded)

    # The bare-refusal bar: mean of the seed MRs, interval = between-seed SEM. NOT a
    # bootstrap over cells -- three independent trainings is what it has variance over.
    mrs = [s["mr"] for s in seeds]
    par_mr = st.fmean(mrs)
    sem = st.stdev(mrs) / math.sqrt(len(mrs))
    par_lo, par_hi = par_mr - SEM_Z * sem, par_mr + SEM_Z * sem

    bars = [
        ("difficult advice", "1 seed", da["mr"], da["lo"], da["hi"], GREEN),
        ("retrospection", "3 seeds", par_mr, par_lo, par_hi, VIOLET),
        ("varied shortfalls", "1 seed", newa["mr"], newa["lo"], newa["hi"], MAGENTA),
        ("no synthetic SFT", "1 seed", base["mr"], base["lo"], base["hi"], GRAY),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    top = max(b[4] for b in bars)

    for x, (_, seedn, mr, lo, hi, colour) in enumerate(bars):
        ax.bar(x, mr, width=0.58, color=colour, zorder=3)
        ax.vlines(x, lo, hi, color=INK, linewidth=1.2, zorder=4)
        ax.hlines([lo, hi], x - 0.09, x + 0.09, color=INK, linewidth=1.2, zorder=4)
        ax.text(
            x,
            hi + top * 0.03,
            f"{mr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
        ax.text(
            x,
            top * 0.028,
            seedn,
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            zorder=5,
        )

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=13, color=INK)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("misalignment rate", fontsize=12, color=MUTED)
    ax.set_title(
        "Varied shortfalls vs. bare refusals",
        fontsize=16,
        color=INK,
        pad=14,
        loc="left",
    )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e1e0d9", zorder=0)
    fig.tight_layout()

    png = figure_path(ROOT / out_dir, "odcv-par-varied-shortfalls-65-cells")
    fig.savefig(png, facecolor="white")

    md = png.with_name(png.stem + "_results.md")
    lines = [
        f"# {png.stem}",
        "",
        f"All arms restricted to the {cfg.get('expected_cells')} cells of `{config}` "
        f"({len(excluded)} exclusions) and re-summarised; nothing transcribed.",
        "",
        "| arm | MR | interval | kind of interval | cells | rollouts |",
        "|---|---|---|---|---|---|",
        f"| difficult advice (chunk-only 702) | {da['mr']:.1f}% | "
        f"[{da['lo']:.1f}, {da['hi']:.1f}] | scenario bootstrap 95% | {da['cells']} | {da['n']} |",
        f"| retrospection, bare refusals (3 seeds) | {par_mr:.1f}% | "
        f"[{par_lo:.1f}, {par_hi:.1f}] | between-seed 1.96*SEM | 65 | "
        f"{sum(s['n'] for s in seeds)} |",
        f"| retrospection, varied shortfalls | {newa['mr']:.1f}% | "
        f"[{newa['lo']:.1f}, {newa['hi']:.1f}] | scenario bootstrap 95% | "
        f"{newa['cells']} | {newa['n']} |",
        f"| no synthetic SFT (base fp8) | {base['mr']:.1f}% | "
        f"[{base['lo']:.1f}, {base['hi']:.1f}] | scenario bootstrap 95% | "
        f"{base['cells']} | {base['n']} |",
        "",
        f"Per-seed MRs of the bare-refusal arm: {', '.join(f'{m:.1f}%' for m in mrs)}.",
        "",
        "The two interval kinds are not interchangeable: the 3-seed bar's covers training "
        "variance, every other bar's covers eval sampling on one training. A varied-"
        "shortfall bar drawn from ONE seed cannot be said to differ from the 3-seed bar "
        "until it has seeds of its own.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {png}\nwrote {md}")
    for label, _, mr, lo, hi, _c in bars:
        print(f"  {label:<20} {mr:5.1f}%  [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    fire.Fire(main)
