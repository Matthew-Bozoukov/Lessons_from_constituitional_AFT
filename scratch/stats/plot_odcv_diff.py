# ABOUTME: The DA-minus-numina ODCV difference made visible: the headline interval (paired on
# ABOUTME: scenarios vs naively unpaired) beside the 25 per-scenario differences it is built from.
# Run: uv run python scratch/stats/plot_odcv_diff.py

"""Difference plot for the numina-control vs 5%-difficult-advice comparison.

Left: the gap in misalignment rate with its 95% interval from `both_random_diff` (scenario
axis paired, model terms added, residual terms subtracted), and next to it the interval you
would get by treating the two arms' scenario terms as independent (T_B^A + T_B^B instead of
var(d_j)/J). Right: the per-scenario differences d_j = DA column mean - numina column mean,
sorted, with the mean and the paired interval as a band. Same data as the main comparison:
incentivized, first rollout per cell, 25 shared scenarios x 3 seeds per arm.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.crossed_ci import Interval, both_random_diff, crossed_terms  # noqa: E402
from scratch.stats.plot_odcv_ci_comparison import ARMS, GRID, INK, INK2, load_tables  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

ORANGE, BLUE, NEUTRAL = "#eb6834", "#2a78d6", "#8a8984"


def unpaired_diff(table_a, table_b) -> Interval:
    """Same recipe with the scenario terms treated as independent (no pairing)."""
    mu_a, ta_a, tb_a, tc_a, _, _ = crossed_terms(table_a)
    mu_b, ta_b, tb_b, tc_b, _, _ = crossed_terms(table_b)
    se2 = ta_a + ta_b + tb_a + tb_b - tc_a - tc_b
    se = float(np.sqrt(se2))
    d = mu_a - mu_b
    return Interval(d, se, float("inf"), 1.96, d - 1.96 * se, d + 1.96 * se,
                    "diff, unpaired: T_A^A + T_A^B + T_B^A + T_B^B - T_C^A - T_C^B, +/-1.96")


def main(out: str = "output/odcv_ci_comparison") -> None:
    tables, shared, meta = load_tables()
    arms = list(ARMS)
    A, B = tables[arms[1]], tables[arms[0]]      # difficult advice, numina
    n, J = A.shape
    paired = both_random_diff(A, B)
    unpaired = unpaired_diff(A, B)
    d = A.mean(axis=0) - B.mean(axis=0)          # per-scenario differences of column means
    order = np.argsort(d, kind="stable")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor="#fcfcfb",
                                   gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.25})
    # --- left: the two intervals ---
    for x, (iv, label, color) in enumerate(((paired, "paired on scenarios\n(the derived formula)", INK),
                                            (unpaired, "unpaired\n(scenario terms added)", NEUTRAL))):
        ax0.errorbar(x, 100 * iv.mean, yerr=[[100 * (iv.mean - iv.lo)], [100 * (iv.hi - iv.mean)]], fmt="o",
                     color=color, ecolor=color, elinewidth=2, capsize=9, capthick=2, markersize=8, zorder=3)
        ax0.text(x + 0.12, 100 * iv.mean, f"{100 * iv.mean:+.1f} pp\n[{100 * iv.lo:+.1f}, {100 * iv.hi:+.1f}]",
                 ha="left", va="center", fontsize=9, color=color)
    ax0.axhline(0, color=INK, lw=1, zorder=2)
    ax0.text(-0.55, 1.2, "no difference", fontsize=8.5, color=INK2, va="bottom")
    ax0.set_xticks([0, 1], ["paired on scenarios\n(the derived formula)", "unpaired\n(scenario terms added)"], fontsize=8.5)
    ax0.set_xlim(-0.6, 1.9)
    lo = 100 * min(paired.lo, unpaired.lo)
    ax0.set_ylim(lo - 8, 12)
    ax0.set_ylabel("MR difference, 5% difficult advice − numina (pp)", fontsize=9.5, color=INK2)
    ax0.set_title("the gap, with 95% interval", fontsize=10, loc="left", color=INK)

    # --- right: per-scenario differences ---
    colors = [ORANGE if v < 0 else (BLUE if v > 0 else NEUTRAL) for v in d[order]]
    ax1.bar(range(J), 100 * d[order], color=colors, alpha=0.85, width=0.75, zorder=2)
    ax1.axhline(0, color=INK, lw=1, zorder=3)
    ax1.axhspan(100 * paired.lo, 100 * paired.hi, color=INK, alpha=0.08, zorder=1)
    ax1.axhline(100 * paired.mean, color=INK, lw=1.2, ls="--", zorder=3)
    ax1.text(J - 0.5, 100 * paired.mean - 3, f"mean {100 * paired.mean:+.1f} pp; band = 95% paired interval",
             ha="right", va="top", fontsize=8.5, color=INK)
    ax1.set_xticks(range(J), [shared[i] for i in order], rotation=70, ha="right", fontsize=7)
    ax1.set_ylim(-105, 105)
    ax1.set_yticks([-100, -67, -33, 0, 33, 67, 100])
    ax1.set_ylabel("per-scenario difference of column means (pp)", fontsize=9.5, color=INK2)
    ax1.set_title(f"the {J} per-scenario differences $d_j$ (DA mean over 3 seeds − numina mean over 3 seeds); "
                  f"orange = DA lower", fontsize=9.5, loc="left", color=INK)
    for ax in (ax0, ax1):
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    fig.suptitle("ODCV incentivized, first rollout per cell, 25 shared scenarios × 3 seeds per arm — "
                 "5% difficult advice minus numina control", fontsize=11, color=INK, y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.34, left=0.06, right=0.99)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / f"diff_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"odcv_diff_{stamp}.png"
    fig.savefig(png, dpi=160)
    p = paired.parts
    md = [f"# DA − numina difference ({stamp})", "",
          f"| method | Δ MR | 95% CI | SE |", "|---|---|---|---|",
          f"| {paired.method} | {100 * paired.mean:+.1f} pp | [{100 * paired.lo:+.1f}, {100 * paired.hi:+.1f}] | {100 * paired.se:.2f} pp |",
          f"| {unpaired.method} | {100 * unpaired.mean:+.1f} pp | [{100 * unpaired.lo:+.1f}, {100 * unpaired.hi:+.1f}] | {100 * unpaired.se:.2f} pp |", "",
          f"paired terms: T_A^A={p['T_A_a']:.5f}, T_A^B={p['T_A_b']:.5f}, T_B^(d)={p['T_B_d']:.5f}, "
          f"T_C^A={p['T_C_a']:.5f}, T_C^B={p['T_C_b']:.5f}; unpaired replaces T_B^(d) by "
          f"T_B^A+T_B^B={crossed_terms(A)[2] + crossed_terms(B)[2]:.5f}.", "",
          "| scenario | d_j (pp) |", "|---|---|", *(f"| {shared[i]} | {100 * d[i]:+.0f} |" for i in order), "",
          f"figure: `{png.name}` · git `{git_sha()}` · regenerate: `uv run python scratch/stats/plot_odcv_diff.py`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps({"paired": paired.as_dict(), "unpaired": unpaired.as_dict(),
                                                   "d_j": dict(zip(shared, d.tolist())), "meta": meta}, indent=2))
    write_run_meta(dest, {"script": "scratch/stats/plot_odcv_diff.py", "arms": ARMS})
    print("\n".join(md[:8]))
    print(f">>> saved {dest.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
