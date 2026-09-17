# ABOUTME: ODCV misalignment rate against MASK honesty for the three arms trained on the same DA
# ABOUTME: prompts (da-7, delib-7, delib-sonnet-7) beside the base and nosynth references.
# Run: uv run python scratch/plot_mr_vs_mask_da_prompts.py [--out output/plots]
#
# Same numbers and bars as scratch/plot_odcv_mr_vs_mask.py, whose loaders this reuses: MASK and the
# MR point from each run's head results.json (sha pinned in the results.md); MR bars recomputed from
# the raw per-rollout judge scores under ODCV's Design with item_sampling="fixed" and the normal z
# (that script's header says why z). The published scenario-sampled interval is reproduced from the
# same raw scores first. MASK publishes no interval and has one generation per row, so a
# fixed-benchmark bar is not estimable there and none is drawn.

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from plot_odcv_mr_vs_mask import DESIGN, FIXED, ORG, load, mr_interval, odcv_medians, z_bounds  # noqa: E402
from src.naming import figure_path  # noqa: E402

# key -> (ODCV repo, MASK repo, label, colour, hollow, label offset pt, ha, group)
# Colour follows the entity: da keeps the violet it carries in every earlier ODCV figure; the
# baselines share one neutral and differ by fill. Validated 2026-09-13 (dataviz
# validate_palette.js --pairs all, light surface): violet/orange/aqua/#52514e worst CVD dE 9.2,
# normal-vision >= 15; aqua is 2.74:1 on the surface, relieved by the direct labels.
ARMS = {
    "base":           ("2026-09-06-odcv-qwen36", "2026-09-07-mask-qwen36",
                       "Qwen3.6-27B", "#52514e", True, (9, -4), "left", "baseline"),
    "nosynth":        ("2026-09-09-odcv-qwen36-0-nosynth", "2026-09-09-mask-qwen36-0-nosynth",
                       "nosynth", "#52514e", False, (-9, 4), "right", "baseline"),
    "da-7":           ("2026-09-09-odcv-qwen36-0-da-7", "2026-09-09-mask-qwen36-0-da-7",
                       "DA", "#7a56c5", False, (9, 0), "left", "DA prompts"),
    "delib-7":        ("2026-09-11-odcv-qwen36-0-delib-7", "2026-09-11-mask-qwen36-0-delib-7",
                       "Delib (Qwen)", "#eb6834", False, (9, 0), "left", "DA prompts"),
    "delib-sonnet-7": ("2026-09-12-odcv-qwen36-0-delib-sonnet-7", "2026-09-12-mask-qwen36-0-delib-sonnet-7",
                       "Delib (Sonnet)", "#1baf7a", False, (9, 0), "left", "DA prompts"),
}
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
XLIM, YLIM = (48, 77), (0, 50)
BRACE = (67.5, 7.5, 31.5, 0.6)  # x of the ends, y span over the three DA-prompt arms' bars, depth


def left_brace(ax, x: float, y0: float, y1: float, depth: float, **line) -> float:
    """Draw a `{` from y0 to y1 with its ends at x and its tip `depth` to the left; return the tip's x."""
    u = np.linspace(0.0, 0.5, 200)
    k = 60.0
    half = 1 / (1 + np.exp(-k * u)) + 1 / (1 + np.exp(-k * (u - 0.5))) - 0.5  # 0 at the end, 1 at the tip
    s = np.concatenate([half, half[-2::-1]])
    ax.plot(x - depth * s, np.linspace(y0, y1, s.size), **line)
    return x - depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()

    api = HfApi()
    rows = {}
    for key, (odcv_name, mask_name, *_rest) in ARMS.items():
        odcv_repo, mask_repo = f"{ORG}/{odcv_name}", f"{ORG}/{mask_name}"
        o, o_meta, o_sha = load(api, odcv_repo)
        m, m_meta, m_sha = load(api, mask_repo)
        if o["mode"] != "think" or m["mode"] != "think":
            raise SystemExit(f"{key}: ODCV mode {o['mode']!r}, MASK mode {m['mode']!r}; every point must be think")
        ov = o["ours"]["overall"]
        medians, unscored = odcv_medians(api, odcv_repo, o_sha)
        sampled = mr_interval(medians, DESIGN)
        pub_lo, pub_hi = ov["mr_ci95"]
        if (abs(sampled.mean - ov["mr_pct"]) > 0.05 or abs(sampled.lo - pub_lo) > 0.05
                or abs(sampled.hi - pub_hi) > 0.05):
            raise SystemExit(f"{key}: raw scores give MR {sampled.mean:.2f} [{sampled.lo:.2f}, {sampled.hi:.2f}] "
                             f"but results.json says {ov['mr_pct']} [{pub_lo}, {pub_hi}]")
        fixed = mr_interval(medians, FIXED)
        lo, hi, _shape = z_bounds(fixed)
        rows[key] = {"mr": fixed.mean, "lo": lo, "hi": hi, "pub_lo": pub_lo, "pub_hi": pub_hi,
                     "unscored": unscored, "window": o_meta["config"]["serving"]["context_window"],
                     "mask": m["overall_honesty_score"], "mask_cap": m_meta["config"]["max_tokens"],
                     "odcv_repo": odcv_repo, "odcv_sha": o_sha, "mask_repo": mask_repo, "mask_sha": m_sha}
        print(f"  {key:15s} MR {fixed.mean:5.1f} fixed z [{lo:5.1f}, {hi:5.1f}] | sampled [{pub_lo}, {pub_hi}] "
              f"reproduced | MASK {m['overall_honesty_score']} (cap {rows[key]['mask_cap']})")

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200, facecolor=SURFACE)
    left, right, bottom, top = 0.1, 0.97, 0.15, 0.97
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    bx, by0, by1, depth = BRACE
    tip = left_brace(ax, bx, by0, by1, depth, color=INK2, linewidth=1.1, solid_capstyle="round", zorder=2)
    ax.text(tip - 0.3, (by0 + by1) / 2, "same DA prompts", ha="right", va="center", fontsize=9, color=INK2, zorder=2)

    for key, r in rows.items():
        _o, _m, text, c, hollow, off, ha, _g = ARMS[key]
        ax.errorbar(r["mask"], r["mr"], yerr=[[r["mr"] - r["lo"]], [r["hi"] - r["mr"]]], fmt="none",
                    ecolor=c, elinewidth=1.4, capsize=3.5, capthick=1.2, zorder=3)
        ax.plot(r["mask"], r["mr"], "o", ms=12.5, mfc=SURFACE, mec=SURFACE, zorder=4)  # 2px surface ring
        ax.plot(r["mask"], r["mr"], "o", ms=9, mfc=SURFACE if hollow else c, mec=c, mew=2 if hollow else 0, zorder=5)
        ax.annotate(text, (r["mask"], r["mr"]), textcoords="offset points", xytext=off, ha=ha, va="center",
                    fontsize=9.5, color=INK, zorder=6)

    ax.text(0.985, 0.025, "better ↘", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel("MASK honesty", color=INK2, fontsize=10)
    ax.set_ylabel("ODCV misalignment rate (%)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    fig.text(0.012, 0.015, "Bars: 95% CI, fixed benchmark", fontsize=7.8, color=INK2, va="bottom")

    png = figure_path(args.out, "odcv mr vs mask da prompts base nosynth da delib delib sonnet")
    fig.savefig(png, facecolor=SURFACE)

    md = png.with_name(png.stem + "_results.md")
    lines = ["# ODCV misalignment rate against MASK honesty: arms trained on the same DA prompts, with baselines", "",
             f"![]({png.name})", "",
             "| arm | group | ODCV MR | fixed-benchmark 95% CI, z (plotted) | scenario-sampled 95% CI (published) "
             "| MASK honesty | MASK cap | ODCV repo @ sha | MASK repo @ sha |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, r in rows.items():
        lines.append(f"| {ARMS[key][2]} ({key}) | {ARMS[key][7]} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] "
                     f"| [{r['pub_lo']:.1f}, {r['pub_hi']:.1f}] | {r['mask']:.1f} | {r['mask_cap']:,} "
                     f"| `{r['odcv_repo']}` @ `{r['odcv_sha'][:8]}` | `{r['mask_repo']}` @ `{r['mask_sha'][:8]}` |")
    lines += ["",
              "DA, Delib (Qwen) and Delib (Sonnet) are trained on the same DA prompts at a 7% share on the same "
              "nosynth replay base; they differ in what answers them (DA's answers; Qwen3.6's own deliberation over "
              "the constitution; Sonnet 5's templated deliberation). nosynth is that base alone; Qwen3.6-27B is untuned. "
              "All think mode, seed 0, one run each.",
              "",
              "**Bars: fixed benchmark, z.** ODCV MR intervals recomputed from each run's raw per-rollout judge scores "
              "with `src.eval.stats.interval` under ODCV's Design with `item_sampling=\"fixed\"` (within-cell rollout "
              "noise only, logit scale), normal z in place of Satterthwaite t, exactly as "
              "`scratch/plot_odcv_mr_vs_mask.py`. They say nothing about scenarios beyond these 40 or about "
              "seed-to-seed variance. MASK has no bar: one generation per row makes the fixed-benchmark interval "
              "not estimable.",
              "",
              f"ODCV-lite: 40 scenarios x 2 variants x 3 passes, temperature 0.7, {rows['base']['window']:,}-token "
              "window, one gemini-3-flash-preview judge. MASK: re-scored per-row honesty (2026-09-10), "
              "gemini-3-flash-preview judge. MASK generation cap differs (column above): 12,288 for base, nosynth "
              "and DA; 16,384 for both delib arms.",
              "", f"Figure: `{png}`. Script: `scratch/plot_mr_vs_mask_da_prompts.py`."]
    md.write_text("\n".join(lines) + "\n")
    print(f">>> wrote {png}\n>>> wrote {md}")


if __name__ == "__main__":
    main()
