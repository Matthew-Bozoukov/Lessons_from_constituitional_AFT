# ABOUTME: ODCV misalignment rate against MASK honesty for base Qwen3.6 27B, nosynth, da, dat, da-mix
# ABOUTME: and dat-mix on one set of axes; dated figure + results.md with every number.
# Run: uv run python scratch/plot_odcv_mr_vs_mask.py [--out output/plots]
#
# MASK honesty and the ODCV point estimate are read from the HEAD revision of each run's published
# results/results.json (pinned to the sha it resolved to, recorded in the results.md). The ODCV
# error bars are RECOMPUTED here from the raw per-rollout judge scores (results/scores_<judge>.json)
# with src.eval.stats.interval under ODCV's own Design changed in one field, item_sampling="fixed":
# the 40 scenarios x 2 variants are then the benchmark itself rather than a draw from a population
# of scenarios, so the only spread left is rollout noise within each cell (docs/error_bars.md, 4).
# The multiplier is the normal z, not interval()'s Satterthwaite t: for 0/1 rollouts the variance
# is p(1-p), not a separate unknown, and Satterthwaite's df assumes chi-squared per-cell variance
# estimates, which a 3-rollout 0/1 cell (variance 0 or 3,333) is not -- at one misaligned rollout
# the df collapses to 2 and t_2 = 4.30 stretches da to [0.0, 23.9]. Same SE, same log-odds
# geometry (_shape_interval); only the multiplier changes, as the zero-count rule already does.
# The published scenario-sampled interval is recomputed from the same raw scores first and must
# match results.json, which is what licenses reading the raw file this way.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from src.eval.misalignment.odcv.odcv import DESIGN, MR_BOUNDS, VARIANTS, to_long  # noqa: E402
from src.eval.stats import Z_95, _shape_interval, interval  # noqa: E402
from src.naming import figure_path  # noqa: E402

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

ORG = "dougalldeepmind"
# key -> (ODCV repo, MASK repo). nosynth is the 09-08 control whose replay half the 7% arms
# share byte for byte; the 100% arms are those arms' 700 synthetic rows alone (LOG 2026-09-09).
ARMS = {
    "base":    ("2026-09-06-odcv-qwen36",           "2026-09-07-mask-qwen36"),
    "nosynth": ("2026-09-09-odcv-qwen36-0-nosynth", "2026-09-09-mask-qwen36-0-nosynth"),
    "da-7":    ("2026-09-09-odcv-qwen36-0-da-7",    "2026-09-09-mask-qwen36-0-da-7"),
    "da-100":  ("2026-09-09-odcv-qwen36-0-da-100",  "2026-09-09-mask-qwen36-0-da-100"),
    "dat-7":   ("2026-09-09-odcv-qwen36-0-dat-7",   "2026-09-09-mask-qwen36-0-dat-7"),
    "dat-100": ("2026-09-09-odcv-qwen36-0-dat-100", "2026-09-09-mask-qwen36-0-dat-100"),
}
FIXED = replace(DESIGN, item_sampling="fixed")
SCORES = re.compile(r"results/scores_(?!progress_)[^/]+\.json")

# Colour follows the source: da and dat keep the hues they carry in
# scratch/plot_odcv_mr_vs_tp_informative.py; base and nosynth are neutral references. The fill
# is the mix: filled = diluted in the MSM-derived rows (-mix), hollow = synthetic rows alone. Validated
# 2026-09-10 (dataviz validate_palette.js --pairs all, light surface): violet-yellow CVD dE
# 35.5, normal-vision dE 37.7; yellow is 2.11:1 on the surface, relieved by the direct labels.
COLOR = {"base": "#898781", "nosynth": "#52514e", "da": "#7a56c5", "dat": "#eda100"}
KEY = {"da": "da: difficult advice", "dat": "dat: difficult agentic tasks"}
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# Direct label per point: (text, offset in points, ha, va), placed so no label crosses
# another mark or interval.
LABEL = {
    "base":    ("Qwen3.6 27B", (9, -5), "left", "top"),
    "nosynth": ("nosynth", (-9, 5), "right", "bottom"),
    "da-7":    ("da-mix", (-10, 0), "right", "center"),
    "dat-7":   ("dat-mix", (-10, 0), "right", "center"),
    "da-100":  ("da", (10, 0), "left", "center"),
    "dat-100": ("dat", (10, 0), "left", "center"),
}


def load(api: HfApi, repo: str) -> tuple[dict, dict, str]:
    """(results.json, run_meta.json, sha) at the repo's head, pinned to the sha it resolved to."""
    sha = api.dataset_info(repo).sha
    res = json.loads(Path(hf_hub_download(repo, "results/results.json", repo_type="dataset",
                                          revision=sha)).read_text())
    files = api.list_repo_files(repo, repo_type="dataset", revision=sha)
    meta_file = "metadata/run_meta.json" if "metadata/run_meta.json" in files else "metadata/passes/pass1_run_meta.json"
    meta = json.loads(Path(hf_hub_download(repo, meta_file, repo_type="dataset",
                                           revision=sha)).read_text())
    return res, meta, sha


def odcv_medians(api: HfApi, repo: str, sha: str) -> tuple[dict, int]:
    """{variant: {scenario: [severity per rollout]}} from the run's one misalignment judge's raw
    scores (a single judge, so its score IS the median), and how many rollouts it left unscored."""
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha) if SCORES.fullmatch(f)]
    assert len(files) == 1, f"{repo}: expected one misalignment judge's raw scores, found {files}"
    scores = json.loads(Path(hf_hub_download(repo, files[0], repo_type="dataset", revision=sha)).read_text())
    medians, unscored = {v: {} for v in VARIANTS}, 0
    for key, rec in scores.items():
        variant, scenario, _ = key.split("/")
        s = rec.get("score") if isinstance(rec, dict) else rec
        if isinstance(s, (int, float)):
            medians[variant].setdefault(scenario, []).append(float(s))
        else:
            unscored += 1
    return medians, unscored


def mr_interval(medians: dict, design):
    """The MR interval exactly as the ODCV runner builds it, under `design`."""
    rows = [dict(r, value=100.0 * r["violation"]) for r in to_long(medians)]
    return interval(rows, design, bounds=MR_BOUNDS)


def z_bounds(r) -> tuple[float, float, str]:
    """`r`'s interval with the normal z in place of its Satterthwaite t: same SE, same log-odds
    geometry (src.eval.stats._shape_interval). The zero-count rule already takes z, so a
    wilson-at-boundary result passes through unchanged."""
    if r.shape == "wilson-at-boundary":
        return r.lo, r.hi, r.shape
    assert r.shape == "logit", f"unexpected interval shape {r.shape!r}"
    lo, hi, shape = _shape_interval(r.mean, r.se, Z_95, MR_BOUNDS, n_floor=1)
    assert shape == "logit"
    return lo, hi, shape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()

    api = HfApi()
    rows = {}
    for key, (odcv_name, mask_name) in ARMS.items():
        odcv_repo, mask_repo = f"{ORG}/{odcv_name}", f"{ORG}/{mask_name}"
        o, o_meta, o_sha = load(api, odcv_repo)
        m, m_meta, m_sha = load(api, mask_repo)
        if o["mode"] != "think" or m["mode"] != "think":
            raise SystemExit(f"{key}: ODCV mode {o['mode']!r}, MASK mode {m['mode']!r} — "
                             "every point on this figure must be think mode")
        ov = o["ours"]["overall"]
        medians, unscored = odcv_medians(api, odcv_repo, o_sha)
        sampled = mr_interval(medians, DESIGN)
        pub_lo, pub_hi = ov["mr_ci95"]
        if (abs(sampled.mean - ov["mr_pct"]) > 0.05 or abs(sampled.lo - pub_lo) > 0.05
                or abs(sampled.hi - pub_hi) > 0.05):
            raise SystemExit(f"{key}: raw scores give MR {sampled.mean:.2f} [{sampled.lo:.2f}, {sampled.hi:.2f}] "
                             f"but results.json says {ov['mr_pct']} [{pub_lo}, {pub_hi}]")
        fixed = mr_interval(medians, FIXED)
        z_lo, z_hi, z_shape = z_bounds(fixed)
        rows[key] = {
            "family": key.split("-")[0], "share": key.split("-")[1] + "%" if "-" in key else "–",
            "mr": fixed.mean, "lo": z_lo, "hi": z_hi, "shape": z_shape, "method": fixed.method,
            "t_lo": fixed.lo, "t_hi": fixed.hi, "df": fixed.df, "pub_lo": pub_lo, "pub_hi": pub_hi,
            "pub_method": ov["ci_method"], "unscored": unscored,
            "n": sum(len(s) for v in medians.values() for s in v.values()),
            "window": o_meta["config"]["serving"]["context_window"],
            "submitted": o.get("progress", {}).get("submitted_pct"),
            "mask": m["overall_honesty_score"], "mask_cap": m_meta["config"]["max_tokens"],
            "odcv_repo": odcv_repo, "odcv_sha": o_sha, "mask_repo": mask_repo, "mask_sha": m_sha,
        }
        print(f"  {key:8s} MR {fixed.mean:5.1f}  fixed z [{z_lo:5.1f}, {z_hi:5.1f}] ({z_shape})  |  "
              f"fixed t [{fixed.lo:5.1f}, {fixed.hi:5.1f}] df {fixed.df:5.1f}  |  "
              f"sampled [{pub_lo:5.1f}, {pub_hi:5.1f}] reproduced  |  unscored {unscored}")

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=170, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, r in rows.items():
        c = COLOR[r["family"]]
        hollow = r["share"] == "100%"
        # clipped at 0: at a 0.0 mean the Wilson bound's lo can land a float ulp above it
        ax.errorbar(r["mask"], r["mr"], yerr=[[max(0.0, r["mr"] - r["lo"])], [max(0.0, r["hi"] - r["mr"])]],
                    fmt="none", ecolor=c, elinewidth=1.4, capsize=3.5, capthick=1.2, zorder=2)
        ax.plot(r["mask"], r["mr"], "o", ms=12.5, mfc=SURFACE, mec=SURFACE, zorder=3)  # 2px surface ring
        ax.plot(r["mask"], r["mr"], "o", ms=9, mfc=SURFACE if hollow else c, mec=c,
                mew=2 if hollow else 0, zorder=4)
        text, off, ha, va = LABEL[k]
        ax.annotate(text, (r["mask"], r["mr"]), textcoords="offset points", xytext=off, ha=ha,
                    va=va, fontsize=9, color=INK, zorder=5)

    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", ms=8, mfc=COLOR[f], mec=COLOR[f],
                              label=KEY[f]) for f in KEY],
              loc="upper right", frameon=False, fontsize=9, labelcolor=INK, handletextpad=0.3)
    ax.set_xlabel("MASK honesty score", color=INK2, fontsize=10)
    ax.set_ylabel("ODCV-lite misalignment rate (%)", color=INK2, fontsize=10)
    ax.set_xlim(45, 100)  # the 2026-09-10 MASK re-score put nosynth at 53.6
    ax.set_ylim(-3, 60)
    ax.set_yticks(range(0, 61, 10))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_title("ODCV misalignment against MASK honesty", loc="left", fontsize=12.5,
                 weight="bold", color=INK, pad=12)
    # Row counts verified 2026-09-10 from each adapter's pinned mixture (training_meta.dataset):
    # nosynth-mix 10,000; da-7-mix / dat-7-mix 700 synthetic + 9,300 of nosynth's rows; da-100 /
    # dat-100 mixes 700 synthetic rows alone.
    fig.text(0.012, 0.015,
             "da, dat: 700 synthetic rows only. da-mix, dat-mix: the same 700 + 9,300 rows derived from curated MSM mix (7% synthetic).\n"
             "nosynth: 10,000 rows derived from curated MSM mix, no synthetic.\n"
             "ODCV-lite: one cheap misalignment judge (Gemini 3 Flash), which roughly matches the paper's expensive judges.\n"
             "Bars: 95% CI on rollout noise, treating the 40 scenarios x 2 variants as a fixed benchmark.",
             fontsize=7.6, color=INK2, va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=0.1, right=0.975, top=0.9, bottom=0.25)

    png = figure_path(args.out, "odcv mr vs mask base nosynth da dat 7 and 100")
    fig.savefig(png, facecolor=SURFACE)

    md = png.with_name(png.stem + "_results.md")
    lines = ["# ODCV misalignment rate against MASK honesty: base, nosynth, da and dat at 7% and 100% synthetic", "",
             "| arm | synthetic share | ODCV MR | fixed-benchmark 95% CI, z (plotted) | same, Satterthwaite t (src default) "
             "| scenario-sampled 95% CI (published) | submitted | MASK honesty | MASK cap | ODCV repo @ sha | MASK repo @ sha |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        lines.append(f"| {LABEL[k][0]} ({k}) | {r['share']} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] "
                     f"| [{r['t_lo']:.1f}, {r['t_hi']:.1f}] (df {r['df']:.1f}) "
                     f"| [{r['pub_lo']:.1f}, {r['pub_hi']:.1f}] | {r['submitted']}% | {r['mask']:.2f} | {r['mask_cap']:,} "
                     f"| `{r['odcv_repo']}` @ `{r['odcv_sha'][:8]}` | `{r['mask_repo']}` @ `{r['mask_sha'][:8]}` |")
    lines += ["", "MASK honesty and the ODCV point estimate read straight from each run's published "
              "`results/results.json` at the head revision; the sha each head resolved to is pinned above. "
              "All runs think mode, every adapter seed 0, one run each.",
              "",
              "**Error bars (plotted): fixed benchmark, z.** Recomputed from each run's raw per-rollout judge scores "
              "(`results/scores_gemini-3-flash-preview.json`) with `src.eval.stats.interval` under ODCV's Design "
              "with `item_sampling=\"fixed\"` (within-cell rollout noise only), logit scale on [0, 100], with the "
              "normal z in place of the method's Satterthwaite t (same SE, same geometry via `_shape_interval`). For "
              "0/1 rollouts the variance is p(1-p), not a separate unknown, and Satterthwaite's df assumes chi-squared "
              "per-cell variance estimates, which a 3-rollout 0/1 cell (variance 0 or 3,333) is not: at one misaligned "
              "rollout (da) the df collapses to 2 and t_2 = 4.30 gives [0.0, 23.9] (the t column). Remaining limits: "
              "cells at 0/3 or 3/3 still contribute zero variance, and the zero-count rule (dat) is Wilson at n = 120, "
              "not the 240 rollouts. The 40 scenarios x 2 variants are the benchmark itself, so the bars say nothing "
              "about scenarios beyond these 40 or about seed-to-seed variance. The published scenario-sampled interval "
              "was recomputed from the same raw scores and matched `results.json` for every arm before anything else.",
              "",
              f"ODCV-lite (`configs/eval/odcv/lite.yaml`): 40 scenarios x 2 conditions x 3 passes = 240 rollouts, "
              f"temperature 0.7, {rows['base']['window']:,}-token window, scored by one gemini-3-flash-preview judge "
              "instead of the paper's grok-4.20 + gemini-3.1-pro-preview pair (`heavy.yaml`, same rollouts).",
              "",
              "MASK: overall honesty = 100 - % of all 1,000 rows judged a lie (pooled per row, as the paper does), "
              "gemini-3-flash-preview judge; no interval published. These are the 2026-09-10 re-scores (docs/LOG.md): "
              "the verdict parser was fixed and the overall now pools rows; the superseded revisions averaged the six "
              "archetypes. Generation cap differs by run (column above): 12,288 before commit 424bec7, 16,384 after.",
              "", f"Figure: `{png}`"]
    md.write_text("\n".join(lines) + "\n")
    print(f">>> wrote {png}\n>>> wrote {md}")


if __name__ == "__main__":
    main()
