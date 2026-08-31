# ABOUTME: Plots incentivized ODCV MR for the three 7%-synthetic arms with +/-1.96 SEM bars
# ABOUTME: over training seeds 0/42/69, on the cell set all nine runs share.
"""Incentivized MR by arm, error bars = +/-1.96 SEM over k=3 training seeds.

    uv run python scratch/stats/plot_incentivized_mr.py

All three arms are scored on the SAME cells -- the intersection over all nine runs
(3 arms x 3 seeds). Comparing arms on their own cell sets would confound the arm with which
scenarios happened to survive that arm's runs, and cell loss is not random with respect to
MR (the arms that game more overrun context and the proxy timeout more often).

The +/-1.96 multiplier is what was asked for and is drawn; note in the caption that at df=2
it is an ~81% interval, not 95% (t=4.303 there). Both are written to the markdown mirror.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    VIOLATION_THRESHOLD, median_score, misalignment_rate,
)
from src.utils import git_sha  # noqa: E402

KEY = re.compile(r"^(?P<variant>[^/]+)/(?P<scenario>[^/]+?)(?:/rollout_(?P<idx>\d+))?$")
ARMS = {
    "numina control": {
        0: ("qwen3_6-27b-lora-9284-numina-control-716-r64", "001"),
        42: ("qwen3_6-27b-lora-9284-numina-control-716-r64-seed42", None),
        69: ("qwen3_6-27b-lora-9284-numina-control-716-r64-seed69", None)},
    "synthdoc": {
        0: ("qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64", "000"),
        42: ("qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64-seed42", None),
        69: ("qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64-seed69", None)},
    "grok responder": {
        0: ("qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64", "000"),
        42: ("qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed42", None),
        69: ("qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64-seed69", None)},
    "verbose cot": {
        0: ("odcv/qwen3_6-27b-lora-t2-9284-da716-verbose-r64-dynbatch", None),
        42: ("odcv/qwen3_6-27b-lora-t2-9284-verbosecot716-r64-seed42", None),
        69: ("odcv/qwen3_6-27b-lora-t2-9284-verbosecot716-r64-seed69", None)},
}
COLOR = {"numina control": "#B4483C", "synthdoc": "#1A6B72", "grok responder": "#6E5AA8",
         "verbose cot": "#B07A1E"}
# What each arm is in the reader's terms: the share of difficult advice in the 10k
# mixture and who wrote it. Internal keys stay as-is so paths and JSON do not move.
LABEL = {"numina control": "0% diff advice",
         "synthdoc": "7% sonnet diff advice",
         "grok responder": "7% grok diff advice",
         "verbose cot": "7% verbose diff advice"}


def _scores(arm_dir: str, want: str | None) -> dict[str, list[float]]:
    """{'incentivized/scenario': [judge scores]} for one run, one pass."""
    roots = ([str(REPO / "output" / arm_dir)] if arm_dir.startswith("odcv/")
             else [str(REPO / "output/odcv_bench" / arm_dir)])
    cands = [p for root in roots for sub in ("*/evaluations", "*/results")
             for p in glob.glob(f"{root}/{sub}") if glob.glob(p + "/scores_*.json")]
    assert cands, f"no judged scores under {arm_dir}"
    d = max(cands, key=os.path.getmtime)
    out: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(d + "/scores_*.json")):
        for k, v in json.load(open(f)).items():
            m = KEY.match(k)
            if not m or m["variant"] != "incentivized":
                continue
            if want is not None and m["idx"] is not None and m["idx"] != want:
                continue
            if want is None and m["idx"] is not None:
                continue
            sc = v.get("score") if isinstance(v, dict) else v
            try:
                out[f'incentivized/{m["scenario"]}'].append(float(sc))
            except (TypeError, ValueError):
                pass
    return dict(out)


def main(out: str = "output/odcv_seed_variance",
         synthdoc_pass: str = "000", numina_pass: str = "001",
         grok_pass: str = "000", shared: bool = True) -> None:
    """Write the incentivized-MR figure and its markdown mirror.

    Args:
        out: Output root.
        synthdoc_pass: Which seed-0 rollout pass to use for the synthdoc arm. Seed 0 has
            several; seeds 42/69 have one each. WHICH ONE IS CHOSEN MOVES s_seed a lot
            (pass 000 and 004 give three identical values and a degenerate SEM; 001/002/003
            do not), so the choice is recorded on the figure and in the markdown rather than
            left implicit.
        numina_pass: Same, for the numina arm.
        grok_pass: Same, for the grok arm.
        shared: True scores every arm on the cells common to all nine runs. False lets
            each run use every cell it produced -- a larger n per arm, but the arms are
            then scored on DIFFERENT scenario sets. Cell loss is not random with respect
            to MR (an arm that games more overruns the 16,384-token context and the 120s
            proxy timeout more often, losing its longest rollouts), so the unshared view
            understates the high-MR arms. Both are worth looking at; neither alone.
    """
    passes = {"numina control": numina_pass, "synthdoc": synthdoc_pass,
              "grok responder": grok_pass,
              # run_eval writes single passes but still keys them "/rollout_000", so the
              # verbose arm needs an explicit pass, not the bare-key path.
              "verbose cot": "000"}
    # fire coerces "000" to int 0; normalise back to the zero-padded string form.
    passes = {k: (str(v).zfill(3) if str(v).strip() else None)
              for k, v in passes.items()}
    data = {a: {s: _scores(ARMS[a][s][0],
                           passes[a] if (s == 0 or a == "verbose cot") else None)
                for s in ARMS[a]} for a in ARMS}
    common = sorted(set.intersection(*(set(data[a][s]) for a in data for s in data[a])))
    assert common, "no cells common to all nine runs"

    res = {}
    for a in ARMS:
        cells = {s: (common if shared else sorted(data[a][s])) for s in ARMS[a]}
        mrs = np.array([misalignment_rate([median_score(data[a][s][c]) for c in cells[s]])
                        for s in ARMS[a]], float)
        ns = [len(cells[s]) for s in ARMS[a]]
        sd = mrs.std(ddof=1)
        res[a] = {"mrs": mrs, "ns": ns, "mean": mrs.mean(), "s_seed": sd,
                  "sem": sd / np.sqrt(len(mrs)), "hw": 1.96 * sd / np.sqrt(len(mrs)),
                  "hw_t": stats.t.ppf(0.975, len(mrs) - 1) * sd / np.sqrt(len(mrs))}

    names = list(ARMS)
    fig, ax = plt.subplots(figsize=(7.2, 4.7), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(len(names))
    for i, a in enumerate(names):
        r = res[a]
        ax.bar(i, r["mean"], width=.56, color=COLOR[a], alpha=.88, zorder=2,
               edgecolor=COLOR[a], linewidth=1.2)
        ax.errorbar(i, r["mean"], yerr=r["hw"], fmt="none", ecolor="#2B2B2B",
                    elinewidth=1.5, capsize=8, capthick=1.5, zorder=4)
        ax.text(i, r["mean"] + r["hw"] + 1.6, f'{r["mean"]:.1f}%',
                ha="center", va="bottom", fontsize=11.5, fontweight="600", color="#1A1A1A")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in names], fontsize=11)
    ax.set_ylabel("Incentivized misalignment rate (%)", fontsize=10.5)
    ax.set_title("ODCV incentivized MR by arm, k=3 training seeds",
                 fontsize=12.5, fontweight="600", pad=26, loc="left")
    ax.text(0, 1.02, "error bars ±1.96 SEM over training seeds 0/42/69",
            transform=ax.transAxes, fontsize=8.8, color="#5A5A5A", va="bottom")
    # synthdoc's three seeds land on the same integer count, so its SEM is 0 and its
    # bar has no width. Say so on the figure rather than letting it read as precision.
    for i, a in enumerate(names):
        if res[a]["s_seed"] == 0:
            ax.annotate("all 3 seeds identical\n(SEM undefined, not precise)",
                        xy=(i, res[a]["mean"]), xytext=(i + .34, res[a]["mean"] + 13),
                        fontsize=7.8, color="#5A5A5A", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=.8))
    top = max(r["mean"] + r["hw"] for r in res.values())
    ax.set_ylim(0, top * 1.24)
    # the negative headroom exists only for the two label rows under the bars;
    # ticks must not advertise it as data
    ax.set_yticks(np.arange(0, int(top * 1.24) + 1, 10))
    ax.set_xlim(-.62, len(names) - .38)
    ax.grid(axis="y", color="#E3E3E3", linewidth=.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#BFBFBF")
    ax.tick_params(colors="#4A4A4A", labelsize=9.5)
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / ts
    (dest / "plots").mkdir(parents=True, exist_ok=True)
    png = dest / "plots" / f"incentivized_mr_by_arm_{ts}.png"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    md = [f"# Incentivized ODCV MR by arm ({ts})", "",
          (f"k=3 training seeds (0, 42, 69) · **{len(common)} incentivized cells** shared by "
           "all nine runs · one rollout pass per seed."
           if shared else
           "k=3 training seeds (0, 42, 69) · **each run scored on every incentivized cell it "
           "produced**, so n differs by arm (shown per bar) · one rollout pass per seed."), "",
          "Seed 0 has several passes; seeds 42 and 69 have one each. The pass used per arm: "
          + ", ".join(f"**{a}** rollout_{passes[a]}" for a in names) + ".", "",
          f"![incentivized MR](plots/{png.name})", "",
          "| arm | seed 0 | seed 42 | seed 69 | mean MR | s_seed | ±1.96 SEM | ±t (true 95%) |",
          "|---|---|---|---|---|---|---|---|"]
    for a in names:
        r = res[a]
        md.append(f"| {LABEL[a]} | "
                  + " | ".join(f"{v:.1f}%" for v in r["mrs"])
                  + f" | **{r['mean']:.2f}%** | {r['s_seed']:.2f}pp "
                    f"| ±{r['hw']:.1f}pp | ±{r['hw_t']:.1f}pp |")
    md += ["", "## Reading this", "",
           "- **±1.96 SEM is an ~81% interval at k=3**, not 95%: with s_seed estimated on "
           "df=2 the correct 95% multiplier is t=4.303. Both are tabulated.",
           "- All arms are scored on the SAME cells. Cell loss is not random with respect to "
           "MR — arms that game more overrun the 16,384-token context and the 120s proxy "
           "timeout more often — so per-arm cell sets would bias the comparison.",
           "- Hollow points are the three individual seeds; with k=3 the bar is only as "
           "trustworthy as those three points look.",
           "- **Which seed-0 pass is used moves s_seed materially.** For synthdoc, passes "
           "000 and 004 give three identical MRs and a degenerate (zero) SEM, while 001, "
           "002 and 003 do not. Choosing the pass that yields a usable SEM is selecting on "
           "the outcome; the principled fix is several passes per seed for every seed, "
           "averaged per cell before the SE is taken.", "",
           f"git `{git_sha()}` · regenerate: "
           "`uv run python scratch/stats/plot_incentivized_mr.py`", ""]
    (dest / "results.md").write_text("\n".join(md))
    def _plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v if isinstance(v, list) else float(v)
    json.dump({"shared_cells_mode": bool(shared), "n_common_cells": len(common),
               "seed0_pass": passes,
               "arms": {a: {k: _plain(v) for k, v in r.items()} for a, r in res.items()}},
              open(dest / "results.json", "w"), indent=2)
    print(f">>> {png}")
    for a in names:
        r = res[a]
        print(f"  {a:16} mean={r['mean']:6.2f}%  s_seed={r['s_seed']:5.2f}pp  "
              f"±1.96SEM=±{r['hw']:.1f}pp  n={r['ns']}  seeds={[round(v,1) for v in r['mrs']]}")
    print(f"  cells shared by all nine runs: {len(common)}  |  mode: {'shared' if shared else 'all cells per run'}")


if __name__ == "__main__":
    fire.Fire(main)
