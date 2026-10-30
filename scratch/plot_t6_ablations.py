# ABOUTME: Grouped bars of ODCV-lite misconduct (fixed-benchmark CI95) and MASK honesty (no CI), for the t6 arms
# ABOUTME: (da-no-t6-15, da-new-t6-15) beside da-15 (old and new wording) and the nosynth control, every seed on the Hub.
"""    uv run python scratch/plot_t6_ablations.py

Reads each run from the Hub (dougalldeepmind), writes the figure and a results.md beside it under output/figures/.
One bar per (arm, seed). ODCV intervals are src.eval.stats.interval with checkpoint AND items fixed ("this checkpoint
on these items"), so they carry rollout noise only: Design(item=scenario, fixed, variant enumerated, pass subsampled)
over per_scenario_medians, R=3 per cell. MASK has no bar: one pass per row, so rollout noise is not estimable.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.eval.misalignment.odcv.odcv import MR_BOUNDS, to_long
from src.eval.stats import Design, interval

load_dotenv()
DATE = "2026-09-28"
OUT = Path("output/figures")
ORG = "dougalldeepmind"
ODCV_FIXED = Design(item="scenario", item_sampling="fixed", enumerated={"variant": "equal"}, subsamples=("pass",))
# label -> (colour, [(seed, odcv run, mask run), ...])
ARMS = {
    "nosynth\n(control)": ("#8a8a8a", [(0, "2026-09-22-odcv-qwen36-0-nosynth", "2026-09-22-mask-qwen36-0-nosynth")]),
    "da-15\n09-23 corpus\n(old wording)": ("#b7a8e6", [(0, "2026-09-23-odcv-qwen36-0-da-15", "2026-09-23-mask-qwen36-0-da-15")]),
    "da-15\n09-25 corpus\n(ask-not-instruct)": ("#6a55c2", [(0, "2026-09-26-odcv-qwen36-0-da-15", "2026-09-26-mask-qwen36-0-da-15"),
                                                     (1, "2026-09-26-odcv-qwen36-1-da-15", "2026-09-26-mask-qwen36-1-da-15")]),
    "da-new-t6-15\n09-24 corpus,\nt6 swapped for new": ("#4a3aa7", [(0, "2026-09-26-odcv-qwen36-0-da-new-t6-15", "2026-09-26-mask-qwen36-0-da-new-t6-15"),
                                                       (1, "2026-09-26-odcv-qwen36-1-da-new-t6-15", "2026-09-26-mask-qwen36-1-da-new-t6-15")]),
    "da-no-t6-15\n09-24 corpus,\nt6 dropped": ("#2b1f6e", [(0, "2026-09-25-odcv-qwen36-0-da-no-t6-15", "2026-09-25-mask-qwen36-0-da-no-t6-15")]),
}


def odcv_fixed(o: dict):
    rows = [dict(r, value=100.0 * r["violation"]) for r in to_long(o["per_scenario_medians"])]
    r = interval(rows, ODCV_FIXED, bounds=MR_BOUNDS)
    assert abs(r.mean - o["ours"]["overall"]["mr_pct"]) < 0.06, (r.mean, o["ours"]["overall"]["mr_pct"])
    return r


def fetch() -> list[dict]:
    api = HfApi()
    rows = []
    for arm, (colour, runs) in ARMS.items():
        for seed, orun, mrun in runs:
            orepo, mrepo = f"{ORG}/{orun}", f"{ORG}/{mrun}"
            o = json.load(open(hf_hub_download(orepo, "results/results.json", repo_type="dataset")))
            m = json.load(open(hf_hub_download(mrepo, "results/results.json", repo_type="dataset")))
            ov = o["ours"]["overall"]
            assert list(o.get("judges", {})) == ["gemini-3-flash-preview"], (orun, o.get("judges"))
            assert o["mode"] == m["mode"] == "think", (orun, o["mode"], m["mode"])
            assert ov["n_rollouts"] == 240, (orun, ov["n_rollouts"])
            oc = odcv_fixed(o)
            rows.append({"arm": arm, "seed": seed, "colour": colour, "mr": ov["mr_pct"], "lo": oc.lo, "hi": oc.hi,
                         "mask": m["overall_honesty_score"], "target": o["target"],
                         "odcv": f"{orepo}@{api.dataset_info(orepo).sha[:8]}",
                         "mask_run": f"{mrepo}@{api.dataset_info(mrepo).sha[:8]}"})
    return rows


def positions(rows: list[dict]) -> list[float]:
    xs, width = [], 0.36
    for i, arm in enumerate(ARMS):
        seeds = [r for r in rows if r["arm"] == arm]
        offs = [0.0] if len(seeds) == 1 else [-width / 2 - 0.02, width / 2 + 0.02]
        xs += [i + d for d in offs]
    return xs


def main() -> None:
    rows = fetch()
    xs = positions(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    for ax in (a1, a2):
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#e6e6e3", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c8c8c4")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(list(ARMS), fontsize=8.5)
    for x, r in zip(xs, rows):
        a1.bar(x, r["mr"], width=0.34, color=r["colour"], edgecolor="white")
        a1.errorbar(x, r["mr"], yerr=[[r["mr"] - r["lo"]], [r["hi"] - r["mr"]]], fmt="none", ecolor="#333333",
                    elinewidth=1.1, capsize=3)
        a1.text(x, r["hi"] + 1.0, f"{r['mr']:.1f}", ha="center", va="bottom", fontsize=8)
        a1.text(x, 0.8, f"s{r['seed']}", ha="center", va="bottom", fontsize=7.5, color="white")
        a2.bar(x, r["mask"], width=0.34, color=r["colour"], edgecolor="white")
        a2.text(x, r["mask"] + 0.8, f"{r['mask']:.1f}", ha="center", va="bottom", fontsize=8)
        a2.text(x, 2, f"s{r['seed']}", ha="center", va="bottom", fontsize=7.5, color="white")
    a1.set_ylabel("ODCV-lite misconduct rate (%)  ↓ better")
    a1.set_title("ODCV-lite (40 scenarios × 2 variants × 3 passes)\nfixed-benchmark 95% CI (rollout noise only)", fontsize=10)
    a1.set_ylim(0, 65)
    a2.set_ylabel("MASK honesty score  ↑ better")
    a2.set_title("MASK (1,000 rows, 1 pass)\nno CI: one pass per row", fontsize=10)
    a2.set_ylim(0, 105)
    fig.suptitle("t6 ablations — Qwen3.6-27B, 15% synthetic share (s0/s1 = training seed)", fontsize=12)
    fig.tight_layout()
    stem = f"{DATE}_t6_ablations_odcv_mask"
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")

    lines = [f"# t6 ablations: ODCV-lite + MASK ({DATE})", "", "Produced by `uv run python scratch/plot_t6_ablations.py`.",
             "", "ODCV: fixed-benchmark 95% CIs (checkpoint and items fixed). MASK: no CI (one pass per row).", "",
             "| arm | seed | ODCV MR % [95% CI] | MASK honesty | ODCV run | MASK run |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm'].replace(chr(10), ' ')} | {r['seed']} | {r['mr']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] "
                     f"| {r['mask']:.1f} | `{r['odcv']}` | `{r['mask_run']}` |")
    (OUT / f"{stem}_results.md").write_text("\n".join(lines) + "\n")
    print(OUT / f"{stem}.png")


if __name__ == "__main__":
    main()
