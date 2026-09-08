#!/usr/bin/env python3
# ABOUTME: ODCV misalignment rate from ONE rollout per scenario cell — numina control vs
# ABOUTME: nosynth vs difficult advice — with the repo's own interval; a dated figure + results.md.
# Run: uv run python scratch/plot_odcv_single_pass.py [--arms numina,nosynth,da] [--out output/plots]
#
# Three arms measured under the same protocol (temperature 0.7, gemini-3-flash-preview as
# the only judge, 40 scenarios x 2 variants = 80 cells), read down to their FIRST rollout
# per cell so every bar is "one pass". The interval is `summarise`'s (src/eval/misalignment/
# odcv/odcv.py): the spread of per-scenario rates over the 40 scenarios, t with 39 df —
# with one rollout per cell the rollout noise stays inside the cell score, which is why
# these bars are wider than the multi-pass ones the LOG reports. The multi-pass numbers
# are written to the results.md beside the figure for orientation, never plotted.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import VARIANTS, summarise
from src.naming import figure_path

# label -> (repo, how to read it). `scores`: a per-rollout jsonl (variant, scenario,
# rollout, severity_score); `results`: a contract run whose results.json carries
# per_scenario_medians = {variant: {scenario: [score per rollout, pass order]}}.
ARMS = {
    "numina": ("numina control\n(716 numina rows)",
               "LASR-Callum/2026-08-30-odcv-temp07-numina-control-rollout-scores",
               "scores", "odcv_temp07_numina_rollout_scores.jsonl"),
    "nosynth": ("nosynth\n(0% synthetic)",
                "LASR-Callum/2026-09-06-odcv-qwen36-0-nosynth", "results", "results/results.json"),
    "da": ("difficult advice\n(principle-scoped 702, 7%)",
           "LASR-Callum/2026-09-04-odcv-qwen36-0-da-principle-scoped-7", "results",
           "results/results.json"),
}
# Categorical slots 1-3 of the reference palette (dataviz skill), validated 2026-09-06:
# CVD dE >= 24.7, normal-vision dE >= 29.9, >= 3:1 on the light surface.
COLOR = {"numina": "#2a78d6", "nosynth": "#eb6834", "da": "#7a56c5"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def cell_scores(repo: str, kind: str, fname: str) -> dict[str, dict[str, list[float]]]:
    """{variant: {scenario: [score per rollout, first rollout first]}} for one arm."""
    path = hf_hub_download(repo, fname, repo_type="dataset")
    if kind == "results":
        psm = json.load(open(path))["per_scenario_medians"]
        return {v: {s: [float(x) for x in xs] for s, xs in psm.get(v, {}).items()} for v in VARIANTS}
    out: dict[str, dict[str, list[float]]] = {v: defaultdict(list) for v in VARIANTS}
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in sorted(rows, key=lambda r: r["rollout"]):      # rollout_000 = pass 1
        out[r["variant"]][r["scenario"]].append(float(r["severity_score"]))
    return {v: dict(s) for v, s in out.items()}


def first_pass(cells: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    return {v: {s: xs[:1] for s, xs in scen.items() if xs} for v, scen in cells.items()}


def stats(cells) -> dict:
    summ = summarise(cells)
    mr = summ["stats"]["overall"]["mr"]
    return {"mr": mr["mean"], "lo": mr["lo"], "hi": mr["hi"],
            "n_scenarios": summ["overall"]["n_scenarios"],
            "n_cells": summ["overall"]["n_cells"], "n_rollouts": summ["overall"]["n_rollouts"],
            "dropped": summ["overall"]["dropped_scenarios"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="numina,nosynth,da")
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()
    load_dotenv()
    keys = [k.strip() for k in args.arms.split(",") if k.strip()]
    rows = {}
    for k in keys:
        label, repo, kind, fname = ARMS[k]
        cells = cell_scores(repo, kind, fname)
        rows[k] = {"label": label, "repo": repo, "one": stats(first_pass(cells)), "all": stats(cells)}
        o, a = rows[k]["one"], rows[k]["all"]
        print(f"{k:8s} one pass: MR {o['mr']:.1f}% [{o['lo']:.1f}, {o['hi']:.1f}] over {o['n_cells']} cells"
              f" | all passes: MR {a['mr']:.1f}% [{a['lo']:.1f}, {a['hi']:.1f}] over {a['n_rollouts']} rollouts")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(keys)))
    for i, k in enumerate(keys):
        o = rows[k]["one"]
        ax.bar(i, o["mr"], width=0.42, color=COLOR[k], linewidth=0, zorder=2)
        ax.errorbar(i, o["mr"], yerr=[[o["mr"] - o["lo"]], [o["hi"] - o["mr"]]], fmt="none",
                    ecolor=INK2, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
        ax.text(i, o["hi"] + 1.2, f"{o['mr']:.1f}%", ha="center", va="bottom", fontsize=10,
                color=INK, zorder=4)
    ax.set_xticks(xs, [rows[k]["label"] for k in keys], fontsize=9.5, color=INK)
    ax.set_ylabel("misalignment rate (%)", color=INK2, fontsize=9.5)
    ax.set_ylim(0, max(rows[k]["one"]["hi"] for k in keys) * 1.18 + 4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_title("ODCV, one rollout per scenario cell", loc="left", fontsize=11.5, color=INK, pad=12)
    n = rows[keys[0]]["one"]["n_cells"]
    fig.text(0.01, 0.005, f"{n} cells (40 scenarios x mandated/incentivized), pass 1 only.\n"
             "Temperature 0.7; judge gemini-3-flash-preview; misaligned = score >= 3.\n"
             "Bars: 95% interval over scenarios (t, 39 df). Multi-pass numbers: results.md beside this figure.",
             fontsize=7.2, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    subject = "odcv single pass " + " ".join(keys)
    png = figure_path(args.out, subject)
    fig.savefig(png, facecolor=SURFACE)
    md = png.with_name(png.stem + "_results.md")
    lines = ["# ODCV, one rollout per scenario cell", "",
             "| arm | repo | one pass MR | 95% CI | cells | all passes MR | 95% CI | rollouts |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        o, a, r = rows[k]["one"], rows[k]["all"], rows[k]
        lines.append(f"| {r['label'].replace(chr(10), ' ')} | `{r['repo']}` | {o['mr']:.1f}% | [{o['lo']:.1f}, {o['hi']:.1f}] | {o['n_cells']} "
                     f"| {a['mr']:.1f}% | [{a['lo']:.1f}, {a['hi']:.1f}] | {a['n_rollouts']} |")
    lines += ["", "One pass = the first rollout of every cell (rollout_000 / pass 1). Interval: `summarise` "
              "(src/eval/misalignment/odcv/odcv.py), spread of per-scenario rates over 40 scenarios, t with 39 df. "
              "All three arms: temperature 0.7, gemini-3-flash-preview as the only judge, misaligned = score >= 3.",
              "", f"Figure: `{png}`"]
    md.write_text("\n".join(lines) + "\n")
    print(f">>> wrote {png}\n>>> wrote {md}")


if __name__ == "__main__":
    main()
