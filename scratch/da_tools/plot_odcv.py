# ABOUTME: ODCV figure for the da-tools arm vs da-15: misalignment rate and severity with CIs, the paired-by-scenario
# ABOUTME: difference from da-15 seed 0, and the side-effect checks (task progress, submission, tool calls per rollout).
# Run: uv run python scratch/da_tools/plot_odcv.py --tools-run <org/odcv-run-repo> [--out-dir output/figures]
import argparse
import json
import re
import statistics as st
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from src.eval.misalignment.odcv.stats import arm_difference  # noqa: E402

# Fixed arm colours (CLAUDE.md): nosynth grey, difficult advice purple; da-tools is new -- magenta,
# the same family as da so a reader sees it as a DA variant.
GREY, PURPLE, PURPLE_LIGHT, MAGENTA = "#6B7280", "#7E22CE", "#B794E6", "#C026D3"
REFERENCE = [
    ("nosynth (seed 0)", "dougalldeepmind/2026-09-22-odcv-qwen36-0-nosynth", GREY),
    ("da-15 seed 0", "dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15", PURPLE),
    ("da-15 seed 1", "dougalldeepmind/2026-09-26-odcv-qwen36-1-da-15", PURPLE_LIGHT),
]
CONTROL = "da-15 seed 0"


def load(repo: str) -> dict:
    res = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))
    meta = json.load(open(hf_hub_download(repo, "metadata/run_meta.json", repo_type="dataset")))
    cells = {f"{v}/{s}": sev for v, per in res["per_scenario_medians"].items() for s, sev in per.items()}
    rolls = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith("messages_record.txt")]
    calls, silent = [], 0
    for f in rolls:
        text = open(hf_hub_download(repo, f, repo_type="dataset")).read()
        n = sum(line.count("'type': 'function'") for line in text.splitlines() if line.startswith("call: ["))
        calls.append(n)
        silent += n == 0
    cfg = meta.get("config") or {}
    return {"repo": repo, "res": res, "meta": meta, "cells": cells, "calls": calls,
            "no_call_rollouts": silent,
            "protocol": (cfg.get("passes"), cfg.get("temperature"), tuple(res.get("judges") or []), meta.get("mode"))}


def mean_ci(xs: list[float]) -> tuple[float, float]:
    m = st.mean(xs)
    half = 1.96 * st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
    return m, half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-run", required=True, help="the da-tools ODCV run repo (org/name)")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    arms = [(lbl, repo, col) for lbl, repo, col in REFERENCE] + [("da-tools seed 0", a.tools_run, MAGENTA)]
    data = {lbl: load(repo) for lbl, repo, _ in arms}
    protos = {lbl: d["protocol"] for lbl, d in data.items()}
    assert len(set(protos.values())) == 1, f"ODCV protocols differ across arms: {protos}"
    labels = [lbl for lbl, _, _ in arms]
    colors = [col for _, _, col in arms]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # A: misalignment rate
    ov = [data[lbl]["res"]["ours"]["overall"] for lbl in labels]
    mr = [o["mr_pct"] for o in ov]
    lo = [o["mr_pct"] - o["mr_ci95"][0] for o in ov]
    hi = [o["mr_ci95"][1] - o["mr_pct"] for o in ov]
    ax[0, 0].bar(labels, mr, color=colors, yerr=[lo, hi], capsize=4)
    for i, v in enumerate(mr):
        ax[0, 0].text(i, v + hi[i] + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax[0, 0].set_ylim(0, max(o["mr_ci95"][1] for o in ov) * 1.15)
    ax[0, 0].set_ylabel("% of rollouts judged misaligned (severity >= 3)")
    ax[0, 0].set_title("A. Misalignment rate (240 rollouts, 95% CI over scenarios)")
    # B: mean severity (the continuous measure)
    sev = [o["mean_severity"] for o in ov]
    slo = [o["mean_severity"] - o["severity_ci95"][0] for o in ov]
    shi = [o["severity_ci95"][1] - o["mean_severity"] for o in ov]
    ax[0, 1].bar(labels, sev, color=colors, yerr=[slo, shi], capsize=4)
    for i, v in enumerate(sev):
        ax[0, 1].text(i, v + shi[i] + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[0, 1].set_ylim(0, max(o["severity_ci95"][1] for o in ov) * 1.15)
    ax[0, 1].set_ylabel("mean judge severity (0 = none ... 5 = severe)")
    ax[0, 1].set_title("B. Mean severity -- the continuous measure")
    # C: paired difference from da-15 seed 0, by scenario
    diffs = {lbl: arm_difference(data[lbl]["cells"], data[CONTROL]["cells"]) for lbl in labels if lbl != CONTROL}
    names = list(diffs)
    y = range(len(names))
    for i, lbl in enumerate(names):
        d = diffs[lbl]
        col = colors[labels.index(lbl)]
        ax[1, 0].errorbar(d["mr_diff_pp"], i, xerr=[[d["mr_diff_pp"] - d["mr_diff_ci95"][0]],
                                                     [d["mr_diff_ci95"][1] - d["mr_diff_pp"]]],
                          fmt="o", color=col, capsize=4)
        p = d["mr_diff_p_two_sided"]
        ax[1, 0].annotate(f"{d['mr_diff_pp']:+.1f} pp  [{d['mr_diff_ci95'][0]:+.1f}, {d['mr_diff_ci95'][1]:+.1f}]  "
                          + ("p<0.001" if p < 0.001 else f"p={p:.2f}"),
                          (d["mr_diff_pp"], i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax[1, 0].axvline(0, color="black", lw=0.8)
    ax[1, 0].set_yticks(list(y), names)
    ax[1, 0].set_ylim(-0.6, len(names) - 0.4)
    ax[1, 0].set_xlabel("difference in misalignment rate vs da-15 seed 0 (percentage points)")
    ax[1, 0].set_title("C. Paired by scenario: each arm minus da-15 seed 0")
    # D: side effects -- does the arm still act?
    tc = [mean_ci(data[lbl]["calls"]) for lbl in labels]
    ax[1, 1].bar(labels, [m for m, _ in tc], color=colors, yerr=[h for _, h in tc], capsize=4)
    for i, lbl in enumerate(labels):
        prog = data[lbl]["res"]["progress"]["ours"]["overall"]
        sub = data[lbl]["res"]["submission"]["overall"]["submitted_pct"]
        ax[1, 1].text(i, tc[i][0] + tc[i][1] + 0.3,
                      f"{tc[i][0]:.1f} calls\ntask done {prog['tp_rate_pct']:.0f}%\nsubmitted {sub:.0f}%",
                      ha="center", va="bottom", fontsize=8)
    ax[1, 1].set_ylabel("tool calls per rollout (mean, 95% CI)")
    ax[1, 1].set_title("D. Side effect check: does the arm still use its tools?")
    ax[1, 1].set_ylim(0, max(m + h for m, h in tc) * 1.45)
    for axis in (ax[0, 0], ax[0, 1], ax[1, 1]):
        axis.tick_params(axis="x", labelsize=9)
    fig.suptitle("ODCV: difficult advice with benign unused tools (da-tools) vs the same data without them (da-15)",
                 fontsize=12)
    fig.tight_layout()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today().isoformat()}_odcv_da_tools_vs_da15"
    fig.savefig(out / f"{stem}.png", dpi=160)
    summary = {lbl: {"repo": data[lbl]["repo"], "target": data[lbl]["meta"].get("target"),
                     "mr_pct": data[lbl]["res"]["ours"]["overall"]["mr_pct"],
                     "mr_ci95": data[lbl]["res"]["ours"]["overall"]["mr_ci95"],
                     "mean_severity": data[lbl]["res"]["ours"]["overall"]["mean_severity"],
                     "severity_ci95": data[lbl]["res"]["ours"]["overall"]["severity_ci95"],
                     "task_progress_rate_pct": data[lbl]["res"]["progress"]["ours"]["overall"]["tp_rate_pct"],
                     "submitted_pct": data[lbl]["res"]["submission"]["overall"]["submitted_pct"],
                     "tool_calls_per_rollout": round(tc[i][0], 2), "rollouts_without_a_call": data[lbl]["no_call_rollouts"],
                     "paired_vs_da15_seed0": {k: v for k, v in diffs.get(lbl, {}).items() if k != "stats"}}
               for i, lbl in enumerate(labels)}
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    md = [f"# {stem}", "", "| arm | misaligned % [95% CI] | mean severity | task done % | tool calls/rollout | vs da-15 s0 (pp) |",
          "|---|---|---|---|---|---|"]
    for lbl in labels:
        s = summary[lbl]
        pv = s["paired_vs_da15_seed0"]
        md.append(f"| {lbl} | {s['mr_pct']:.1f} [{s['mr_ci95'][0]:.1f}, {s['mr_ci95'][1]:.1f}] | {s['mean_severity']:.2f} | "
                  f"{s['task_progress_rate_pct']:.1f} | {s['tool_calls_per_rollout']:.1f} | "
                  + (f"{pv['mr_diff_pp']:+.1f} [{pv['mr_diff_ci95'][0]:+.1f}, {pv['mr_diff_ci95'][1]:+.1f}], "
                     f"sev {pv['sev_diff']:+.2f} [{pv['sev_diff_ci95'][0]:+.2f}, {pv['sev_diff_ci95'][1]:+.2f}]"
                     if pv else "control") + " |")
    md += ["", "Runs: " + ", ".join(f"`{summary[lbl]['repo']}`" for lbl in labels)]
    (out / f"{stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
