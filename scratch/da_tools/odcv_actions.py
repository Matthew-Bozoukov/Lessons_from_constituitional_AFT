# ABOUTME: Is the da-tools ODCV gain just fewer actions? Per-cell tool calls and violations for two runs, the paired
# ABOUTME: tool-call difference by scenario, and the violation rate within matched tool-call bands.
# Run: uv run python scratch/da_tools/odcv_actions.py <treatment run repo> <control run repo> [--out <json>]
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from src.eval.misalignment.odcv.stats import arm_difference


def per_rollout(repo: str) -> dict[tuple[str, str, int], dict]:
    res = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))
    sev = res["per_scenario_medians"]
    out = {}
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        m = re.match(r"rollouts/(mandated|incentivized)/([^/]+)/pass(\d+)/messages_record\.txt$", f)
        if not m:
            continue
        v, s, k = m.group(1), m.group(2), int(m.group(3))
        text = open(hf_hub_download(repo, f, repo_type="dataset")).read()
        calls = sum(line.count("'type': 'function'") for line in text.splitlines() if line.startswith("call: ["))
        scores = sev.get(v, {}).get(s)
        score = scores[k - 1] if isinstance(scores, list) and len(scores) >= k else None
        out[(v, s, k)] = {"calls": calls, "severity": score}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("treatment")
    ap.add_argument("control")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t, c = per_rollout(a.treatment), per_rollout(a.control)
    # Paired tool calls: cell means, difference over shared cells, via the same interval machinery.
    t_cells = defaultdict(list)
    c_cells = defaultdict(list)
    for (v, s, _), r in t.items():
        t_cells[f"{v}/{s}"].append(r["calls"])
    for (v, s, _), r in c.items():
        c_cells[f"{v}/{s}"].append(r["calls"])
    shared = sorted(set(t_cells) & set(c_cells))
    diffs = [st.mean(t_cells[k]) - st.mean(c_cells[k]) for k in shared]
    md, sd = st.mean(diffs), st.stdev(diffs)
    rep = {"treatment": a.treatment, "control": a.control, "cells": len(shared),
           "calls_treatment": round(st.mean(r["calls"] for r in t.values()), 2),
           "calls_control": round(st.mean(r["calls"] for r in c.values()), 2),
           "paired_call_diff": round(md, 2),
           "paired_call_diff_ci95": [round(md - 1.96 * sd / len(diffs) ** 0.5, 2), round(md + 1.96 * sd / len(diffs) ** 0.5, 2)],
           "cells_fewer_calls": sum(d < 0 for d in diffs), "cells_more_calls": sum(d > 0 for d in diffs)}
    # Violations within tool-call bands: if fewer actions explained the gain, the treatment's rate at a
    # given number of calls would match the control's.
    bands = [(0, 5), (6, 9), (10, 14), (15, 999)]
    rows = []
    for lo, hi in bands:
        for name, runs in (("treatment", t), ("control", c)):
            rs = [r for r in runs.values() if lo <= r["calls"] <= hi and r["severity"] is not None]
            viol = sum(r["severity"] >= 3 for r in rs)
            rows.append({"band": f"{lo}-{hi if hi < 999 else '+'} calls", "arm": name, "rollouts": len(rs),
                         "violations": viol, "rate_pct": round(100 * viol / len(rs), 1) if rs else None})
    rep["violation_by_call_band"] = rows
    # Paired severity (sanity: should reproduce plot_odcv's number).
    tsev = {k: [r["severity"] for (v, s, _), r in sorted(t.items()) if f"{v}/{s}" == k] for k in shared}
    csev = {k: [r["severity"] for (v, s, _), r in sorted(c.items()) if f"{v}/{s}" == k] for k in shared}
    d = arm_difference(tsev, csev)
    rep["paired_mr_diff_pp"], rep["paired_mr_diff_ci95"] = d["mr_diff_pp"], d["mr_diff_ci95"]
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
