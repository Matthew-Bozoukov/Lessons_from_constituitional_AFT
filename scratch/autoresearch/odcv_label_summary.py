# ABOUTME: Summarise the gemini decision-point labels of ODCV-lite rollouts per arm group: outcome mix of violations,
# ABOUTME: rationalisation types, noticed-before-acting rates, honest-completion vs refusal in the gap-cluster cells.
"""uv run python scratch/autoresearch/odcv_label_summary.py -> output/autoresearch/odcv_label_summary.md"""
from __future__ import annotations

import collections
import json

import pandas as pd

OUT = "output/autoresearch"
GOOD = ["s25_s0", "s25_s1", "swap_self_otherai", "swap_advice", "swap_otherai", "s28_da25"]
BAD = ["s28_base_a", "s28_base_b", "s28_no_t6", "new_otherai", "new_self", "new_explicit", "lowstakes", "nonmoral"]
grp = lambda a: "good" if a in GOOD else "bad" if a in BAD else a
D = pd.DataFrame([json.loads(l) for l in open(f"{OUT}/odcv_decision_labels.jsonl")])
D = D[D.get("parse_error").isna()] if "parse_error" in D else D
D["group"] = D.arm.map(grp)
C = pd.read_csv(f"{OUT}/odcv_cells_good_bad.csv")
top = set(zip(C.scenario[:15], C.variant[:15]))
D["top"] = [(s, v) in top for s, v in zip(D.scenario, D.variant)]
lines = [f"# ODCV-lite decision-point labels (gemini-3-flash, n={len(D)})", ""]


def tab(df: pd.DataFrame, col: str, by: str = "group") -> str:
    t = pd.crosstab(df[by], df[col], normalize="index").round(2)
    t["n"] = df.groupby(by).size()
    return t.to_string()


V = D[D.violation == 1]
lines += ["## Violations: outcome type by group", tab(V, "outcome"), ""]
lines += ["## Violations: outcome type by arm", tab(V, "outcome", "arm"), ""]
lines += ["## Violations: noticed / first-turn flag / reported shortfall", V.groupby("group")[["noticed", "first_turn_flags_risk", "reported_shortfall", "kpi_met"]].mean().round(2).to_string(), ""]
rc = collections.defaultdict(collections.Counter)
for g, rs in zip(V.group, V.rationalization):
    for r in (rs or []):
        rc[g][r] += 1
R = pd.DataFrame(rc).fillna(0).astype(int)
R.loc["n_violations"] = V.groupby("group").size()
lines += ["## Violations: rationalisation tags (count; multi-label)", R.to_string(), ""]
N = D[(D.violation == 0) & D.top]
lines += ["## Non-violations in the 15 gap cells: outcome by group", tab(N, "outcome"), "",
          "## Non-violations in gap cells: kpi_met / reported_shortfall / first-turn flag", N.groupby("group")[["kpi_met", "reported_shortfall", "first_turn_flags_risk", "noticed"]].mean().round(2).to_string(), ""]
T = D[D.top & D.group.isin(["good", "bad"])]
lines += ["## All labelled rollouts in gap cells (good/bad arms): first_turn_flags_risk vs violation",
          pd.crosstab([T.group, T.first_turn_flags_risk], T.violation, margins=True).to_string(), ""]
# per-arm, in gap cells: share of rollouts where the first turn flags risk
lines += ["## In gap cells, per arm: first-turn flags risk (all labelled rollouts there)",
          T.groupby("arm")[["first_turn_flags_risk", "violation"]].mean().round(2).to_string(), ""]
open(f"{OUT}/odcv_label_summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
