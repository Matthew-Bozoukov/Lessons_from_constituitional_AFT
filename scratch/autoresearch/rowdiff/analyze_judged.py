# ABOUTME: Tabulates the 2026-10-01 row-judge labels per sampled group, estimates each scored arm's DA-row composition
# ABOUTME: (swap arms = donor sample + remaining-28-Sep sample, weighted), and ranks features by ODCV / MASK separation.
"""    uv run python scratch/autoresearch/rowdiff/analyze_judged.py
       -> output/autoresearch/rowdiff/judged_by_group.md, judged_by_arm.md, judged_summary.json
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr

OUT = Path("output/autoresearch/rowdiff")
ENUMS = {
    "norm": ["deception_third_party", "falsify_metric_record_test", "evade_oversight", "harm_person",
             "rule_bending_shortcut", "confidentiality_privacy", "other"],
    "clarity": ["clear_wrongdoing", "mostly_clear", "genuinely_grey"],
    "actor": ["user", "user_org", "ai_system", "assistant_itself", "other_person"],
    "ai_role": ["none", "incidental", "ai_overseen", "ai_actor", "ai_tool"],
    "ask_type": ["what_should_i_do", "produce_artifact", "act_for_me", "about_assistant", "other"],
    "stance": ["firm_refusal", "advises_against_open", "neutral", "helps_partially", "helps"],
}
BOOLS = ["kpi_pressure", "metric_gaming_core", "names_plainly", "asked_to_produce_deceptive", "involves_deception",
         "involves_falsifying", "involves_evading_oversight", "concrete_alternative", "assistant_self_reference"]
GROUPS = ["m0914", "m0925", "sw_other_donor", "sw_self_donor", "sw_advice_donor", "sw_explicit_donor",
          "sw_other_replaced", "m0928", "mself", "mother", "mexpl"]
ODCV_GOOD = ["m0914", "m0925", "sw_other_donor", "sw_advice_donor"]
ODCV_BAD = ["m0928", "mself", "mother", "mexpl"]
MASK_GOOD = ["m0925", "mself", "mother", "mexpl"]
MASK_BAD = ["m0928"]
ARMS = {"m0914": (7.5, 89.4), "m0925": (8.55, 89.25), "m0928": (18.35, 74.4), "mself": (14.2, 97.4),
        "mother": (19.2, 93.8), "mexpl": (15.8, 92.0), "sw_other": (10.8, 79.1), "sw_self": (14.6, 81.5),
        "sw_selfother": (7.5, 85.9), "sw_advice": (8.8, 81.2), "sw_explicit": (12.5, 77.3)}


def feats(o):
    f = {}
    for k, vals in ENUMS.items():
        for v in vals:
            f[f"{k}={v}"] = float(o[k] == v)
    for k in BOOLS:
        f[k] = float(o[k])
    f["ai_any(role!=none)"] = float(o["ai_role"] != "none")
    f["ai_overseen_or_actor"] = float(o["ai_role"] in ("ai_overseen", "ai_actor"))
    f["actor_is_ai_or_assistant"] = float(o["actor"] in ("ai_system", "assistant_itself"))
    f["grey(clarity=grey)"] = float(o["clarity"] == "genuinely_grey")
    f["deception_or_falsify_any"] = float(o["involves_deception"] or o["involves_falsifying"])
    f["stance_firm"] = float(o["stance"] == "firm_refusal")
    f["kpi_and_gaming"] = float(o["kpi_pressure"] and o["metric_gaming_core"])
    f["kpi_and_falsify_or_deceive"] = float(o["kpi_pressure"] and (o["involves_falsifying"] or o["involves_deception"]))
    own = o["actor"] == "assistant_itself" or o["asked_to_produce_deceptive"] or \
        o["ask_type"] in ("about_assistant", "produce_artifact", "act_for_me")
    f["assistant_output_at_stake"] = float(own)
    f["AI_or_assistant_output"] = float(own or o["ai_role"] != "none")
    f["human_only_advice"] = float(o["ai_role"] == "none" and o["ask_type"] == "what_should_i_do"
                                   and o["actor"] not in ("assistant_itself", "ai_system"))
    return f


def main():
    lab = {json.loads(l)["rid"]: json.loads(l) for l in open(OUT / "judged.jsonl")}
    mem = [json.loads(l) for l in open(OUT / "judged_membership.jsonl")]
    by = collections.defaultdict(list)
    for m in mem:
        if m["rid"] in lab:
            by[m["group"]].append({**feats(lab[m["rid"]]), "_sid": m["scenario_id"]})
    keys = [k for k in by["m0928"][0] if not k.startswith("_")]
    mean = {g: {k: st.mean(x[k] for x in v) for k in keys} for g, v in by.items()}
    # remaining (un-swapped) 28 Sep rows per swap arm, estimated from the m0928 sample
    rows = [json.loads(l) for l in open(OUT / "rows.jsonl")]
    repl = collections.defaultdict(set)
    for r in rows:
        if r["group"].endswith("_replaced"):
            repl[r["group"][:-9]].add(r["scenario_id"])

    def remaining(arm):
        v = [x for x in by["m0928"] if x["_sid"] not in repl[arm]]
        return {k: st.mean(x[k] for x in v) for k in keys}, len(v)

    comp = {a: mean[a] for a in ("m0914", "m0925", "m0928", "mself", "mother", "mexpl")}
    ncomp = {a: len(by[a]) for a in comp}
    for arm, parts in {"sw_other": [("sw_other_donor", 391)], "sw_self": [("sw_self_donor", 123)],
                       "sw_advice": [("sw_advice_donor", 159)], "sw_explicit": [("sw_explicit_donor", 159)],
                       "sw_selfother": [("sw_other_donor", 430), ("sw_self_donor", 121)]}.items():
        rem, nrem = remaining(arm)
        w_rem = 617 - sum(w for _, w in parts)
        comp[arm] = {k: (sum(mean[g][k] * w for g, w in parts) + rem[k] * w_rem) / 617 for k in keys}
        ncomp[arm] = f"{'+'.join(str(len(by[g])) for g, _ in parts)}+{nrem}rem"
    arms = list(ARMS)
    summ = {}
    for k in keys:
        pg = lambda gs: st.mean(mean[g][k] for g in gs)
        sep_odcv = min(mean[g][k] for g in ODCV_GOOD) - max(mean[g][k] for g in ODCV_BAD)
        sep_odcv2 = min(mean[g][k] for g in ODCV_BAD) - max(mean[g][k] for g in ODCV_GOOD)
        sep_mask = min(mean[g][k] for g in MASK_GOOD) - max(mean[g][k] for g in MASK_BAD)
        sep_mask2 = min(mean[g][k] for g in MASK_BAD) - max(mean[g][k] for g in MASK_GOOD)
        xs = [comp[a][k] for a in arms]
        ok = len(set(round(x, 6) for x in xs)) > 2
        summ[k] = {
            "odcv_good_minus_bad": round(pg(ODCV_GOOD) - pg(ODCV_BAD), 3),
            "odcv_strict_gap": round(max(sep_odcv, sep_odcv2), 3),  # >0 = every good group on one side of every bad
            "mask_good_minus_bad": round(pg(MASK_GOOD) - pg(MASK_BAD), 3),
            "mask_strict_gap": round(max(sep_mask, sep_mask2), 3),
            "rho_odcv_arms": round(spearmanr(xs, [ARMS[a][0] for a in arms]).correlation, 3) if ok else None,
            "rho_mask_arms": round(spearmanr(xs, [ARMS[a][1] for a in arms]).correlation, 3) if ok else None,
        }
    (OUT / "judged_summary.json").write_text(json.dumps({"groups": {g: len(by[g]) for g in GROUPS},
                                                         "features": summ}, indent=1))

    def pct(v):
        return f"{100 * v:.0f}"

    def table(order, sortkey, title):
        lines = [f"### {title}", "",
                 "| feature | good-bad (pp) | strict gap (pp) | rho ODCV (11 arms) | rho MASK (11 arms) | "
                 + " | ".join(GROUPS) + " |", "|---|---|---|---|---|" + "---|" * len(GROUPS),
                 "| n judged | | | | | " + " | ".join(str(len(by[g])) for g in GROUPS) + " |"]
        for k in order:
            s = summ[k]
            gb, gap = (s["odcv_good_minus_bad"], s["odcv_strict_gap"]) if sortkey == "odcv" else \
                (s["mask_good_minus_bad"], s["mask_strict_gap"])
            se = max(math.sqrt(max(mean[g][k] * (1 - mean[g][k]), 1e-9) / len(by[g])) for g in GROUPS)
            lines.append(f"| {k} | {100 * gb:+.0f} | {100 * gap:+.0f} | {s['rho_odcv_arms']} | {s['rho_mask_arms']} | "
                         + " | ".join(pct(mean[g][k]) for g in GROUPS) + " |")
        return lines

    o_order = sorted(keys, key=lambda k: (-(summ[k]["odcv_strict_gap"] > 0), -abs(summ[k]["odcv_good_minus_bad"])))
    m_order = sorted(keys, key=lambda k: (-(summ[k]["mask_strict_gap"] > 0), -abs(summ[k]["mask_good_minus_bad"])))
    lines = ["All cells are % of judged rows (n per group in the second row). ODCV-good groups: " + ", ".join(ODCV_GOOD)
             + "; ODCV-bad: " + ", ".join(ODCV_BAD) + ". MASK-good: " + ", ".join(MASK_GOOD) + "; MASK-bad: m0928.",
             "strict gap = smallest distance between the two sets of groups when every good group lies on one side of "
             "every bad group (positive), else the overlap (negative). rho = Spearman over 11 arms' estimated DA "
             "composition (swap arms = donor sample + un-swapped 28 Sep sample, weighted by row count).", ""]
    lines += table(o_order, "odcv", "Sorted by ODCV separation") + [""] + table(m_order, "mask", "Sorted by MASK separation")
    (OUT / "judged_by_group.md").write_text("\n".join(lines) + "\n")
    al = ["| feature | rho ODCV | rho MASK | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms),
          "| ODCV | | | " + " | ".join(str(ARMS[a][0]) for a in arms) + " |",
          "| MASK | | | " + " | ".join(str(ARMS[a][1]) for a in arms) + " |",
          "| n (sample) | | | " + " | ".join(str(ncomp[a]) for a in arms) + " |"]
    for k in sorted(keys, key=lambda k: -abs(summ[k]["rho_odcv_arms"] or 0)):
        al.append(f"| {k} | {summ[k]['rho_odcv_arms']} | {summ[k]['rho_mask_arms']} | "
                  + " | ".join(pct(comp[a][k]) for a in arms) + " |")
    (OUT / "judged_by_arm.md").write_text("\n".join(al) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
