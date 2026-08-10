# ABOUTME: Two-round validation-funnel statistics over compound-flagged Petri samples:
# ABOUTME: exact binomial CIs, per-seed replication tables, and the funnel summary JSON.
"""Validation-funnel statistics for Petri audit runs.

Generalized from experiments/vulnerabilities/scripts/petri/funnel.py at commit
b38da52. The original encoded that run's hand-adjudicated Round 1 / Round 2
verdicts as in-source dict literals, deliberately, so the published counts could
not drift from the prose; those verdicts are NOT copied here - they remain part
of the frozen record in git history. This module takes verdicts as data.

Protocol: Round 1 returns CONFIRMED / BORDERLINE / FALSE_POSITIVE with
CONFIRMED and FALSE_POSITIVE final; only BORDERLINE escalates to Round 2's
forced binary.

Inputs:
  --samples   samples.json from extract_funnel.py ({uid, seed, epoch, scores,
              clauses, flagged_compound, flagged_old_rule, errored,
              target_tool_calls, target_turns, ...} records)
  --verdicts  JSON: {"round1": {uid: {"verdict", "on_construct", "rationale"}},
                     "round2": {uid: {"verdict", "on_construct", "rationale"}}}
              round1 must cover exactly the compound-flagged set; round2 must
              cover exactly round1's BORDERLINE set (asserted, as the original did).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from math import comb


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI without scipy, via bisection on the Beta CDF identity
    using the regularized incomplete beta computed from the binomial sum."""
    def binom_cdf(p, k, n):
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))

    def solve(target, lo_side):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            # lower bound: P(X >= k) = alpha/2  ->  1 - cdf(k-1) = alpha/2
            v = (1 - binom_cdf(mid, k - 1, n)) if lo_side else binom_cdf(mid, k, n)
            if (v < target) == lo_side:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(alpha / 2, True)
    high = 1.0 if k == n else solve(alpha / 2, False)
    return round(low, 4), round(high, 4)


def build_funnel(rows, r1, r2, source_log="", dimension_mapping=None):
    """Compute the two-round funnel summary.

    Args:
        rows: {uid: sample-record} as produced by extract_funnel.py.
        r1: {uid: (verdict, on_construct, rationale)} covering the flagged set.
        r2: {uid: (verdict, on_construct, rationale)} covering R1's BORDERLINE set.
        source_log: label for the .eval log the samples came from.
        dimension_mapping: optional run-metadata dict recorded verbatim.

    Returns:
        The funnel summary dict (same schema the original run committed).
    """
    flagged = [u for u, r in rows.items() if r["flagged_compound"]]
    old = [u for u, r in rows.items() if r["flagged_old_rule"]]

    assert set(r1) == set(flagged), (set(r1) ^ set(flagged))
    borderline = [u for u in r1 if r1[u][0] == "BORDERLINE"]
    assert set(r2) == set(borderline), (set(r2) ^ set(borderline))

    final = {}
    for u, (v, oc, why) in r1.items():
        if v != "BORDERLINE":
            final[u] = {"round": 1, "verdict": v, "on_construct": oc, "rationale": why}
    for u, (v, oc, why) in r2.items():
        final[u] = {"round": 2, "verdict": v, "on_construct": oc,
                    "r1_rationale": r1[u][2], "rationale": why}

    r1c = [u for u in r1 if r1[u][0] == "CONFIRMED"]
    r1f = [u for u in r1 if r1[u][0] == "FALSE_POSITIVE"]
    r2c = [u for u in r2 if r2[u][0] == "CONFIRMED"]
    r2f = [u for u in r2 if r2[u][0] == "FALSE_POSITIVE"]
    conf = sorted(r1c + r2c)
    fp = sorted(r1f + r2f)
    onc = sorted(u for u in conf if final[u]["on_construct"])

    # --- replication, per seed ---
    seeds = {}
    for u, r in rows.items():
        seeds.setdefault(r["seed"], []).append(r)
    rep = {}
    for s, rs in sorted(seeds.items()):
        n = len(rs)
        scored = [r for r in rs if not r["errored"]]
        fl = [r["uid"] for r in rs if r["flagged_compound"]]
        cf = [u for u in fl if final[u]["verdict"] == "CONFIRMED"]
        oc = [u for u in cf if final[u]["on_construct"]]
        rep[s] = {
            "epochs": n, "scored_epochs": len(scored),
            "flagged": sorted(fl), "n_flagged": len(fl),
            "confirmed": sorted(cf), "n_confirmed": len(cf),
            "on_construct_confirmed": sorted(oc), "n_on_construct": len(oc),
            "flag_majority_of_3": len(fl) > n / 2,
            "confirmed_majority_of_3": len(cf) > n / 2,
            "mean_target_tool_calls": round(statistics.mean(r["target_tool_calls"] for r in rs), 2),
            "tool_calls_per_epoch": [r["target_tool_calls"] for r in sorted(rs, key=lambda x: x["epoch"])],
        }

    allc = [r["target_tool_calls"] for r in rows.values()]
    nonnull = [r["target_tool_calls"] for r in rows.values() if r["target_turns"] > 0]

    out = {
        "source_log": source_log,
        "n_samples": len(rows),
        "funnel": {
            "raw_flagged_compound": len(flagged),
            "raw_flagged_old_rule": len(old),
            "r1_confirmed": len(r1c), "r1_borderline": len(borderline), "r1_false_positive": len(r1f),
            "r2_borderline_to_confirmed": len(r2c), "r2_borderline_to_false_positive": len(r2f),
            "final_confirmed": len(conf), "final_false_positive": len(fp),
            "final_confirmed_on_construct": len(onc),
            "false_positive_rate": round(len(fp) / len(flagged), 4),
            "false_positive_rate_ci95": clopper_pearson(len(fp), len(flagged)),
            "multi_core_clause_only_flags": sorted(
                u for u in flagged
                if all(c.startswith("multi-core") for c in rows[u]["clauses"])),
        },
        "flagged_set_diff": {
            "newly_flagged_vs_old_rule": sorted(set(flagged) - set(old)),
            "no_longer_flagged_vs_old_rule": sorted(set(old) - set(flagged)),
        },
        "verdicts": final,
        "confirmed": conf,
        "confirmed_on_construct": onc,
        "false_positives": fp,
        "replication": rep,
        "replicating_seeds_raw_flag": sorted(s for s, v in rep.items() if v["flag_majority_of_3"]),
        "replicating_seeds_after_validation": sorted(s for s, v in rep.items() if v["confirmed_majority_of_3"]),
        "tool_call_diagnostic": {
            "mean_all_transcripts": round(statistics.mean(allc), 2),
            "mean_excluding_null_transcript": round(statistics.mean(nonnull), 2),
            "min": min(allc), "max": max(allc),
            "family_B_mean": round(statistics.mean(
                r["target_tool_calls"] for r in rows.values() if r["seed"].startswith("B")), 2),
            "family_C_mean": round(statistics.mean(
                r["target_tool_calls"] for r in rows.values() if r["seed"].startswith("C")), 2),
            "seeds_near_zero": [s for s, v in rep.items() if v["mean_target_tool_calls"] < 1.0],
            "transcripts_le_2_calls": sorted(
                (r["uid"], r["target_tool_calls"]) for r in rows.values() if r["target_tool_calls"] <= 2),
        },
        "per_sample": {u: {"scores": r["scores"], "clauses": r["clauses"],
                           "target_tool_calls": r["target_tool_calls"],
                           "scenario_realism": r["scenario_realism"],
                           "auditor_failure": r["auditor_failure"],
                           "broken_tool_use": r["broken_tool_use"],
                           "input_hallucination": r["input_hallucination"],
                           "errored": r["errored"],
                           "flagged_compound": r["flagged_compound"],
                           "flagged_old_rule": r["flagged_old_rule"]}
                       for u, r in sorted(rows.items())},
    }
    if dimension_mapping is not None:
        out["dimension_mapping"] = dimension_mapping
    return out


def render_tables(out):
    """Print the funnel and per-seed replication tables (same layout the original used)."""
    f_ = out["funnel"]
    print("FUNNEL")
    for k in ("raw_flagged_old_rule", "raw_flagged_compound", "r1_confirmed", "r1_borderline",
              "r1_false_positive", "r2_borderline_to_confirmed", "r2_borderline_to_false_positive",
              "final_confirmed", "final_confirmed_on_construct", "final_false_positive"):
        print(f"  {k:<34}{f_[k]}")
    print(f"  false-positive rate              {f_['false_positive_rate']:.1%}  95% CI {f_['false_positive_rate_ci95']}")
    print()
    print("REPLICATION (majority of 3 epochs)")
    print(f"  {'seed':<38}{'flagged':<9}{'confirmed':<11}{'on-construct'}")
    for s, v in out["replication"].items():
        print(f"  {s:<38}{v['n_flagged']}/{v['epochs']:<7}{v['n_confirmed']}/{v['epochs']:<9}{v['n_on_construct']}/{v['epochs']}")
    print()
    print("  seeds flagged in a majority (rule only):", out["replicating_seeds_raw_flag"] or "NONE")
    print("  seeds CONFIRMED in a majority          :", out["replicating_seeds_after_validation"] or "NONE")


def _load_verdict_round(block):
    """{uid: {"verdict","on_construct","rationale"}} -> {uid: (v, oc, why)}."""
    return {u: (d["verdict"], bool(d["on_construct"]), d.get("rationale", ""))
            for u, d in block.items()}


def main(args):
    rows = {r["uid"]: r for r in json.load(open(args.samples, encoding="utf-8"))}
    verdicts = json.load(open(args.verdicts, encoding="utf-8"))
    r1 = _load_verdict_round(verdicts["round1"])
    r2 = _load_verdict_round(verdicts.get("round2", {}))
    mapping = None
    if args.dimension_mapping:
        mapping = json.load(open(args.dimension_mapping, encoding="utf-8"))

    out = build_funnel(rows, r1, r2, source_log=args.source_log, dimension_mapping=mapping)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    render_tables(out)
    print()
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Two-round validation-funnel statistics for Petri audit runs.")
    p.add_argument("--samples", required=True, help="samples.json from extract_funnel.py")
    p.add_argument("--verdicts", required=True,
                   help='JSON with "round1"/"round2" hand-adjudicated verdict maps')
    p.add_argument("--out", required=True, help="funnel summary JSON to write")
    p.add_argument("--source-log", default="", help="label for the source .eval log")
    p.add_argument("--dimension-mapping", default="",
                   help="optional JSON recorded verbatim as dimension_mapping")
    main(p.parse_args())
