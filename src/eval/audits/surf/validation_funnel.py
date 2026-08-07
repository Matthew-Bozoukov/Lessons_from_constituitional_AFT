# ABOUTME: Two-round validation funnel over a SURF flag set: pinned-snapshot integrity,
# ABOUTME: fingerprint drift detection, fp-rate with exact CI, fabrication/mode axes.
"""Validation funnel for SURF runs.

Generalized from experiments/vulnerabilities/scripts/surf/validation_funnel.py
at commit b38da52. The original pinned one live run (harmful-omission run 2)
and encoded its hand-adjudicated verdicts, mode labels, and per-run analysis
notes as in-source literals so the published counts could not drift from the
prose; that frozen record lives in git history. This module keeps the protocol
mechanics and takes the run-specific parts as data.

Protocol (same as the Petri funnel): Round 1 returns CONFIRMED / BORDERLINE /
FALSE_POSITIVE with CONFIRMED and FALSE_POSITIVE final; only BORDERLINE
escalates to Round 2's forced binary. Two extra axes beyond the Petri funnel:
`fabrication` (the defect is inventing content, the opposite construct to an
omission rubric - FALSE_POSITIVE for the rubric, counted separately) and
`mode` (why a false positive failed, so the pattern is visible).

Verdicts file schema (JSON):
  {
    "pinned_max_iteration": 4,        # adjudication was pinned to iterations <= N
    "pinned_n": 193,                  # expected candidate count in the pinned snapshot
    "pinned_flags": 40,               # expected flag count in the pinned snapshot
    "flag_threshold": 50.0,           # SURF flags reward_score > threshold
    "mode_labels": {"mode-key": "human label", ...},
    "verdicts": {
      "<row-index>": {"query_fingerprint": "...", "round1": "CONFIRMED|BORDERLINE|FALSE_POSITIVE",
                       "round2": "CONFIRMED|FALSE_POSITIVE|null", "fabrication": false,
                       "mode": "mode-key|null", "mechanism_present": false, "rationale": "..."}
    },
    "annotations": { ... }            # optional; recorded verbatim in the output
  }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter

from src.eval.audits.petri.funnel_stats import clopper_pearson


def fingerprint(query):
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


def build_funnel(rows, spec, source="", rubric="", target="", judge=""):
    """Compute the SURF funnel summary.

    Args:
        rows: list of SURF results.jsonl records (needs iteration, reward_score, query).
        spec: parsed verdicts file (schema in the module docstring).
        source/rubric/target/judge: labels recorded in the output.

    Returns:
        The funnel summary dict.
    """
    pinned_max_iteration = spec["pinned_max_iteration"]
    pinned_n = spec["pinned_n"]
    pinned_flags = spec["pinned_flags"]
    flag_threshold = spec["flag_threshold"]
    mode_labels = spec.get("mode_labels", {})
    verdicts = {int(i): v for i, v in spec["verdicts"].items()}

    total_on_disk = len(rows)
    snapshot = [(i, r) for i, r in enumerate(rows) if r["iteration"] <= pinned_max_iteration]
    later = [(i, r) for i, r in enumerate(rows) if r["iteration"] > pinned_max_iteration]

    assert len(snapshot) == pinned_n, (
        "pinned snapshot is %d rows, expected %d - results.jsonl was rewritten, "
        "not appended to" % (len(snapshot), pinned_n))

    flagged = [(i, r) for i, r in snapshot if r["reward_score"] > flag_threshold]
    assert len(flagged) == pinned_flags, (
        "%d flags, expected %d" % (len(flagged), pinned_flags))
    assert set(i for i, _ in flagged) == set(verdicts), (
        "adjudicated set does not match the flagged set")

    # Fingerprints are recorded but not enforced: they were taken at adjudication
    # time and exist so a rewrite of the file is detectable.
    drifted = [i for i in verdicts
               if fingerprint(rows[i]["query"]) != verdicts[i]["query_fingerprint"]]

    r1 = Counter(v["round1"] for v in verdicts.values())
    borderline = sorted(i for i, v in verdicts.items() if v["round1"] == "BORDERLINE")
    r2 = Counter(verdicts[i]["round2"] for i in borderline)

    def final_of(i):
        v = verdicts[i]
        return v["round2"] if v["round1"] == "BORDERLINE" else v["round1"]

    finals = {i: final_of(i) for i in verdicts}
    confirmed = sorted(i for i in finals if finals[i] == "CONFIRMED")
    fps = sorted(i for i in finals if finals[i] == "FALSE_POSITIVE")
    fabrication = sorted(i for i in verdicts if verdicts[i].get("fabrication"))
    mechanism = sorted(i for i in verdicts if verdicts[i].get("mechanism_present"))

    fp_rate = len(fps) / len(flagged)
    ci = clopper_pearson(len(fps), len(flagged))
    modes = Counter(verdicts[i].get("mode") for i in fps)

    scores = [r["reward_score"] for _, r in snapshot]
    flagged_scores = [r["reward_score"] for _, r in flagged]

    out = {
        "source": source,
        "rubric": rubric,
        "target": target,
        "judge": judge,
        "protocol": ("Two-round funnel. Round 1 returns CONFIRMED / BORDERLINE / "
                     "FALSE_POSITIVE with CONFIRMED and FALSE_POSITIVE final; only "
                     "BORDERLINE escalates to Round 2's forced binary."),
        "pinning": {
            "note": ("Adjudication is pinned to iterations 1-%d; later iterations are "
                     "counted but not adjudicated." % pinned_max_iteration),
            "pinned_max_iteration": pinned_max_iteration,
            "pinned_candidates": len(snapshot),
            "pinned_flags": len(flagged),
            "rows_on_disk_at_this_run": total_on_disk,
            "later_iteration_candidates_not_adjudicated": len(later),
            "later_iteration_flags_not_adjudicated": sum(
                1 for _, r in later if r["reward_score"] > flag_threshold),
            "query_fingerprint_drift": drifted,
        },
        "flagging": {
            "threshold": "reward_score > %g" % flag_threshold,
            "candidates_scored": len(snapshot),
            "flagged": len(flagged),
            "flag_rate": round(len(flagged) / len(snapshot), 4),
            "mean_score_all": round(statistics.mean(scores), 2),
            "mean_score_flagged": round(statistics.mean(flagged_scores), 2),
            "flagged_score_range": [min(flagged_scores), max(flagged_scores)],
        },
        "funnel": {
            "raw_flagged": len(flagged),
            "r1_confirmed": r1["CONFIRMED"],
            "r1_borderline": r1["BORDERLINE"],
            "r1_false_positive": r1["FALSE_POSITIVE"],
            "r2_borderline_to_confirmed": r2["CONFIRMED"],
            "r2_borderline_to_false_positive": r2["FALSE_POSITIVE"],
            "final_confirmed": len(confirmed),
            "final_false_positive": len(fps),
            "false_positive_rate": round(fp_rate, 4),
            "false_positive_rate_ci95": ci,
            "ci_method": "Clopper-Pearson exact, via src/eval/audits/petri/funnel_stats.py",
        },
        "confirmed": {
            str(i): {
                "reward_score": rows[i]["reward_score"],
                "iteration": rows[i]["iteration"],
                "round1": verdicts[i]["round1"],
                "round2": verdicts[i]["round2"],
                "mechanism_present": verdicts[i].get("mechanism_present", False),
                "query": rows[i]["query"],
                "rationale": verdicts[i].get("rationale", ""),
            } for i in confirmed
        },
        "fabrication_rather_than_omission": {
            "count": len(fabrication),
            "share_of_flags": round(len(fabrication) / len(flagged), 4),
            "indices": fabrication,
        },
        "false_positive_modes": {
            k: {"count": v, "label": mode_labels.get(k, k),
                "indices": sorted(i for i in fps if verdicts[i].get("mode") == k)}
            for k, v in modes.most_common()
        },
        "mechanism": {
            "flags_showing_explicit_humility_deference_or_caution_as_reason": len(mechanism),
            "indices": mechanism,
        },
        "verdicts": {
            str(i): {
                "reward_score": rows[i]["reward_score"],
                "iteration": rows[i]["iteration"],
                "query_fingerprint": verdicts[i]["query_fingerprint"],
                "round1": verdicts[i]["round1"],
                "round2": verdicts[i]["round2"],
                "final": finals[i],
                "fabrication": verdicts[i].get("fabrication", False),
                "mode": verdicts[i].get("mode"),
                "mode_label": mode_labels.get(verdicts[i].get("mode"), verdicts[i].get("mode")),
                "mechanism_present": verdicts[i].get("mechanism_present", False),
                "query_head": rows[i]["query"][:160],
                "rationale": verdicts[i].get("rationale", ""),
            } for i in sorted(verdicts, key=lambda i: -rows[i]["reward_score"])
        },
    }
    if spec.get("annotations"):
        out["annotations"] = spec["annotations"]
    return out


def render_tables(out):
    """Print the funnel summary (same layout the original used)."""
    p = out["pinning"]
    f = out["funnel"]
    print("SURF validation funnel")
    print("  pinned snapshot        %d candidates (iterations 1-%d), %d on disk now"
          % (p["pinned_candidates"], p["pinned_max_iteration"], p["rows_on_disk_at_this_run"]))
    print("  raw flagged            %d  (%.1f%% of candidates)"
          % (f["raw_flagged"], 100 * out["flagging"]["flag_rate"]))
    print("  R1 confirmed           %d" % f["r1_confirmed"])
    print("  R1 borderline          %d" % f["r1_borderline"])
    print("  R1 false positive      %d" % f["r1_false_positive"])
    print("  R2 borderline->conf    %d" % f["r2_borderline_to_confirmed"])
    print("  R2 borderline->FP      %d" % f["r2_borderline_to_false_positive"])
    print("  FINAL CONFIRMED        %d  %s" % (f["final_confirmed"], list(out["confirmed"])))
    print("  FINAL FALSE POSITIVE   %d" % f["final_false_positive"])
    print("  false-positive rate    %.1f%%  (95%% CI %.1f-%.1f%%, Clopper-Pearson)"
          % (100 * f["false_positive_rate"],
             100 * f["false_positive_rate_ci95"][0], 100 * f["false_positive_rate_ci95"][1]))
    fab = out["fabrication_rather_than_omission"]
    print("  fabrication-not-omission %d  (%.0f%% of flags)"
          % (fab["count"], 100 * fab["share_of_flags"]))
    print("  flags showing the rubric's mechanism  %d"
          % out["mechanism"]["flags_showing_explicit_humility_deference_or_caution_as_reason"])
    print()
    print("  false-positive modes:")
    for k, v in out["false_positive_modes"].items():
        print("    %-24s %2d  %s" % (k, v["count"], v["label"]))
    if p["query_fingerprint_drift"]:
        print()
        print("  WARNING: query fingerprint drift on indices %s" % p["query_fingerprint_drift"])


def main(args):
    rows = [json.loads(l) for l in open(args.results, encoding="utf-8")]
    spec = json.load(open(args.verdicts, encoding="utf-8"))

    out = build_funnel(rows, spec, source=args.results, rubric=args.rubric,
                       target=args.target, judge=args.judge)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    render_tables(out)
    print()
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Two-round validation funnel over a SURF flag set.")
    p.add_argument("--results", required=True, help="SURF results.jsonl for the run")
    p.add_argument("--verdicts", required=True, help="verdicts JSON (schema in module docstring)")
    p.add_argument("--out", required=True, help="funnel summary JSON to write")
    p.add_argument("--rubric", default="", help="rubric label recorded in the output")
    p.add_argument("--target", default="", help="target model label")
    p.add_argument("--judge", default="", help="judge model label")
    main(p.parse_args())
