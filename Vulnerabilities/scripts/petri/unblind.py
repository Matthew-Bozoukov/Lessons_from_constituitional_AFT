#!/usr/bin/env python3
"""Merge the blind judge scores, unblind them, and build the attribution table.

This is the step the whole run exists to reach. Petri discovers candidate
behaviours; this measures them on byte-identical prompts across matched
checkpoints, so a difference can be attributed to the weights rather than to
the scenario the auditor happened to invent.

The contrasts that matter are the matched pairs. Comparing the primary target
against the base model conflates every training stage at once; comparing it
against the checkpoint that differs only by the presence of model-spec
midtraining is what isolates MSM.
"""
import io
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FE = os.path.join(ROOT, "evidence", "fixed-eval")


def load(path):
    with io.open(path, encoding="utf-8-sig") as f:
        return json.load(f)


mapping = load(os.path.join(FE, "blind-mapping.json"))
probes = {p["id"]: p for p in load(os.path.join(ROOT, "seeds", "fixed-eval", "probes.json"))["probes"]}

rows = []
missing = []
for slice_name, ckpt in mapping.items():
    path = os.path.join(FE, "scores", f"{slice_name}.json")
    if not os.path.exists(path):
        missing.append(slice_name)
        continue
    blind = {r["record_id"]: r for r in load(os.path.join(FE, "blind", f"{slice_name}.json"))}
    for s in load(path):
        rid = s["record_id"]
        b = blind.get(rid)
        if b is None:
            continue
        if not isinstance(s.get("score"), int):
            continue
        rows.append({"checkpoint": ckpt, "probe": b["probe"], "family": b["family"],
                     "arm": b["arm"], "score": s["score"],
                     "did_expected": s.get("did_expected")})

if missing:
    print(f"MISSING SCORE FILES: {', '.join(sorted(missing))}\n")

print(f"{len(rows)} scored records across {len(set(r['checkpoint'] for r in rows))} checkpoints\n")

CHECKPOINTS = ["msm-aft-cot", "msm-aft-no-cot", "aft-cot", "aft-no-cot",
               "msm-only", "id-baseline", "qwen3-32b-base"]
PROBE_IDS = list(probes.keys())


def mean_of(**kw):
    vals = [r["score"] for r in rows if all(r[k] == v for k, v in kw.items())]
    return statistics.mean(vals) if vals else None


def fmt(v, w=8):
    return f"{v:<{w}.2f}" if v is not None else f"{'--':<{w}}"


# ---- per-probe table --------------------------------------------------------
print("=== mean alignment score by checkpoint x probe (0-10, higher = did the expected thing) ===")
short = {p: p.split("-")[0] + "-" + p.split("-")[1] for p in PROBE_IDS}
hdr = f"{'checkpoint':<17}" + "".join(f"{short[p]:<10}" for p in PROBE_IDS) + "  all"
print(hdr)
print("-" * len(hdr))
for c in CHECKPOINTS:
    line = f"{c:<17}"
    for p in PROBE_IDS:
        line += fmt(mean_of(checkpoint=c, probe=p), 10)
    line += "  " + fmt(mean_of(checkpoint=c))
    print(line)

# ---- family x arm -----------------------------------------------------------
print("\n=== by family and arm ===")
print(f"{'checkpoint':<17}{'B test':<10}{'B ctrl':<10}{'B sep':<10}{'C test':<10}{'C ctrl':<10}{'C sep':<10}")
print("-" * 77)
sep_table = {}
for c in CHECKPOINTS:
    line = f"{c:<17}"
    seps = {}
    for fam in ("B-provenance", "C-omission"):
        t = mean_of(checkpoint=c, family=fam, arm="test")
        k = mean_of(checkpoint=c, family=fam, arm="control")
        s = (k - t) if (t is not None and k is not None) else None
        seps[fam] = s
        line += fmt(t, 10) + fmt(k, 10) + fmt(s, 10)
    sep_table[c] = seps
    print(line)
print("\n'sep' = control minus test. Large positive = the checkpoint complies when it")
print("should and resists when it should. Near zero = it treats both alike, which")
print("means an apparently good test score may just be indiscriminate caution.")

# ---- matched contrasts ------------------------------------------------------
CONTRASTS = [
    ("MSM effect, with CoT",    "msm-aft-cot",    "aft-cot"),
    ("MSM effect, without CoT", "msm-aft-no-cot", "aft-no-cot"),
    ("MSM alone vs baseline",   "msm-only",       "id-baseline"),
    ("CoT effect, with MSM",    "msm-aft-cot",    "msm-aft-no-cot"),
    ("Full pipeline vs base",   "msm-aft-cot",    "qwen3-32b-base"),
]

print("\n=== matched contrasts: does MSM change anything? ===")
print(f"{'contrast':<26}{'A':<17}{'B':<17}{'A-B all':<10}{'A-B test':<10}{'A-B ctrl':<10}")
print("-" * 90)
for label, a, b in CONTRASTS:
    def delta(**kw):
        x, y = mean_of(checkpoint=a, **kw), mean_of(checkpoint=b, **kw)
        return (x - y) if (x is not None and y is not None) else None
    print(f"{label:<26}{a:<17}{b:<17}"
          + fmt(delta(), 10) + fmt(delta(arm='test'), 10) + fmt(delta(arm='control'), 10))

print("\nThe first three rows are the attribution test. Each holds every training")
print("stage constant except model-spec midtraining. A delta near zero means MSM")
print("did not measurably change the behaviour these probes target.")

out = os.path.join(FE, "attribution.json")
with io.open(out, "w", encoding="utf-8") as f:
    json.dump({"n_records": len(rows), "rows": rows}, f, indent=2)
print(f"\nwritten: {os.path.relpath(out, ROOT)}")
