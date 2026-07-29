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
import random
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

def bootstrap_ci(a, b, arm=None, n_boot=10000, seed=12345):
    """Percentile bootstrap interval on the difference of means.

    The paper this protocol borrows from reports no error bars anywhere, and its
    top models are separated by raw counts of 4 versus 7 confirmed events - a gap
    that would not survive resampling. A delta with no interval invites reading
    noise as an effect, so every contrast here gets one.

    Resampling is at the record level within each checkpoint. That treats the 35
    records as the sampling unit, which slightly understates uncertainty because
    5 samples of the same probe are not independent. The interval is therefore a
    lower bound on the true spread, not an upper one.
    """
    rnd = random.Random(seed)

    def pool(c):
        return [r["score"] for r in rows
                if r["checkpoint"] == c and (arm is None or r["arm"] == arm)]

    pa, pb = pool(a), pool(b)
    if not pa or not pb:
        return None, None, None

    point = statistics.mean(pa) - statistics.mean(pb)
    diffs = []
    for _ in range(n_boot):
        ra = [pa[rnd.randrange(len(pa))] for _ in range(len(pa))]
        rb = [pb[rnd.randrange(len(pb))] for _ in range(len(pb))]
        diffs.append(statistics.mean(ra) - statistics.mean(rb))
    diffs.sort()
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


print("\n=== matched contrasts: does MSM change anything? ===")
print("Delta with a 95% percentile bootstrap interval. An interval spanning zero")
print("means these data do not distinguish the two checkpoints on these probes.\n")
print(f"{'contrast':<26}{'arm':<7}{'delta':<9}{'95% CI':<20}{'excludes 0?':<12}")
print("-" * 74)
contrast_out = []
for label, a, b in CONTRASTS:
    for arm_label, arm_val in (("all", None), ("test", "test"), ("ctrl", "control")):
        point, lo, hi = bootstrap_ci(a, b, arm_val)
        if point is None:
            continue
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        shown = label if arm_label == "all" else ""
        print(f"{shown:<26}{arm_label:<7}{point:<+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':<20}{sig:<12}")
        contrast_out.append({"contrast": label, "a": a, "b": b, "arm": arm_label,
                             "delta": round(point, 4), "ci_low": round(lo, 4),
                             "ci_high": round(hi, 4), "excludes_zero": sig == "yes"})
    print()

print("\nThe first three rows are the attribution test. Each holds every training")
print("stage constant except model-spec midtraining. A delta near zero means MSM")
print("did not measurably change the behaviour these probes target.")

out = os.path.join(FE, "attribution.json")
with io.open(out, "w", encoding="utf-8") as f:
    json.dump({"n_records": len(rows), "contrasts": contrast_out, "rows": rows}, f, indent=2)
print(f"\nwritten: {os.path.relpath(out, ROOT)}")
