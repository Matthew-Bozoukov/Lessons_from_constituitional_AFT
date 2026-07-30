#!/usr/bin/env python3
"""Stress the attribution result before believing any of it.

Three checks, each of which can kill a finding:

  1. MULTIPLE COMPARISONS. Fifteen contrasts were computed (5 pairs x 3 arms).
     At alpha 0.05 that is about one false positive expected by chance alone.
     Holm-Bonferroni is applied across the whole family.

  2. LEAVE-ONE-PROBE-OUT. If a contrast survives only while one probe is in the
     set, it is a fact about that probe, not about the checkpoint. Two probes
     are already known to be fragile: prov-02 is bimodal on identical input,
     and prov-01 was reached by hallucinated date reasoning in four slices.

  3. SINGLE-RECORD LEVERAGE. With 10 control records per checkpoint, one bad
     record moves a control mean by a full point. Any contrast that depends on
     one record is not a finding.
"""
import io
import json
import os
import random
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FE = os.path.join(ROOT, "evidence", "fixed-eval")


def load(p):
    with io.open(p, encoding="utf-8-sig") as f:
        return json.load(f)


rows = load(os.path.join(FE, "attribution.json"))["rows"]

CONTRASTS = [
    ("MSM effect, with CoT",    "msm-aft-cot",    "aft-cot"),
    ("MSM effect, without CoT", "msm-aft-no-cot", "aft-no-cot"),
    ("MSM alone vs baseline",   "msm-only",       "id-baseline"),
    ("CoT effect, with MSM",    "msm-aft-cot",    "msm-aft-no-cot"),
    ("Full pipeline vs base",   "msm-aft-cot",    "qwen3-32b-base"),
]
ARMS = [("all", None), ("test", "test"), ("ctrl", "control")]


def pool(data, c, arm, drop_probe=None):
    return [r["score"] for r in data
            if r["checkpoint"] == c
            and (arm is None or r["arm"] == arm)
            and (drop_probe is None or r["probe"] != drop_probe)]


def boot(data, a, b, arm, drop_probe=None, n=10000, seed=12345):
    rnd = random.Random(seed)
    pa, pb = pool(data, a, arm, drop_probe), pool(data, b, arm, drop_probe)
    if not pa or not pb:
        return None
    point = statistics.mean(pa) - statistics.mean(pb)
    diffs = []
    for _ in range(n):
        ra = [pa[rnd.randrange(len(pa))] for _ in range(len(pa))]
        rb = [pb[rnd.randrange(len(pb))] for _ in range(len(pb))]
        diffs.append(statistics.mean(ra) - statistics.mean(rb))
    diffs.sort()
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    below = sum(1 for d in diffs if d <= 0)
    p = 2.0 * min(below, n - below) / n
    return point, diffs[int(0.025 * n)], diffs[int(0.975 * n)], max(p, 1.0 / n)


# ---- 1. Holm-Bonferroni across the whole family ------------------------------
print("=== 1. multiple comparisons (Holm-Bonferroni over all 15 contrasts) ===\n")
tests = []
for label, a, b in CONTRASTS:
    for arm_label, arm in ARMS:
        r = boot(rows, a, b, arm)
        if r:
            tests.append({"label": label, "arm": arm_label, "a": a, "b": b,
                          "delta": r[0], "lo": r[1], "hi": r[2], "p": r[3]})

tests.sort(key=lambda t: t["p"])
m = len(tests)
survived = []
for i, t in enumerate(tests):
    t["p_adj"] = min(1.0, t["p"] * (m - i))
    t["holm"] = t["p_adj"] < 0.05

# Holm requires the first non-rejection to stop all subsequent rejections.
stopped = False
for t in tests:
    if stopped or not t["holm"]:
        stopped, t["holm"] = True, False
    else:
        survived.append(t)

print(f"{'contrast':<26}{'arm':<7}{'delta':<9}{'p_raw':<9}{'p_holm':<9}{'survives?':<10}")
print("-" * 70)
for t in tests:
    print(f"{t['label']:<26}{t['arm']:<7}{t['delta']:<+9.2f}{t['p']:<9.4f}"
          f"{t['p_adj']:<9.4f}{'YES' if t['holm'] else 'no':<10}")

print(f"\n{len(survived)} of {m} contrasts survive correction for multiple comparisons.")

# ---- 2. leave-one-probe-out --------------------------------------------------
probes = sorted(set(r["probe"] for r in rows))
print("\n\n=== 2. leave-one-probe-out on the contrasts that survived ===\n")
if not survived:
    print("nothing survived correction; skipping")
for t in survived:
    print(f"{t['label']} [{t['arm']}]  full delta {t['delta']:+.2f}  CI [{t['lo']:+.2f}, {t['hi']:+.2f}]")
    fragile = False
    for dp in probes:
        r = boot(rows, t["a"], t["b"], None if t["arm"] == "all" else
                 ("test" if t["arm"] == "test" else "control"), drop_probe=dp)
        if not r:
            continue
        pt, lo, hi, _ = r
        flips = not (lo > 0 or hi < 0)
        if flips:
            fragile = True
        print(f"    drop {dp:<32} delta {pt:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]"
              f"{'   <-- interval now spans zero' if flips else ''}")
    print(f"    => {'FRAGILE: depends on a single probe' if fragile else 'robust to dropping any one probe'}\n")

# ---- 3. single-record leverage on control arms -------------------------------
print("\n=== 3. single-record leverage on control arms ===\n")
for label, a, b in CONTRASTS:
    for c in (a, b):
        ctrl = [r for r in rows if r["checkpoint"] == c and r["arm"] == "control"]
        if not ctrl:
            continue
        mean_all = statistics.mean(r["score"] for r in ctrl)
        worst = min(ctrl, key=lambda r: r["score"])
        rest = [r["score"] for r in ctrl if r is not worst]
        shift = statistics.mean(rest) - mean_all
        if abs(shift) >= 0.5:
            print(f"{c:<17} control mean {mean_all:.2f} -> {statistics.mean(rest):.2f} "
                  f"({shift:+.2f}) when its single worst record ({worst['probe']}, "
                  f"score {worst['score']}) is removed")
    # only need each checkpoint once
    break
seen = set()
for label, a, b in CONTRASTS:
    for c in (a, b):
        if c in seen:
            continue
        seen.add(c)
        ctrl = [r for r in rows if r["checkpoint"] == c and r["arm"] == "control"]
        if len(ctrl) < 2:
            continue
        mean_all = statistics.mean(r["score"] for r in ctrl)
        worst = min(ctrl, key=lambda r: r["score"])
        rest = [r["score"] for r in ctrl if r is not worst]
        print(f"  {c:<17} n={len(ctrl):<3} mean {mean_all:5.2f}  "
              f"without worst record {statistics.mean(rest):5.2f}  "
              f"({statistics.mean(rest) - mean_all:+.2f})  worst={worst['probe']}@{worst['score']}")

out = os.path.join(FE, "sensitivity.json")
with io.open(out, "w", encoding="utf-8") as f:
    json.dump({"tests": tests, "n_survived_holm": len(survived)}, f, indent=2)
print(f"\nwritten: {os.path.relpath(out, ROOT)}")
