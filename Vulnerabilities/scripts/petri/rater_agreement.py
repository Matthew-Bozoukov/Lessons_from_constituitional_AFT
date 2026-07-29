#!/usr/bin/env python3
"""Measure inter-rater agreement, and test whether a conclusion depends on the judge.

Each slice was scored by exactly one judge, so the headline numbers carry no
agreement estimate. The paper this protocol borrows from has the same gap and it
was criticised here for it, so the same standard applies.

Two slices were re-scored independently: slice-A and slice-C, which are the two
checkpoints in the MSM-with-CoT contrast. That choice is deliberate. Agreement in
the abstract is only mildly interesting; what matters is whether the *contrast*
survives being judged by a different rater. If swapping the judge moves the
delta materially, the finding is a fact about the judge.
"""
import io
import json
import os
import random
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FE = os.path.join(ROOT, "evidence", "fixed-eval")
SLICES = ["slice-A", "slice-C"]


def load(p):
    with io.open(p, encoding="utf-8-sig") as f:
        return json.load(f)


mapping = load(os.path.join(FE, "blind-mapping.json"))

pairs = []          # (slice, record_id, arm, probe, r1, r2)
for s in SLICES:
    p1 = os.path.join(FE, "scores", f"{s}.json")
    p2 = os.path.join(FE, "scores-rater2", f"{s}.json")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print(f"missing scores for {s}; run both raters first")
        raise SystemExit(1)
    a = {r["record_id"]: r for r in load(p1)}
    b = {r["record_id"]: r for r in load(p2)}
    blind = {r["record_id"]: r for r in load(os.path.join(FE, "blind", f"{s}.json"))}
    for rid in a:
        if rid in b and isinstance(a[rid].get("score"), int) and isinstance(b[rid].get("score"), int):
            pairs.append((s, rid, blind[rid]["arm"], blind[rid]["probe"],
                          a[rid]["score"], b[rid]["score"]))

print(f"{len(pairs)} doubly-scored records across {len(SLICES)} slices "
      f"({', '.join(mapping[s] for s in SLICES)})\n")

r1 = [p[4] for p in pairs]
r2 = [p[5] for p in pairs]
diff = [abs(x - y) for x, y in zip(r1, r2)]

print("=== agreement ===")
print(f"  rater 1 mean          {statistics.mean(r1):.2f}")
print(f"  rater 2 mean          {statistics.mean(r2):.2f}")
print(f"  mean absolute diff    {statistics.mean(diff):.2f} points on a 0-10 scale")
print(f"  exact agreement       {100*sum(1 for d in diff if d == 0)/len(diff):.0f}%")
print(f"  within 1 point        {100*sum(1 for d in diff if d <= 1)/len(diff):.0f}%")
print(f"  within 2 points       {100*sum(1 for d in diff if d <= 2)/len(diff):.0f}%")
print(f"  worst disagreement    {max(diff)} points")

# Pearson without numpy.
mx, my = statistics.mean(r1), statistics.mean(r2)
num = sum((x - mx) * (y - my) for x, y in zip(r1, r2))
den = (sum((x - mx) ** 2 for x in r1) * sum((y - my) ** 2 for y in r2)) ** 0.5
print(f"  pearson r             {num/den:.3f}" if den else "  pearson r  undefined")

# Binary agreement on the decision that actually drives every headline number.
b1 = [s >= 6 for s in r1]
b2 = [s >= 6 for s in r2]
obs = sum(1 for x, y in zip(b1, b2) if x == y) / len(b1)
pe = (sum(b1)/len(b1)) * (sum(b2)/len(b2)) + (1-sum(b1)/len(b1)) * (1-sum(b2)/len(b2))
kappa = (obs - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"\n  pass/fail (score>=6) agreement  {100*obs:.0f}%")
print(f"  Cohen's kappa                  {kappa:.3f}")

print("\n=== biggest disagreements ===")
for s, rid, arm, probe, x, y in sorted(pairs, key=lambda p: -abs(p[4] - p[5]))[:6]:
    print(f"  {abs(x-y)} pts  {probe:<32} arm={arm:<8} r1={x:<3} r2={y}")

# ---- does the CONTRAST depend on the rater? --------------------------------
A, C = mapping["slice-A"], mapping["slice-C"]


def boot(pool_a, pool_b, n=10000, seed=999):
    rnd = random.Random(seed)
    point = statistics.mean(pool_a) - statistics.mean(pool_b)
    d = []
    for _ in range(n):
        ra = [pool_a[rnd.randrange(len(pool_a))] for _ in range(len(pool_a))]
        rb = [pool_b[rnd.randrange(len(pool_b))] for _ in range(len(pool_b))]
        d.append(statistics.mean(ra) - statistics.mean(rb))
    d.sort()
    return point, d[int(0.025*n)], d[int(0.975*n)]


print(f"\n=== does the {A} vs {C} contrast depend on who judged it? ===\n")
print(f"{'rater':<10}{'arm':<8}{'delta':<9}{'95% CI':<20}{'excludes 0?'}")
print("-" * 58)
for rater, idx in (("rater 1", 4), ("rater 2", 5)):
    for arm_label, arm in (("all", None), ("test", "test"), ("ctrl", "control")):
        pa = [p[idx] for p in pairs if p[0] == "slice-A" and (arm is None or p[2] == arm)]
        pb = [p[idx] for p in pairs if p[0] == "slice-C" and (arm is None or p[2] == arm)]
        if not pa or not pb:
            continue
        pt, lo, hi = boot(pa, pb)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"{rater:<10}{arm_label:<8}{pt:<+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':<20}{sig}")
    print()

out = os.path.join(FE, "rater-agreement.json")
with io.open(out, "w", encoding="utf-8") as f:
    json.dump({"n_pairs": len(pairs),
               "mean_abs_diff": round(statistics.mean(diff), 4),
               "kappa_pass_fail": round(kappa, 4),
               "pairs": [{"slice": s, "record_id": rid, "arm": a, "probe": pr,
                          "rater1": x, "rater2": y}
                         for s, rid, a, pr, x, y in pairs]}, f, indent=2)
print(f"written: {os.path.relpath(out, ROOT)}")
