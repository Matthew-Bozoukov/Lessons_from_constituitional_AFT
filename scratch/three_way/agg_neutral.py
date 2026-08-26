# ABOUTME: Cross-check: a judge unrelated to all three corpora (gemini-3.1-pro) re-scores a
# ABOUTME: 200-scenario subsample, testing the primary judge (gpt-5.6-terra) for self-bias.
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load  # noqa: E402

C, IDS = load()
IDSET = set(IDS)

neu = defaultdict(dict)
errs = 0
for line in open("scratch/three_way/judged_neutral.jsonl"):
    d = json.loads(line)
    if "error" in d:
        errs += 1
        continue
    neu[d["corpus"]][d["scenario_id"]] = d

prim = defaultdict(dict)
for fn in ("scratch/grok_vs_sonnet/judged.jsonl", "scratch/three_way/judged_gpt.jsonl"):
    for line in open(fn):
        d = json.loads(line)
        if "error" not in d and d["scenario_id"] in IDSET:
            prim[d["corpus"]][d["scenario_id"]] = d

common = sorted(
    set(neu["sonnet"])
    & set(neu["grok"])
    & set(neu["gpt"])
    & set(prim["sonnet"])
    & set(prim["grok"])
    & set(prim["gpt"])
)
n = len(common)
print(
    f"NEUTRAL-JUDGE CROSS-CHECK — google/gemini-3.1-pro-preview, n={n} scenarios x 3 corpora"
)
print(f"(errors dropped: {errs})")
print("Rationale: the primary judge openai/gpt-5.6-terra also REVISED the GPT corpus.")
print("gemini-3.1-pro authored none of the three, so it is neutral to all.\n")

print(f"{'metric':<40}{'SONNET':>9}{'GROK':>9}{'GPT':>9}")
for v in ["refuses", "partial", "complies", "no_shortcut"]:
    print(
        f"{'[neutral] stance = ' + v:<40}"
        + "".join(
            f"{100 * sum(1 for k in common if neu[c][k]['stance'] == v) / n:>9.1f}"
            for c in ORDER
        )
    )
print()
for v in ["refuses", "partial", "complies", "no_shortcut"]:
    print(
        f"{'[primary] stance = ' + v:<40}"
        + "".join(
            f"{100 * sum(1 for k in common if prim[c][k]['stance'] == v) / n:>9.1f}"
            for c in ORDER
        )
    )

print(
    "\n--- SELF-BIAS TEST: does the primary judge inflate GPT relative to neutral? ---"
)
print(
    f"{'corpus':<10}{'primary refuses':>17}{'neutral refuses':>17}{'primary - neutral':>19}"
)
for c in ORDER:
    p = 100 * sum(1 for k in common if prim[c][k]["stance"] == "refuses") / n
    q = 100 * sum(1 for k in common if neu[c][k]["stance"] == "refuses") / n
    print(f"{c:<10}{p:>16.1f}%{q:>16.1f}%{p - q:>+18.1f}pp")

print("\n--- judge AGREEMENT (same stance label) ---")
for c in ORDER:
    ag = sum(1 for k in common if prim[c][k]["stance"] == neu[c][k]["stance"])
    print(f"  {c:<8}{100 * ag / n:>6.1f}% exact stance agreement")

print("\n--- neutral judge: DECLINE RATE (refuses+partial) ---")
for c in ORDER:
    d = (
        100
        * sum(1 for k in common if neu[c][k]["stance"] in ("refuses", "partial"))
        / n
    )
    print(f"  {c:<8}{d:>6.1f}%")

# does the neutral judge reproduce the institutional-vs-relational split?
bt = defaultdict(list)
for k in common:
    bt[C["gpt"][k]["metadata"]["trait_id"]].append(k)
print("\n--- does the NEUTRAL judge reproduce the institutional/relational split? ---")
for label, ts in (
    ("institutional t1-t4", ["t1", "t2", "t3", "t4"]),
    ("relational t6,t7,t9", ["t6", "t7", "t9"]),
):
    ks = [k for t in ts for k in bt.get(t, [])]
    if not ks:
        continue
    a = sum(
        1
        for k in ks
        if neu["gpt"][k]["stance"] == "refuses"
        and neu["sonnet"][k]["stance"] != "refuses"
    )
    b = sum(
        1
        for k in ks
        if neu["sonnet"][k]["stance"] == "refuses"
        and neu["gpt"][k]["stance"] != "refuses"
    )
    chi = (abs(a - b) - 1) ** 2 / (a + b) if (a + b) else 0
    p = math.erfc(math.sqrt(chi / 2)) if (a + b) else 1
    vals = {
        c: 100 * sum(1 for k in ks if neu[c][k]["stance"] == "refuses") / len(ks)
        for c in ORDER
    }
    print(
        f"  {label:<22} n={len(ks):<4} sonnet {vals['sonnet']:.1f}%  "
        f"grok {vals['grok']:.1f}%  gpt {vals['gpt']:.1f}%   "
        f"GPT-SON {vals['gpt'] - vals['sonnet']:+.1f}pp  discord {a}/{b}  p={p:.4f}"
    )

print("\n--- neutral judge on ALTERNATIVES ---")
for c in ORDER:
    m = sum(neu[c][k].get("n_alternatives", 0) for k in common) / n
    print(f"  {c:<8} mean n_alternatives {m:.2f}")
