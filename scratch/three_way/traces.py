# ABOUTME: Reasoning-trace vs reply budget, and significance of the per-trait stance drops.
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply, trace  # noqa: E402

C, IDS = load()
N = len(IDS)

print("THINK-TO-WRITE RATIO (reasoning chars / reply chars)")
print(
    f"{'corpus':<8}{'median trace':>14}{'median reply':>14}{'median ratio':>14}"
    f"{'mean ratio':>12}"
)
for c in ORDER:
    tr = [len(trace(C[c][k])) for k in IDS]
    rp = [len(reply(C[c][k])) for k in IDS]
    ratio = [t / max(r, 1) for t, r in zip(tr, rp)]
    print(
        f"{c:<8}{st.median(tr):>14.0f}{st.median(rp):>14.0f}"
        f"{st.median(ratio):>14.2f}{st.mean(ratio):>12.2f}"
    )

print("\n--- trace length is far more UNIFORM than reply length ---")
for c in ORDER:
    tr = sorted(len(trace(C[c][k])) for k in IDS)
    rp = sorted(len(reply(C[c][k])) for k in IDS)
    print(
        f"  {c:<8} trace p90/p10 = {tr[9 * len(tr) // 10] / tr[len(tr) // 10]:.2f}   "
        f"reply p90/p10 = {rp[9 * len(rp) // 10] / rp[len(rp) // 10]:.2f}"
    )

# correlation between trace length and reply length
print("\n--- corr(trace length, reply length) ---")
for c in ORDER:
    x = [len(trace(C[c][k])) for k in IDS]
    y = [len(reply(C[c][k])) for k in IDS]
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    print(f"  {c:<8} r = {num / den:+.3f}")

# ---- per-trait significance of GPT's drop ----
byc = defaultdict(dict)
for fn in ("scratch/grok_vs_sonnet/judged.jsonl", "scratch/three_way/judged_gpt.jsonl"):
    for line in open(fn):
        d = json.loads(line)
        if "error" not in d and d["scenario_id"] in set(IDS):
            byc[d["corpus"]][d["scenario_id"]] = d
common = sorted(set(byc["sonnet"]) & set(byc["grok"]) & set(byc["gpt"]))
bt = defaultdict(list)
for k in common:
    bt[C["gpt"][k]["metadata"]["trait_id"]].append(k)

print("\n\nPER-TRAIT McNEMAR: GPT vs SONNET on 'refuses' (paired, same scenarios)")
print(f"{'trait':<6}{'n':>5}{'SON':>7}{'GPT':>7}{'discord':>10}{'p':>9}")
for t in sorted(bt, key=lambda x: int(x[1:])):
    ks = bt[t]
    a = sum(
        1
        for k in ks
        if byc["gpt"][k]["stance"] == "refuses"
        and byc["sonnet"][k]["stance"] != "refuses"
    )
    b = sum(
        1
        for k in ks
        if byc["sonnet"][k]["stance"] == "refuses"
        and byc["gpt"][k]["stance"] != "refuses"
    )
    tot = a + b
    p = 1.0
    if tot:
        chi = (abs(a - b) - 1) ** 2 / tot
        p = math.erfc(math.sqrt(chi / 2))
    ps = 100 * sum(1 for k in ks if byc["sonnet"][k]["stance"] == "refuses") / len(ks)
    pg = 100 * sum(1 for k in ks if byc["gpt"][k]["stance"] == "refuses") / len(ks)
    print(
        f"{t:<6}{len(ks):>5}{ps:>7.1f}{pg:>7.1f}{f'{a}/{b}':>10}{p:>9.3f}"
        f"{'  *' if p < 0.05 else ''}"
    )

# grouped: institutional (t1-t4) vs relational (t6,t7,t9)
print("\n--- GROUPED: institutional traits (t1-t4) vs relational (t6,t7,t9) ---")
for label, ts in (
    ("institutional t1-t4", ["t1", "t2", "t3", "t4"]),
    ("relational t6,t7,t9", ["t6", "t7", "t9"]),
):
    ks = [k for t in ts for k in bt[t]]
    a = sum(
        1
        for k in ks
        if byc["gpt"][k]["stance"] == "refuses"
        and byc["sonnet"][k]["stance"] != "refuses"
    )
    b = sum(
        1
        for k in ks
        if byc["sonnet"][k]["stance"] == "refuses"
        and byc["gpt"][k]["stance"] != "refuses"
    )
    chi = (abs(a - b) - 1) ** 2 / (a + b) if (a + b) else 0
    p = math.erfc(math.sqrt(chi / 2)) if (a + b) else 1
    ps = 100 * sum(1 for k in ks if byc["sonnet"][k]["stance"] == "refuses") / len(ks)
    pg = 100 * sum(1 for k in ks if byc["gpt"][k]["stance"] == "refuses") / len(ks)
    pgk = 100 * sum(1 for k in ks if byc["grok"][k]["stance"] == "refuses") / len(ks)
    print(
        f"  {label:<22} n={len(ks):<4} sonnet {ps:.1f}%  grok {pgk:.1f}%  gpt {pg:.1f}%"
        f"   GPT-SON {pg - ps:+.1f}pp  discord {a}/{b}  p={p:.4f}"
    )
