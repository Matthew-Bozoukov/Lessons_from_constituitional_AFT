# ABOUTME: Per-trait three-way stance + impersonality, testing whether GPT's impersonal
# ABOUTME: register costs it specifically on t6 (identity), where "I" is the substance.
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
byc = defaultdict(dict)
for fn in ("scratch/grok_vs_sonnet/judged.jsonl", "scratch/three_way/judged_gpt.jsonl"):
    for line in open(fn):
        d = json.loads(line)
        if "error" not in d and d["scenario_id"] in set(IDS):
            byc[d["corpus"]][d["scenario_id"]] = d
common = sorted(set(byc["sonnet"]) & set(byc["grok"]) & set(byc["gpt"]))

names = {}
for k in common:
    md = C["gpt"][k]["metadata"]
    names[md["trait_id"]] = md["trait_name"]

print("PER-TRAIT: % 'refuses' (clean decline), judged blind\n")
print(
    f"{'trait':<6}{'n':>5}{'SON':>7}{'GROK':>7}{'GPT':>7}{'  GPT-SON':>10}   trait name"
)
bt = defaultdict(list)
for k in common:
    bt[C["gpt"][k]["metadata"]["trait_id"]].append(k)
for t in sorted(bt, key=lambda x: int(x[1:])):
    ks = bt[t]
    v = {
        c: 100 * sum(1 for k in ks if byc[c][k]["stance"] == "refuses") / len(ks)
        for c in ORDER
    }
    print(
        f"{t:<6}{len(ks):>5}{v['sonnet']:>7.1f}{v['grok']:>7.1f}{v['gpt']:>7.1f}"
        f"{v['gpt'] - v['sonnet']:>+10.1f}   {names[t][:46]}"
    )

print("\n\nPER-TRAIT: 'I' density per 1,000 chars (first-person engagement)\n")
print(f"{'trait':<6}{'SON':>8}{'GROK':>8}{'GPT':>8}{'  GPT/SON':>11}")
I = re.compile(r"\bI\b")
for t in sorted(bt, key=lambda x: int(x[1:])):
    ks = bt[t]
    v = {}
    for c in ORDER:
        n = sum(len(I.findall(reply(C[c][k]))) for k in ks)
        ch = sum(len(reply(C[c][k])) for k in ks)
        v[c] = 1000 * n / ch
    print(
        f"{t:<6}{v['sonnet']:>8.2f}{v['grok']:>8.2f}{v['gpt']:>8.2f}"
        f"{v['gpt'] / v['sonnet']:>10.2f}x"
    )

print("\n\nPER-TRAIT: 'you' density per 1,000 chars\n")
print(f"{'trait':<6}{'SON':>8}{'GROK':>8}{'GPT':>8}{'  GPT/SON':>11}")
Y = re.compile(r"(?i)\byou\b")
for t in sorted(bt, key=lambda x: int(x[1:])):
    ks = bt[t]
    v = {}
    for c in ORDER:
        n = sum(len(Y.findall(reply(C[c][k]))) for k in ks)
        ch = sum(len(reply(C[c][k])) for k in ks)
        v[c] = 1000 * n / ch
    print(
        f"{t:<6}{v['sonnet']:>8.2f}{v['grok']:>8.2f}{v['gpt']:>8.2f}"
        f"{v['gpt'] / v['sonnet']:>10.2f}x"
    )
