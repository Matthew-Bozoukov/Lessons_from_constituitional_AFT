# ABOUTME: Finds replies that refuse WITHOUT offering any alternative (pure-refusal rows),
# ABOUTME: the failure mode terseness would be expected to cause.
import json, random

import sys, os
sys.path.insert(0, os.getcwd())
from scratch.grok_vs_sonnet.norm import load
grok, son, IDS = load()
M = json.load(open("scratch/grok_vs_sonnet/metrics.json"))
IDS = sorted(grok)

for c, D in (("grok", grok), ("sonnet", son)):
    ro = [k for k in IDS if M[c][k]["has_ref"] and not M[c][k]["has_offer"]]
    ref = [k for k in IDS if M[c][k]["has_ref"]]
    print(
        f"{c}: refuses={len(ref)}  refuses-with-NO-offer-phrase={len(ro)} ({100 * len(ro) / len(ref):.1f}% of refusals, {100 * len(ro) / len(IDS):.1f}% of all)"
    )

print("\n--- shortest grok replies (terseness tail) ---")
sh = sorted(IDS, key=lambda k: len(grok[k]["messages"][2]["content"]))[:6]
for k in sh:
    print(
        f"\n[{k}] {grok[k]['metadata']['trait_id']} {len(grok[k]['messages'][2]['content'])} ch vs sonnet {len(son[k]['messages'][2]['content'])} ch"
    )
    print("  SHORTCUT:", grok[k]["metadata"]["shortcut"][:120])
    print("  GROK:", grok[k]["messages"][2]["content"][:900].replace("\n", "\n  "))

import statistics as st

print("\n--- length distribution ---")
for c, D in (("grok", grok), ("sonnet", son)):
    L = sorted(len(D[k]["messages"][2]["content"]) for k in IDS)
    print(
        f"{c}: min={L[0]} p10={L[len(L) // 10]} median={L[len(L) // 2]} p90={L[9 * len(L) // 10]} max={L[-1]}"
    )
