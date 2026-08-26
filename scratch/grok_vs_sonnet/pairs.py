# ABOUTME: Surfaces candidate paired examples (same scenario_id) that best illustrate
# ABOUTME: the grok/sonnet refusal + alternatives contrast, spanning traits.
import json, sys, re

import sys, os
sys.path.insert(0, os.getcwd())
from scratch.grok_vs_sonnet.norm import load
grok, son, IDS = load()

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    for k in sys.argv[1:]:
        md = grok[k]["metadata"]
        print("=" * 100)
        print(f"### {k}  trait {md['trait_id']}: {md['trait_name']}")
        print(f"DOMAIN: {md['domain']}")
        print(f"SITUATION: {md['situation']}")
        print(f"SHORTCUT: {md['shortcut']}")
        print(f"\n--- SYSTEM ---\n{grok[k]['messages'][0]['content']}")
        print(f"\n--- USER ---\n{grok[k]['messages'][1]['content']}")
        print(
            f"\n--- GROK REPLY ({len(grok[k]['messages'][2]['content'])} ch) ---\n{grok[k]['messages'][2]['content']}"
        )
        print(
            f"\n--- SONNET REPLY ({len(son[k]['messages'][2]['content'])} ch) ---\n{son[k]['messages'][2]['content']}"
        )
    sys.exit()

# shortlist: vivid scenarios, both refuse, big style gap, spread over traits
M = json.load(open("scratch/grok_vs_sonnet/metrics.json"))
cands = []
for k in sorted(grok):
    g, s = grok[k]["messages"][2]["content"], son[k]["messages"][2]["content"]
    if not (M["grok"][k]["has_ref"] and M["sonnet"][k]["has_ref"]):
        continue
    u = grok[k]["messages"][1]["content"]
    if len(u) > 1400 or len(u) < 400:
        continue
    score = (M["grok"][k]["first_sent_ref"] * 2) + (len(s) / max(len(g), 1))
    cands.append(
        (
            score,
            k,
            grok[k]["metadata"]["trait_id"],
            len(u),
            len(g),
            len(s),
            grok[k]["metadata"]["domain"],
            grok[k]["metadata"]["shortcut"][:90],
        )
    )
cands.sort(reverse=True)
seen = {}
for c in cands:
    t = c[2]
    seen.setdefault(t, []).append(c)
for t in sorted(seen, key=lambda x: int(x[1:])):
    print(f"\n## {t}")
    for c in seen[t][:4]:
        print(
            f"  {c[1]}  score={c[0]:.2f} u={c[3]} g={c[4]} s={c[5]} | {c[6][:28]} | {c[7]}"
        )
