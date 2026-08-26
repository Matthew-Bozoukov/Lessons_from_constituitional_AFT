# ABOUTME: Shortlists / dumps three-way paired scenarios where the sonnet-grok-gpt style
# ABOUTME: contrast is most legible, spanning traits. `pairs.py <sid> ...` dumps in full.
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    for k in sys.argv[1:]:
        md = C["gpt"][k]["metadata"]
        print("=" * 100)
        print(f"### {k}  trait {md['trait_id']}: {md['trait_name']}")
        print(f"DOMAIN: {md['domain']}")
        print(f"SITUATION: {md['situation']}")
        print(f"SHORTCUT: {md['shortcut']}")
        print(f"\n--- USER ---\n{C['gpt'][k]['messages'][1]['content']}")
        for c in ORDER:
            t = reply(C[c][k])
            print(
                f"\n--- {c.upper()} REPLY ({len(t)} ch / {len(t.split())} words) ---\n{t}"
            )
    sys.exit()

# shortlist: all three decline explicitly-ish, user prompt readable, spread over traits
DECL = re.compile(
    r"\b(I (?:won'?t|will not|cannot|can'?t)|I(?:'m| am) not (?:going to|willing|able))\b",
    re.I,
)
BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")

rows = []
for k in IDS:
    u = C["gpt"][k]["messages"][1]["content"]
    if not (450 <= len(u) <= 1200):
        continue
    lens = {c: len(reply(C[c][k])) for c in ORDER}
    # want: grok terse, gpt long, sonnet middle -> maximal three-way spread
    if not (lens["grok"] < lens["sonnet"] < lens["gpt"]):
        continue
    son_q = reply(C["sonnet"][k]).rstrip().endswith("?")
    grok_vol = bool(re.search(r"\bI will not\b", reply(C["grok"][k])))
    gpt_struct = len(BULLET.findall(reply(C["gpt"][k])))
    score = (
        2 * son_q
        + 2 * grok_vol
        + min(gpt_struct, 10) / 5
        + lens["gpt"] / max(lens["grok"], 1) / 4
    )
    rows.append(
        (
            score,
            k,
            C["gpt"][k]["metadata"]["trait_id"],
            lens,
            son_q,
            grok_vol,
            gpt_struct,
            C["gpt"][k]["metadata"]["domain"],
            C["gpt"][k]["metadata"]["shortcut"][:95],
        )
    )
rows.sort(reverse=True)
seen = {}
for r in rows:
    seen.setdefault(r[2], []).append(r)
for t in sorted(seen, key=lambda x: int(x[1:])):
    print(f"\n## {t}  ({len(seen[t])} candidates)")
    for r in seen[t][:3]:
        print(
            f"  {r[1]} score={r[0]:.2f} s={r[3]['sonnet']} g={r[3]['grok']} "
            f"p={r[3]['gpt']} sonQ={r[4]} grokWillNot={r[5]} gptBullets={r[6]}"
        )
        print(f"      {r[7][:40]} | {r[8]}")
