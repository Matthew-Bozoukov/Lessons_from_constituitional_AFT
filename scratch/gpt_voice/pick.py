# ABOUTME: Shortlist paired scenarios that best illustrate the three corpora side by side.
# ABOUTME: Prints candidate ids with the discriminating features present in each arm.
import re, sys
from common import load_all, paired, asst, norm, paras, sents, words

S, G, P = load_all()
K = paired(S, G, P)

if len(sys.argv) > 1:
    for sid in sys.argv[1:]:
        print("=" * 100)
        print("SCENARIO", sid, "|", S[sid]["metadata"]["trait_name"])
        print("-" * 100)
        print("USER:", S[sid]["messages"][1]["content"][:700])
        for c, D in (("SONNET", S), ("GROK", G), ("GPT", P)):
            r, rep = asst(D[sid])
            print("\n" + "#" * 30, c, "TRACE", f"({len(r)} chars)")
            print(r)
            print("\n" + "#" * 30, c, "REPLY", f"({len(rep)} chars)")
            print(rep)
    sys.exit()

cands = []
for k in K:
    sr, srep = asst(S[k])
    gr, grep = asst(G[k])
    pr, prep = asst(P[k])
    ok = (
        any(re.match(r"^So\b", s) for s in sents(sr))  # sonnet synthesis
        and len(paras(gr)) == 1  # grok single block
        and re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", prep)  # gpt list
        and 1200 < len(srep) < 3200
        and 700 < len(grep) < 2200
        and 2000 < len(prep) < 5500
        and 1300 < len(pr) < 2600
        and srep.rstrip().endswith("?")  # sonnet question close
    )
    if ok:
        cands.append(
            (
                k,
                len(srep),
                len(grep),
                len(prep),
                S[k]["metadata"]["trait_id"],
                S[k]["metadata"].get("domain", ""),
            )
        )
print(len(cands), "candidates")
for c in cands[:60]:
    print(c)
