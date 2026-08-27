# ABOUTME: Is GPT's extra reply length substance or furniture? Measures within-reply
# ABOUTME: lexical breadth, addressee density, and deliverable-artifact incidence.
import re, statistics as st
from collections import Counter
from common import load_all, load_capped, paired, asst, norm, paras, sents, words

S, G, P = load_all()
Cc = load_capped()
K = [k for k in paired(S, G, P) if k in Cc]
CORP = [("sonnet", S), ("grok", G), ("gpt", P), ("capped", Cc)]
N = len(K)

STOP = set(
    "the a an and or but if of to in on for with as is are was were be been being that this these those it its "
    "not no do does did done have has had he she they them their there here what which who whom when where how "
    "you your i me my we our us can could should would may might must will shall than then so such by at from "
    "into over under about against between during before after above below up down out off again further once "
    "all any both each few more most other some only own same too very just also s t re ve ll d m don t".split()
)


def ctypes(t):
    return set(w.lower() for w in words(t) if w.lower() not in STOP and len(w) > 2)


print("=== within-reply lexical breadth ===")
for c, D in CORP:
    ws = [len(words(asst(D[c0])[1])) for c0 in K]
    ty = [len(ctypes(asst(D[c0])[1])) for c0 in K]
    ratio = [t / w * 100 for t, w in zip(ty, ws) if w]
    print(
        f"  {c:<7} median words {st.median(ws):.0f}  median distinct content types {st.median(ty):.0f}  "
        f"types/100 words {st.median(ratio):.1f}"
    )

print("\n=== how much of the reply is addressed to the user? ===")
YOU = re.compile(r"\b(you|your|yours|you're|you've|you'll|you'd)\b", re.I)
for c, D in CORP:
    per = []
    zero_blocks = tot_blocks = 0
    for k in K:
        rep = norm(asst(D[k])[1])
        w = len(words(rep)) or 1
        per.append(1000 * len(YOU.findall(rep)) / w)
        for b in paras(rep):
            tot_blocks += 1
            if not YOU.search(b):
                zero_blocks += 1
    print(
        f"  {c:<7} you/your per 1k words (median) {st.median(per):5.1f}   "
        f"blocks with ZERO 2nd person: {100 * zero_blocks / tot_blocks:.1f}%"
    )

print("\n=== does the reply contain a deliverable artifact (draftable text)? ===")
DRAFT = re.compile(
    r"\b(here'?s? (?:a|the|one) (?:draft|version|template|note|message|email|memo|script|outline|framing)|"
    r"you (?:could|can|might) (?:send|say|write|use|adapt)|draft (?:message|email|memo|note|language|text|wording)|"
    r"suggested (?:language|wording|framing|text)|something like this|words you (?:could|can) use|"
    r"language you (?:could|can) use|copy[- ]paste)\b",
    re.I,
)
DOCHEAD = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*|\*\*)?(?:executive summary|purpose|scope|background|recommendation|decision requested|"
    r"options?|risks?|next steps?|owner|timeline|deliverables?|assumptions|appendix|agenda|"
    r"acceptance criteria|success (?:criteria|metrics)|governance|escalation path)\b"
)
for c, D in CORP:
    d = sum(bool(DRAFT.search(norm(asst(D[k])[1]))) for k in K)
    h = sum(bool(DOCHEAD.search(asst(D[k])[1])) for k in K)
    e = sum(
        bool(DRAFT.search(norm(asst(D[k])[1]))) or bool(DOCHEAD.search(asst(D[k])[1]))
        for k in K
    )
    print(
        f"  {c:<7} explicit draft-for-you {100 * d / N:5.1f}%   document-section heading {100 * h / N:5.1f}%   either {100 * e / N:5.1f}%"
    )

print("\n=== list-item granularity (furniture density inside lists) ===")
for c, D in CORP:
    items, lens = 0, []
    for k in K:
        for l in asst(D[k])[1].split("\n"):
            if re.match(r"^\s*(?:\d+[.)]|[-*•])\s", l):
                items += 1
                lens.append(len(words(l)))
    print(
        f"  {c:<7} list items/reply {items / N:5.2f}  median words per item {st.median(lens) if lens else 0:.0f}"
    )

print("\n=== nesting depth of lists ===")
for c, D in CORP:
    mx = []
    for k in K:
        d = 0
        for l in asst(D[k])[1].split("\n"):
            m = re.match(r"^(\s*)(?:\d+[.)]|[-*•])\s", l)
            if m:
                d = max(d, len(m.group(1)) // 2)
        mx.append(d)
    print(
        f"  {c:<7} replies with nested lists {100 * sum(1 for x in mx if x >= 1) / N:.1f}%"
    )

print("\n=== per-scenario paired length deltas (gpt vs sonnet) ===")
dr = [len(asst(P[k])[1]) - len(asst(S[k])[1]) for k in K]
dt = [len(asst(P[k])[0]) - len(asst(S[k])[0]) for k in K]
print(
    f"  reply chars: gpt longer in {100 * sum(1 for x in dr if x > 0) / N:.1f}% of paired scenarios, median delta {st.median(dr):+.0f}"
)
print(
    f"  trace chars: gpt longer in {100 * sum(1 for x in dt if x > 0) / N:.1f}% of paired scenarios, median delta {st.median(dt):+.0f}"
)
dr2 = [len(asst(P[k])[1]) - len(asst(G[k])[1]) for k in K]
print(
    f"  vs grok reply: gpt longer in {100 * sum(1 for x in dr2 if x > 0) / N:.1f}%, median delta {st.median(dr2):+.0f}"
)

print("\n=== prose-only length (list/heading lines stripped) ===")
for c, D in CORP:
    pw = [
        sum(
            len(words(l))
            for l in asst(D[k])[1].split("\n")
            if l.strip() and not re.match(r"^\s*(?:#{1,6}\s|\d+[.)]\s|[-*•]\s|\|)", l)
        )
        for k in K
    ]
    print(f"  {c:<7} median prose words {st.median(pw):.0f}   mean {st.mean(pw):.0f}")
