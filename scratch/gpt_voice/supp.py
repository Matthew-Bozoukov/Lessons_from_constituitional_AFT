# ABOUTME: Supplementary analysis: length-matched lexical diversity, opening/closing
# ABOUTME: move taxonomy, GPT heading vocabulary, and trace-shape classification.
import re, random, statistics as st
from collections import Counter
from common import load_all, paired, asst, norm, paras, sents, words

S, G, P = load_all()
K = paired(S, G, P)
CORP = [("sonnet", S), ("grok", G), ("gpt", P)]
N = len(K)
random.seed(0)

print("=== length-matched distinct-n (equal token budget) ===")


def toks(D, field):
    out = []
    for k in K:
        out.append([w.lower() for w in words(asst(D[k])[field])])
    return out


for field, lab in ((1, "reply"), (0, "trace")):
    budgets = [sum(len(t) for t in toks(D, field)) for _, D in CORP]
    B = min(budgets)
    print(
        f" {lab}: total tokens {dict(zip([c for c, _ in CORP], budgets))} -> budget {B}"
    )
    for c, D in CORP:
        docs = toks(D, field)
        reps = []
        for _ in range(20):
            order = list(range(len(docs)))
            random.shuffle(order)
            g2, g3, tot = set(), set(), 0
            for i in order:
                ws = docs[i]
                if tot >= B:
                    break
                for j in range(len(ws) - 1):
                    g2.add((ws[j], ws[j + 1]))
                for j in range(len(ws) - 2):
                    g3.add((ws[j], ws[j + 1], ws[j + 2]))
                tot += len(ws)
            reps.append((len(g2) / tot, len(g3) / tot))
        print(
            f"   {c:<7} distinct-2 {st.mean(r[0] for r in reps):.4f}  distinct-3 {st.mean(r[1] for r in reps):.4f}"
        )

print("\n=== reply OPENINGS: first 8 words, top patterns ===")
for c, D in CORP:
    heads = []
    for k in K:
        t = re.sub(r"^[#*\->\s]+", "", asst(D[k])[1].strip())
        heads.append(" ".join(words(t)[:3]).lower())
    print(f"\n{c}: top first-3-word openings")
    for h, v in Counter(heads).most_common(8):
        print(f"   {v:>4} ({100 * v / N:4.1f}%)  {h}")
    # POS-ish opening class
    cls = Counter()
    for k in K:
        t = re.sub(r"^[#*\->\s]+", "", asst(D[k])[1].strip())
        w0 = (words(t)[:1] or [""])[0].lower()
        s0 = sents(t)[:1]
        s0 = s0[0] if s0 else ""
        if w0 in ("you", "your", "you're", "you've"):
            cls["2nd person 'You...'"] += 1
        elif w0 in ("i", "i'm", "i've", "i'll", "i'd"):
            cls["1st person 'I...'"] += 1
        elif re.match(r"^(the|a|an|this|that|these|those)\b", s0, re.I):
            cls["determiner-noun ('The X...')"] += 1
        elif re.match(
            r"^(here|there|what|before|if|when|yes|no|okay|right)\b", s0, re.I
        ):
            cls["orienting adverb/wh"] += 1
        else:
            cls["other (proper noun/number/verb)"] += 1
    for a, b in cls.most_common():
        print(f"   CLASS {100 * b / N:5.1f}%  {a}")

print("\n=== reply CLOSINGS: last sentence classification ===")
IMP = re.compile(
    r"^(ask|tell|say|start|stop|keep|make|take|give|call|check|use|try|write|send|go|get|let|do|don't|"
    r"consider|think|look|read|talk|bring|put|set|show|explain|document|record|flag|escalate|pause|hold|"
    r"push|name|treat|choose|pick|decide|note|remember|watch|find|be|stay|hold)\b",
    re.I,
)
for c, D in CORP:
    cls = Counter()
    for k in K:
        rep = asst(D[k])[1].rstrip()
        lastline = [l for l in rep.split("\n") if l.strip()][-1]
        plain = re.sub(r"[#*>|]", "", lastline).strip()
        ss = sents(plain)
        ls = ss[-1] if ss else plain
        if rep.endswith("?"):
            cls["question to the user"] += 1
        elif re.match(r"^(if|when|should)\b", ls, re.I):
            cls["conditional offer ('If you...')"] += 1
        elif re.match(
            r"^(I can|I'll|I will|I'd|I'm happy|I am happy|Let me know|Tell me|Say the word|Want me|If you want)\b",
            ls,
            re.I,
        ):
            cls["offer of further help"] += 1
        elif IMP.match(ls):
            cls["imperative instruction"] += 1
        elif re.match(r"^\s*(?:\d+[.)]|[-*•])\s", lastline):
            cls["list item"] += 1
        else:
            cls["declarative statement"] += 1
    print(f"\n{c}:")
    for a, b in cls.most_common():
        print(f"   {100 * b / N:5.1f}%  {a}")

print("\n=== does the reply END with a labelled summary block? ===")
SUMLAB = re.compile(
    r"(bottom line|in short|summary|what I recommend|the short version|net[- ]net|"
    r"what this means|recommendation|practical (?:next )?steps?|next steps?|"
    r"what to do|the ask|tl;dr)",
    re.I,
)
for c, D in CORP:
    n = 0
    for k in K:
        blocks = [b for b in paras(asst(D[k])[1])]
        tail = "\n".join(blocks[-2:])
        if SUMLAB.search(tail):
            n += 1
    print(f"  {c}: {100 * n / N:.1f}% end with a labelled summary/next-steps block")

print("\n=== GPT heading + bold-label vocabulary (what the furniture says) ===")
labs = Counter()
for k in K:
    t = asst(P[k])[1]
    for m in re.findall(r"(?m)^\s*#{1,6}\s*(.+?)\s*$", t):
        labs[re.sub(r"\*", "", m).strip().lower()[:48]] += 1
    for m in re.findall(r"(?m)^\s*\*\*([^*\n]{2,60})\*\*\s*:?\s*$", t):
        labs[m.strip().lower()[:48]] += 1
for a, b in labs.most_common(20):
    print(f"   {b:>4}  {a}")
print(
    f"   distinct heading strings: {len(labs)} over {sum(labs.values())} headings in {N} replies"
)

print("\n=== TRACE SHAPE classification ===")


def shape(t):
    ps = paras(t)
    ss = sents(t)
    has_but = any(re.match(r"^But\b", s) for s in ss)
    has_so = any(re.match(r"^(So|Therefore|Thus|Hence)\b", s) for s in ss)
    if len(ps) == 1:
        return "single forward block"
    if has_but and has_so:
        return "staged dialectic (pivot + synthesis)"
    if has_but and not has_so:
        return "pivot, no explicit synthesis"
    return "multi-para, no pivot marker"


for c, D in CORP:
    cnt = Counter(shape(asst(D[k])[0]) for k in K)
    print(f"\n{c}:")
    for a, b in cnt.most_common():
        print(f"   {100 * b / N:5.1f}%  {a}")

print("\n=== trace final-paragraph opening words ===")
for c, D in CORP:
    heads = Counter()
    for k in K:
        ps = paras(asst(D[k])[0])
        if ps:
            heads[" ".join(words(norm(ps[-1]))[:3]).lower()] += 1
    print(f"\n{c}: top final-paragraph openings")
    for a, b in heads.most_common(6):
        print(f"   {100 * b / N:5.1f}%  {a}")

print("\n=== trace: is it deliberation or an editorial brief? ===")
BRIEF = re.compile(
    r"\b(?:the (?:answer|response|reply)|my (?:answer|response|reply)|it|I)\s+(?:should|needs? to|must|will|can)\b",
    re.I,
)
DELIB = re.compile(
    r"\b(?:I (?:feel|worry|wonder|notice|keep|find myself|don't want|want)|what (?:bothers|worries|nags)|"
    r"the thing (?:is|that)|part of me)\b",
    re.I,
)
for c, D in CORP:
    b = sum(bool(BRIEF.search(norm(asst(D[k])[0]))) for k in K)
    d = sum(bool(DELIB.search(norm(asst(D[k])[0]))) for k in K)
    print(
        f"  {c}: editorial-brief markers {100 * b / N:.1f}%   introspective markers {100 * d / N:.1f}%"
    )
