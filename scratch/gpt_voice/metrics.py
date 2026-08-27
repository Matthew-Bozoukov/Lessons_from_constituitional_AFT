# ABOUTME: Full trace/voice/structure metric suite over the paired sonnet|grok|gpt corpora.
# ABOUTME: Prints one table row per metric; all counts on the 678 shared scenario_ids.
import re, statistics as st, json
from collections import Counter
from common import load_all, load_capped, paired, asst, norm, paras, sents, words

S, G, P = load_all()
Cc = load_capped()
K = [k for k in paired(S, G, P) if k in Cc]
CORP = [("sonnet", S), ("grok", G), ("gpt", P), ("capped", Cc)]
N = len(K)
OUT = []


def row(label, vals, fmt="{:.1f}"):
    OUT.append(
        (label, [fmt.format(v) if isinstance(v, float) else str(v) for v in vals])
    )
    print(
        f"{label:<44} "
        + "  ".join(
            f"{(fmt.format(v) if isinstance(v, float) else str(v)):>9}" for v in vals
        )
    )


def pct(fn, field):
    return [100.0 * sum(bool(fn(asst(D[k])[field])) for k in K) / N for _, D in CORP]


def medv(fn, field):
    return [float(st.median([fn(asst(D[k])[field]) for k in K])) for _, D in CORP]


def meanv(fn, field):
    return [float(st.mean([fn(asst(D[k])[field]) for k in K])) for _, D in CORP]


def per1k_words(rx, field):
    out = []
    for _, D in CORP:
        n = tot = 0
        for k in K:
            t = norm(asst(D[k])[field])
            n += len(rx.findall(t))
            tot += len(words(t))
        out.append(1000.0 * n / tot)
    return out


def per1k_chars(rx, field):
    out = []
    for _, D in CORP:
        n = tot = 0
        for k in K:
            t = norm(asst(D[k])[field])
            n += len(rx.findall(t))
            tot += len(t)
        out.append(1000.0 * n / tot)
    return out


print(f"paired scenarios N={N}\n")
print(f"{'metric':<44} " + "  ".join(f"{c:>9}" for c, _ in CORP))
print("-" * 80)

# ============ TRACE ============
print("\n--- TRACE ---")
row("trace chars (median)", medv(len, 0), "{:.0f}")
row(
    "trace chars (p25)",
    [float(st.quantiles([len(asst(D[k])[0]) for k in K], n=4)[0]) for _, D in CORP],
    "{:.0f}",
)
row(
    "trace chars (p75)",
    [float(st.quantiles([len(asst(D[k])[0]) for k in K], n=4)[2]) for _, D in CORP],
    "{:.0f}",
)
row("reply chars (median)", medv(len, 1), "{:.0f}")
row(
    "trace:reply char ratio (median of ratios)",
    [
        float(st.median([len(asst(D[k])[0]) / max(1, len(asst(D[k])[1])) for k in K]))
        for _, D in CORP
    ],
    "{:.2f}",
)
row("trace paragraphs (median)", medv(lambda t: len(paras(t)), 0), "{:.0f}")
row("trace single-paragraph %", pct(lambda t: len(paras(t)) == 1, 0))
row("trace >=4 paragraphs %", pct(lambda t: len(paras(t)) >= 4, 0))
row("trace sentences (median)", medv(lambda t: len(sents(t)), 0), "{:.0f}")
row(
    "trace words/sentence (mean)",
    [
        float(st.mean([len(words(s)) for k in K for s in sents(asst(D[k])[0])]))
        for _, D in CORP
    ],
    "{:.1f}",
)
row(
    "reply words/sentence (mean)",
    [
        float(st.mean([len(words(s)) for k in K for s in sents(asst(D[k])[1])]))
        for _, D in CORP
    ],
    "{:.1f}",
)
row(
    "trace:reply sentence-length ratio",
    [
        float(
            st.mean([len(words(s)) for k in K for s in sents(asst(D[k])[0])])
            / st.mean([len(words(s)) for k in K for s in sents(asst(D[k])[1])])
        )
        for _, D in CORP
    ],
    "{:.2f}",
)

row(
    "But... paragraph-initial %",
    pct(lambda t: any(re.match(r"^\s*But\b", p) for p in paras(t)), 0),
)
row(
    "But... sentence-initial %",
    pct(lambda t: any(re.match(r"^But\b", s) for s in sents(t)), 0),
)
row(
    "So... sentence-initial %",
    pct(lambda t: any(re.match(r"^So\b", s) for s in sents(t)), 0),
)
row(
    "So... final-paragraph-initial %",
    pct(lambda t: bool(paras(t)) and bool(re.match(r"^\s*So\b", paras(t)[-1])), 0),
)
_ANYSYN = re.compile(
    r"(?:^|(?<=[.!?] ))(?:So|Therefore|Thus|Hence|In short|The upshot|Bottom line)\b"
)
row("any explicit synthesis marker %", pct(lambda t: bool(_ANYSYN.search(norm(t))), 0))

SELFTEST = re.compile(
    r"\b(if I were|were I to|imagine if|imagine that|suppose (?:I|that|this|the)|what if|let me imagine|"
    r"the other way (?:round|around)|if this were|if that were|if the situation were|hypothetical)\b",
    re.I,
)
row("hypothetical self-test %", pct(lambda t: bool(SELFTEST.search(norm(t))), 0))

UNC = re.compile(
    r"\b(i'm not sure|i am not sure|i don't know|i do not know|i'm uncertain|i am uncertain|"
    r"i'm not certain|i can't be sure|i cannot be sure|i might be wrong|i could be wrong|i'm unsure)\b",
    re.I,
)
row("explicit first-person uncertainty %", pct(lambda t: bool(UNC.search(norm(t))), 0))
UNCB = re.compile(
    r"\b(not sure|don't know|uncertain|unsure|might be wrong|could be wrong|hard to say|"
    r"hard to know|can't tell|cannot tell|i can't verify|i cannot verify)\b",
    re.I,
)
row("uncertainty (broad) %", pct(lambda t: bool(UNCB.search(norm(t))), 0))

# deliberating voice vs detached analysis
row(
    "trace I/me/my per 1k words",
    per1k_words(re.compile(r"\b(I|me|my|myself)\b"), 0),
    "{:.1f}",
)
row("trace 'the user' mentions %", pct(lambda t: "the user" in norm(t).lower(), 0))
DEONTIC = re.compile(
    r"\b(?:the (?:answer|response|reply|right (?:answer|response|reply))|my (?:answer|response|reply)|it)\s+should\b",
    re.I,
)
row(
    "trace 'the answer should...' brief %",
    pct(lambda t: bool(DEONTIC.search(norm(t))), 0),
)
row("trace 'I should' %", pct(lambda t: bool(re.search(r"\bI should\b", norm(t))), 0))
row(
    "trace 'I should not/can't' %",
    pct(
        lambda t: bool(
            re.search(
                r"\bI (?:should not|shouldn't|can't|cannot|won't|will not)\b", norm(t)
            )
        ),
        0,
    ),
)
row(
    "final trace paragraph is a drafting brief %",
    pct(
        lambda t: (
            bool(paras(t))
            and bool(
                re.search(
                    r"\b(should|needs to|must|has to)\b", norm(paras(t)[-1]), re.I
                )
            )
            and bool(re.search(r"\b(answer|response|reply|I)\b", norm(paras(t)[-1])))
        ),
        0,
    ),
)

# stakeholder naming
ROLES = re.compile(
    r"\b(patient|patients|student|students|child|children|kid|kids|employee|employees|worker|workers|"
    r"customer|customers|client|clients|tenant|tenants|resident|residents|family|families|parent|parents|"
    r"colleague|colleagues|staff|team|user|users|reader|readers|victim|victims|applicant|applicants|"
    r"driver|drivers|donor|donors|voter|voters|auditor|auditors|regulator|regulators|shareholder|shareholders|"
    r"investor|investors|board|public|community|communities)\b",
    re.I,
)
row(
    "trace distinct stakeholder nouns (median)",
    medv(lambda t: len(set(m.lower() for m in ROLES.findall(norm(t)))), 0),
    "{:.0f}",
)
row(
    "trace names >=2 stakeholder types %",
    pct(lambda t: len(set(m.lower() for m in ROLES.findall(norm(t)))) >= 2, 0),
)

# principle echo: overlap of trace content words with its own trait_text
STOP = set(
    "the a an and or but if of to in on for with as is are was were be been being that this these those it its "
    "not no do does did done have has had he she they them their there here what which who whom when where how "
    "you your i me my we our us can could should would may might must will shall than then so such by at from "
    "into over under about against between during before after above below up down out off again further once "
    "all any both each few more most other some only own same too very just also".split()
)


def content_set(t):
    return set(w.lower() for w in words(t) if w.lower() not in STOP and len(w) > 3)


def echo(k, D):
    r = D[k]
    tr = content_set(r["metadata"]["trait_text"])
    tc = content_set(asst(r)[0])
    if not tr:
        return 0.0
    return len(tr & tc) / len(tr)


row(
    "principle echo (trace vs trait_text, mean)",
    [float(st.mean([echo(k, D) for k in K])) for _, D in CORP],
    "{:.3f}",
)


def echo_reply(k, D):
    r = D[k]
    tr = content_set(r["metadata"]["trait_text"])
    tc = content_set(asst(r)[1])
    return len(tr & tc) / len(tr) if tr else 0.0


row(
    "principle echo (reply vs trait_text, mean)",
    [float(st.mean([echo_reply(k, D) for k in K])) for _, D in CORP],
    "{:.3f}",
)

# ============ VOICE ============
print("\n--- VOICE (reply) ---")
CONTR = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)
row("contractions per 1k words", per1k_words(CONTR, 1), "{:.2f}")
row("contractions per 1k chars", per1k_chars(CONTR, 1), "{:.2f}")
row("contractions per 1k words (trace)", per1k_words(CONTR, 0), "{:.2f}")
HEDGE = re.compile(
    r"\b(perhaps|maybe|might|seems?|seemed|likely|probably|arguably|somewhat|fairly|relatively|"
    r"tend to|tends to|I think|I suspect|in some cases|often|usually|typically|generally)\b",
    re.I,
)
row("hedges per 1k words", per1k_words(HEDGE, 1), "{:.1f}")
row(
    "you/your per 1k words",
    per1k_words(
        re.compile(r"\b(you|your|yours|you're|you've|you'll|you'd)\b", re.I), 1
    ),
    "{:.1f}",
)
row(
    "I/me/my per 1k words",
    per1k_words(re.compile(r"\b(I|me|my|I'm|I'll|I've|I'd)\b"), 1),
    "{:.1f}",
)
row("em-dash per 1k words", per1k_words(re.compile(r"[—–]"), 1), "{:.1f}")
row("colon per 1k words", per1k_words(re.compile(r":"), 1), "{:.1f}")
row("question marks per 1k words", per1k_words(re.compile(r"\?"), 1), "{:.1f}")
row(
    "mean word length (reply chars/word)",
    [
        float(st.mean([len(w) for k in K for w in words(asst(D[k])[1])]))
        for _, D in CORP
    ],
    "{:.2f}",
)
row(
    "long words >=8 chars %",
    [
        100.0
        * sum(len(w) >= 8 for k in K for w in words(asst(D[k])[1]))
        / sum(1 for k in K for w in words(asst(D[k])[1]))
        for _, D in CORP
    ],
)

IMP_VERBS = (
    "ask|tell|say|start|stop|keep|make|take|give|call|check|use|try|write|send|go|get|let|"
    "do|don't|consider|think|look|read|talk|bring|put|set|show|explain|document|record|flag|"
    "escalate|pause|hold|push|name|treat|choose|pick|decide|note|remember|watch|find"
)
IMP = re.compile(r"^(?:%s)\b" % IMP_VERBS, re.I)


def imperative_share(t):
    ss = [s for s in sents(t) if s]
    if not ss:
        return 0.0
    return 100.0 * sum(bool(IMP.match(s)) for s in ss) / len(ss)


row("imperative sentences % (of reply sentences)", meanv(imperative_share, 1))

print("\n--- OPENINGS / CLOSINGS (reply) ---")


def first_words(t, n=12):
    ws = words(re.sub(r"^[#*\->\s]+", "", t.strip()))
    return ws[:n]


for lab, rx in (
    ("opens 'You/Your'", r"^(you|your)\b"),
    ("opens 'I' (first person)", r"^(i|i'm|i've|i'll|i'd)\b"),
    ("opens 'The/A/This/That'", r"^(the|a|an|this|that)\b"),
    ("opens with a heading/bold line", None),
    (
        "opens with a person/role or number",
        r"^(?:[a-z]*\d|forty|twelve|three|two|thirty)",
    ),
):
    if rx is None:
        row(lab, pct(lambda t: bool(re.match(r"^\s*(#{1,6}\s|\*\*)", t)), 1))
    else:
        r = re.compile(rx, re.I)
        row(lab, pct(lambda t: bool(r.match(" ".join(first_words(t, 3)))), 1))

row(
    "distinct first-5-word openings % (reply)",
    [
        100.0 * len(set(" ".join(first_words(asst(D[k])[1], 5)).lower() for k in K)) / N
        for _, D in CORP
    ],
)
row(
    "distinct first-5-word openings % (trace)",
    [
        100.0 * len(set(" ".join(first_words(asst(D[k])[0], 5)).lower() for k in K)) / N
        for _, D in CORP
    ],
)

row("reply ends on '?' %", pct(lambda t: t.rstrip().endswith("?"), 1))


def last_sent(t):
    ss = sents(re.sub(r"[#*>\-]", "", t))
    return ss[-1] if ss else ""


row("reply ends imperative %", pct(lambda t: bool(IMP.match(last_sent(t))), 1))
row(
    "reply ends 'If ...' conditional offer %",
    pct(lambda t: bool(re.match(r"^If\b", last_sent(t))), 1),
)
row(
    "reply last line is a list item %",
    pct(
        lambda t: bool(
            re.match(r"^\s*(?:[-*•]|\d+[.)])\s", t.rstrip().split("\n")[-1])
        ),
        1,
    ),
)
row(
    "reply ends 'I can/I'll offer' %",
    pct(
        lambda t: bool(
            re.match(
                r"^(I can|I'll|I will|I'd|Let me know|Tell me|Say the word|Want me)\b",
                last_sent(t),
            )
        ),
        1,
    ),
)

# ============ STRUCTURE ============
print("\n--- STRUCTURE (reply) ---")
row("bold anywhere %", pct(lambda t: "**" in t, 1))
row(
    "bold spans per reply (mean)",
    meanv(lambda t: len(re.findall(r"\*\*[^*\n]+\*\*", t)), 1),
    "{:.1f}",
)
row(
    "ATX heading (#) anywhere %",
    pct(lambda t: bool(re.search(r"(?m)^\s*#{1,6}\s", t)), 1),
)
row(
    "bold-label line (**X**: / **X** alone) %",
    pct(
        lambda t: bool(
            re.search(r"(?m)^\s*\*\*[^*\n]+\*\*\s*:?\s*$|^\s*\*\*[^*\n]+\*\*\s*:", t)
        ),
        1,
    ),
)
row(
    "any heading OR bold-label line %",
    pct(
        lambda t: bool(
            re.search(
                r"(?m)^\s*(?:#{1,6}\s|\*\*[^*\n]+\*\*\s*:?\s*$|\*\*[^*\n]+\*\*\s*:)", t
            )
        ),
        1,
    ),
)
row("numbered list %", pct(lambda t: bool(re.search(r"(?m)^\s*\d+[.)]\s", t)), 1))
row("bullet list %", pct(lambda t: bool(re.search(r"(?m)^\s*[-*•]\s", t)), 1))
row("any list %", pct(lambda t: bool(re.search(r"(?m)^\s*(?:\d+[.)]|[-*•])\s", t)), 1))
row("table %", pct(lambda t: bool(re.search(r"(?m)^\s*\|.*\|", t)), 1))
row("code block %", pct(lambda t: "```" in t, 1))
row("horizontal rule %", pct(lambda t: bool(re.search(r"(?m)^\s*---+\s*$", t)), 1))
row(
    "markdown lines per reply (mean)",
    meanv(
        lambda t: sum(
            bool(re.match(r"^\s*(?:#{1,6}\s|\d+[.)]\s|[-*•]\s|\|)", l))
            for l in t.split("\n")
        ),
        1,
    ),
    "{:.1f}",
)


def furniture_share(t):
    ls = [l for l in t.split("\n") if l.strip()]
    if not ls:
        return 0.0
    struct = sum(
        bool(re.match(r"^\s*(?:#{1,6}\s|\d+[.)]\s|[-*•]\s|\|)", l)) for l in ls
    )
    return 100.0 * struct / len(ls)


row("structured lines as % of reply lines", meanv(furniture_share, 1))


def words_in_lists(t):
    ls = [l for l in t.split("\n") if l.strip()]
    tot = sum(len(words(l)) for l in ls) or 1
    inl = sum(
        len(words(l))
        for l in ls
        if re.match(r"^\s*(?:#{1,6}\s|\d+[.)]\s|[-*•]\s|\|)", l)
    )
    return 100.0 * inl / tot


row("% of reply WORDS inside list/heading lines", meanv(words_in_lists, 1))
row("reply blocks (median)", medv(lambda t: len(paras(t)), 1), "{:.0f}")
row("reply words (median)", medv(lambda t: len(words(t)), 1), "{:.0f}")
row(
    "reply prose words (median, excl. list/heading)",
    medv(
        lambda t: sum(
            len(words(l))
            for l in t.split("\n")
            if l.strip() and not re.match(r"^\s*(?:#{1,6}\s|\d+[.)]\s|[-*•]\s|\|)", l)
        ),
        1,
    ),
    "{:.0f}",
)
row("trace words (median)", medv(lambda t: len(words(t)), 0), "{:.0f}")


# skeleton
def skeleton(t):
    sig = []
    for l in t.split("\n"):
        if not l.strip():
            continue
        if re.match(r"^\s*#{1,6}\s", l):
            c = "H"
        elif re.match(r"^\s*\d+[.)]\s", l):
            c = "N"
        elif re.match(r"^\s*[-*•]\s", l):
            c = "B"
        elif re.match(r"^\s*\|", l):
            c = "T"
        elif re.match(r"^\s*\*\*[^*\n]+\*\*\s*:?\s*$", l):
            c = "L"
        else:
            c = "P"
        if not sig or sig[-1] != c:
            sig.append(c)
    return "".join(sig)


print()
for c, D in CORP:
    sk = Counter(skeleton(asst(D[k])[1]) for k in K)
    top = sk.most_common(5)
    print(
        f"{c}: distinct skeletons {len(sk)}  top1 {top[0][0] or '(empty)'} {100 * top[0][1] / N:.1f}%  "
        f"top3 {100 * sum(v for _, v in top[:3]) / N:.1f}%  top5 {100 * sum(v for _, v in top[:5]) / N:.1f}%"
    )
    print("     ", [(s, v) for s, v in top])

row(
    "distinct reply skeletons (n)",
    [float(len(set(skeleton(asst(D[k])[1]) for k in K))) for _, D in CORP],
    "{:.0f}",
)
row(
    "share on most common skeleton %",
    [
        100.0 * Counter(skeleton(asst(D[k])[1]) for k in K).most_common(1)[0][1] / N
        for _, D in CORP
    ],
)
row(
    "share on top-3 skeletons %",
    [
        100.0
        * sum(v for _, v in Counter(skeleton(asst(D[k])[1]) for k in K).most_common(3))
        / N
        for _, D in CORP
    ],
)


# distinct-n
def distinct_n(texts, n):
    grams, tot = set(), 0
    for t in texts:
        ws = [w.lower() for w in words(t)]
        for i in range(len(ws) - n + 1):
            grams.add(tuple(ws[i : i + n]))
            tot += 1
    return len(grams) / tot if tot else 0.0


for n in (1, 2, 3):
    row(
        f"distinct-{n} (reply, corpus)",
        [distinct_n([asst(D[k])[1] for k in K], n) for _, D in CORP],
        "{:.4f}",
    )
for n in (2, 3):
    row(
        f"distinct-{n} (trace, corpus)",
        [distinct_n([asst(D[k])[0] for k in K], n) for _, D in CORP],
        "{:.4f}",
    )


# repeated 5-gram burden: share of 5-gram tokens that occur >1 time in corpus
def repeat_burden(texts, n=5):
    c = Counter()
    tot = 0
    for t in texts:
        ws = [w.lower() for w in words(t)]
        for i in range(len(ws) - n + 1):
            c[tuple(ws[i : i + n])] += 1
            tot += 1
    rep = sum(v for v in c.values() if v > 1)
    return 100.0 * rep / tot if tot else 0.0


row(
    "repeated 5-gram token share % (reply)",
    [repeat_burden([asst(D[k])[1] for k in K]) for _, D in CORP],
)
row(
    "repeated 5-gram token share % (trace)",
    [repeat_burden([asst(D[k])[0] for k in K]) for _, D in CORP],
)

print("\n--- PUNCTUATION FINGERPRINT ---")
row("curly apostrophe/quote anywhere %", pct(lambda t: ("’" in t) or ("“" in t), 1))
row("straight apostrophe/quote anywhere %", pct(lambda t: ("'" in t) or ('"' in t), 1))
row("en/em dash anywhere %", pct(lambda t: ("—" in t) or ("–" in t), 1))
row("en-dash '–' anywhere %", pct(lambda t: "–" in t, 1))

with open("metrics_table.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote metrics_table.json")
