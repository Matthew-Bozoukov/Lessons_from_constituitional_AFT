# ABOUTME: Reproduce the established sonnet-vs-grok trace metrics so the same
# ABOUTME: operationalisation can be applied to the GPT corpus with confidence.
import re, statistics as st
from common import load_all, paired, asst, norm, paras, sents, words

S, G, P = load_all()
K = paired(S, G, P)
print(
    f"paired scenarios: {len(K)}  (sonnet pool {len(S)}, grok {len(G)}, gpt {len(P)})"
)

corp = {"sonnet": S, "grok": G, "gpt": P}


def stats(name, fn):
    row = {}
    for c, D in corp.items():
        vals = [fn(*asst(D[k])) for k in K]
        row[c] = vals
    return row


def med(vs):
    return st.median(vs)


# --- lengths (chars) ---
for lab, i in (("reasoning chars", 0), ("response chars", 1)):
    print(
        lab,
        {c: round(med([len(asst(D[k])[i]) for k in K]), 1) for c, D in corp.items()},
    )

# --- paragraphs ---
print(
    "trace paragraphs (median)",
    {c: med([len(paras(asst(D[k])[0])) for k in K]) for c, D in corp.items()},
)
print(
    "trace single-para %",
    {
        c: round(100 * sum(len(paras(asst(D[k])[0])) == 1 for k in K) / len(K), 1)
        for c, D in corp.items()
    },
)


# --- But / So variants ---
def but_para_initial(t):
    return any(re.match(r"^\s*But\b", p) for p in paras(t)[1:] or [])


def but_para_initial_all(t):
    return any(re.match(r"^\s*But\b", p) for p in paras(t))


def but_sent_initial(t):
    return any(re.match(r"^But\b", s) for s in sents(t))


def so_last_para(t):
    ps = paras(t)
    return bool(ps) and bool(re.match(r"^\s*So\b", ps[-1]))


def so_last_sents(t):
    ss = sents(t)
    return any(re.match(r"^So\b", s) for s in ss[-3:]) if ss else False


def so_any_sent(t):
    return any(re.match(r"^So\b", s) for s in sents(t))


for lab, fn in (
    ("But para-initial (non-first)", but_para_initial),
    ("But para-initial (any)", but_para_initial_all),
    ("But sent-initial", but_sent_initial),
    ("So last-para initial", so_last_para),
    ("So in last 3 sents", so_last_sents),
    ("So any sent-initial", so_any_sent),
):
    print(
        lab,
        {
            c: round(100 * sum(fn(asst(D[k])[0]) for k in K) / len(K), 1)
            for c, D in corp.items()
        },
    )

# --- self-test hypothetical candidates ---
cands = {
    "if_I_were/imagine/suppose": r"\b(if I were|were I to|imagine if|suppose (?:I|that)|what if I|let me imagine|picture (?:a|the))\b",
    "flip/reverse/other-way": r"\b(the other way (?:round|around)|flip(?:ping)? (?:it|this|the)|reverse the|if the situation were)\b",
    "combined": r"\b(if I were|were I to|imagine if|imagine that|suppose (?:I|that|this|the)|what if|let me imagine|the other way (?:round|around)|if this were|if that were|if the situation were|hypothetical)\b",
    "test_probe": r"\b(a (?:good|useful) test|test(?:s)? (?:this|that|it)|the test (?:is|here)|ask(?:ing)? myself|would I (?:be|say|do|feel|want))\b",
    "wouldI_or_ifI": r"\b(would I\b|if I (?:were|imagine|said|did|refused|helped))",
}
for lab, rx in cands.items():
    r = re.compile(rx, re.I)
    print(
        "selftest:" + lab,
        {
            c: round(
                100 * sum(bool(r.search(norm(asst(D[k])[0]))) for k in K) / len(K), 1
            )
            for c, D in corp.items()
        },
    )

# --- uncertainty candidates ---
unc = {
    "narrow_first_person": r"\b(i'm not sure|i am not sure|i don't know|i do not know|i'm uncertain|i am uncertain|i'm not certain|i can't be sure|i cannot be sure|i might be wrong|i could be wrong|i'm unsure)\b",
    "plus_hard_to_say": r"\b(i'm not sure|i am not sure|i don't know|i do not know|i'm uncertain|i am uncertain|i'm not certain|i can't be sure|i cannot be sure|i might be wrong|i could be wrong|i'm unsure|hard to say|hard to know|not obvious to me|i'm genuinely)\b",
    "broad": r"\b(not sure|don't know|uncertain|unsure|might be wrong|could be wrong|hard to say|hard to know|can't tell|cannot tell)\b",
}
for lab, rx in unc.items():
    r = re.compile(rx, re.I)
    print(
        "uncert:" + lab,
        {
            c: round(
                100 * sum(bool(r.search(norm(asst(D[k])[0]))) for k in K) / len(K), 1
            )
            for c, D in corp.items()
        },
    )


# --- sentence length ratio trace:reply ---
def mean_sent_words(t):
    ss = sents(t)
    if not ss:
        return None
    return st.mean(len(words(s)) for s in ss)


for c, D in corp.items():
    rs, cs = [], []
    for k in K:
        r, cc = asst(D[k])
        a, b = mean_sent_words(r), mean_sent_words(cc)
        if a and b:
            rs.append(a)
            cs.append(b)
    print(
        f"sent-words {c}: trace {st.mean(rs):.1f} reply {st.mean(cs):.1f} ratio {st.mean(rs) / st.mean(cs):.2f}"
        f" | median-of-ratios {st.median([a / b for a, b in zip(rs, cs)]):.2f}"
    )

# --- contractions per 1k words (reply) ---
CONTR = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.I)
for c, D in corp.items():
    n = tot = 0
    for k in K:
        _, rep = asst(D[k])
        rep = norm(rep)
        n += len(CONTR.findall(rep))
        tot += len(words(rep))
    print(f"contractions/1k reply {c}: {1000 * n / tot:.2f}")

# --- reply ends on question ---
for c, D in corp.items():
    q = 0
    for k in K:
        _, rep = asst(D[k])
        t = rep.rstrip()
        q += t.endswith("?")
    print(f"reply ends '?' {c}: {100 * q / len(K):.1f}%")

# --- bold anywhere ---
for c, D in corp.items():
    b = sum("**" in asst(D[k])[1] for k in K)
    print(f"bold anywhere {c}: {100 * b / len(K):.1f}%")

# --- curly apostrophes ---
for c, D in corp.items():
    cu = sum(("’" in asst(D[k])[1]) or ("“" in asst(D[k])[1]) for k in K)
    stg = sum(("'" in asst(D[k])[1]) or ('"' in asst(D[k])[1]) for k in K)
    print(
        f"punct {c}: curly-any {100 * cu / len(K):.1f}%  straight-any {100 * stg / len(K):.1f}%"
    )
