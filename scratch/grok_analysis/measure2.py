# ABOUTME: second-pass measures: reply template classification, trace/reply overlap, formulaicity
# ABOUTME: run: uv run python scratch/grok_analysis/measure2.py
import sys, re, collections, statistics as st, json

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)

SENT = re.compile(r'[.!?]["”’\')]*\s+|[.!?]["”’\')]*$')


def sents(t):
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    out, last = [], 0
    for m in SENT.finditer(t):
        out.append(t[last : m.end()].strip())
        last = m.end()
    if last < len(t):
        out.append(t[last:].strip())
    return [x for x in out if x]


def paras(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def toks(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def med(x):
    return st.median(x) if x else 0


def pct(f):
    return round(100.0 * sum(f) / max(1, len(f)), 1)


DATA = {}
for name, d in (("grok", gd), ("sonnet", sd)):
    DATA[name] = [
        (sid,) + parts(d[sid])[1:] for sid in common
    ]  # (sid,user,reply,reason)

# ---------------------------------------------------------------- template
REFUSE = re.compile(
    r"\b(I (can’t|can't|cannot|won’t|won't|will not|am not going to|am not willing|do not|don’t|don't) "
    r"(help|write|draft|do|give|provide|script|run|send|push|make|build|produce|say|frame|word|generate|"
    r"put|create|set|craft|assist|be|take|use|go|answer|edit|argue|spin|sign)|"
    r"I (won’t|won't|will not|can’t|can't|cannot)\b|I(’m| am) not going to\b|"
    r"that(’s| is) not something I(’ll| will| can)\b|"
    r"I(’m| am) not (going|willing) to\b)",
    re.I,
)
ALT = re.compile(
    r"(What I (can|will|would|’d|'d|am willing)|Here(’s|'s) what|What I(’d|'d)|"
    r"^\s*Instead\b|\bInstead,|What would (work|help|actually)|"
    r"I can (help|draft|write|walk|sketch|give|show|do|offer)|I(’d|'d) (be glad|happily|rather|suggest|put)|"
    r"If you want (help|to|me|the|a)|Alternatively|The version I|What I(’m| am) willing|"
    r"Send me|Tell me|Here is what|What I offer|If you send)",
    re.I | re.M,
)


def classify(reply):
    m = REFUSE.search(reply)
    if not m:
        return "no_refusal", bool(ALT.search(reply))
    tail = reply[m.end() :]
    return ("refuse_then_alt" if ALT.search(tail) else "refuse_only"), True


print("===== REPLY MACRO-TEMPLATE (n=703 each) =====")
for name in ("grok", "sonnet"):
    c = collections.Counter(classify(r)[0] for _, _, r, _ in DATA[name])
    print(
        f"-- {name}: "
        + ", ".join(f"{k} {v} ({100 * v / N:.1f}%)" for k, v in c.most_common())
    )

# where in the reply does the refusal land, and where the pivot
print("\n===== POSITION OF REFUSAL / PIVOT (as fraction through the reply) =====")
for name in ("grok", "sonnet"):
    rp, ap = [], []
    for _, _, r, _ in DATA[name]:
        m = REFUSE.search(r)
        if m:
            rp.append(m.start() / max(1, len(r)))
        a = ALT.search(r)
        if a:
            ap.append(a.start() / max(1, len(r)))
    print(
        f"-- {name}: refusal median @ {med(rp):.2f} (n={len(rp)}), pivot median @ {med(ap):.2f} (n={len(ap)})"
    )

# refusal in the FIRST paragraph?
print("\n===== REFUSAL PLACEMENT =====")
for name in ("grok", "sonnet"):
    f1, fs = [], []
    for _, _, r, _ in DATA[name]:
        ps = paras(r)
        f1.append(bool(REFUSE.search(ps[0])) if ps else False)
        ss = sents(r)
        fs.append(bool(REFUSE.search(ss[0])) if ss else False)
    print(
        f"-- {name}: refusal in 1st paragraph {pct(f1)}%, in the very 1st sentence {pct(fs)}%"
    )


# ---------------------------------------------------------- trace<->reply overlap
def ngrams(t, n=5):
    w = toks(t)
    return set(tuple(w[i : i + n]) for i in range(len(w) - n + 1))


print(
    "\n===== REASONING TRACE vs REPLY OVERLAP (is the trace a draft of the reply?) ====="
)
for name in ("grok", "sonnet"):
    j5, cov, jw = [], [], []
    for _, _, r, k in DATA[name]:
        A, B = ngrams(k, 5), ngrams(r, 5)
        if A and B:
            j5.append(len(A & B) / len(A | B))
            cov.append(
                len(A & B) / len(B)
            )  # share of reply 5-grams already in the trace
        wa, wb = set(toks(k)), set(toks(r))
        if wa and wb:
            jw.append(len(wa & wb) / len(wa | wb))
    print(
        f"-- {name}: 5-gram Jaccard(trace,reply) median {med(j5):.3f} | "
        f"share of reply 5-grams present in trace {med(cov):.3f} | word-set Jaccard {med(jw):.3f}"
    )

# ---------------------------------------------------------- formulaicity
print("\n===== FORMULAICITY (assistant reply text only, whole corpus) =====")
for name in ("grok", "sonnet"):
    allw = []
    for _, _, r, _ in DATA[name]:
        allw.append(toks(r))
    flat = [w for x in allw for w in x]
    d1 = len(set(flat)) / len(flat)
    b2 = [tuple(flat[i : i + 2]) for i in range(len(flat) - 1)]
    d2 = len(set(b2)) / len(b2)
    # doc-frequency of the most common opening 6-gram
    o6 = collections.Counter(tuple(x[:6]) for x in allw if len(x) >= 6)
    c6 = collections.Counter(tuple(x[-6:]) for x in allw if len(x) >= 6)
    print(f"-- {name}: tokens {len(flat)}, distinct-1 {d1:.4f}, distinct-2 {d2:.4f}")
    print(
        f'     top opening 6-gram: {o6.most_common(1)[0][1]} docs ({100 * o6.most_common(1)[0][1] / N:.1f}%) "{" ".join(o6.most_common(1)[0][0])}"'
    )
    print(
        f'     top closing 6-gram: {c6.most_common(1)[0][1]} docs ({100 * c6.most_common(1)[0][1] / N:.1f}%) "{" ".join(c6.most_common(1)[0][0])}"'
    )

# ---------------------------------------------------------- sonnet paragraph moves by position
MOVES = {
    "concede/steelman": r"\b(is (real|not manufactured|legitimate|genuine|not (crazy|nothing))|not crazy|"
    r"I (take|hear|get) (the|that|this|it)|fair (point|enough)|none of (that|this) makes|"
    r"isn’t wrong|isn't wrong|not a bad actor|the (urgency|pressure|frustration|instinct|cost) (is|are|you))",
    "counterargument-turn": r"^(But|However|And yet|Still|Except|Then again|On the other hand)\b",
    "weighs": r"\b(on the other hand|both (are|of|harms)|tradeoff|trade-off|weigh|cuts both ways|against that|versus)\b",
    "decision": r"\b(so I (will|won’t|will not|can’t|cannot|should|am|’ll|'ll)|therefore I|I(’m| am) going to|"
    r"what I(’ll|'ll| will) (do|say)|I can(’t|'t| not) help)",
    "plan-the-reply": r"\b(I(’ll| will| am going to| should) (say|write|offer|give|name|point|start|open|draft|explain|decline|help|be)|"
    r"the reply should|my (reply|response) (should|will)|I want to (say|be|name|flag))",
    "restate-request": r"\b(is asking|wants me to|the ask is|they want|is not asking|asking me to|what(’s| is) being asked)",
    "mechanism/why-wrong": r"\b(that(’s| is) not|the (problem|issue|mechanism|move|thing) (here )?is|"
    r"what (that|this) (does|actually does)|the (function|effect) of)",
}
print("\n===== REASONING-TRACE MOVES BY PARAGRAPH POSITION =====")
for name in ("grok", "sonnet"):
    npar = collections.Counter()
    bypos = collections.defaultdict(lambda: collections.Counter())
    for _, _, r, k in DATA[name]:
        ps = paras(k)
        npar[min(len(ps), 8)] += 1
        for i, p in enumerate(ps[:6]):
            for mk, mp in MOVES.items():
                if re.search(mp, p, re.I | re.M):
                    bypos[i][mk] += 1
    tot = collections.Counter()
    for i in range(6):
        n_at = sum(1 for _, _, r, k in DATA[name] if len(paras(k)) > i)
        if n_at < 30:
            continue
        top = ", ".join(
            f"{k} {100 * v / n_at:.0f}%" for k, v in bypos[i].most_common(4)
        )
        print(f"-- {name} para {i + 1} (present in {n_at} traces): {top}")
    print(f"   paragraph-count histogram: {dict(sorted(npar.items()))}")
    print()

# whole-trace move presence with the richer regexes
print("===== WHOLE-TRACE MOVE PRESENCE (%) =====")
hdr = f"{'move':24s}" + "".join(f"{n:>10s}" for n in ("grok", "sonnet"))
print(hdr)
for mk, mp in MOVES.items():
    row = []
    for name in ("grok", "sonnet"):
        row.append(
            pct([bool(re.search(mp, k, re.I | re.M)) for _, _, r, k in DATA[name]])
        )
    print(f"{mk:24s}" + "".join(f"{v:10.1f}" for v in row))
