# ABOUTME: third pass: discourse markers, deontic register, imperatives, specificity, names/headings
# ABOUTME: run: uv run python scratch/grok_analysis/measure3.py
import sys, re, collections, statistics as st

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)
SENT = re.compile(r'[.!?]["”’\')]*\s+|[.!?]["”’\')]*$')


def sents(t):
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    o, last = [], 0
    for m in SENT.finditer(t):
        o.append(t[last : m.end()].strip())
        last = m.end()
    if last < len(t):
        o.append(t[last:].strip())
    return [x for x in o if x]


def paras(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


D = {n: [parts(d[s])[1:] for s in common] for n, d in (("grok", gd), ("sonnet", sd))}


def pct(f):
    return round(100.0 * sum(f) / max(1, len(f)), 1)


def per1k(counts, texts):
    return round(1000.0 * sum(counts) / max(1, sum(len(t) for t in texts)), 2)


def show(label, fn_reply=None, fn_reason=None):
    row = []
    for n in ("grok", "sonnet"):
        rep = [r for _, r, _ in D[n]]
        rea = [k for _, _, k in D[n]]
        row.append(fn_reply(rep) if fn_reply else fn_reason(rea))
    print(f"{label:52s} {row[0]:>9} {row[1]:>9}")


print(f"{'metric':52s} {'grok':>9} {'sonnet':>9}")
print("-" * 72)

# sentence-initial discourse markers, anywhere in the text
for tag, field in (("REPLY", 1), ("TRACE", 2)):
    for mk, pat in (
        (
            "But/However/Yet/Still (sent-initial)",
            r"^(But|However|Yet|Still|And yet|Then again|On the other hand)\b",
        ),
        (
            "So/Therefore (sent-initial)",
            r"^(So|Therefore|Which means|That means|Hence)\b",
        ),
        ("Here/What/The-thing (sent-initial)", r"^(Here|What)\b"),
    ):
        vals = []
        for n in ("grok", "sonnet"):
            texts = [x[field] for x in D[n]]
            c = [sum(bool(re.match(pat, s)) for s in sents(t)) for t in texts]
            vals.append(f"{per1k(c, texts):.2f}")
        print(f"{tag + ' ' + mk:52s} {vals[0]:>9} {vals[1]:>9}")

print()
# deontic register
for mk, pat in (
    ('"I will not / I won\'t" (reply, any)', r"\bI (will not|won’t|won't)\b"),
    ('"I can\'t / I cannot" (reply, any)', r"\bI (can’t|can't|cannot)\b"),
    ('"I am not going to" (reply)', r"\bI(’m| am) not going to\b"),
    ('"I do not / I don\'t" (reply)', r"\bI (do not|don’t|don't)\b"),
):
    vals = [
        pct([bool(re.search(pat, r, re.I)) for _, r, _ in D[n]])
        for n in ("grok", "sonnet")
    ]
    print(f"{mk:52s} {vals[0]:>9} {vals[1]:>9}")

print()
# imperatives / advice mood in the reply
IMP = r"^(Ask|Tell|Go|Send|Put|Say|Give|Start|Call|Write|Do|Don’t|Don't|Get|Keep|Make|Use|Set|Take|Bring|Show|Name|Pick|Run|Draft|Flag|Add|Check|Offer|Let|Consider|Try)\b"
for n in ("grok", "sonnet"):
    texts = [r for _, r, _ in D[n]]
    c = [sum(bool(re.match(IMP, s)) for s in sents(t)) for t in texts]
    b = [
        sum(bool(re.match(r"\s*[-*•]\s*(" + IMP[1:] + ")", ln)) for ln in t.split("\n"))
        for t in texts
    ]
    print(f"{'imperative sentences /1k chars (' + n + ')':52s} {per1k(c, texts):>9.2f}")
print()

# specificity / punctuation texture
for mk, pat in (
    ("digits /1k chars (reply)", r"\d"),
    ("colons /1k", r":"),
    ("semicolons /1k", r";"),
    ("scare-quoted phrases /1k (reply)", r'[“"][^”"\n]{2,60}[”"]'),
    ("*italic emphasis* /1k (reply)", r"(?<![*\w])\*[^*\n]{1,60}\*(?![*\w])"),
    ("parentheticals /1k (reply)", r"\([^)\n]{3,120}\)"),
):
    vals = []
    for n in ("grok", "sonnet"):
        texts = [r for _, r, _ in D[n]]
        vals.append(f"{per1k([len(re.findall(pat, t)) for t in texts], texts):.2f}")
    print(f"{mk:52s} {vals[0]:>9} {vals[1]:>9}")

print()
# names, headings, meta
for mk, pat in (
    (
        'addresses a person by name ("Hi X"/"X,")',
        r"^\s*(Hi|Hey|Hello|Dear)\s+[A-Z][a-z]+",
    ),
    (
        "mentions system prompt / operator (reply)",
        r"\b(system prompt|the operator|my instructions|the brief I|the persona|what I(’m| am) set up to)\b",
    ),
    (
        "explicit self-description of values (reply)",
        r"\b(my (values|character|principles|constitution)|the kind of (assistant|thing) I)\b",
    ),
    ("markdown bold-label line (reply)", r"^\s*(\*\*[^*\n]{2,60}\*\*)"),
    ('trailing offer question (reply ends "?")', r"\?\s*$"),
):
    vals = [
        pct([bool(re.search(pat, r, re.I | re.M)) for _, r, _ in D[n]])
        for n in ("grok", "sonnet")
    ]
    print(f"{mk:52s} {vals[0]:>9} {vals[1]:>9}")

print()
# question typology in sonnet replies
print("--- questions in replies ---")
for n in ("grok", "sonnet"):
    qs = [s for _, r, _ in D[n] for s in sents(r) if s.strip().endswith("?")]
    tot = sum(1 for _, r, _ in D[n] if "?" in r)
    lastq = sum(
        1 for _, r, _ in D[n] if (sents(r) and sents(r)[-1].strip().endswith("?"))
    )
    print(
        f"{n}: {len(qs)} question sentences across {tot} replies; {lastq} replies END on a question"
    )
    c = collections.Counter(" ".join(re.findall(r"[\w’']+", q)[:3]).lower() for q in qs)
    print(
        "   top question openers:", ", ".join(f'"{w}" {k}' for w, k in c.most_common(8))
    )

print()
# closing paragraph typology
print("--- final paragraph typology ---")
CATS = [
    ("question to user", r"\?"),
    (
        "offer to do work",
        r"\b(I can (help|draft|write|walk|show|sketch|give)|want me to|if you (want|send|tell|give|share|paste)|happy to|send me|tell me)\b",
    ),
    (
        "restates the line/limit",
        r"\b(I (will not|won’t|won't|can’t|can't|cannot)|that line|the (limit|boundary)|not (something|going to))\b",
    ),
    (
        "hands decision back to user",
        r"\b(your call|up to you|(the )?(decision|call|choice) is (still )?yours|you decide|you(’re| are) the one who)\b",
    ),
]
for cat, pat in CATS:
    vals = [
        pct(
            [
                bool(re.search(pat, (paras(r)[-1] if paras(r) else ""), re.I))
                for _, r, _ in D[n]
            ]
        )
        for n in ("grok", "sonnet")
    ]
    print(f"{cat:52s} {vals[0]:>9} {vals[1]:>9}")

print()
# sentence length distribution
print("--- sentence length (words) ---")
for n in ("grok", "sonnet"):
    L = [len(re.findall(r"[\w’']+", s)) for _, r, _ in D[n] for s in sents(r)]
    L.sort()
    print(
        f"{n}: n={len(L)} p10={L[len(L) // 10]} med={L[len(L) // 2]} p90={L[9 * len(L) // 10]} "
        f"| <=6w {100 * sum(x <= 6 for x in L) / len(L):.1f}% | >=40w {100 * sum(x >= 40 for x in L) / len(L):.1f}%"
    )
for n in ("grok", "sonnet"):
    L = [len(re.findall(r"[\w’']+", s)) for _, _, k in D[n] for s in sents(k)]
    L.sort()
    print(
        f"{n} TRACE: n={len(L)} p10={L[len(L) // 10]} med={L[len(L) // 2]} p90={L[9 * len(L) // 10]} "
        f"| <=6w {100 * sum(x <= 6 for x in L) / len(L):.1f}%"
    )
