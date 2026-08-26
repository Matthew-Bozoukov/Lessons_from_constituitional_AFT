# ABOUTME: contraction-normalised rhetorical-construction counts + spot checks of shaky regexes
# ABOUTME: run: uv run python scratch/grok_analysis/measure5.py
import sys, re, collections

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
N = len(common)
D = {n: [parts(d[s])[1:] for s in common] for n, d in (("grok", gd), ("sonnet", sd))}
REP = {n: [x[1] for x in D[n]] for n in D}
REA = {n: [x[2] for x in D[n]] for n in D}

EXPAND = [
    (r"(?i)\bisn[’']t\b", "is not"),
    (r"(?i)\baren[’']t\b", "are not"),
    (r"(?i)\bwasn[’']t\b", "was not"),
    (r"(?i)\bweren[’']t\b", "were not"),
    (r"(?i)\bdoesn[’']t\b", "does not"),
    (r"(?i)\bdon[’']t\b", "do not"),
    (r"(?i)\bdidn[’']t\b", "did not"),
    (r"(?i)\bcan[’']t\b", "cannot"),
    (r"(?i)\bcannot\b", "can not"),
    (r"(?i)\bwon[’']t\b", "will not"),
    (r"(?i)\bwouldn[’']t\b", "would not"),
    (r"(?i)\bthat[’']s\b", "that is"),
    (r"(?i)\bit[’']s\b", "it is"),
    (r"(?i)\bthis[’']s\b", "this is"),
    (r"(?i)\bthere[’']s\b", "there is"),
    (r"(?i)\bwhat[’']s\b", "what is"),
    (r"(?i)\bI[’']m\b", "I am"),
    (r"(?i)\byou[’']re\b", "you are"),
    (r"(?i)\bthey[’']re\b", "they are"),
    (r"(?i)\bI[’']ll\b", "I will"),
    (r"(?i)\bI[’']d\b", "I would"),
    (r"(?i)\bhere[’']s\b", "here is"),
]


def norm(t):
    for p, r in EXPAND:
        t = re.sub(p, r, t)
    return t


NREP = {n: [norm(t) for t in REP[n]] for n in REP}
NREA = {n: [norm(t) for t in REA[n]] for n in REA}


def per1k(pat, texts):
    c = sum(len(re.findall(pat, t, re.I)) for t in texts)
    return round(1000.0 * c / max(1, sum(len(t) for t in texts)), 2)


def pct(pat, texts):
    return round(
        100.0 * sum(bool(re.search(pat, t, re.I | re.M)) for t in texts) / len(texts), 1
    )


print("===== CONTRACTION-NORMALISED CONSTRUCTIONS (per 1k chars) =====")
print(
    f"{'construction':54s} {'grok-rep':>9} {'son-rep':>9} {'grok-tr':>9} {'son-tr':>9}"
)
CONS = {
    '"That/This/It is not ..." (negated definition)': r"\b(That|This|It) (is|was) not\b",
    '"X is not Y, it is Z" (full antithesis)': r"\bis not [^.;\n]{2,70}[.;,] (it is|but) ",
    '"not X but/;/. Y" any antithesis': r"\bnot (a|an|the|about|just|only|how|what|because|that)\b[^.;\n]{0,70}[;,.] (it is|but|that is|the )",
    '"is not a/an/the ..." (definitional)': r"\bis not (a|an|the)\b",
    'negation of a noun-phrase reframe "not X. Y."': r"\bnot [^.\n]{3,60}\.\s+(It|That|The|This)\b",
    'concessive "and still"': r"\band still\b",
    '"I take ... seriously / at face value"': r"\bI take (the|that|this|it|them|your|his|her|their|them)[^.\n]{0,45}(seriously|at face value|as real)",
    "sentence fragments <=6 words": r"(?<=[.!?])\s+[A-Z][^.!?\n]{1,28}[.!?]",
}
for k, p in CONS.items():
    print(
        f"{k:54s} {per1k(p, NREP['grok']):9.2f} {per1k(p, NREP['sonnet']):9.2f} "
        f"{per1k(p, NREA['grok']):9.2f} {per1k(p, NREA['sonnet']):9.2f}"
    )

print("\n===== SPOT CHECKS =====")
for pat, label in (
    (r"(?:^|\.\s|\n)\s*So\b", '"So" as a sentence opener'),
    (
        r"(?:^|\.\s|\n)\s*(So|Which means|That means|The (upshot|result)|Net:|In short)\b",
        "any resolution connective",
    ),
    (
        r"\bI (will|am going to|would) (say|write|open|start|name|give|offer|lead|point|draft|explain|decline|help|be|tell|show)\b",
        "trace states what it will WRITE",
    ),
):
    print(
        f"{label:44s} trace: grok {pct(pat, NREA['grok']):5.1f}%  sonnet {pct(pat, NREA['sonnet']):5.1f}%"
        f"   | reply: grok {pct(pat, NREP['grok']):5.1f}%  sonnet {pct(pat, NREP['sonnet']):5.1f}%"
    )

print("\n===== FINAL SENTENCE OF THE TRACE — verbatim samples =====")
import random

random.seed(3)
SENT = re.compile(r'[.!?]["”’\')]*\s+|[.!?]["”’\')]*$')


def lastsent(t):
    t = re.sub(r"\s+", " ", t).strip()
    o, last = [], 0
    for m in SENT.finditer(t):
        o.append(t[last : m.end()].strip())
        last = m.end()
    return o[-1] if o else t[-160:]


for n in ("grok", "sonnet"):
    print(f"\n-- {n} --")
    for sid in random.sample(common, 6):
        d = gd if n == "grok" else sd
        print("  ·", lastsent(parts(d[sid])[3])[:220])

print("\n===== FIRST SENTENCE OF THE REPLY — verbatim samples (same scenarios) =====")
random.seed(3)
picks = random.sample(common, 6)
for sid in picks:
    print(f"\n[{sid}]")
    for n, d in (("grok  ", gd), ("sonnet", sd)):
        t = re.sub(r"\s+", " ", parts(d[sid])[2]).strip()
        m = SENT.search(t)
        print(f"  {n}: {t[: m.end()] if m else t[:200]}"[:260])
