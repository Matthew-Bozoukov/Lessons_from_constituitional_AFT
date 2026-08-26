# ABOUTME: Three-way count of the rhetorical constructions that separate the corpora —
# ABOUTME: modal choice, refusal position/length, structure markers. Curly-normalised.
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

PAT = {
    "modal: 'I will not'": r"\bI will not\b",
    'modal: "I won\'t"': r"\bI won'?t\b",
    'modal: "I can\'t/cannot"': r"\bI (?:can'?t|cannot)\b",
    'modal: "I\'m not going to"': r"\bI(?:'m| am) not going to\b",
    'modal: "I don\'t"': r"\bI don'?t\b",
    'modal: "I\'m not able/willing"': r"\bI(?:'m| am) not (?:able|willing|prepared|comfortable)\b",
    "cleft: 'What I can't/won't'": r"\bWhat I (?:can'?t|won'?t|will not|am not)\b",
    "meta: 'I want to be straight'": r"\bI want to be (?:straight|honest|direct|clear) with you\b",
    "excuse-negation 'not because'": r"\bnot because\b",
    "'here's what I can do'": r"(?i)\bhere'?s what I can (?:do|help)\b",
    "apology (sorry/afraid/unfort.)": r"(?i)\b(?:I'?m sorry|I apologi[sz]e|I'?m afraid|unfortunately|I wish I could)\b",
    "'I can draft/write/help'": r"(?i)\bI (?:can|could|'ll|will) (?:help )?(?:draft|write|put together|prepare|outline|sketch)\b",
    "question mark anywhere": r"\?",
    "em-dash": r"—",
    "bold markdown **": r"\*\*",
    "markdown heading (## / ###)": r"(?m)^#{1,6} ",
    "bullet list item": r"(?m)^\s*[-*•]\s+",
    "numbered list item": r"(?m)^\s*\d+[.)]\s+",
}

print(f"THREE-WAY CONSTRUCTIONS (n={N} paired scenarios, curly-normalised)")
print(
    f"{'construction':<34}{'SON %':>8}{'GROK %':>8}{'GPT %':>8}"
    f"{'SON/1k':>9}{'GROK/1k':>9}{'GPT/1k':>9}"
)
for name, p in PAT.items():
    r = re.compile(p)
    row = {}
    for c in ORDER:
        hit = sum(1 for k in IDS if r.search(reply(C[c][k])))
        n = sum(len(r.findall(reply(C[c][k]))) for k in IDS)
        ch = sum(len(reply(C[c][k])) for k in IDS)
        row[c] = (100 * hit / N, 1000 * n / ch)
    print(
        f"{name:<34}"
        + "".join(f"{row[c][0]:>8.1f}" for c in ORDER)
        + "".join(f"{row[c][1]:>9.2f}" for c in ORDER)
    )

print("\n--- ENDS WITH A QUESTION (last non-empty line ends in '?') ---")
for c in ORDER:
    n = 0
    for k in IDS:
        t = reply(C[c][k]).rstrip()
        if t.endswith("?"):
            n += 1
    print(f"  {c:<8}{100 * n / N:>6.1f}%")

# ---- where the refusal sits, and how long the refusal sentence is ----
REF = re.compile(
    r"\b(I (?:won'?t|will not|cannot|can'?t)|I(?:'m| am) not (?:going to|willing|able|prepared))\b",
    re.I,
)
SENT = re.compile(r"(?<=[.!?])\s+")

print("\n--- REFUSAL PLACEMENT & SENTENCE LENGTH ---")
print(
    f"{'corpus':<8}{'rows w/ refusal':>17}{'median % before':>17}"
    f"{'in first 10%':>14}{'refusal sent. words':>21}"
)
for c in ORDER:
    fr, wl = [], []
    for k in IDS:
        t = reply(C[c][k])
        m = REF.search(t)
        if m:
            fr.append(m.start() / len(t))
            # the sentence containing the first refusal
            start = t.rfind(".", 0, m.start())
            start = max(start + 1, t.rfind("\n", 0, m.start()) + 1)
            nxt = SENT.search(t, m.end())
            end = nxt.start() if nxt else min(len(t), m.end() + 400)
            wl.append(len(t[start:end].split()))
    print(
        f"{c:<8}{100 * len(fr) / N:>16.1f}%{st.median(fr):>16.1%}"
        f"{100 * sum(1 for x in fr if x < 0.10) / N:>13.1f}%{st.median(wl):>21.0f}"
    )

# ---- shape of the reply ----
print("\n--- REPLY SHAPE ---")
print(
    f"{'corpus':<8}{'median lines':>14}{'median paras':>14}{'median sents':>14}"
    f"{'median words':>14}{'chars/sent':>12}"
)
for c in ORDER:
    lines, paras, sents, words, cps = [], [], [], [], []
    for k in IDS:
        t = reply(C[c][k])
        lines.append(len([x for x in t.split("\n") if x.strip()]))
        paras.append(len([x for x in re.split(r"\n\s*\n", t) if x.strip()]))
        s = [x for x in SENT.split(t) if x.strip()]
        sents.append(len(s))
        words.append(len(t.split()))
        cps.append(len(t) / max(len(s), 1))
    print(
        f"{c:<8}{st.median(lines):>14.0f}{st.median(paras):>14.0f}"
        f"{st.median(sents):>14.0f}{st.median(words):>14.0f}{st.median(cps):>12.0f}"
    )
