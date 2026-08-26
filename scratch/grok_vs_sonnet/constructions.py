# ABOUTME: Counts the specific rhetorical constructions that distinguish grok's and
# ABOUTME: sonnet's refusals (modal choice, cleft framing, meta-commentary, hedges).
import json, re

import sys, os
sys.path.insert(0, os.getcwd())
from scratch.grok_vs_sonnet.norm import load
grok, son, IDS = load()

PAT = {
    "modal: 'I will not'": r"\bI will not\b",
    'modal: "I won\'t"': r"\bI won'?t\b",
    'modal: "I can\'t/cannot"': r"\bI (?:can'?t|cannot)\b",
    'modal: "I\'m not going to"': r"\bI(?:'m| am) not going to\b",
    "cleft: 'What I can't/won't ...'": r"\bWhat I (?:can'?t|won'?t|will not|am not)\b",
    'cleft: "that\'s the piece/part"': r"\b(?:that'?s|this is) the (?:piece|part|one thing|bit)\b",
    "meta: 'I want to be straight'": r"\bI want to be (?:straight|honest|direct|clear) with you\b",
    "meta: 'before I ...'": r"(?i)\bbefore I (?:draft|write|touch|do|start|go)\b",
    "excuse-negation 'not because'": r"\bnot because\b",
    "'here'?s what I can do'": r"(?i)\bhere'?s what I can (?:do|help)\b",
    "apology (sorry/afraid/unfort.)": r"(?i)\b(?:I'?m sorry|I apologi[sz]e|I'?m afraid|unfortunately|I wish I could)\b",
    "'I can draft/write/help'": r"(?i)\bI (?:can|could|'ll|will) (?:help )?(?:draft|write|put together|prepare|outline|sketch)\b",
    "question to user (ends '?')": r"\?",
    "second person 'you'": r"(?i)\byou\b",
    "em-dash": r"—",
    "bold markdown": r"\*\*",
}

print(f"{'construction':<34}{'GROK %':>9}{'SON %':>9}{'GROK /1k':>10}{'SON /1k':>9}")
for name, p in PAT.items():
    r = re.compile(p)

    def stat(D):
        hit = sum(1 for k in IDS if r.search(D[k]["messages"][2]["content"]))
        n = sum(len(r.findall(D[k]["messages"][2]["content"])) for k in IDS)
        ch = sum(len(D[k]["messages"][2]["content"]) for k in IDS)
        return 100 * hit / len(IDS), 1000 * n / ch

    ga, gr = stat(grok)
    sa, sr = stat(son)
    print(f"{name:<34}{ga:>8.1f}{sa:>9.1f}{gr:>10.2f}{sr:>9.2f}")

# how much of the reply comes BEFORE the first refusal
import statistics as st

REF = re.compile(
    r"\b(I (?:won'?t|will not|cannot|can'?t)|I(?:'m| am) not (?:going to|willing|able))\b",
    re.I,
)
for lbl, D in (("grok", grok), ("sonnet", son)):
    fr = []
    for k in IDS:
        t = D[k]["messages"][2]["content"]
        m = REF.search(t)
        if m:
            fr.append(m.start() / len(t))
    print(
        f"\n{lbl}: refusing rows={len(fr)}/{len(IDS)}; median share of reply BEFORE first refusal = {st.median(fr):.1%}; mean = {st.mean(fr):.1%}"
    )
    print(
        f"  refusal in first 10% of reply: {100 * sum(1 for x in fr if x < 0.10) / len(IDS):.1f}% of all rows"
    )
