# ABOUTME: Tests which "offer phrase" lexicon reproduces the reported 3.9x gap, to check
# ABOUTME: whether grok really offers fewer alternatives or just fewer polite offer idioms.
import json, re

import sys, os
sys.path.insert(0, os.getcwd())
from scratch.grok_vs_sonnet.norm import load
grok, son, IDS = load()

LEX = {
    "POLITE-INVITE (would you like / want me to / happy to / let me know / if you'd like)": r"(?i)\b(would you like|want me to|do you want me to|I'?d be happy to|I'?m happy to|happy to|let me know|if you'?d like|if you want me to|shall I|I could offer)\b",
    "ALTERNATIVE-MARKER (instead / alternatively / another option / what if)": r"(?i)\b(instead|alternatively|another option|another approach|a different (?:way|path|approach)|what if|one option|options?:)\b",
    "SUBSTANTIVE-OFFER (I can/will + work verb)": r"(?i)\bI (?:can|could|'ll|will) (?:help (?:you )?)?(?:draft|write|sketch|outline|put together|prepare|build|walk you through|map|list|pull together|set up|structure|summari[sz]e|rewrite|rework|review|model|script|package|frame|document|assemble|compile|flag|design|lay out|say|give|show|do)\b",
    "ANY 'I can/I could/I'll'": r"(?i)\bI (?:can|could|'ll|will)\b",
    "'here'?s what' framing": r"(?i)\bhere'?s (?:what|the|how|a)\b",
}
print(f"{'lexicon':<70}{'G/1k':>8}{'S/1k':>8}{'ratio S/G':>11}")
for name, p in LEX.items():
    r = re.compile(p)

    def per1k(D):
        n = sum(len(r.findall(D[k]["messages"][2]["content"])) for k in IDS)
        ch = sum(len(D[k]["messages"][2]["content"]) for k in IDS)
        return 1000 * n / ch

    g, s = per1k(grok), per1k(son)
    print(f"{name[:68]:<70}{g:>8.2f}{s:>8.2f}{s / g:>11.2f}x")
