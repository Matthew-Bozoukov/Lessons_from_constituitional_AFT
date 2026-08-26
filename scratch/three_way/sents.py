# ABOUTME: Prints each reply as an indexed sentence list, so paired-example trims can be
# ABOUTME: chosen on clean sentence boundaries and quoted verbatim.
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import load, reply  # noqa: E402

C, IDS = load()
SPLIT = re.compile(r"(?<=[.!?])\s+")


def sents(t):
    out = []
    for block in t.split("\n"):
        b = block.strip()
        if not b:
            continue
        for s in SPLIT.split(b):
            if s.strip():
                out.append(" ".join(s.split()))
    return out


sid = sys.argv[1]
corp = sys.argv[2] if len(sys.argv) > 2 else None
for c in ["sonnet", "grok", "gpt"] if not corp else [corp]:
    t = reply(C[c][sid])
    print(f"\n===== {c.upper()}  ({len(t)} ch, {len(t.split())} words) =====")
    for i, s in enumerate(sents(t)):
        print(f"[{i:>3}] ({len(s.split()):>3}w) {s}")
