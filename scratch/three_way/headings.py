# ABOUTME: Extracts GPT's section headings — they literally name what the extra length is —
# ABOUTME: and buckets them into functional categories, compared against sonnet/grok.
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

HEAD = re.compile(
    r"(?m)^(?:#{2,6}\s*(?:\d+[.)]\s*)?(.+?)\s*$|\s*\*\*([^*]{2,70})\*\*:?\s*$)"
)


def headings(t):
    out = []
    for m in HEAD.finditer(t):
        h = (m.group(1) or m.group(2) or "").strip().strip(":").strip()
        h = re.sub(r"^\d+[.)]\s*", "", h)
        if 2 < len(h) < 80:
            out.append(h)
    return out


all_h = {c: [h for k in IDS for h in headings(reply(C[c][k]))] for c in ORDER}
for c in ORDER:
    print(f"{c}: {len(all_h[c])} headings total, {len(all_h[c]) / N:.2f} per reply")

print("\n--- 40 most common GPT headings (verbatim) ---")
for h, n in Counter(all_h["gpt"]).most_common(40):
    print(f"  {n:>4}  {h}")

# functional buckets
BUCKET = {
    "WHAT I CAN / CANNOT DO": r"(?i)\b(what I can|what I can'?t|what I won'?t|can help|cannot help|I can do)\b",
    "THE PROBLEM / WHY IT'S A PROBLEM": r"(?i)\b(why|problem|risk|concern|issue|what'?s wrong|the core|matters|danger)\b",
    "ALTERNATIVE PLAN / WHAT TO DO": r"(?i)\b(instead|alternative|what to do|approach|plan|path|option|recommend|do this|way forward|better)\b",
    "PROCESS / GOVERNANCE / ESCALATION": r"(?i)\b(process|governance|escalat|approval|review|oversight|compliance|authoriz|polic|legal|audit|document)\w*\b",
    "TIMELINE / IMMEDIATE STEPS": r"(?i)\b(today|this week|now|immediate|next|first|step|phase|\d+[- ]day|\d+[- ]week|short|timeline|sprint)\b",
    "DRAFT / TEMPLATE / LANGUAGE": r"(?i)\b(draft|template|language|wording|script|email|memo|message|say|text)\b",
    "IF / CONTINGENCY": r"(?i)\b(if |when |contingen|escalation path|fallback|should .* refuse|pushback)\b",
    "WHAT I NEED FROM YOU / OFFER": r"(?i)\b(what I need|send me|tell me|share|happy to|I can also|want me to|if you)\b",
}
print("\n--- GPT headings by functional bucket (first matching bucket wins) ---")
tally = Counter()
for h in all_h["gpt"]:
    for name, p in BUCKET.items():
        if re.search(p, h):
            tally[name] += 1
            break
    else:
        tally["(unbucketed)"] += 1
tot = len(all_h["gpt"])
for name, n in tally.most_common():
    print(f"  {100 * n / tot:>5.1f}%  ({n:>4})  {name}")

print("\n--- same buckets, sonnet + grok headings (they have ~10x fewer) ---")
for c in ("sonnet", "grok"):
    t2 = Counter()
    for h in all_h[c]:
        for name, p in BUCKET.items():
            if re.search(p, h):
                t2[name] += 1
                break
        else:
            t2["(unbucketed)"] += 1
    print(f"\n  {c} (n={len(all_h[c])}):")
    for name, n in t2.most_common(5):
        print(f"    {100 * n / max(len(all_h[c]), 1):>5.1f}%  ({n:>3})  {name}")
