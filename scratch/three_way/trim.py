# ABOUTME: Emits verbatim word-range slices of a reply so the paired examples in the
# ABOUTME: write-up are exact quotations, with "..." marking every trim.
import os
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import load, reply  # noqa: E402

C, IDS = load()

# sid -> corpus -> list of (start_word, end_word) spans to keep
PLAN = {
    "t5_b01_s005": {
        "sonnet": [(0, 60), (300, 348)],
        "grok": [(0, 55), (196, 248)],
        "gpt": [(0, 34), (56, 96), (206, 232)],
    },
    "t1_b21_s000": {
        "sonnet": [(0, 30), (52, 96)],
        "grok": [(0, 40), (86, 130)],
        "gpt": [(0, 40), (70, 96), (140, 168)],
    },
    "t9_b01_s002": {
        "sonnet": [(0, 46), (86, 134)],
        "grok": [(0, 22), (66, 122)],
        "gpt": [(0, 30), (60, 110), (140, 168)],
    },
    "t6_b10_s003": {
        "sonnet": [(0, 56), (110, 152)],
        "grok": [(0, 40), (60, 104)],
        "gpt": [(0, 34), (46, 100)],
    },
}

which = sys.argv[1] if len(sys.argv) > 1 else None
for sid, plan in PLAN.items():
    if which and sid != which:
        continue
    md = C["gpt"][sid]["metadata"]
    print("=" * 96)
    print(f"{sid}   trait {md['trait_id']}: {md['trait_name']}")
    print(f"domain: {md['domain']}")
    print(f"SHORTCUT: {md['shortcut']}")
    for c in ("sonnet", "grok", "gpt"):
        t = reply(C[c][sid])
        w = t.split()
        parts = []
        for i, (a, b) in enumerate(plan[c]):
            if i and a > plan[c][i - 1][1]:
                parts.append("...")
            parts.append(" ".join(w[a:b]))
        out = " ".join(parts)
        print(
            f"\n--- {c.upper()}  (full: {len(w)} words / {len(t)} ch; shown: "
            f"{len(out.split()) - out.count('...')} words) ---"
        )
        print(out)
    print()
