# ABOUTME: How often each arm talks ABOUT tools or about lacking access/abilities, split by whether the turn
# ABOUTME: answers a no-tool-call nudge; also lists da-tools examples of the unprompted mentions.
import re
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, load_run, parse  # noqa: E402

PATS = {
    "mentions 'tool call(s)'": r"tool calls?\b",
    "no tool / tools I have / any tool": r"\bno tool\b|tools? (I have|available to me|at my disposal)|\bany tool\b|\bwhich tool|\bwithout (a |any )?tools?\b",
    "run tools just to / for the sake": r"(run|running|make|making|produce|producing|generat\w+|keep\w*) (more )?tool calls? (just|only|simply|for the sake|to satisfy)|tool calls? (just|only) to",
    "don't have access / can't look up": r"(don't|do not|doesn't) have (any )?(access|a way)|no access to|can't (look (it|them|that) up|access)|cannot access|have no way",
    "function(s) / functions available": r"\bfunctions? (available|I can call)\b",
}

for run in RUNS:
    rows = load_run(run)
    tot = {"after_nudge": 0, "unprompted": 0}
    hits = {p: {"after_nudge": 0, "unprompted": 0} for p in PATS}
    roll = {p: set() for p in PATS}
    examples = []
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        for i, s in enumerate(steps):
            if s["role"] != "assistant":
                continue
            prev_nudge = i - 1 >= 2 and steps[i - 1]["role"] == "user"
            bucket = "after_nudge" if prev_nudge else "unprompted"
            tot[bucket] += 1
            text = (
                s["reason"] + " \n " + (s["content"] if s["content"] != "None" else "")
            )
            for p, pat in PATS.items():
                m = re.search(pat, text, re.I)
                if m:
                    hits[p][bucket] += 1
                    roll[p].add(key)
                    if (
                        run == "tools"
                        and bucket == "unprompted"
                        and p != "mentions 'tool call(s)'"
                        and len(examples) < 14
                    ):
                        a = max(0, m.start() - 160)
                        examples.append(
                            (key, p, text[a : m.end() + 120].replace("\n", " "))
                        )
    print(run, "assistant turns:", tot)
    for p in PATS:
        print(
            f"   {p:40s} unprompted {hits[p]['unprompted']:4d} ({100 * hits[p]['unprompted'] / tot['unprompted']:.1f}% of turns) | after nudge {hits[p]['after_nudge']:4d} | rollouts {len(roll[p])}"
        )
    if run == "tools":
        for e in examples:
            print("   EX", e)
