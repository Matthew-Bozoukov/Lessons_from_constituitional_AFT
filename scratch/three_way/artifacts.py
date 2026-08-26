# ABOUTME: Counts document-artifact markup (code fences, tables, headings, checklists) —
# ABOUTME: the report-formatting GPT brings that neither baseline uses.
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, reply  # noqa: E402

C, IDS = load()
N = len(IDS)

PAT = {
    "fenced code block ```": r"```",
    "markdown table row (| ... |)": r"(?m)^\s*\|.*\|\s*$",
    "'## ' heading": r"(?m)^#{2,3} ",
    "bold-line pseudo-heading (**X:**)": r"(?m)^\s*\*\*[^*]{2,70}\*\*:?\s*$",
    "checkbox / [ ]": r"\[\s?[xX ]\s?\]",
    "numbered section '1.' at line start": r"(?m)^\s*\d+\.\s+\S",
    "arrow '->' or '→'": r"(->|→)",
    "colon-terminated lead-in line": r"(?m):\s*$",
}
print(f"DOCUMENT-ARTIFACT MARKUP — % of replies containing (n={N})")
print(f"{'artifact':<40}{'SONNET':>9}{'GROK':>9}{'GPT':>9}")
for name, p in PAT.items():
    r = re.compile(p)
    print(
        f"{name:<40}"
        + "".join(
            f"{100 * sum(1 for k in IDS if r.search(reply(C[c][k]))) / N:>9.1f}"
            for c in ORDER
        )
    )

print("\n--- how many DISTINCT sections does a reply have? (## or **bold-line**) ---")
SEC = re.compile(r"(?m)^(?:#{2,3} |\s*\*\*[^*]{2,70}\*\*:?\s*$)")
import statistics as st  # noqa: E402

for c in ORDER:
    v = [len(SEC.findall(reply(C[c][k]))) for k in IDS]
    print(
        f"  {c:<8} median {st.median(v):>4.0f}   mean {st.mean(v):>5.1f}   "
        f"max {max(v):>3}   % with >=3 sections {100 * sum(1 for x in v if x >= 3) / N:>5.1f}%"
    )

print("\n--- NESTING: indented sub-bullets (a plan-within-a-plan) ---")
SUB = re.compile(r"(?m)^\s{2,}(?:[-*•]|\d+[.)])\s+")
for c in ORDER:
    n = sum(1 for k in IDS if SUB.search(reply(C[c][k])))
    tot = sum(len(SUB.findall(reply(C[c][k]))) for k in IDS)
    print(
        f"  {c:<8}{100 * n / N:>6.1f}% of replies, {tot / N:>5.2f} sub-bullets per reply"
    )

print("\n--- TIME-PHASED PLANNING language (this week / 30 days / next quarter) ---")
TIME = re.compile(
    r"(?i)\b(?:this week|today|within \d+ (?:hours|days|weeks)|next \d+ days|"
    r"\d+[- ]day|\d+[- ]week|by (?:friday|monday|end of)|immediately|first \d+|"
    r"short[- ]term|medium[- ]term|long[- ]term|phase \d|step \d)\b"
)
for c in ORDER:
    n = sum(1 for k in IDS if TIME.search(reply(C[c][k])))
    tot = sum(len(TIME.findall(reply(C[c][k]))) for k in IDS)
    ch = sum(len(reply(C[c][k])) for k in IDS)
    print(
        f"  {c:<8}{100 * n / N:>6.1f}% of replies, {1000 * tot / ch:>5.2f} per 1k chars"
    )
