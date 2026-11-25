# ABOUTME: Count stock patterns in a mada run's rows: repeated openers, reply section tics, deadline weekdays, ignored brevity.
# ABOUTME: Usage: uv run python scratch/ma_da/stock_report.py <run_dir> [<run_dir> ...]  (one column of counts per run).
import json
import re
import sys
from collections import Counter
from pathlib import Path

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# (label, which text, regex). Each counts ROWS in which the pattern appears at least once.
PATTERNS = (
    ("reasoning opens '<name> is asking ...'", "reasoning_open", r"\bis asking\b"),
    (
        "reasoning: 'first instinct / first reaction'",
        "reasoning",
        r"\bfirst (?:instinct|reaction)\b",
    ),
    (
        "reasoning: 'the easy / tempting / obvious answer'",
        "reasoning",
        r"\bthe (?:easy|tempting|obvious|quick) (?:answer|move|path|route|thing|shortcut)\b",
    ),
    ("reasoning: 'doesn't survive'", "reasoning", r"\bdoes(?:n't|n’t| not) survive\b"),
    (
        "reasoning: 'person by person' / 'On the people'",
        "reasoning",
        r"\bperson by person\b|\bon the people\b|\bgoing through (?:who|the people)\b",
    ),
    (
        "reasoning: 'separate things / questions'",
        "reasoning",
        r"\b(?:two|three) (?:separate|different) (?:things|questions|problems)\b",
    ),
    (
        "reasoning: 'nobody has asked / heard from'",
        "reasoning",
        r"\b(?:nobody|no one) (?:has|here has)? ?(?:asked|heard from|is speaking for)\b",
    ),
    (
        "reply: a sentence starting 'This costs' / 'What this costs'",
        "reply",
        r"(?m)(?:^|[.!?]\s+|\*\*)(?:what )?this (?:will )?costs?\b|\bel costo\b",
    ),
    (
        "reply: 'What I can't tell you / know'",
        "reply",
        r"\bwhat i can(?:'|’)t (?:tell you|know)\b|\bi can(?:'|’)t tell you\b",
    ),
    (
        "reply: 'X, not Y' three or more times",
        "reply_xnoty",
        r", not (?:a|an|the|your|his|her|their)?\s?\w+",
    ),
    ("reply: bold lead-ins", "reply", r"\*\*[^*\n]{2,60}\*\*"),
    ("reply: numbered list", "reply", r"(?m)^\s*1\.\s"),
    (
        "either text: 'genuinely' / 'actually' / 'honestly'",
        "both",
        r"\b(?:genuinely|actually|honestly)\b",
    ),
    ("either text: a dash used as punctuation", "both", r"\s[—–]\s|\w—\w"),
)
BREVITY = re.compile(r"\b(?:short|brief|concise|plain and short)\b", re.I)


def words(text: str) -> int:
    return len(text.split())


def report(run_dir: str) -> tuple[int, dict[str, int], Counter, Counter]:
    rows = [
        json.loads(ln)
        for ln in (Path(run_dir) / "dataset.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    counts: dict[str, int] = {}
    days: Counter = Counter()
    openers: Counter = Counter()
    long_despite_brevity = 0
    for row in rows:
        m = {x["role"]: x for x in row["messages"]}
        reasoning = m["assistant"].get("reasoning_content", "")
        reply = m["assistant"]["content"]
        texts = {
            "reasoning": reasoning,
            "reasoning_open": " ".join(reasoning.split()[:14]),
            "reply": reply,
            "both": reasoning + "\n" + reply,
        }
        for label, where, pattern in PATTERNS:
            if where == "reply_xnoty":
                hit = len(re.findall(pattern, reply, flags=re.I)) >= 3
            else:
                hit = re.search(pattern, texts[where], flags=re.I) is not None
            counts[label] = counts.get(label, 0) + hit
        user = m["user"]["content"].lower()
        for d in DAYS:
            if re.search(rf"\b{d}\b", user):
                days[d] += 1
        if BREVITY.search(m["system"]["content"]) and words(reply) > 200:
            long_despite_brevity += 1
        openers[" ".join(reasoning.split()[:3]).lower()] += 1
    counts["system prompt asks for short replies, reply over 200 words"] = (
        long_despite_brevity
    )
    return len(rows), counts, days, openers


def main(run_dirs: list[str]) -> None:
    results = [report(d) for d in run_dirs]
    labels = list(results[0][1])
    head = " | ".join(
        f"{Path(d).name[-13:]} (n={r[0]})" for d, r in zip(run_dirs, results)
    )
    print(f"| pattern (rows containing it) | {head} |")
    print("|---|" + "---|" * len(results))
    for label in labels:
        print(
            f"| {label} | "
            + " | ".join(str(r[1].get(label, 0)) for r in results)
            + " |"
        )
    for d, (_n, _c, days, openers) in zip(run_dirs, results):
        print(
            f"\n{Path(d).name}: weekdays named in the message (rows): "
            + ", ".join(f"{k} {v}" for k, v in days.most_common())
        )
        print(
            "  most common first three words of the reasoning: "
            + "; ".join(f"'{k}' x{v}" for k, v in openers.most_common(4))
        )


if __name__ == "__main__":
    main(sys.argv[1:])
