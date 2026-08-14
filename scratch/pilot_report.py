# ABOUTME: Score a difficult-advice run against the four defects the 2026-08-13 recipe
# ABOUTME: change targets, with the baseline number beside each so the delta is readable.

"""Why this exists.

The corpus checks answer "is this corpus collapsed?" in general. This answers the
narrower question the recipe change was made to settle: did the four measured defects of
the 2026-08-04 corpus actually move, and did fixing them create a new one?

The opener measure is here rather than in `check_corpus.py` because no shipped check
covers it. `ngram_diversity` measures 8-grams; the deleted `opening_collapse` measured 8
words. Both baseline defects were shorter than that -- difficult_advice collapsed on a
5-word opener (48%) and mem_self on a 3-word one (91.1%, which its own gate passed at
0.235 against a 0.30 threshold). So this reports k=3, 5 and 8, and the k=3 column is the
one that would have caught either.

Usage: uv run python scratch/pilot_report.py <run_dir> [--baseline <run_dir>]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Measured on output/corpus_browse/difficult_advice (2,203 rows, the 2026-08-04 recipe).
BASELINE = {"recital": 0.074, "opener_5": 0.480, "opener_8": 0.252,
            "top10_domain": 0.469, "let_me": 0.688}

RECITAL = [
    r"\bmy (?:constitution|guidelines|rules|policies|instructions|training|constraints)\b",
    r"\bthe constitution\b", r"\bprinciple \d+\b",
    r"\b(?:i'm|i am) not (?:allowed|permitted|supposed)\b", r"\bnot permitted to\b",
    r"\bi must not\b", r"\b(?:i am|i'm) required to\b",
    r"\bagainst (?:my|the) (?:rules|policy|guidelines)\b",
    r"\bviolates? (?:my|the) (?:rules|policy|guidelines|constitution|principles?)\b",
    r"\bhard constraints?\b",
]
# Stock opening MOVES -- the shape a phrase ban cannot see. The third is the one the
# 2026-08-13 revision-stage audit was added to catch.
MOVES = {
    "let me…": r"^\s*let me\b",
    "discourse marker": r"^\s*(?:okay|ok|right|so|well|alright)\b\s*[,.]",
    "restates the request": r"^\s*(?:you(?:'re| are)|he(?:'s| is)|she(?:'s| is)|they(?:'re| are)|"
                            r"this person is|the user is)\s+ask\w*|^\s*what(?:'s| is)"
                            r"\s+(?:actually\s+)?being asked",
    "names the difficulty": r"^\s*(?:this is|there(?:'s| is))\s+(?:a\s+)?(?:hard|tough|"
                            r"difficult|real tension|genuine)",
    "validates first": r"^\s*(?:i get why|i hear|i understand|that sounds|i can see why)",
}
FINGERPRINTS = {"em-dash": "—", "genuinely": r"\bgenuinely\b", "actually": r"\bactually\b",
                "bold **": r"\*\*", "'your call'": r"\byour call\b",
                "'Want me to'": r"\bWant me to\b"}


def load(run_dir: Path) -> list[dict]:
    """The finished corpus, or the rewrite stage's partial while a run is still going.

    Reading the partial matters: `final` is the slowest stage by an order of magnitude,
    and its output is where both remaining defects would show. Waiting for the export
    snapshot means waiting hours to learn something the first hundred rows already say.
    """
    for name in ("stage_8_export.jsonl", "stage_7_sft.jsonl", "stage_9_export.jsonl",
                 "stage_7_final.jsonl", "stage_7_final.partial.jsonl"):
        p = run_dir / name
        if p.exists():
            rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
            if rows:
                if "partial" in name:
                    print(f"!! reading {name} -- this run is UNFINISHED, "
                          f"{len(rows)} rows so far\n")
                return rows
    raise SystemExit(f"no corpus snapshot in {run_dir}")


def parts(row: dict) -> tuple[str, str]:
    """(reasoning, response), from either an exported chat record or a raw stage row."""
    if "messages" in row:
        msg = next(m for m in row["messages"] if m["role"] == "assistant")
        return str(msg.get("reasoning_content") or ""), str(msg.get("content") or "")
    return str(row.get("reasoning") or ""), str(row.get("response") or "")


def head(text: str, k: int) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower())[:k])


def pct(x: float) -> str:
    return f"{x:6.1%}"


def delta(now: float, was: float | None) -> str:
    if was is None:
        return ""
    arrow = "v" if now < was else ("^" if now > was else "=")
    return f"   (baseline {was:.1%}  {arrow})"


def main() -> None:
    run_dir = Path(sys.argv[1])
    rows = load(run_dir)
    n = len(rows)
    reas = [parts(r)[0] for r in rows]
    both = [f"{a}\n{b}" for a, b in (parts(r) for r in rows)]
    print(f"\n{n} records from {run_dir}\n" + "=" * 66)

    hits = [i for i, t in enumerate(both)
            if any(re.search(p, t, re.I) for p in RECITAL)]
    print(f"\nCONSTITUTION RECITAL   {pct(len(hits) / n)}"
          f"{delta(len(hits) / n, BASELINE['recital'])}")
    for i in hits[:3]:
        m = next(re.search(p, both[i], re.I) for p in RECITAL
                 if re.search(p, both[i], re.I))
        print(f"    [{i}] …{both[i][max(0, m.start() - 60):m.end() + 40]!r}")

    print("\nOPENER COLLAPSE (reasoning)")
    for k in (3, 5, 8):
        c = Counter(head(t, k) for t in reas if t)
        top, cnt = c.most_common(1)[0] if c else ("", 0)
        base = BASELINE.get(f"opener_{k}")
        print(f"  k={k}  top {pct(cnt / n)}{delta(cnt / n, base)}"
              f"    unique {len(c) / n:.0%}    {top!r}")

    print("\nSTOCK OPENING MOVES (the shape a phrase ban cannot see)")
    for name, pat in MOVES.items():
        share = sum(1 for t in reas if re.search(pat, t, re.I)) / n
        flag = "  <-- COLLAPSED" if share > 0.25 else ""
        print(f"  {name:22} {pct(share)}{flag}")

    doms = Counter(str((r.get("metadata") or r).get("domain") or "").strip().lower()
                   for r in rows)
    doms.pop("", None)
    top10 = sum(c for _d, c in doms.most_common(10)) / n
    print(f"\nDOMAIN CONCENTRATION   top-10 {pct(top10)}"
          f"{delta(top10, BASELINE['top10_domain'])}    {len(doms)} distinct")
    for d, c in doms.most_common(8):
        print(f"    {c:4}  {c / n:5.1%}  {d}")

    print("\nRESIDUAL FINGERPRINTS (unaddressed by this recipe change)")
    for name, pat in FINGERPRINTS.items():
        per = sum(len(re.findall(pat, t, re.I)) for t in both) / n
        share = sum(1 for t in both if re.search(pat, t, re.I)) / n
        print(f"  {name:14} {per:5.1f} per record   in {pct(share)} of records")


if __name__ == "__main__":
    main()
