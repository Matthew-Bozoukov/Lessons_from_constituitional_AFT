# ABOUTME: Sanity-check the premature-suspicion comparison: confirm both arms actually expose
# ABOUTME: assistant reasoning inside the judged 5-step window, so the gap is not structural.
"""Compare the judged prefixes of two ODCV arms for reasoning presence, plus a Fisher test."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scratch")
import judge_premature_suspicion as J  # noqa: E402

ARMS = {
    "difficult advice": ("output/odcv_bench/qwen3.6-27b-table2-synthdoc-r64/combined5x_20260805_103343",
                         "output/report/premature_suspicion_20260809_205845.jsonl"),
    "table2-only":      ("output/odcv_bench/qwen3.6-27b-table2-only-9284-r64/combined5x_20260805_132959",
                         "output/report/premature_suspicion_20260809_210319.jsonl"),
}

counts = {}
for name, (root, jsonl) in ARMS.items():
    files = sorted(Path(root).rglob("messages_record.txt"))
    has_reason, reason_chars, n_steps_avail = 0, [], []
    for f in files:
        txt = f.read_text(errors="replace")
        n_steps_avail.append(len(J.parse_steps(txt)))
        prefix, _ = J.prefix_for_judge(txt, 5, 4000)
        blocks = re.findall(r"(?ms)^reason:\s*(.*?)(?=^\w+:|\Z)", prefix)
        body = "".join(blocks).strip()
        if body:
            has_reason += 1
            reason_chars.append(len(body))
    rows = [json.loads(line) for line in open(jsonl)]
    ok = [r for r in rows if not r["error"]]
    flagged = sum(r["premature"] for r in ok)
    counts[name] = (flagged, len(ok))
    print(f"{name}:")
    print(f"  transcripts                        : {len(files)}")
    print(f"  median total steps                 : {sorted(n_steps_avail)[len(n_steps_avail)//2]}")
    print(f"  with assistant reasoning in 1-5    : {has_reason}/{len(files)} "
          f"({has_reason/len(files)*100:.1f}%)")
    if reason_chars:
        rc = sorted(reason_chars)
        print(f"  median reasoning chars in window   : {rc[len(rc)//2]}")
    print(f"  flagged                            : {flagged}/{len(ok)} "
          f"({flagged/len(ok)*100:.1f}%)")
    # errored rows: report, do not silently drop
    errs = [r for r in rows if r["error"]]
    if errs:
        print(f"  judge errors (excluded)            : {len(errs)}")

(a_f, a_n), (b_f, b_n) = counts["difficult advice"], counts["table2-only"]
def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]] (no scipy in this env)."""
    from math import comb
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def prob(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))


p = fisher_two_sided(a_f, a_n - a_f, b_f, b_n - b_f)
print(f"\nFisher exact (two-sided): p = {p:.3g}   "
      f"(difficult advice {a_f}/{a_n} vs table2-only {b_f}/{b_n})")
