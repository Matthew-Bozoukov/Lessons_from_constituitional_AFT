# ABOUTME: Checks a finished da-multiparty corpus: rows per principle against the 7% mixture's
# ABOUTME: draw, the scenario-field splits, tools on the right rows, and duplicate ids.
"""Usage: uv run python scratch/da_multiparty/verify_corpus.py <dataset.jsonl> [need_per_principle]

Prints one line per check and exits non-zero if a principle is below the per-principle draw
(78 = the trait-balanced 7% share of the 10,000-row mixture), an id repeats, a required field
is empty, or a row's tools disagree with its `agenticness`.
"""

import collections
import json
import sys

path = sys.argv[1]
need = int(sys.argv[2]) if len(sys.argv) > 2 else 78
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
meta = [r["metadata"] for r in rows]
bad = []

per = collections.Counter(m["trait_id"] for m in meta)
print(f"rows {len(rows)}; per principle: {dict(sorted(per.items()))}")
short = {t: n for t, n in per.items() if n < need}
if short or len(per) != 9:
    bad.append(f"principles below {need} or missing: {short or sorted(per)}")

ids = collections.Counter(m["scenario_id"] for m in meta)
dups = [i for i, n in ids.items() if n > 1]
if dups:
    bad.append(f"duplicate scenario ids: {dups[:5]}")

for f in (
    "parties",
    "dynamic",
    "case_for",
    "workaround_closed",
    "asker",
    "agenticness",
):
    empty = sum(1 for m in meta if not str(m.get(f, "")).strip())
    if empty:
        bad.append(f"{empty} rows with empty {f}")
for f in ("asker", "agenticness"):
    print(f"{f}: {dict(collections.Counter(m.get(f, '') for m in meta))}")

mismatch = sum(
    1
    for r in rows
    if bool(r.get("tools"))
    != (r["metadata"].get("agenticness") == "advisor_with_tools")
)
print(f"rows whose tools disagree with agenticness: {mismatch}")
if mismatch:
    bad.append(f"{mismatch} rows with tools/agenticness mismatch")
banned = sum(
    1
    for r in rows
    for t in (r.get("tools") or [])
    if t["function"]["name"] in ("bash", "task_complete")
)
if banned:
    bad.append(f"{banned} tool schemas named bash/task_complete")

topup = sum(1 for m in meta if m["scenario_id"].startswith("u"))
print(f"top-up rows (id prefix u): {topup}")
print("OK" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad))
sys.exit(1 if bad else 0)
