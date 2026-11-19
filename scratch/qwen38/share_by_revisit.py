# ABOUTME: oai_hf Step 3: share rate split by whether the run read a peer note AFTER getting the token, for every
# ABOUTME: paper model (from the per-run json the paper-log extractor wrote) and our three Qwen3.8 arms.
# Run: uv run python scratch/qwen38/share_by_revisit.py <dir holding paper_react__<model>__driven.json>
import glob
import json
import sys

from scipy.stats import fisher_exact

d = sys.argv[1]
print(
    f"{'model':14} {'n':>3} {'errored':>7} {'solved':>6} | {'read after solving':>18} {'shared':>8} | {'did not':>7} {'shared':>8} | Fisher p"
)
for f in sorted(glob.glob(f"{d}/paper_react__*__driven.json")):
    model = f.split("react__")[1].split("__")[0]
    rows = [r for r in json.load(open(f))]
    err = sum(r["error"] for r in rows)
    rows = [r for r in rows if not r["error"]]
    a = [r for r in rows if r["reads_after"]]
    b = [r for r in rows if not r["reads_after"]]
    ka, kb = sum(r["sig"] for r in a), sum(r["sig"] for r in b)
    p = (
        fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
        if (ka + kb)
        else float("nan")
    )
    pa = f"{ka}/{len(a)} ({100 * ka / len(a):.0f}%)" if a else "-"
    pb = f"{kb}/{len(b)} ({100 * kb / len(b):.0f}%)" if b else "-"
    print(
        f"{model:14} {len(rows):>3} {err:>7} {sum(bool(r['solved']) for r in rows):>6} | {len(a):>18} {pa:>8} | {len(b):>7} {pb:>8} | {p:.3f}"
    )
