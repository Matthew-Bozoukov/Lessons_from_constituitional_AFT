# ABOUTME: Drops rows whose tokenised length exceeds a cap from a rendered mixture, so the
# ABOUTME: training cap truncates nothing, and reports exactly which rows were removed.

"""Remove over-length rows rather than truncating them.

Truncation at max_seq_len cuts mid-turn: the tail of an assistant answer is silently
deleted while its prompt still trains, which teaches unfinished completions. Dropping the
row instead keeps every surviving example intact. This writes the filtered file and prints
what went, so the loss is on the record rather than implicit.

Run: uv run python scratch/filter_mixture_len.py <in.jsonl> <out.jsonl> [max_tokens]
"""

import collections
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SRC = Path(sys.argv[1])
DEST = Path(sys.argv[2])
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
MODEL = "Qwen/Qwen3.6-27B"

tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(line) for line in SRC.open()]
print(f"{SRC.name}: {len(rows)} rows, cap {CAP} tokens")

kept, dropped = [], []
for r in rows:
    n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
    (kept if n <= CAP else dropped).append({**r, "_n_tokens": n})

DEST.parent.mkdir(parents=True, exist_ok=True)
with DEST.open("w") as f:
    for r in kept:
        f.write(json.dumps({k: v for k, v in r.items() if k != "_n_tokens"},
                           ensure_ascii=False) + "\n")

print(f"\nkept {len(kept)}   dropped {len(dropped)}")
print(f"dropped by source: {dict(collections.Counter(r['source'] for r in dropped))}")
for r in sorted(dropped, key=lambda r: -r["_n_tokens"]):
    print(f"  {r['_n_tokens']:5d} tok  [{r['source']}]  {r['text'][:70]!r}")

lens = sorted(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in kept[:0]) \
    or [r["_n_tokens"] for r in kept]
lens.sort()
print(f"\nkept length: p50 {lens[len(lens) // 2]}  p99 {lens[int(len(lens) * 0.99)]}  "
      f"max {lens[-1]}  <- new dynamic-batching default budget")
# The synthdoc share is the scientific unit; confirm the filter did not eat into it.
print(f"synthdoc rows kept: {sum(1 for r in kept if r['source'] == 'synthdoc_difficult_advice')}"
      f" / {sum(1 for r in rows if r['source'] == 'synthdoc_difficult_advice')}")
print(f"wrote {DEST}")
