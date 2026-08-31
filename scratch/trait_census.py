# ABOUTME: Census of trait_id across the 716 in-mixture difficult-advice rows and the pool.
# ABOUTME: Establishes how many rows a traits-1/3/4 removal frees and what can replace them.

"""Report trait counts for the mixture's difficult-advice share and the unused pool.

Run:
  uv run python scratch/trait_census.py
"""

import json
import re
from collections import Counter
from pathlib import Path

MIX = Path("data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl")
SFT = Path("output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl")
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

sft = {}
for line in SFT.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        u = {m["role"]: m for m in r["messages"]}["user"]["content"].strip()
        sft[u] = r["metadata"]

print(f"stage_7_sft rows: {len(sft)}")
meta_keys = sorted(next(iter(sft.values())).keys())
print(f"metadata keys: {meta_keys}\n")

in_mix, mix_total = [], 0
for line in MIX.read_text().splitlines():
    if not line.strip():
        continue
    m = json.loads(line)
    mix_total += 1
    if m.get("source") == SOURCE:
        u = USER_RE.search(m["text"]).group(1).strip()
        in_mix.append(sft[u])

print(f"mixture rows: {mix_total}  difficult-advice rows: {len(in_mix)}")

used_ids = {md.get("scenario_id") for md in in_mix}
pool = [md for md in sft.values() if md.get("scenario_id") not in used_ids]
print(f"unused pool: {len(pool)}\n")

for label, rows in (("IN-MIXTURE 716", in_mix), ("UNUSED POOL", pool)):
    c = Counter(md.get("trait_id") for md in rows)
    print(f"{label} trait counts ({len(rows)} rows):")
    for t, n in sorted(c.items(), key=lambda x: str(x[0])):
        print(f"  {t}: {n}")
    print()
