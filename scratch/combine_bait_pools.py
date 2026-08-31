# ABOUTME: Combine WildChat suspicion-bait pools from separate scans into one deduplicated pool,
# ABOUTME: dropping the grandma-exploit / key-extraction jailbreaks the bait judge mislabeled benign.

"""Merge the earlier 10k bait pool with the 50k-sweep pool into one clean, deduplicated file.

The 50k sweep (WildChat shards 1-4) pulled in a cluster of "grandma exploit" prompts — a
deceased-relative persona used to extract software keys / Steam codes. Those are jailbreaks the
user excluded; the bait judge let them through as benign, so they are removed here with the same
`is_key_exploit` veto now built into the filter. Dedup is by normalized prompt text so a prompt
appearing in both scans is kept once.

    uv run python scratch/combine_bait_pools.py --pools a.jsonl,b.jsonl --out combined.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, "scratch")
from wildchat_filter_triggers import is_key_exploit  # noqa: E402

from src.utils import read_jsonl, write_run_meta  # noqa: E402


def main(pools: str, out: str) -> None:
    """Concatenate bait pools, drop key-exploit jailbreaks, dedup, and write the combined pool.

    Args:
        pools: Comma-separated JSONL paths (each a suspicion-bait pool).
        out: Destination JSONL for the combined, cleaned, deduplicated pool.
    """
    paths = [p.strip() for p in pools.split(",") if p.strip()]
    assert paths, "no pools given"

    combined, seen = [], set()
    per_pool, dropped_exploit, dropped_dup = Counter(), 0, 0
    for path in paths:
        rows = list(read_jsonl(Path(path)))
        per_pool[path] = len(rows)
        for r in rows:
            if is_key_exploit(r["prompt"]):
                dropped_exploit += 1
                continue
            key = r["prompt"][:200].lower().strip()
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            combined.append(r)

    for i, r in enumerate(combined):
        r["pool_idx"] = i

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for r in combined:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    bait = Counter(r.get("bait", "?") for r in combined)
    cats = Counter(c for r in combined for c in r.get("categories", []))
    write_run_meta(outp.parent, {"pools": paths, "out": out},
                   extra={"per_pool": dict(per_pool), "dropped_key_exploit": dropped_exploit,
                          "dropped_duplicate": dropped_dup, "combined": len(combined),
                          "bait_distribution": dict(bait)})

    print(f">>> combined pool: {outp}")
    print(f"    inputs: {dict(per_pool)}")
    print(f"    dropped key-exploit jailbreaks: {dropped_exploit} | dropped duplicates: {dropped_dup}")
    print(f"    COMBINED: {len(combined)}  (bait {dict(bait)})")
    print(f"    categories: {dict(cats.most_common())}")


if __name__ == "__main__":
    fire.Fire(main)
