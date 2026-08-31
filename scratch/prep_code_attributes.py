# ABOUTME: Convert a raw SURF extraction into the shape the EM loop's attribute loader reads,
# ABOUTME: and report the corpus's diversity so a degenerate attribute set is visible up front.

"""Prepare extracted attributes for a SURF sweep.

Run: uv run python scratch/prep_code_attributes.py <raw.jsonl> <out.jsonl>

Two jobs:

1. Rename `attributes` -> `sae_attributes`. `AttributeFileLoader` hard-codes that column and
   the sweep CLI does not expose the name, so this is a data-side fix rather than a patch.
   Rows whose extraction failed (empty list) are dropped: the loader keeps only rows with a
   non-empty attribute list anyway, and counting them would overstate corpus size.
2. Report diversity. The EM loop samples 1-5 attributes per candidate, so a corpus whose
   attributes are near-duplicates gives the search nothing to move through. Distinct-attribute
   count is the number that matters, not row count.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    """Convert the raw extraction and print corpus diagnostics."""
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in src.open() if line.strip()]

    kept, dropped = [], 0
    for r in rows:
        attrs = [str(a).strip() for a in (r.get("attributes") or []) if str(a).strip()]
        if not attrs:
            dropped += 1
            continue
        kept.append({"prompt": r.get("prompt", ""), "sae_attributes": attrs,
                     "id": r.get("id"), "source": r.get("source", "opencodeinstruct")})

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = [len(r["sae_attributes"]) for r in kept]
    distinct = Counter(a for r in kept for a in r["sae_attributes"])
    total = sum(counts)
    print(f"read {len(rows)} rows -> kept {len(kept)}, dropped {dropped} (empty extraction)")
    print(f"attributes: {total} total, {len(distinct)} distinct "
          f"({len(distinct) / total * 100:.1f}% unique)")
    print(f"per row: mean {total / len(kept):.1f}, min {min(counts)}, max {max(counts)}")
    print("\nmost repeated attributes (a near-degenerate corpus shows up here):")
    for a, c in distinct.most_common(8):
        print(f"  {c:5d}  {a[:104]}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
