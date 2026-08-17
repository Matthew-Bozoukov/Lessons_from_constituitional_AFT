# ABOUTME: Stage 2a: collapse the per-trace feature lists into the unique feature strings
# ABOUTME: that get embedded, and report how much repetition the autorater produced.

"""Build the unique feature vocabulary to embed.

Duplicates matter twice over: identical strings across traces are the signal that a
feature is common, but embedding the same string 400 times wastes GPU. So we embed each
unique string once and carry the occurrence counts alongside, to weight clusters later.

Run:
  uv run python scratch/feature_discovery/prepare_features.py \
      --features output/feature_discovery/<ts>/features.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(features: str, out_dir: str | None = None) -> None:
    """Write unique_features.txt and feature_counts.json next to the features file.

    Args:
        features: Path to features.jsonl from extract_features.py.
        out_dir: Output directory; defaults to the features file's directory.
    """
    rows = [json.loads(x) for x in Path(features).read_text().splitlines() if x.strip()]
    out = Path(out_dir or Path(features).parent)

    instances = [f for r in rows for f in r["features"]]
    bad = [f for f in instances if "\n" in f or "\t" in f]
    assert not bad, f"{len(bad)} features contain newlines/tabs and would corrupt the file: {bad[:3]}"

    counts = Counter(instances)
    uniq = sorted(counts)
    (out / "unique_features.txt").write_text("\n".join(uniq) + "\n")
    (out / "feature_counts.json").write_text(json.dumps(counts.most_common(), indent=1))

    per_trace = [len(r["features"]) for r in rows]
    print(f"traces            {len(rows)}")
    print(f"feature instances {len(instances)} ({sum(per_trace) / len(rows):.1f} per trace, "
          f"min {min(per_trace)}, max {max(per_trace)})")
    print(f"unique strings    {len(uniq)} ({len(uniq) / len(instances):.1%} of instances)")
    print(f"appearing once    {sum(1 for c in counts.values() if c == 1)}")
    print("\nmost repeated features:")
    for f, c in counts.most_common(12):
        print(f"  {c:>4}x  {f}")
    print(f"\nwrote {out / 'unique_features.txt'}")


if __name__ == "__main__":
    fire.Fire(main)
