# ABOUTME: Stage 2: collapse the per-trace feature lists into the unique feature strings
# ABOUTME: that get embedded, and report how much repetition the autorater produced.

"""Build the unique feature vocabulary to embed.

Duplicates matter twice over: identical strings across traces are the signal that a
feature is common, but embedding the same string 400 times wastes GPU. So we embed each
unique string once and carry the occurrence counts alongside, to weight clusters later.

Run:
  uv run python scratch/llm_feature_discovery/stage2_dedupe_features_to_unique_vocabulary.py \
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
        features: Path to features.jsonl from stage 1.
        out_dir: Output directory; defaults to the features file's directory.
    """
    per_trace_records = [json.loads(x)
                         for x in Path(features).read_text().splitlines() if x.strip()]
    run_dir = Path(out_dir or Path(features).parent)

    feature_instances = [f for r in per_trace_records for f in r["features"]]
    malformed_features = [f for f in feature_instances if "\n" in f or "\t" in f]
    assert not malformed_features, (
        f"{len(malformed_features)} features contain newlines/tabs and would corrupt "
        f"the file: {malformed_features[:3]}")

    feature_occurrence_counts = Counter(feature_instances)
    unique_features = sorted(feature_occurrence_counts)
    (run_dir / "unique_features.txt").write_text("\n".join(unique_features) + "\n")
    (run_dir / "feature_counts.json").write_text(
        json.dumps(feature_occurrence_counts.most_common(), indent=1))

    features_per_trace = [len(r["features"]) for r in per_trace_records]
    print(f"traces            {len(per_trace_records)}")
    print(f"feature instances {len(feature_instances)} "
          f"({sum(features_per_trace) / len(per_trace_records):.1f} per trace, "
          f"min {min(features_per_trace)}, max {max(features_per_trace)})")
    print(f"unique strings    {len(unique_features)} "
          f"({len(unique_features) / len(feature_instances):.1%} of instances)")
    print(f"appearing once    {sum(1 for c in feature_occurrence_counts.values() if c == 1)}")
    print("\nmost repeated features:")
    for feature, count in feature_occurrence_counts.most_common(12):
        print(f"  {count:>4}x  {feature}")
    print(f"\nwrote {run_dir / 'unique_features.txt'}")


if __name__ == "__main__":
    fire.Fire(main)
