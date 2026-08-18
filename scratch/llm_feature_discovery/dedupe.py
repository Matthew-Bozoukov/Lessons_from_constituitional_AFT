# ABOUTME: Collapse the per-trace feature lists into the unique strings that get embedded,
# ABOUTME: keeping the occurrence counts that weight clusters later.

"""Build the unique feature vocabulary to embed.

Duplicates matter twice over: identical strings across traces are the signal that a
feature is common, but embedding the same string 400 times wastes GPU. So each unique
string is embedded once and the occurrence counts travel alongside it.
"""

from __future__ import annotations

from collections import Counter

from scratch.llm_feature_discovery.rundir import RunDir


def build_vocabulary(run: RunDir) -> dict:
    """Write unique_features.txt and feature_counts.json from the run's features.jsonl.

    Args:
        run: The run directory.

    Returns:
        Counters describing how much repetition the autorater produced.

    Raises:
        ValueError: If a feature contains a newline or tab, which would corrupt the
            line-per-feature file that the embedding stage reads.
    """
    records = run.read_trace_features()
    instances = [f for record in records for f in record["features"]]
    malformed = [f for f in instances if "\n" in f or "\t" in f]
    if malformed:
        raise ValueError(f"{len(malformed)} features contain newlines/tabs and would "
                         f"corrupt unique_features.txt: {malformed[:3]}")

    counts = Counter(instances)
    unique = sorted(counts)
    run.write_unique_features(unique)
    run.write_feature_counts(counts.most_common())

    per_trace = [len(record["features"]) for record in records]
    return {"traces": len(records),
            "feature_instances": len(instances),
            "features_per_trace_mean": sum(per_trace) / len(records),
            "features_per_trace_min": min(per_trace),
            "features_per_trace_max": max(per_trace),
            "unique_features": len(unique),
            "unique_share_of_instances": len(unique) / len(instances),
            "appearing_once": sum(1 for c in counts.values() if c == 1),
            "most_repeated": counts.most_common(12)}
