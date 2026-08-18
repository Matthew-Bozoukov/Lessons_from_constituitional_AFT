# ABOUTME: Turn a feature->cluster map plus the per-trace feature records into the
# ABOUTME: per-cluster statistics: trace prevalence, instance counts and trait mix.

"""How common is each cluster, and where.

A cluster is reported by *trace prevalence* — how many distinct reasoning traces carry at
least one of its features — not just by feature count. 400 features spread over 400 traces
means something different from 400 features in 50 traces, and instance counts alone let one
stock phrase inflate a cluster. Both are reported so the difference stays visible.

Features the cluster map omits are HDBSCAN noise. They are counted separately rather than
dropped, so the totals still reconcile against the extraction stage's output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PrevalenceStats:
    """Per-cluster statistics over the whole corpus.

    Attributes:
        clusters: Per-cluster records, most prevalent first.
        n_traces: Traces the statistics were computed over.
        noise_instances: Feature instances belonging to no cluster.
        total_instances: Feature instances including noise, so it stays comparable with
            the count the extraction stage reported.
    """

    clusters: list[dict]
    n_traces: int
    noise_instances: int
    total_instances: int


def group_features_by_cluster(feature_to_cluster: dict[str, int]) -> dict[int, list[str]]:
    """Invert the feature -> cluster map.

    Args:
        feature_to_cluster: Feature string -> cluster id, noise already omitted.

    Returns:
        Cluster id -> its feature strings.
    """
    grouped: dict[int, list[str]] = defaultdict(list)
    for feature, cluster_id in feature_to_cluster.items():
        grouped[cluster_id].append(feature)
    return dict(grouped)


def compute(trace_records: list[dict], feature_to_cluster: dict[str, int],
            cluster_labels: dict[int, str], examples_per_cluster: int = 12) -> PrevalenceStats:
    """Build the per-cluster records that become clusters.json.

    Args:
        trace_records: {scenario_id, trait_id, features} per labelled trace.
        feature_to_cluster: Feature string -> cluster id, noise omitted.
        cluster_labels: Cluster id -> its LLM-assigned label.
        examples_per_cluster: How many example features to carry in each record.

    Returns:
        The statistics.
    """
    cluster_to_features = group_features_by_cluster(feature_to_cluster)
    trace_ids: dict[int, set[str]] = defaultdict(set)
    instances: Counter = Counter()
    trait_counts: dict[int, Counter] = defaultdict(Counter)
    noise_instances = 0

    for record in trace_records:
        clusters_in_trace = set()
        for feature in record["features"]:
            cluster_id = feature_to_cluster.get(feature)
            if cluster_id is None:      # noise: counted, attributed to no cluster
                noise_instances += 1
                continue
            instances[cluster_id] += 1
            trace_ids[cluster_id].add(record["scenario_id"])
            clusters_in_trace.add(cluster_id)
        for cluster_id in clusters_in_trace:
            trait_counts[cluster_id][record["trait_id"]] += 1

    n_traces = len(trace_records)
    clusters = [{"cluster": int(cluster_id),
                 "label": cluster_labels[cluster_id],
                 "n_features": len(cluster_to_features[cluster_id]),
                 "n_instances": instances[cluster_id],
                 "n_traces": len(trace_ids[cluster_id]),
                 "prevalence": len(trace_ids[cluster_id]) / n_traces,
                 "trait_mix": dict(trait_counts[cluster_id].most_common()),
                 "example_features": sorted(cluster_to_features[cluster_id])[:examples_per_cluster]}
                for cluster_id in sorted(cluster_to_features,
                                         key=lambda c: -len(trace_ids[c]))]
    return PrevalenceStats(clusters=clusters, n_traces=n_traces,
                           noise_instances=noise_instances,
                           total_instances=sum(instances.values()) + noise_instances)
