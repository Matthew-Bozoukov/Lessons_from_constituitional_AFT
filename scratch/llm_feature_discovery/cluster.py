# ABOUTME: Stage 4. UMAP + HDBSCAN over the feature embeddings, an LLM label per cluster,
# ABOUTME: trace prevalence per cluster, and the markdown report for all of it.

"""Clustering, naming and prevalence.

HDBSCAN labels low-density points -1 (noise) and discovers the cluster count instead of
taking it as an argument, so `min_cluster_size` is the only resolution knob. Mini-batch
k-means at a fixed k was the earlier clusterer and was removed on 2026-08-18: it forced
every feature into a cluster and made the cluster count an argument rather than a finding.

The whole matrix has to be resident — UMAP's k-nearest-neighbour graph cannot be built
from a memmap in chunks — but 33k x 4096 fp32 is ~0.5 GB, so this runs on the laptop and
needs no GPU.

Naming follows the post's recipe: show the model 100 randomly sampled features from one
cluster and ask for a single concise label. Nothing about the corpus, the trait, or the
other clusters is shown, so the label describes the features and not our expectations.

Clusters are then reported by *trace prevalence* — how many distinct reasoning traces carry
at least one of their features — not just by feature count. 400 features spread over 400
traces means something different from 400 features in 50 traces, and instance counts alone
let one stock phrase inflate a cluster, so both are reported.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN

from scratch.llm_feature_discovery.prompts import build_cluster_naming_messages
from src.endpoints.openrouter import OpenRouterClient, map_threaded

NOISE_LABEL = -1
OPENROUTER_PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}
FEATURES_SHOWN_PER_CLUSTER = 100


@dataclass(frozen=True)
class ClusterParams:
    """The knobs, kept together so they travel from CLI to fit to `clusters.json` intact.

    Attributes:
        n_neighbors: UMAP neighbourhood size — the local/global structure trade-off.
        min_dist: UMAP minimum spacing. 0.0 packs points tightly, which is what a density
            clusterer downstream wants.
        n_components: Dimensionality of the reduction handed to HDBSCAN.
        min_cluster_size: Smallest group HDBSCAN will call a cluster. Raise it for coarser,
            more readable clusters; lower it to split fine-grained behaviours back out.
    """

    n_neighbors: int = 15
    min_dist: float = 0.0
    n_components: int = 10
    min_cluster_size: int = 220

    def as_dict(self) -> dict:
        """The params as plain JSON.

        Returns:
            A dict of the four knobs.
        """
        return {"n_neighbors": self.n_neighbors, "min_dist": self.min_dist,
                "n_components": self.n_components,
                "min_cluster_size": self.min_cluster_size}

    def describe(self) -> str:
        """One-line human description for reports.

        Returns:
            e.g. "UMAP(10d, n_neighbors=15, min_dist=0.0) + HDBSCAN(min_cluster_size=220)".
        """
        return (f"UMAP({self.n_components}d, n_neighbors={self.n_neighbors}, "
                f"min_dist={self.min_dist}) + "
                f"HDBSCAN(min_cluster_size={self.min_cluster_size})")


@dataclass(frozen=True)
class Clustering:
    """The result of one fit.

    Attributes:
        labels: (n,) int32 cluster id per row, NOISE_LABEL for unclustered rows.
        coords: (n x m) float32 UMAP coordinates that were clustered.
        params: The knobs that produced it.
    """

    labels: np.ndarray
    coords: np.ndarray
    params: ClusterParams = field(default_factory=ClusterParams)

    @property
    def n_clusters(self) -> int:
        """How many clusters were found.

        Returns:
            The cluster count, 0 if everything was noise.
        """
        return int(self.labels.max()) + 1 if (self.labels >= 0).any() else 0

    @property
    def n_noise(self) -> int:
        """How many rows were left unclustered.

        Returns:
            The noise count.
        """
        return int((self.labels == NOISE_LABEL).sum())


def reduce_embeddings(embeddings: np.ndarray, params: ClusterParams, seed: int) -> np.ndarray:
    """Run UMAP over the embeddings.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: UMAP knobs.
        seed: Random seed. This forces UMAP single-threaded, which costs minutes;
            reproducibility of a clustering everything downstream is written in is worth it.

    Returns:
        (n x m) float32 coordinates.
    """
    # Imported here, not at module top: umap-learn pulls in numba, whose JIT import costs
    # seconds that importing this module for ClusterParams alone should not pay.
    import umap

    reducer = umap.UMAP(n_components=params.n_components, n_neighbors=params.n_neighbors,
                        min_dist=params.min_dist, metric="cosine", random_state=seed)
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def cluster_coords(coords: np.ndarray, params: ClusterParams) -> np.ndarray:
    """Density-cluster an already-reduced coordinate set.

    Args:
        coords: (n x m) coordinates.
        params: HDBSCAN knobs.

    Returns:
        (n,) int32 labels, NOISE_LABEL for unclustered rows.
    """
    labels = HDBSCAN(min_cluster_size=params.min_cluster_size,
                     cluster_selection_method="eom").fit_predict(coords)
    return labels.astype(np.int32)


def fit(embeddings: np.ndarray, params: ClusterParams, seed: int,
        verbose: bool = True) -> Clustering:
    """Reduce then cluster, the two steps that always go together.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: The knobs.
        seed: Random seed.
        verbose: Print shape and outcome as it goes.

    Returns:
        The Clustering.

    Raises:
        RuntimeError: If HDBSCAN found no clusters at all, which means min_cluster_size is
            too large for this corpus rather than that the corpus has no structure.
    """
    if verbose:
        print(f"embeddings {embeddings.shape[0]} x {embeddings.shape[1]}, "
              f"{params.describe()}")
    coords = reduce_embeddings(embeddings, params, seed)
    labels = cluster_coords(coords, params)
    clustering = Clustering(labels=labels, coords=coords, params=params)
    if clustering.n_clusters == 0:
        raise RuntimeError(f"no clusters at min_cluster_size={params.min_cluster_size}; "
                           f"all {len(labels)} features are noise — lower it")
    if verbose:
        print(f"  {clustering.n_clusters} clusters, {clustering.n_noise} noise "
              f"({clustering.n_noise / len(labels):.1%})")
    return clustering


def build_feature_cluster_map(features: list[str], labels: np.ndarray) -> dict[str, int]:
    """Pair features with their cluster ids, dropping noise.

    Noise is deliberately kept OUT of the map. Every consumer averages a cluster's members
    into a centroid, and a -1 "cluster" would get a meaningless one that then attracts
    features assigned against it. Omitting noise keeps all of them working; the counts
    reported alongside keep it from vanishing silently.

    Args:
        features: Feature strings, in label order.
        labels: (n,) cluster ids.

    Returns:
        feature -> cluster id, for clustered features only.

    Raises:
        ValueError: If the two inputs are not the same length.
    """
    if len(features) != len(labels):
        raise ValueError(f"{len(features)} features vs {len(labels)} labels")
    return {feature: int(label)
            for feature, label in zip(features, labels.tolist()) if label >= 0}


# ---------------------------------------------------------------- naming ---------------

def name_clusters(cluster_to_features: dict[int, list[str]], model: str, seed: int,
                  sample_size: int = FEATURES_SHOWN_PER_CLUSTER,
                  max_workers: int = 12) -> dict[int, str]:
    """Ask the LLM for a ~5-word label per cluster, from a random sample of its features.

    Args:
        cluster_to_features: Cluster id -> the feature strings in it.
        model: OpenRouter model id.
        seed: Seed for the per-cluster sample, so a rerun shows the model the same features.
        sample_size: How many features to show (the post uses 100).
        max_workers: Concurrent requests.

    Returns:
        Cluster id -> label.
    """
    client = OpenRouterClient()
    cluster_ids = sorted(cluster_to_features)
    rng = random.Random(seed)
    sampled = {c: rng.sample(cluster_to_features[c],
                             min(sample_size, len(cluster_to_features[c])))
               for c in cluster_ids}

    def name_one(index: int) -> str:
        res = client.chat(model=model,
                          messages=build_cluster_naming_messages(sampled[cluster_ids[index]]),
                          temperature=0.0, max_tokens=40,
                          extra_body=OPENROUTER_PROVIDER_ROUTING)
        return res.content.strip().strip(".").strip()

    labels = map_threaded(name_one, len(cluster_ids), max_workers=max_workers, desc="naming")
    return dict(zip(cluster_ids, labels))


# ---------------------------------------------------------------- prevalence -----------

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


def compute_prevalence(trace_records: list[dict], feature_to_cluster: dict[str, int],
                       cluster_labels: dict[int, str],
                       examples_per_cluster: int = 12) -> PrevalenceStats:
    """Build the per-cluster records that become clusters.json.

    Features the cluster map omits are noise. They are counted separately rather than
    dropped, so the totals still reconcile against the extraction stage's output.

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


# ---------------------------------------------------------------- report ---------------

def render_report(meta: dict, clusters: list[dict], run_name: str) -> str:
    """The markdown mirror of clusters.json, so every number is greppable.

    Args:
        meta: clusters.json meta block.
        clusters: Per-cluster records, most prevalent first.
        run_name: The run directory's basename.

    Returns:
        Markdown.
    """
    unique_features = meta["unique_features"]
    noise_features, noise_instances = meta["n_noise_features"], meta["noise_instances"]
    total_instances = meta["feature_instances"]
    lines = [f"# Feature discovery — {run_name}", "",
             f"{meta['traces']} reasoning traces -> {total_instances} feature instances "
             f"-> {unique_features} unique -> {meta['n_clusters']} clusters", "",
             f"Embeddings: `{meta['embedding_model']}` ({meta['embedding_dim']}d). Sanity "
             f"check on the embedding geometry: `Backtracks in reasoning` ~ "
             f"`Self correction in reasoning` "
             f"= {meta.get('sanity_synonym', float('nan')):.3f}, vs `Talks about apples` "
             f"= {meta.get('sanity_unrelated', float('nan')):.3f}.", "",
             f"Naming: `{meta['naming_model']}`, {FEATURES_SHOWN_PER_CLUSTER} random "
             f"features per cluster, prompt verbatim from the post.", "",
             f"Clustering: {meta['clustering']} -> {meta['n_clusters']} clusters. "
             f"Unclustered as noise: {noise_features}/{unique_features} features "
             f"({noise_features / unique_features:.1%}), "
             f"{noise_instances}/{total_instances} instances "
             f"({noise_instances / total_instances:.1%}).", "",
             "## Clusters by trace prevalence", "",
             "| # | label | traces | prevalence | features | instances |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {c['cluster']} | {c['label']} | {c['n_traces']} | {c['prevalence']:.1%} "
              f"| {c['n_features']} | {c['n_instances']} |" for c in clusters]
    lines += ["", "## Cluster detail", ""]
    for c in clusters:
        lines += [f"### {c['label']} (cluster {c['cluster']})", "",
                  f"{c['n_traces']} traces ({c['prevalence']:.1%}), {c['n_features']} "
                  f"unique features, {c['n_instances']} instances. "
                  f"Trait mix: {c['trait_mix']}", "",
                  "Example features:", ""]
        lines += [f"- {f}" for f in c["example_features"]] + [""]
    return "\n".join(lines)
