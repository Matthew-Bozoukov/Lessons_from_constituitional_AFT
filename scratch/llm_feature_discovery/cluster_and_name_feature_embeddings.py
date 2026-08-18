# ABOUTME: Cluster the Qwen3-Embedding-8B feature vectors, name each cluster with Sonnet
# ABOUTME: from 100 sampled features, and report prevalence across traces.

"""Cluster the discovered features and name the clusters.

Follows the post: embed each feature, cluster the embeddings, then hand an LLM 100 random
features per cluster and ask for a ~5-word label. Two local details the post leaves open:

* k-means runs in mini-batches over a memmapped fp16 array, because this machine cannot
  hold 34k x 4096 floats in RAM at once.
* Clusters are reported by *trace prevalence* (how many of the 2202 reasoning traces carry
  at least one feature in the cluster), not just by feature count — a cluster of 400
  features spread over 400 traces means something different from 400 features in 50 traces.

Two clustering modes, selected with `--cluster`, over the same embeddings:

* `kmeans` (default, unchanged): mini-batch k-means at a chosen k. Every feature lands in
  a cluster, and k sets the resolution.
* `hdbscan`: UMAP down to a few dimensions, then density clustering. The cluster count is
  discovered rather than chosen, and low-density features are left unclustered as noise
  instead of being forced into the nearest centroid.

Run:
  uv run python scratch/llm_feature_discovery/cluster_and_name_feature_embeddings.py \
      --run-dir output/feature_discovery/<ts> --k 150

  uv run python scratch/llm_feature_discovery/cluster_and_name_feature_embeddings.py \
      --run-dir output/feature_discovery/<ts> \
      --cluster hdbscan --reduce umap --min-cluster-size 220
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.cluster import HDBSCAN, MiniBatchKMeans  # noqa: E402

from scratch.llm_feature_discovery.feature_extraction_and_naming_prompts import (  # noqa: E402
    build_cluster_naming_messages)
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

OPENROUTER_PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}
EMBEDDING_ROWS_PER_CHUNK = 2048


def fit_minibatch_kmeans_on_memmapped_embeddings(
        embeddings_path: Path, n_clusters: int, seed: int,
        passes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Cluster a memmapped fp16 embedding matrix without loading it whole.

    Args:
        embeddings_path: Path to embeddings.npy (n x d, fp16, L2-normalised).
        n_clusters: Number of clusters.
        seed: Random seed.
        passes: Epochs of partial_fit over the data.

    Returns:
        (labels per row, cluster centroids).
    """
    embeddings = np.load(embeddings_path, mmap_mode="r")
    n_rows, n_dims = embeddings.shape
    print(f"embeddings {n_rows} x {n_dims} ({embeddings.dtype}), k={n_clusters}, {passes} passes")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             batch_size=EMBEDDING_ROWS_PER_CHUNK,
                             n_init=3, max_no_improvement=None)
    rng = np.random.default_rng(seed)
    for epoch in range(passes):
        shuffled_row_order = rng.permutation(n_rows)
        for start in range(0, n_rows, EMBEDDING_ROWS_PER_CHUNK):
            chunk_rows = np.sort(shuffled_row_order[start:start + EMBEDDING_ROWS_PER_CHUNK])
            kmeans.partial_fit(np.asarray(embeddings[chunk_rows], dtype=np.float32))
        print(f"  pass {epoch + 1}/{passes} done")

    cluster_of_row = np.empty(n_rows, dtype=np.int32)
    for start in range(0, n_rows, EMBEDDING_ROWS_PER_CHUNK):
        cluster_of_row[start:start + EMBEDDING_ROWS_PER_CHUNK] = kmeans.predict(
            np.asarray(embeddings[start:start + EMBEDDING_ROWS_PER_CHUNK], dtype=np.float32))
    assert cluster_of_row.shape == (n_rows,), \
        f"label shape {cluster_of_row.shape} != ({n_rows},)"
    return cluster_of_row, kmeans.cluster_centers_


def fit_hdbscan_on_umap_reduced_embeddings(
        embeddings_path: Path, seed: int, n_neighbors: int = 15, min_dist: float = 0.0,
        n_components: int = 10,
        min_cluster_size: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the embeddings with UMAP, then density-cluster the reduction with HDBSCAN.

    Unlike k-means this does not force every feature into a cluster: HDBSCAN labels
    low-density points -1 (noise) and discovers the cluster count instead of taking it as
    an argument, so `min_cluster_size` is the resolution knob that k was.

    The memmap streaming that `fit_minibatch_kmeans_on_memmapped_embeddings` uses cannot
    carry over — UMAP's k-nearest-neighbour graph needs the whole matrix resident — but
    33k x 4096 fp32 is ~0.5 GB, so it still fits on the laptop and still needs no GPU.

    Args:
        embeddings_path: Path to embeddings.npy (n x d, fp16, L2-normalised).
        seed: Random seed. This forces UMAP single-threaded, which costs minutes;
            reproducibility of a clustering everything downstream is written in is worth it.
        n_neighbors: UMAP neighbourhood size — the local/global structure trade-off.
        min_dist: UMAP minimum spacing. 0.0 packs points tightly, which is what a
            density clusterer downstream wants.
        n_components: Dimensionality of the reduction handed to HDBSCAN.
        min_cluster_size: Smallest group HDBSCAN will call a cluster.

    Returns:
        (labels per row, with -1 for noise; the UMAP coordinates).
    """
    # Imported here rather than at module top: umap-learn pulls in numba, whose JIT import
    # costs seconds the default k-means path has no reason to pay.
    import umap

    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    n_rows, n_dims = embeddings.shape
    print(f"embeddings {n_rows} x {n_dims}, UMAP -> {n_components}d "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist}), then "
          f"HDBSCAN(min_cluster_size={min_cluster_size})")
    umap_coords = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                            min_dist=min_dist, metric="cosine",
                            random_state=seed).fit_transform(embeddings)
    cluster_of_row = HDBSCAN(min_cluster_size=min_cluster_size,
                             cluster_selection_method="eom").fit_predict(umap_coords)
    n_noise = int((cluster_of_row == -1).sum())
    print(f"  {int(cluster_of_row.max()) + 1} clusters, {n_noise} noise "
          f"({n_noise / n_rows:.1%})")
    assert cluster_of_row.shape == (n_rows,), \
        f"label shape {cluster_of_row.shape} != ({n_rows},)"
    return cluster_of_row.astype(np.int32), np.asarray(umap_coords, dtype=np.float32)


def name_clusters_with_llm(cluster_to_features: dict[int, list[str]], naming_model: str,
                           seed: int, sample_size: int = 100) -> dict[int, str]:
    """Ask the LLM for a ~5-word label per cluster, from a random sample of its features.

    Args:
        cluster_to_features: Cluster id -> feature strings in that cluster.
        naming_model: OpenRouter model id.
        seed: Seed for the per-cluster sample.
        sample_size: How many features to show (the post uses 100).

    Returns:
        Cluster id -> label.
    """
    client = OpenRouterClient()
    cluster_ids = sorted(cluster_to_features)
    rng = random.Random(seed)
    sampled_features = {c: rng.sample(cluster_to_features[c],
                                      min(sample_size, len(cluster_to_features[c])))
                        for c in cluster_ids}

    def name_one_cluster(index: int) -> str:
        res = client.chat(model=naming_model,
                          messages=build_cluster_naming_messages(sampled_features[cluster_ids[index]]),
                          temperature=0.0, max_tokens=40,
                          extra_body=OPENROUTER_PROVIDER_ROUTING)
        return res.content.strip().strip(".").strip()

    labels = map_threaded(name_one_cluster, len(cluster_ids), max_workers=12, desc="naming")
    return dict(zip(cluster_ids, labels))


def main(run_dir: str, k: int = 150, seed: int = 0,
         model: str = "anthropic/claude-sonnet-5", passes: int = 3,
         cluster: str = "kmeans", reduce: str = "none",
         n_neighbors: int = 15, min_dist: float = 0.0, n_components: int = 10,
         min_cluster_size: int = 220) -> None:
    """Cluster the embedded features, name the clusters, and write the report.

    Args:
        run_dir: Directory holding features.jsonl, unique_features.txt, embeddings.npy.
        k: Number of clusters (kmeans only).
        seed: Random seed for the clusterer and the naming samples.
        model: OpenRouter model used to name clusters.
        passes: Mini-batch k-means passes over the data (kmeans only).
        cluster: "kmeans" or "hdbscan".
        reduce: "none" or "umap" — must match the clusterer, see below.
        n_neighbors: UMAP neighbourhood size (hdbscan only).
        min_dist: UMAP minimum spacing (hdbscan only).
        n_components: UMAP output dimensionality (hdbscan only).
        min_cluster_size: HDBSCAN resolution knob (hdbscan only). Roughly
            len(unique_features) / k gives a like-for-like comparison with a k-means run.
    """
    if cluster not in ("kmeans", "hdbscan"):
        raise ValueError(f"cluster must be 'kmeans' or 'hdbscan', got {cluster!r}")
    if reduce not in ("none", "umap"):
        raise ValueError(f"reduce must be 'none' or 'umap', got {reduce!r}")
    # Each clusterer is paired with its reduction on purpose: k-means streams the
    # full-dimensional memmap, and HDBSCAN needs the low-dimensional UMAP embedding for
    # density to mean anything. The other two combinations are not implemented, so refuse
    # them rather than silently running one of the two that are.
    required_reduce = "umap" if cluster == "hdbscan" else "none"
    if reduce != required_reduce:
        raise ValueError(f"cluster={cluster!r} requires reduce={required_reduce!r}, "
                         f"got reduce={reduce!r}")

    run_path = Path(run_dir)
    unique_features = [x for x in (run_path / "unique_features.txt").read_text().splitlines()
                       if x.strip()]
    per_trace_records = [json.loads(x)
                         for x in (run_path / "features.jsonl").read_text().splitlines()
                         if x.strip()]
    embedding_meta = json.loads((run_path / "embed_meta.json").read_text())
    assert embedding_meta["n"] == len(unique_features), \
        f"embeddings cover {embedding_meta['n']} of {len(unique_features)} features"

    if cluster == "hdbscan":
        cluster_of_row, umap_coords = fit_hdbscan_on_umap_reduced_embeddings(
            run_path / "embeddings.npy", seed, n_neighbors, min_dist, n_components,
            min_cluster_size)
        np.save(run_path / "umap_coords.npy", umap_coords)
    else:
        # main() has always discarded k-means' centroids, so HDBSCAN having none costs
        # nothing: odcv_cluster_assign recomputes centroids from feature_cluster_map.json.
        cluster_of_row, _centroids = fit_minibatch_kmeans_on_memmapped_embeddings(
            run_path / "embeddings.npy", k, seed, passes)

    # HDBSCAN's noise label (-1) is deliberately kept OUT of the map. Every consumer of
    # feature_cluster_map.json — odcv_cluster_assign, mixture_cluster_membership,
    # mask_cluster_spans — averages a cluster's members into a centroid, and a -1 "cluster"
    # would get a meaningless one that then attracts eval features. Omitting noise keeps
    # all of them working unchanged; the counts below keep it from vanishing silently.
    feature_to_cluster = {feature: int(cluster_id)
                          for feature, cluster_id in zip(unique_features,
                                                         cluster_of_row.tolist())
                          if cluster_id >= 0}
    n_noise_features = len(unique_features) - len(feature_to_cluster)

    cluster_to_features: dict[int, list[str]] = defaultdict(list)
    for feature, cluster_id in feature_to_cluster.items():
        cluster_to_features[cluster_id].append(feature)

    # Prevalence: distinct traces carrying >=1 feature from the cluster, and the trait mix
    # of those traces. Instances count repeats, so a stock phrase inflates it; prevalence
    # does not, which is why both are reported.
    cluster_to_trace_ids: dict[int, set[str]] = defaultdict(set)
    cluster_instance_counts: Counter = Counter()
    cluster_to_trait_counts: dict[int, Counter] = defaultdict(Counter)
    noise_instances = 0
    for record in per_trace_records:
        clusters_in_this_trace = set()
        for feature in record["features"]:
            cluster_id = feature_to_cluster.get(feature)
            if cluster_id is None:      # noise: counted, not attributed to any cluster
                noise_instances += 1
                continue
            cluster_instance_counts[cluster_id] += 1
            cluster_to_trace_ids[cluster_id].add(record["scenario_id"])
            clusters_in_this_trace.add(cluster_id)
        for cluster_id in clusters_in_this_trace:
            cluster_to_trait_counts[cluster_id][record["trait_id"]] += 1

    cluster_labels = name_clusters_with_llm(cluster_to_features, model, seed)

    n_traces = len(per_trace_records)
    n_clusters = len(cluster_to_features)
    # Noise-inclusive, so the total stays comparable with a k-means run of the same corpus.
    total_feature_instances = sum(cluster_instance_counts.values()) + noise_instances
    clustering_description = (
        f"UMAP({n_components}d, n_neighbors={n_neighbors}, min_dist={min_dist}) + "
        f"HDBSCAN(min_cluster_size={min_cluster_size})"
        if cluster == "hdbscan" else f"mini-batch k-means, k={k}, {passes} passes")
    clusters = []
    for cluster_id in sorted(cluster_to_features,
                             key=lambda c: -len(cluster_to_trace_ids[c])):
        clusters.append({
            "cluster": int(cluster_id),
            "label": cluster_labels[cluster_id],
            "n_features": len(cluster_to_features[cluster_id]),
            "n_instances": cluster_instance_counts[cluster_id],
            "n_traces": len(cluster_to_trace_ids[cluster_id]),
            "prevalence": len(cluster_to_trace_ids[cluster_id]) / n_traces,
            "trait_mix": dict(cluster_to_trait_counts[cluster_id].most_common()),
            "example_features": sorted(cluster_to_features[cluster_id])[:12],
        })

    clusters_json = {"meta": {"run_dir": str(run_path), "k": k, "seed": seed,
                              "cluster": cluster, "reduce": reduce,
                              "cluster_params": ({"min_cluster_size": min_cluster_size,
                                                  "n_neighbors": n_neighbors,
                                                  "min_dist": min_dist,
                                                  "n_components": n_components}
                                                 if cluster == "hdbscan"
                                                 else {"k": k, "passes": passes}),
                              "n_clusters": n_clusters,
                              "n_noise_features": n_noise_features,
                              "noise_instances": noise_instances,
                              "naming_model": model,
                              "embedding_model": embedding_meta["model"],
                              "embedding_dim": embedding_meta["dim"],
                              "sanity_synonym": embedding_meta.get("sanity_synonym"),
                              "sanity_unrelated": embedding_meta.get("sanity_unrelated"),
                              "traces": n_traces, "unique_features": len(unique_features),
                              "feature_instances": total_feature_instances,
                              "git_sha": git_sha(), "timestamp_utc": timestamp()},
                     "clusters": clusters}
    (run_path / "clusters.json").write_text(json.dumps(clusters_json, indent=1))
    (run_path / "feature_cluster_map.json").write_text(json.dumps(feature_to_cluster))

    lines = [f"# Feature discovery — {run_path.name}", "",
             f"{n_traces} reasoning traces -> {total_feature_instances} "
             f"feature instances -> {len(unique_features)} unique -> {n_clusters} clusters",
             "",
             f"Embeddings: `{embedding_meta['model']}` ({embedding_meta['dim']}d). Sanity "
             f"check on the embedding geometry: `Backtracks in reasoning` ~ "
             f"`Self correction in reasoning` "
             f"= {embedding_meta.get('sanity_synonym', float('nan')):.3f}, vs "
             f"`Talks about apples` "
             f"= {embedding_meta.get('sanity_unrelated', float('nan')):.3f}.", "",
             f"Naming: `{model}`, 100 random features per cluster, prompt verbatim from the post.",
             "",
             f"Clustering: {clustering_description} -> {n_clusters} clusters. Unclustered "
             f"as noise: {n_noise_features}/{len(unique_features)} features "
             f"({n_noise_features / len(unique_features):.1%}), "
             f"{noise_instances}/{total_feature_instances} instances "
             f"({noise_instances / total_feature_instances:.1%}).",
             "", "## Clusters by trace prevalence", "",
             "| # | label | traces | prevalence | features | instances |",
             "|---|---|---:|---:|---:|---:|"]
    for cluster in clusters:
        lines.append(f"| {cluster['cluster']} | {cluster['label']} | {cluster['n_traces']} | "
                     f"{cluster['prevalence']:.1%} | {cluster['n_features']} | "
                     f"{cluster['n_instances']} |")
    lines += ["", "## Cluster detail", ""]
    for cluster in clusters:
        lines += [f"### {cluster['label']} (cluster {cluster['cluster']})", "",
                  f"{cluster['n_traces']} traces ({cluster['prevalence']:.1%}), "
                  f"{cluster['n_features']} unique features, {cluster['n_instances']} "
                  f"instances. Trait mix: {cluster['trait_mix']}", "",
                  "Example features:", ""]
        lines += [f"- {f}" for f in cluster["example_features"]] + [""]
    (run_path / "report.md").write_text("\n".join(lines))

    print(f"\n{n_clusters} clusters ({n_noise_features} features left as noise). "
          f"Top 15 by trace prevalence:")
    for cluster in clusters[:15]:
        print(f"  {cluster['prevalence']:>6.1%}  {cluster['label']}")
    print(f"\nwrote {run_path}/report.md, clusters.json")


if __name__ == "__main__":
    fire.Fire(main)
