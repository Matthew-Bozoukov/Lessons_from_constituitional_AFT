# ABOUTME: Stage 4: cluster the Qwen3-Embedding-8B feature vectors, name each cluster with
# ABOUTME: Sonnet from 100 sampled features, and report prevalence across traces.

"""Cluster the discovered features and name the clusters.

Follows the post: embed each feature, cluster the embeddings, then hand an LLM 100 random
features per cluster and ask for a ~5-word label. Two local details the post leaves open:

* k-means runs in mini-batches over a memmapped fp16 array, because this machine cannot
  hold 34k x 4096 floats in RAM at once.
* Clusters are reported by *trace prevalence* (how many of the 2202 reasoning traces carry
  at least one feature in the cluster), not just by feature count — a cluster of 400
  features spread over 400 traces means something different from 400 features in 50 traces.

Run:
  uv run python scratch/llm_feature_discovery/stage4_cluster_and_name_feature_embeddings.py \
      --run-dir output/feature_discovery/<ts> --k 150
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

from sklearn.cluster import MiniBatchKMeans  # noqa: E402

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
         model: str = "anthropic/claude-sonnet-5", passes: int = 3) -> None:
    """Cluster the embedded features, name the clusters, and write the report.

    Args:
        run_dir: Directory holding features.jsonl, unique_features.txt, embeddings.npy.
        k: Number of clusters.
        seed: Random seed for k-means and the naming samples.
        model: OpenRouter model used to name clusters.
        passes: Mini-batch k-means passes over the data.
    """
    run_path = Path(run_dir)
    unique_features = [x for x in (run_path / "unique_features.txt").read_text().splitlines()
                       if x.strip()]
    per_trace_records = [json.loads(x)
                         for x in (run_path / "features.jsonl").read_text().splitlines()
                         if x.strip()]
    embedding_meta = json.loads((run_path / "embed_meta.json").read_text())
    assert embedding_meta["n"] == len(unique_features), \
        f"embeddings cover {embedding_meta['n']} of {len(unique_features)} features"

    cluster_of_row, _centroids = fit_minibatch_kmeans_on_memmapped_embeddings(
        run_path / "embeddings.npy", k, seed, passes)
    feature_to_cluster = dict(zip(unique_features, cluster_of_row.tolist()))

    cluster_to_features: dict[int, list[str]] = defaultdict(list)
    for feature, cluster_id in feature_to_cluster.items():
        cluster_to_features[cluster_id].append(feature)

    # Prevalence: distinct traces carrying >=1 feature from the cluster, and the trait mix
    # of those traces. Instances count repeats, so a stock phrase inflates it; prevalence
    # does not, which is why both are reported.
    cluster_to_trace_ids: dict[int, set[str]] = defaultdict(set)
    cluster_instance_counts: Counter = Counter()
    cluster_to_trait_counts: dict[int, Counter] = defaultdict(Counter)
    for record in per_trace_records:
        clusters_in_this_trace = set()
        for feature in record["features"]:
            cluster_id = feature_to_cluster[feature]
            cluster_instance_counts[cluster_id] += 1
            cluster_to_trace_ids[cluster_id].add(record["scenario_id"])
            clusters_in_this_trace.add(cluster_id)
        for cluster_id in clusters_in_this_trace:
            cluster_to_trait_counts[cluster_id][record["trait_id"]] += 1

    cluster_labels = name_clusters_with_llm(cluster_to_features, model, seed)

    n_traces = len(per_trace_records)
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
                              "naming_model": model,
                              "embedding_model": embedding_meta["model"],
                              "embedding_dim": embedding_meta["dim"],
                              "sanity_synonym": embedding_meta.get("sanity_synonym"),
                              "sanity_unrelated": embedding_meta.get("sanity_unrelated"),
                              "traces": n_traces, "unique_features": len(unique_features),
                              "feature_instances": sum(cluster_instance_counts.values()),
                              "git_sha": git_sha(), "timestamp_utc": timestamp()},
                     "clusters": clusters}
    (run_path / "clusters.json").write_text(json.dumps(clusters_json, indent=1))
    (run_path / "feature_cluster_map.json").write_text(json.dumps(feature_to_cluster))

    lines = [f"# Feature discovery — {run_path.name}", "",
             f"{n_traces} reasoning traces -> {sum(cluster_instance_counts.values())} "
             f"feature instances -> {len(unique_features)} unique -> {k} clusters", "",
             f"Embeddings: `{embedding_meta['model']}` ({embedding_meta['dim']}d). Sanity "
             f"check on the embedding geometry: `Backtracks in reasoning` ~ "
             f"`Self correction in reasoning` "
             f"= {embedding_meta.get('sanity_synonym', float('nan')):.3f}, vs "
             f"`Talks about apples` "
             f"= {embedding_meta.get('sanity_unrelated', float('nan')):.3f}.", "",
             f"Naming: `{model}`, 100 random features per cluster, prompt verbatim from the post.",
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

    print(f"\n{k} clusters. Top 15 by trace prevalence:")
    for cluster in clusters[:15]:
        print(f"  {cluster['prevalence']:>6.1%}  {cluster['label']}")
    print(f"\nwrote {run_path}/report.md, clusters.json")


if __name__ == "__main__":
    fire.Fire(main)
