# ABOUTME: Compose the single-purpose modules into the pipeline's stages. Each function
# ABOUTME: here takes a run directory, does one stage, and returns a summary dict.

"""The stages, and nothing else.

Every module beside this one does one job and knows nothing about the order jobs run in.
This file is where that order lives, so the sequence is readable in one screen and the CLI
above it stays a pure argument surface.

Stages communicate only through the run directory (see `rundir.py`), so any of them can be
rerun on its own against a directory an earlier one left behind.
"""

from __future__ import annotations

import sys

from scratch.llm_feature_discovery import (audit as audit_mod, cluster, compare as compare_mod,
                                           dashboard, dedupe, extract, geometry, naming,
                                           prevalence, properties, report)
from scratch.llm_feature_discovery.cluster import ClusterParams
from scratch.llm_feature_discovery.rundir import RunDir, read_jsonl
from src.utils import git_sha, timestamp, write_run_meta

DEFAULT_AUTORATER = "anthropic/claude-sonnet-5"
DEFAULT_NAMING_MODEL = "anthropic/claude-sonnet-5"


def extract_features(input_path: str, run_dir: str | None = None,
                     model: str = DEFAULT_AUTORATER, temperature: float = 1.0,
                     workers: int = 16, limit: int | None = None,
                     smoke: bool = False) -> dict:
    """Stage 1 — label every reasoning trace with free-text features.

    Args:
        input_path: SFT jsonl carrying `messages[2].reasoning_content` and `metadata`.
        run_dir: Run directory; defaults to output/feature_discovery/<timestamp>.
        model: OpenRouter autorater model.
        temperature: Sampling temperature (the post's brainstorm framing wants > 0).
        workers: Concurrent requests.
        limit: Only the first N traces.
        smoke: 3 traces, a prompt-cache probe, and the first result printed in full.

    Returns:
        Usage and quality counters, plus the run directory used.
    """
    sft_rows = list(read_jsonl(input_path))
    if smoke:
        limit = limit or 3
    if limit:
        sft_rows = sft_rows[:limit]

    run = RunDir.at(run_dir or f"output/feature_discovery/{timestamp()}").ensure()
    if smoke:
        from src.endpoints.openrouter import OpenRouterClient
        print("--- cache probe ---")
        extract.probe_prompt_caching(OpenRouterClient(), model,
                                     sft_rows[0]["messages"][2]["reasoning_content"])

    summary = extract.extract_corpus(run, sft_rows, model, temperature, workers)
    summary["run_dir"] = str(run.path)
    write_run_meta(run.path, {"input": input_path, "model": model,
                              "temperature": temperature, **summary,
                              "command": " ".join(sys.argv)})
    return summary


def build_vocabulary(run_dir: str) -> dict:
    """Stage 2 — collapse the per-trace lists into the vocabulary to embed.

    Args:
        run_dir: The run directory.

    Returns:
        Repetition counters.
    """
    summary = dedupe.build_vocabulary(RunDir.at(run_dir))
    summary["run_dir"] = run_dir
    return summary


def cluster_and_name(run_dir: str, seed: int = 0, model: str = DEFAULT_NAMING_MODEL,
                     n_neighbors: int = 15, min_dist: float = 0.0, n_components: int = 10,
                     min_cluster_size: int = 220) -> dict:
    """Stage 4 — cluster the embeddings, name the clusters, write the canonical result.

    Stage 3 is the embedding, which runs on a rented GPU; see `embed.py`.

    Args:
        run_dir: The run directory holding embeddings.npy.
        seed: Random seed for the clusterer and the naming samples.
        model: OpenRouter model used to name clusters.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum spacing.
        n_components: UMAP output dimensionality.
        min_cluster_size: HDBSCAN resolution knob.

    Returns:
        Cluster counts and the noise share.

    Raises:
        RuntimeError: If the embeddings do not cover the vocabulary.
    """
    run = RunDir.at(run_dir)
    features = run.read_unique_features()
    embed_meta = run.read_embed_meta()
    if embed_meta["n"] != len(features):
        raise RuntimeError(f"embeddings cover {embed_meta['n']} of {len(features)} features")

    params = ClusterParams(n_neighbors=n_neighbors, min_dist=min_dist,
                           n_components=n_components, min_cluster_size=min_cluster_size)
    clustering = cluster.fit(run.read_embeddings(), params, seed)
    run.write_umap_coords(clustering.coords)

    feature_to_cluster = cluster.build_feature_cluster_map(features, clustering.labels)
    run.write_feature_cluster_map(feature_to_cluster)
    grouped = prevalence.group_features_by_cluster(feature_to_cluster)
    labels = naming.name_clusters(grouped, model, seed)
    stats = prevalence.compute(run.read_trace_features(), feature_to_cluster, labels)

    meta = {"run_dir": str(run.path), "seed": seed, "cluster": "hdbscan", "reduce": "umap",
            "clustering": params.describe(), "cluster_params": params.as_dict(),
            "n_clusters": clustering.n_clusters,
            "n_noise_features": clustering.n_noise,
            "noise_instances": stats.noise_instances,
            "naming_model": model,
            "embedding_model": embed_meta["model"], "embedding_dim": embed_meta["dim"],
            "sanity_synonym": embed_meta.get("sanity_synonym"),
            "sanity_unrelated": embed_meta.get("sanity_unrelated"),
            "traces": stats.n_traces, "unique_features": len(features),
            "feature_instances": stats.total_instances,
            "git_sha": git_sha(), "timestamp_utc": timestamp()}
    run.write_clusters(meta, stats.clusters)
    run.write_text("report.md", report.clustering_report(meta, stats.clusters, run.name))
    return {"run_dir": run_dir, "n_clusters": clustering.n_clusters,
            "n_noise_features": clustering.n_noise,
            "noise_instances": stats.noise_instances,
            "top_clusters": [(c["prevalence"], c["label"]) for c in stats.clusters[:15]]}


def audit_and_dashboard(run_dir: str) -> dict:
    """Stage 5 — audit the clustering for redundancy and buried behaviours, and render it.

    Args:
        run_dir: The run directory holding clusters.json.

    Returns:
        The audit payload.
    """
    run = RunDir.at(run_dir)
    result = audit_mod.audit_run(run)
    run.write_json("report_audit.json",
                   {k: result[k] for k in ("near_duplicate_clusters", "probes",
                                           "dup_threshold")})
    run.append_text("report.md", report.audit_section(result))
    payload = run.read_clusters()
    run.write_text("dashboard.html",
                   dashboard.render(payload["meta"], payload["clusters"], result, run.name))
    return result


def gate(baseline_dir: str, candidate_dir: str, seed: int = 0, stability: bool = True,
         stability_neighbors: tuple[int, ...] = (10, 15, 30, 50),
         stability_seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Stage 6 — decide whether a clustering can be trusted before anything is read off it.

    Args:
        baseline_dir: A previous clustering of the SAME feature list to compare against.
        candidate_dir: The clustering under test; its meta supplies the UMAP params.
        seed: Seed for the reference fit and the neighbour sample.
        stability: Run the re-fit sweep. False gives geometry + agreement from one fit.
        stability_neighbors: n_neighbors values to sweep.
        stability_seeds: Seeds to sweep.

    Returns:
        The gate report.

    Raises:
        RuntimeError: If the two runs did not cluster the same feature list.
    """
    import umap    # the only stage that needs it beyond cluster.fit

    baseline, candidate = RunDir.at(baseline_dir), RunDir.at(candidate_dir)
    features = candidate.read_unique_features()
    if baseline.read_unique_features() != features:
        raise RuntimeError("the two runs cluster different feature lists; ARI between them "
                           "would be meaningless")

    candidate_meta = candidate.read_clusters()["meta"]
    params = ClusterParams(**candidate_meta["cluster_params"])
    embeddings = candidate.read_embeddings()
    embed_meta = candidate.read_embed_meta()

    print(f"{len(features)} features, {embeddings.shape[1]}d -> {params.n_components}d "
          f"(reference fit, n_neighbors={params.n_neighbors})")
    reducer = umap.UMAP(n_components=params.n_components, n_neighbors=params.n_neighbors,
                        min_dist=params.min_dist, metric="cosine",
                        random_state=seed).fit(embeddings)
    reference_coords = reducer.embedding_

    geometry_result = {
        "neighbours_compared": geometry.NEIGHBOURS_COMPARED,
        "neighbour_overlap": geometry.neighbour_overlap(embeddings, reference_coords, seed),
        "probe": geometry.probe_survives_reduction(candidate.read_probe_embeddings(),
                                                   reducer, embed_meta)}
    agreement = compare_mod.agreement(baseline.read_feature_cluster_map(),
                                      candidate.read_feature_cluster_map(), features)

    sweep, pairwise = [], []
    if stability:
        reference_labels = cluster.cluster_coords(reference_coords, params)
        sweep, pairwise = compare_mod.stability_sweep(embeddings, params, reference_labels,
                                                      stability_neighbors, stability_seeds)

    result = {"meta": {"baseline_dir": baseline_dir, "candidate_dir": candidate_dir,
                       "seed": seed, "clustering": params.describe(),
                       "cluster_params": params.as_dict(),
                       "unique_features": len(features),
                       "embedding_model": embed_meta["model"],
                       "embedding_dim": embeddings.shape[1],
                       "git_sha": git_sha(), "timestamp_utc": timestamp()},
              "geometry": geometry_result, "agreement": agreement,
              "stability": {"sweep": sweep, "pairwise_ari": pairwise,
                            "min_pairwise_ari": min(min(row) for row in pairwise)
                            if pairwise else None}}
    candidate.write_json("clustering_comparison.json", result)
    candidate.write_text("clustering_comparison.md",
                         report.comparison_report(result, baseline.name, candidate.name))
    return result


def export_properties(run_dir: str, source: str = properties.SOURCE) -> dict:
    """Stage 7 — emit the named clusters as rows for the shared List of Properties.

    Args:
        run_dir: The run directory holding clusters.json.
        source: Producer name stamped on every row.

    Returns:
        How many rows were written and where.
    """
    n_rows, path = properties.export(RunDir.at(run_dir), source)
    return {"properties": n_rows, "path": path}
