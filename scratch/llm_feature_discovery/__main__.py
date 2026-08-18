# ABOUTME: The module's only entrypoint. One function per stage: it wires the
# ABOUTME: single-purpose modules together, and prints what the stage produced.

"""Command line for the feature-discovery module.

    uv run python -m scratch.llm_feature_discovery <verb> [args]

Verbs, in the order a fresh corpus goes through them:

    extract    SFT jsonl of reasoning traces -> features.jsonl
    dedupe     features.jsonl -> unique_features.txt + feature_counts.json
    embed      create | push | status | fetch | terminate  (rents a GPU; see embed.py)
    cluster    embeddings.npy -> clusters.json + feature_cluster_map.json + report.md
    audit      clusters.json -> report_audit.json + dashboard.html
    gate       two runs -> clustering_comparison.{json,md}   (run before trusting a change)
    export     clusters.json -> properties.jsonl for the shared List of Properties

Every verb after `extract` takes `--run-dir`, and stages talk to each other only through
that directory, so any of them can be rerun on its own. This file is the only one that
knows the order; every other module does one job and knows nothing about the sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

# Allows `python scratch/llm_feature_discovery/__main__.py` as well as `python -m ...`;
# under `-m` from the repository root this is already on the path and the insert is a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.llm_feature_discovery import audit as audit_mod  # noqa: E402
from scratch.llm_feature_discovery import cluster as cluster_mod  # noqa: E402
from scratch.llm_feature_discovery import embed as embed_mod  # noqa: E402
from scratch.llm_feature_discovery import extract as extract_mod  # noqa: E402
from scratch.llm_feature_discovery import properties as properties_mod  # noqa: E402
from scratch.llm_feature_discovery.cluster import ClusterParams  # noqa: E402
from scratch.llm_feature_discovery.rundir import RunDir, read_jsonl  # noqa: E402
from src.utils import git_sha, timestamp, write_run_meta  # noqa: E402

MODULE = "scratch.llm_feature_discovery"
DEFAULT_AUTORATER = "anthropic/claude-sonnet-5"
DEFAULT_NAMING_MODEL = "anthropic/claude-sonnet-5"


def extract(input: str, run_dir: str | None = None, model: str = DEFAULT_AUTORATER,
            temperature: float = 1.0, workers: int = 16, limit: int | None = None,
            smoke: bool = False) -> None:
    """Stage 1 — label every reasoning trace with free-text features.

    Args:
        input: SFT jsonl carrying `messages[2].reasoning_content` and `metadata`.
        run_dir: Run directory; defaults to output/feature_discovery/<timestamp>.
        model: OpenRouter autorater model.
        temperature: Sampling temperature (the post's brainstorm framing wants > 0).
        workers: Concurrent requests.
        limit: Only the first N traces.
        smoke: 3 traces plus a prompt-cache probe.
    """
    sft_rows = list(read_jsonl(input))
    if smoke:
        limit = limit or 3
    if limit:
        sft_rows = sft_rows[:limit]

    run = RunDir.at(run_dir or f"output/feature_discovery/{timestamp()}").ensure()
    if smoke:
        from src.endpoints.openrouter import OpenRouterClient
        print("--- cache probe ---")
        extract_mod.probe_prompt_caching(OpenRouterClient(), model,
                                         sft_rows[0]["messages"][2]["reasoning_content"])

    s = extract_mod.extract_corpus(run, sft_rows, model, temperature, workers)
    write_run_meta(run.path, {"input": input, "model": model, "temperature": temperature,
                              **s, "command": " ".join(sys.argv)})
    print(f"\n{s['traces_labelled_this_run']} traces labelled -> "
          f"{s['feature_instances']} feature instances in {run.path}")
    print(f"tokens in={s['tokens_in']:,} out={s['tokens_out']:,} | "
          f"cost upper bound ${s['cost_upper_bound_usd']:.2f}")
    print(f"features violating the a-z rule: {s['features_violating_letters_only_rule']}"
          + (f" e.g. {s['violation_examples']}" if s["violation_examples"] else ""))
    print(f"next: uv run python -m {MODULE} dedupe --run-dir {run.path}")


def dedupe(run_dir: str) -> None:
    """Stage 2 — collapse the per-trace feature lists into the vocabulary to embed.

    Args:
        run_dir: The run directory.
    """
    s = extract_mod.build_vocabulary(RunDir.at(run_dir))
    print(f"traces            {s['traces']}")
    print(f"feature instances {s['feature_instances']} "
          f"({s['features_per_trace_mean']:.1f} per trace, "
          f"min {s['features_per_trace_min']}, max {s['features_per_trace_max']})")
    print(f"unique strings    {s['unique_features']} "
          f"({s['unique_share_of_instances']:.1%} of instances)")
    print(f"appearing once    {s['appearing_once']}")
    print("\nmost repeated features:")
    for feature, count in s["most_repeated"]:
        print(f"  {count:>4}x  {feature}")
    print(f"\nnext: uv run python -m {MODULE} embed create")


def cluster(run_dir: str, seed: int = 0, model: str = DEFAULT_NAMING_MODEL,
            n_neighbors: int = 15, min_dist: float = 0.0, n_components: int = 10,
            min_cluster_size: int = 220) -> None:
    """Stage 4 — cluster the embeddings with UMAP + HDBSCAN and name the clusters.

    Stage 3 is the embedding, which runs on a rented GPU; see the `embed` verb.

    Args:
        run_dir: The run directory holding embeddings.npy.
        seed: Random seed for the clusterer and the naming samples.
        model: OpenRouter model used to name clusters.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum spacing.
        n_components: UMAP output dimensionality.
        min_cluster_size: HDBSCAN resolution knob; raise for coarser clusters.

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
    clustering = cluster_mod.fit(run.read_embeddings(), params, seed)
    run.write_umap_coords(clustering.coords)

    feature_to_cluster = cluster_mod.build_feature_cluster_map(features, clustering.labels)
    run.write_feature_cluster_map(feature_to_cluster)
    grouped = cluster_mod.group_features_by_cluster(feature_to_cluster)
    labels = cluster_mod.name_clusters(grouped, model, seed)
    stats = cluster_mod.compute_prevalence(run.read_trace_features(), feature_to_cluster,
                                           labels)

    meta = {"run_dir": str(run.path), "seed": seed, "cluster": "hdbscan", "reduce": "umap",
            "clustering": params.describe(), "cluster_params": params.as_dict(),
            "n_clusters": clustering.n_clusters, "n_noise_features": clustering.n_noise,
            "noise_instances": stats.noise_instances, "naming_model": model,
            "embedding_model": embed_meta["model"], "embedding_dim": embed_meta["dim"],
            "sanity_synonym": embed_meta.get("sanity_synonym"),
            "sanity_unrelated": embed_meta.get("sanity_unrelated"),
            "traces": stats.n_traces, "unique_features": len(features),
            "feature_instances": stats.total_instances,
            "git_sha": git_sha(), "timestamp_utc": timestamp()}
    run.write_clusters(meta, stats.clusters)
    run.write_text("report.md",
                   cluster_mod.render_report(meta, stats.clusters, run.name))

    print(f"\n{clustering.n_clusters} clusters ({clustering.n_noise} features left as "
          f"noise). Top 15 by trace prevalence:")
    for c in stats.clusters[:15]:
        print(f"  {c['prevalence']:>6.1%}  {c['label']}")
    print(f"\nwrote {run_dir}/clusters.json, report.md")
    print(f"next: uv run python -m {MODULE} audit --run-dir {run_dir}")


def audit(run_dir: str) -> None:
    """Stage 5 — audit the clustering for redundancy and buried behaviours; build the HTML.

    Args:
        run_dir: The run directory holding clusters.json.
    """
    run = RunDir.at(run_dir)
    result = audit_mod.audit_run(run)
    run.write_json("report_audit.json",
                   {k: result[k] for k in ("near_duplicate_clusters", "probes",
                                           "dup_threshold")})
    run.append_text("report.md", audit_mod.render_audit_section(result))
    payload = run.read_clusters()
    run.write_text("dashboard.html",
                   audit_mod.render_dashboard(payload["meta"], payload["clusters"],
                                              result, run.name))

    pairs = result["near_duplicate_clusters"]
    print(f"near-duplicate cluster pairs (>= {result['dup_threshold']}): {len(pairs)}")
    for pair in pairs[:8]:
        print(f"  {pair['cosine']:.3f}  {pair['label_a']}  ||  {pair['label_b']}")
    print("\nprobes:")
    for name, probe in result["probes"].items():
        print(f"  {probe['prevalence']:>6.1%} of traces  {name}  "
              f"({probe['unique_features']} unique features)")
    print(f"\nwrote {run_dir}/dashboard.html, report_audit.json, appended to report.md")
    print(f"next: uv run python -m {MODULE} export --run-dir {run_dir}")


def gate(baseline_dir: str, candidate_dir: str, seed: int = 0, stability: bool = True,
         stability_neighbors: tuple[int, ...] = (10, 15, 30, 50),
         stability_seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Stage 6 — decide whether a clustering can be trusted before anything is read off it.

    Args:
        baseline_dir: A previous clustering of the SAME feature list to compare against.
        candidate_dir: The clustering under test; its meta supplies the UMAP params.
        seed: Seed for the reference fit and the neighbour sample.
        stability: Run the re-fit sweep. False gives geometry + agreement from one fit.
        stability_neighbors: n_neighbors values to sweep.
        stability_seeds: Seeds to sweep.

    Raises:
        RuntimeError: If the two runs did not cluster the same feature list.
    """
    import umap    # the reducer has to be re-fitted here to transform the probe vectors

    baseline, candidate = RunDir.at(baseline_dir), RunDir.at(candidate_dir)
    features = candidate.read_unique_features()
    if baseline.read_unique_features() != features:
        raise RuntimeError("the two runs cluster different feature lists; ARI between them "
                           "would be meaningless")

    params = ClusterParams(**candidate.read_clusters()["meta"]["cluster_params"])
    embeddings = candidate.read_embeddings()
    embed_meta = candidate.read_embed_meta()

    print(f"{len(features)} features, {embeddings.shape[1]}d -> {params.n_components}d "
          f"(reference fit, n_neighbors={params.n_neighbors})")
    reducer = umap.UMAP(n_components=params.n_components, n_neighbors=params.n_neighbors,
                        min_dist=params.min_dist, metric="cosine",
                        random_state=seed).fit(embeddings)
    reference_coords = reducer.embedding_

    geometry = {"neighbours_compared": audit_mod.NEIGHBOURS_COMPARED,
                "neighbour_overlap": audit_mod.neighbour_overlap(embeddings,
                                                                 reference_coords, seed),
                "probe": audit_mod.probe_survives_reduction(
                    candidate.read_probe_embeddings(), reducer, embed_meta)}
    agreement = audit_mod.label_agreement(baseline.read_feature_cluster_map(),
                                          candidate.read_feature_cluster_map(), features)

    sweep, pairwise = [], []
    if stability:
        reference_labels = cluster_mod.cluster_coords(reference_coords, params)
        sweep, pairwise = audit_mod.stability_sweep(embeddings, params, reference_labels,
                                                    stability_neighbors, stability_seeds)

    report = {"meta": {"baseline_dir": baseline_dir, "candidate_dir": candidate_dir,
                       "seed": seed, "clustering": params.describe(),
                       "cluster_params": params.as_dict(), "unique_features": len(features),
                       "embedding_model": embed_meta["model"],
                       "embedding_dim": embeddings.shape[1],
                       "git_sha": git_sha(), "timestamp_utc": timestamp()},
              "geometry": geometry, "agreement": agreement,
              "stability": {"sweep": sweep, "pairwise_ari": pairwise,
                            "min_pairwise_ari": min(min(row) for row in pairwise)
                            if pairwise else None}}
    candidate.write_json("clustering_comparison.json", report)
    candidate.write_text("clustering_comparison.md",
                         audit_mod.render_gate_report(report, baseline.name, candidate.name))

    print(f"neighbour overlap @{geometry['neighbours_compared']}: "
          f"{geometry['neighbour_overlap']:.3f}")
    probe = geometry["probe"]
    if probe["available"]:
        print(f"probe synonym/unrelated: full-dim {probe['full_dim_synonym']:.3f}/"
              f"{probe['full_dim_unrelated']:.3f} -> reduced "
              f"{probe['reduced_synonym']:.3f}/{probe['reduced_unrelated']:.3f} "
              f"({'gap survives' if probe['gap_survives'] else 'GAP LOST'})")
    else:
        print(f"probe check skipped: {probe['note']}")
    print(f"agreement over {agreement['features_compared']} shared features "
          f"({agreement['features_dropped_as_noise']} dropped as noise): "
          f"ARI {agreement['ari']:.3f}, AMI {agreement['ami']:.3f}")
    print(f"\nwrote {candidate_dir}/clustering_comparison.md, clustering_comparison.json")


def export(run_dir: str, source: str = properties_mod.SOURCE) -> None:
    """Stage 7 — emit the named clusters as rows for the shared List of Properties.

    Args:
        run_dir: The run directory holding clusters.json.
        source: Producer name stamped on every row.
    """
    n_rows, path = properties_mod.export(RunDir.at(run_dir), source)
    print(f"{n_rows} properties -> {path}")
    print("merge this with the other producers' properties.jsonl before the ablation stage")


class Embed:
    """Stage 3 — the rented-GPU embedding step: rent, push, poll, fetch, terminate."""

    def create(self, gpu: str = embed_mod.DEFAULT_GPU_TYPE, disk_gb: int = 80,
               batch: int = 128) -> None:
        """Create the embedding pod.

        Args:
            gpu: RunPod GPU type id.
            disk_gb: Container disk (16GB model + image + HF cache).
            batch: Encoding batch size.
        """
        pod_id = embed_mod.create(gpu=gpu, disk_gb=disk_gb, batch=batch)
        print(f"pod:      {pod_id}")
        print(f"boot log: https://{pod_id}-8080.proxy.runpod.net/boot.log")
        print(f"next:     uv run python -m {MODULE} embed push --pod {pod_id} "
              f"--run-dir <dir>")
        print(f"TEARDOWN: uv run python -m {MODULE} embed terminate --pod {pod_id}")

    def push(self, pod: str, run_dir: str) -> None:
        """Upload the run's feature list; the pod embeds as soon as it lands.

        Args:
            pod: Pod id.
            run_dir: The run directory holding unique_features.txt.
        """
        sent = embed_mod.push(pod, RunDir.at(run_dir))
        print(f"pushed {sent} features (verified); watch "
              f"https://{pod}-8080.proxy.runpod.net/embed.log")

    def status(self, pod: str) -> None:
        """Print pod state and the tail of the most advanced log.

        Args:
            pod: Pod id.
        """
        info = embed_mod.status(pod)
        print(f"status: {info['desiredStatus']}  cost/hr: ${info['costPerHr']}")
        if info["log_name"]:
            print(f"--- {info['log_name']} tail ---\n{info['log_tail']}")
        else:
            print("no logs reachable yet")

    def fetch(self, pod: str, run_dir: str) -> None:
        """Download the embedding artifacts into the run directory.

        Args:
            pod: Pod id.
            run_dir: The run directory to write into.
        """
        print(embed_mod.fetch(pod, RunDir.at(run_dir)))
        print(f"next: uv run python -m {MODULE} embed terminate --pod {pod}   "
              f"# the pod bills by the second")

    def terminate(self, pod: str) -> None:
        """Terminate the pod and report what is still running on the account.

        Args:
            pod: Pod id.
        """
        still_running = embed_mod.terminate(pod)
        print(f"terminated {pod}")
        print(f"still running on this account: {still_running}")


if __name__ == "__main__":
    fire.Fire({"extract": extract, "dedupe": dedupe, "embed": Embed(), "cluster": cluster,
               "audit": audit, "gate": gate, "export": export})
