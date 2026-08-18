# ABOUTME: The module's only entrypoint: one CLI verb per pipeline stage, mapping
# ABOUTME: arguments onto pipeline.py and printing what each stage produced.

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
that directory, so any of them can be rerun on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

# Allows `python scratch/llm_feature_discovery/__main__.py` as well as `python -m ...`;
# under `-m` from the repository root this is already on the path and the insert is a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.llm_feature_discovery import embed as embed_mod  # noqa: E402
from scratch.llm_feature_discovery import pipeline  # noqa: E402
from scratch.llm_feature_discovery.rundir import RunDir  # noqa: E402

MODULE = "scratch.llm_feature_discovery"


def extract(input: str, run_dir: str | None = None,
            model: str = pipeline.DEFAULT_AUTORATER, temperature: float = 1.0,
            workers: int = 16, limit: int | None = None, smoke: bool = False) -> None:
    """Label every reasoning trace with free-text features.

    Args:
        input: SFT jsonl carrying `messages[2].reasoning_content` and `metadata`.
        run_dir: Run directory; defaults to output/feature_discovery/<timestamp>.
        model: OpenRouter autorater model.
        temperature: Sampling temperature.
        workers: Concurrent requests.
        limit: Only the first N traces.
        smoke: 3 traces plus a prompt-cache probe.
    """
    summary = pipeline.extract_features(input, run_dir, model, temperature, workers,
                                        limit, smoke)
    print(f"\n{summary['traces_labelled_this_run']} traces labelled -> "
          f"{summary['feature_instances']} feature instances in {summary['run_dir']}")
    print(f"tokens in={summary['tokens_in']:,} out={summary['tokens_out']:,} | "
          f"cost upper bound ${summary['cost_upper_bound_usd']:.2f}")
    print(f"features violating the a-z rule: "
          f"{summary['features_violating_letters_only_rule']}"
          + (f" e.g. {summary['violation_examples']}"
             if summary["violation_examples"] else ""))
    print(f"next: uv run python -m {MODULE} dedupe --run-dir {summary['run_dir']}")


def dedupe(run_dir: str) -> None:
    """Collapse the per-trace feature lists into the vocabulary to embed.

    Args:
        run_dir: The run directory.
    """
    s = pipeline.build_vocabulary(run_dir)
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


def cluster(run_dir: str, seed: int = 0, model: str = pipeline.DEFAULT_NAMING_MODEL,
            n_neighbors: int = 15, min_dist: float = 0.0, n_components: int = 10,
            min_cluster_size: int = 220) -> None:
    """Cluster the embeddings with UMAP + HDBSCAN and name the clusters.

    Args:
        run_dir: The run directory holding embeddings.npy.
        seed: Random seed for the clusterer and the naming samples.
        model: OpenRouter model used to name clusters.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum spacing.
        n_components: UMAP output dimensionality.
        min_cluster_size: HDBSCAN resolution knob; raise for coarser clusters.
    """
    s = pipeline.cluster_and_name(run_dir, seed, model, n_neighbors, min_dist,
                                  n_components, min_cluster_size)
    print(f"\n{s['n_clusters']} clusters ({s['n_noise_features']} features left as noise). "
          f"Top 15 by trace prevalence:")
    for prevalence, label in s["top_clusters"]:
        print(f"  {prevalence:>6.1%}  {label}")
    print(f"\nwrote {run_dir}/clusters.json, report.md")
    print(f"next: uv run python -m {MODULE} audit --run-dir {run_dir}")


def audit(run_dir: str) -> None:
    """Audit the clustering for redundancy and buried behaviours, and build the dashboard.

    Args:
        run_dir: The run directory holding clusters.json.
    """
    result = pipeline.audit_and_dashboard(run_dir)
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
    """Check a clustering against a baseline before anything is read off it.

    Args:
        baseline_dir: A previous clustering of the SAME feature list.
        candidate_dir: The clustering under test.
        seed: Seed for the reference fit and the neighbour sample.
        stability: Run the re-fit sweep; False gives geometry + agreement from one fit.
        stability_neighbors: n_neighbors values to sweep.
        stability_seeds: Seeds to sweep.
    """
    result = pipeline.gate(baseline_dir, candidate_dir, seed, stability,
                           stability_neighbors, stability_seeds)
    geometry, agreement = result["geometry"], result["agreement"]
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


def export(run_dir: str, source: str = "feature_discovery") -> None:
    """Emit the named clusters as rows for the shared List of Properties.

    Args:
        run_dir: The run directory holding clusters.json.
        source: Producer name stamped on every row.
    """
    s = pipeline.export_properties(run_dir, source)
    print(f"{s['properties']} properties -> {s['path']}")
    print("merge this with the other producers' properties.jsonl before the ablation stage")


class Embed:
    """The rented-GPU embedding stage: rent, push, poll, fetch, terminate."""

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
        meta = embed_mod.fetch(pod, RunDir.at(run_dir))
        print(meta)
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
