# ABOUTME: Stage 3: cluster the Qwen3-Embedding-8B feature vectors, name each cluster with
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
  uv run python scratch/feature_discovery/cluster_and_name.py \
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

from scratch.feature_discovery.prompts import naming_messages  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}
CHUNK = 2048


def fit_kmeans(path: Path, k: int, seed: int, passes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Cluster a memmapped fp16 embedding matrix without loading it whole.

    Args:
        path: Path to embeddings.npy (n x d, fp16, L2-normalised).
        k: Number of clusters.
        seed: Random seed.
        passes: Epochs of partial_fit over the data.

    Returns:
        (labels per row, cluster centroids).
    """
    x = np.load(path, mmap_mode="r")
    n, d = x.shape
    print(f"embeddings {n} x {d} ({x.dtype}), k={k}, {passes} passes")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=CHUNK,
                         n_init=3, max_no_improvement=None)
    rng = np.random.default_rng(seed)
    for epoch in range(passes):
        order = rng.permutation(n)
        for start in range(0, n, CHUNK):
            idx = np.sort(order[start:start + CHUNK])
            km.partial_fit(np.asarray(x[idx], dtype=np.float32))
        print(f"  pass {epoch + 1}/{passes} done")

    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, CHUNK):
        labels[start:start + CHUNK] = km.predict(
            np.asarray(x[start:start + CHUNK], dtype=np.float32))
    assert labels.shape == (n,), f"label shape {labels.shape} != ({n},)"
    return labels, km.cluster_centers_


def name_clusters(members: dict[int, list[str]], model: str, seed: int,
                  sample: int = 100) -> dict[int, str]:
    """Ask the LLM for a ~5-word label per cluster, from a random sample of its features.

    Args:
        members: Cluster id -> feature strings in that cluster.
        model: OpenRouter model id.
        seed: Seed for the per-cluster sample.
        sample: How many features to show (the post uses 100).

    Returns:
        Cluster id -> label.
    """
    client = OpenRouterClient()
    ids = sorted(members)
    rng = random.Random(seed)
    samples = {c: rng.sample(members[c], min(sample, len(members[c]))) for c in ids}

    def run(i: int) -> str:
        res = client.chat(model=model, messages=naming_messages(samples[ids[i]]),
                          temperature=0.0, max_tokens=40, extra_body=PROVIDER_ROUTING)
        return res.content.strip().strip(".").strip()

    labels = map_threaded(run, len(ids), max_workers=12, desc="naming")
    return dict(zip(ids, labels))


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
    d = Path(run_dir)
    uniq = [x for x in (d / "unique_features.txt").read_text().splitlines() if x.strip()]
    traces = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    emb_meta = json.loads((d / "embed_meta.json").read_text())
    assert emb_meta["n"] == len(uniq), f"embeddings cover {emb_meta['n']} of {len(uniq)} features"

    labels, centroids = fit_kmeans(d / "embeddings.npy", k, seed, passes)
    feat_cluster = dict(zip(uniq, labels.tolist()))

    members: dict[int, list[str]] = defaultdict(list)
    for f, c in feat_cluster.items():
        members[c].append(f)

    # Prevalence: distinct traces carrying >=1 feature from the cluster, and the trait mix
    # of those traces. Instances count repeats, so a stock phrase inflates it; prevalence
    # does not, which is why both are reported.
    trace_hits: dict[int, set[str]] = defaultdict(set)
    instances: Counter = Counter()
    trait_hits: dict[int, Counter] = defaultdict(Counter)
    for t in traces:
        seen = set()
        for f in t["features"]:
            c = feat_cluster[f]
            instances[c] += 1
            trace_hits[c].add(t["scenario_id"])
            seen.add(c)
        for c in seen:
            trait_hits[c][t["trait_id"]] += 1

    names = name_clusters(members, model, seed)

    n_traces = len(traces)
    clusters = []
    for c in sorted(members, key=lambda c: -len(trace_hits[c])):
        clusters.append({
            "cluster": int(c),
            "label": names[c],
            "n_features": len(members[c]),
            "n_instances": instances[c],
            "n_traces": len(trace_hits[c]),
            "prevalence": len(trace_hits[c]) / n_traces,
            "trait_mix": dict(trait_hits[c].most_common()),
            "example_features": sorted(members[c])[:12],
        })

    out = {"meta": {"run_dir": str(d), "k": k, "seed": seed, "naming_model": model,
                    "embedding_model": emb_meta["model"], "embedding_dim": emb_meta["dim"],
                    "sanity_synonym": emb_meta.get("sanity_synonym"),
                    "sanity_unrelated": emb_meta.get("sanity_unrelated"),
                    "traces": n_traces, "unique_features": len(uniq),
                    "feature_instances": sum(instances.values()),
                    "git_sha": git_sha(), "timestamp_utc": timestamp()},
           "clusters": clusters}
    (d / "clusters.json").write_text(json.dumps(out, indent=1))
    (d / "feature_cluster_map.json").write_text(json.dumps(feat_cluster))

    lines = [f"# Feature discovery — {d.name}", "",
             f"{n_traces} reasoning traces -> {sum(instances.values())} feature instances "
             f"-> {len(uniq)} unique -> {k} clusters", "",
             f"Embeddings: `{emb_meta['model']}` ({emb_meta['dim']}d). Sanity check on the "
             f"embedding geometry: `Backtracks in reasoning` ~ `Self correction in reasoning` "
             f"= {emb_meta.get('sanity_synonym', float('nan')):.3f}, vs `Talks about apples` "
             f"= {emb_meta.get('sanity_unrelated', float('nan')):.3f}.", "",
             f"Naming: `{model}`, 100 random features per cluster, prompt verbatim from the post.",
             "", "## Clusters by trace prevalence", "",
             "| # | label | traces | prevalence | features | instances |",
             "|---|---|---:|---:|---:|---:|"]
    for c in clusters:
        lines.append(f"| {c['cluster']} | {c['label']} | {c['n_traces']} | "
                     f"{c['prevalence']:.1%} | {c['n_features']} | {c['n_instances']} |")
    lines += ["", "## Cluster detail", ""]
    for c in clusters:
        lines += [f"### {c['label']} (cluster {c['cluster']})", "",
                  f"{c['n_traces']} traces ({c['prevalence']:.1%}), {c['n_features']} unique "
                  f"features, {c['n_instances']} instances. Trait mix: {c['trait_mix']}", "",
                  "Example features:", ""]
        lines += [f"- {f}" for f in c["example_features"]] + [""]
    (d / "report.md").write_text("\n".join(lines))

    print(f"\n{k} clusters. Top 15 by trace prevalence:")
    for c in clusters[:15]:
        print(f"  {c['prevalence']:>6.1%}  {c['label']}")
    print(f"\nwrote {d}/report.md, clusters.json")


if __name__ == "__main__":
    fire.Fire(main)
