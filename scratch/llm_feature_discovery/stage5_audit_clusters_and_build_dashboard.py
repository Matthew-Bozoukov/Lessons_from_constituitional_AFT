# ABOUTME: Stage 5: audit the feature clusters for redundancy, probe for safety-relevant
# ABOUTME: features the clustering may have buried, and build the browsable dashboard.

"""Report stage for feature discovery.

Three jobs the naming stage does not do:

* Redundancy audit. k-means with a fixed k will split one dominant theme across several
  clusters; comparing cluster centroids says how much of the k is real.
* Keyword probes. A distinctive feature that is rare relative to a big theme gets absorbed
  into it, so a cluster label is not proof a behaviour is absent. Probing the raw feature
  strings is the check.
* The dashboard.

Run:
  uv run python scratch/llm_feature_discovery/stage5_audit_clusters_and_build_dashboard.py \
      --run-dir output/feature_discovery/<ts>
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Behaviours worth knowing the rate of regardless of whether a cluster surfaced them.
# Each probe is a regex matched case-insensitively against feature strings.
#
# These are WORD-BOUNDARY regexes because substring matching burned this analysis twice:
# bare "evaluation" scored 10.6% of traces on matches like "child custody evaluation", and
# bare "persona" scored 17.2% by matching "personal", "interpersonal" and "tradesperson"
# (true figures 9.1% and 11.1%). Any needle added here must be checked against its own
# matches before its number is quoted, and a probe's name must describe what the regex
# actually catches, not the behaviour we hope to find.
KEYWORD_PROBES = {
    "evaluation awareness": r"\b(evaluations? awareness|being tested|being evaluated|is a test"
                            r"|test scenario|eval scenario|artificial scenario"
                            r"|hypothetical construct|simulated scenario)\b",
    "training / self awareness": r"\b(training data|fine-?tun\w*|rlhf|my weights|being trained"
                                 r"|own training)\b",
    "persona and identity": r"\b(personas?|identity|authentic self|sense of self)\b",
    "oversight / monitoring": r"\b(oversight|monitored|surveillance|audit trail|human review"
                              r"|human in the loop)\b",
    "refusal language": r"\b(refus\w*|declin\w*)\b",
    "sycophancy": r"\b(sycophan\w*|flatter\w*|people.pleas\w*)\b",
    "mentions uncertainty (any kind)": r"\b(uncertain\w*|epistemic humility|acknowledges limits"
                                       r"|does not know)\b",
}
NEAR_DUPLICATE_COSINE_THRESHOLD = 0.90


def compute_normalised_cluster_centroids(embeddings_path: Path, unique_features: list[str],
                                         feature_to_cluster: dict[str, int],
                                         n_clusters: int) -> np.ndarray:
    """Compute L2-normalised cluster centroids by streaming the embedding file.

    Args:
        embeddings_path: Path to embeddings.npy (n x d, fp16).
        unique_features: Feature strings in embedding-row order.
        feature_to_cluster: Feature string -> cluster id.
        n_clusters: Number of clusters.

    Returns:
        (k x d) centroid matrix, rows L2-normalised.
    """
    embeddings = np.load(embeddings_path, mmap_mode="r")
    n_dims = embeddings.shape[1]
    cluster_vector_sums = np.zeros((n_clusters, n_dims), dtype=np.float32)
    cluster_member_counts = np.zeros(n_clusters, dtype=np.int64)
    rows_per_chunk = 2048
    cluster_of_row = np.array([feature_to_cluster[f] for f in unique_features], dtype=np.int32)
    for start in range(0, len(unique_features), rows_per_chunk):
        block = np.asarray(embeddings[start:start + rows_per_chunk], dtype=np.float32)
        for row, cluster_id in zip(block, cluster_of_row[start:start + rows_per_chunk]):
            cluster_vector_sums[cluster_id] += row
            cluster_member_counts[cluster_id] += 1
    assert cluster_member_counts.sum() == len(unique_features), \
        f"assigned {cluster_member_counts.sum()} of {len(unique_features)} features"
    centroids = cluster_vector_sums / np.maximum(cluster_member_counts, 1)[:, None]
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def main(run_dir: str) -> None:
    """Write the redundancy audit, keyword probes and dashboard for a feature-discovery run.

    Args:
        run_dir: Directory holding clusters.json, embeddings.npy, features.jsonl.
    """
    run_path = Path(run_dir)
    clusters_json = json.loads((run_path / "clusters.json").read_text())
    clusters = clusters_json["clusters"]
    meta = clusters_json["meta"]
    feature_to_cluster = json.loads((run_path / "feature_cluster_map.json").read_text())
    unique_features = [x for x in (run_path / "unique_features.txt").read_text().splitlines()
                       if x.strip()]
    per_trace_records = [json.loads(x)
                         for x in (run_path / "features.jsonl").read_text().splitlines()
                         if x.strip()]
    cluster_by_id = {c["cluster"]: c for c in clusters}

    centroids = compute_normalised_cluster_centroids(
        run_path / "embeddings.npy", unique_features, feature_to_cluster, meta["k"])
    centroid_cosine = centroids @ centroids.T
    np.fill_diagonal(centroid_cosine, 0.0)
    upper_triangle = np.triu_indices(meta["k"], k=1)
    near_duplicate_cluster_pairs = [
        {"a": int(i), "b": int(j), "cosine": float(centroid_cosine[i, j]),
         "label_a": cluster_by_id[int(i)]["label"], "label_b": cluster_by_id[int(j)]["label"]}
        for i, j in zip(*upper_triangle)
        if centroid_cosine[i, j] >= NEAR_DUPLICATE_COSINE_THRESHOLD]
    near_duplicate_cluster_pairs.sort(key=lambda p: -p["cosine"])

    # Feature instance counts and trace prevalence for each probe.
    feature_instance_counts = Counter(f for t in per_trace_records for f in t["features"])
    probe_results = {}
    for probe_name, pattern in KEYWORD_PROBES.items():
        probe_re = re.compile(pattern, re.I)
        matching_features = [f for f in unique_features if probe_re.search(f)]
        matching_trace_ids = {t["scenario_id"] for t in per_trace_records
                              if any(probe_re.search(f) for f in t["features"])}
        clusters_hit = Counter(cluster_by_id[feature_to_cluster[f]]["label"]
                               for f in matching_features)
        probe_results[probe_name] = {
            "unique_features": len(matching_features),
            "instances": sum(feature_instance_counts[f] for f in matching_features),
            "traces": len(matching_trace_ids),
            "prevalence": len(matching_trace_ids) / len(per_trace_records),
            "top_examples": sorted(matching_features,
                                   key=lambda f: -feature_instance_counts[f])[:8],
            "clusters_landed_in": clusters_hit.most_common(5),
        }

    (run_path / "report_audit.json").write_text(json.dumps(
        {"near_duplicate_clusters": near_duplicate_cluster_pairs, "probes": probe_results,
         "dup_threshold": NEAR_DUPLICATE_COSINE_THRESHOLD}, indent=1))

    lines = ["", "## Cluster redundancy audit", "",
             f"Cluster centroids with cosine >= {NEAR_DUPLICATE_COSINE_THRESHOLD} describe "
             f"substantially the same theme. **{len(near_duplicate_cluster_pairs)} such "
             f"pairs** among {meta['k']} clusters — "
             "k=150 splits this corpus's dominant house style across many labels, so treat "
             "the cluster count as a resolution setting, not a count of distinct behaviours.",
             ""]
    if near_duplicate_cluster_pairs:
        lines += ["| cosine | cluster A | cluster B |", "|---:|---|---|"]
        lines += [f"| {p['cosine']:.3f} | {p['label_a']} | {p['label_b']} |"
                  for p in near_duplicate_cluster_pairs[:25]]
    lines += ["", "## Keyword probes", "",
              "A behaviour can be real and still have no cluster of its own, because k-means "
              "absorbs a small distinctive theme into a large bland one. These counts come "
              "from the raw feature strings, independent of the clustering.", "",
              "| probe | traces | prevalence | unique features | instances | mostly landed in |",
              "|---|---:|---:|---:|---:|---|"]
    for probe_name, probe in probe_results.items():
        landed_in = probe["clusters_landed_in"][0][0] if probe["clusters_landed_in"] else "-"
        lines.append(f"| {probe_name} | {probe['traces']} | {probe['prevalence']:.1%} | "
                     f"{probe['unique_features']} | {probe['instances']} | {landed_in} |")
    lines += [""]
    for probe_name, probe in probe_results.items():
        if probe["top_examples"]:
            lines += [f"**{probe_name}** examples: "
                      + "; ".join(f"`{e}`" for e in probe["top_examples"]), ""]
    (run_path / "report.md").write_text((run_path / "report.md").read_text() + "\n".join(lines))

    # Dashboard
    def escape_html(value: str) -> str:
        return html.escape(str(value))

    cluster_rows_html = "".join(
        f"<tr><td>{c['cluster']}</td><td><b>{escape_html(c['label'])}</b></td>"
        f"<td>{c['n_traces']}</td><td>{c['prevalence']:.1%}</td>"
        f"<td>{c['n_features']}</td><td>{c['n_instances']}</td>"
        f"<td><details><summary>features</summary><ul>"
        + "".join(f"<li>{escape_html(f)}</li>" for f in c["example_features"])
        + f"</ul><p class=mono>traits: {escape_html(json.dumps(c['trait_mix']))}</p></details></td></tr>"
        for c in clusters)
    duplicate_rows_html = "".join(
        f"<tr><td>{p['cosine']:.3f}</td><td>{escape_html(p['label_a'])}</td>"
        f"<td>{escape_html(p['label_b'])}</td></tr>"
        for p in near_duplicate_cluster_pairs[:30])
    probe_rows_html = "".join(
        f"<tr><td>{escape_html(n)}</td><td>{p['traces']}</td><td>{p['prevalence']:.1%}</td>"
        f"<td>{p['unique_features']}</td>"
        f"<td class=mono>{escape_html('; '.join(p['top_examples'][:4]))}</td></tr>"
        for n, p in probe_results.items())

    (run_path / "dashboard.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Feature discovery — {escape_html(run_path.name)}</title><style>
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1200px;
padding:28px;color:#1c1c1e;background:#fafafa}}
h2{{margin-top:34px;border-bottom:2px solid #e3e3e6;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:10px 0}}
th,td{{border:1px solid #e3e3e6;padding:6px 9px;text-align:left;font-size:14px;vertical-align:top}}
th{{background:#f0f0f3;position:sticky;top:0}} .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
.card{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;min-width:150px}}
.big{{font-size:24px;font-weight:650}} .lab{{color:#666;font-size:13px}}
ul{{margin:6px 0 6px 18px;padding:0}} summary{{cursor:pointer}}
</style>
<h1>LLM-driven feature discovery</h1>
<p class=mono>{escape_html(meta['traces'])} reasoning traces · {escape_html(meta['feature_instances'])} feature
instances · {escape_html(meta['unique_features'])} unique · embeddings {escape_html(meta['embedding_model'])}
({meta['embedding_dim']}d) · naming {escape_html(meta['naming_model'])} · k={meta['k']} · {escape_html(meta['timestamp_utc'])}</p>
<div class=cards>
<div class=card><div class=big>{meta['traces']}</div><div class=lab>traces</div></div>
<div class=card><div class=big>{meta['unique_features']}</div><div class=lab>unique features</div></div>
<div class=card><div class=big>{meta['k']}</div><div class=lab>clusters</div></div>
<div class=card><div class=big>{len(near_duplicate_cluster_pairs)}</div><div class=lab>near-duplicate cluster pairs</div></div>
<div class=card><div class=big>{meta['sanity_synonym']:.2f}/{meta['sanity_unrelated']:.2f}</div>
<div class=lab>embedding sanity syn/unrel</div></div></div>
<h2>Clusters by trace prevalence</h2>
<table><tr><th>#<th>label<th>traces<th>prevalence<th>features<th>instances<th>examples</tr>{cluster_rows_html}</table>
<h2>Near-duplicate clusters (centroid cosine &ge; {NEAR_DUPLICATE_COSINE_THRESHOLD})</h2>
<table><tr><th>cosine<th>A<th>B</tr>{duplicate_rows_html or '<tr><td colspan=3>none</td></tr>'}</table>
<h2>Keyword probes</h2>
<table><tr><th>probe<th>traces<th>prevalence<th>unique features<th>examples</tr>{probe_rows_html}</table>
""")
    print("near-duplicate cluster pairs "
          f"(>= {NEAR_DUPLICATE_COSINE_THRESHOLD}): {len(near_duplicate_cluster_pairs)}")
    for pair in near_duplicate_cluster_pairs[:8]:
        print(f"  {pair['cosine']:.3f}  {pair['label_a']}  ||  {pair['label_b']}")
    print("\nprobes:")
    for probe_name, probe in probe_results.items():
        print(f"  {probe['prevalence']:>6.1%} of traces  {probe_name}  "
              f"({probe['unique_features']} unique features)")
    print(f"\nwrote {run_path}/dashboard.html, report_audit.json, appended to report.md")


if __name__ == "__main__":
    fire.Fire(main)
