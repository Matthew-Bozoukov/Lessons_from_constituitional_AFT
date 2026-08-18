# ABOUTME: Stages 5-6. Audit a clustering for redundancy and buried behaviours, gate it
# ABOUTME: against a baseline, and render both as markdown and a browsable dashboard.

"""Is this clustering any good?

Four checks the naming stage cannot do for itself, in the order they can disqualify a run:

* **Geometry.** UMAP is free to tear a manifold. If it did, HDBSCAN clustered an artefact
  and every label downstream is meaningless. Measured as nearest-neighbour overlap between
  the full-dimensional and reduced spaces, plus the embedding sanity probe where the run
  kept its vectors. Only the probe's ORDERING carries over: UMAP compresses the space, so
  reduced cosines run higher across the board and the absolute numbers are not comparable.
* **Agreement.** ARI/AMI against a baseline clustering of the same feature list — usually
  the published k-means run. Close agreement means the numbers already reported were sound.
* **Stability.** Re-fit across `n_neighbors` and seeds and take pairwise ARI. A clustering
  that reshuffles when the seed changes is not a finding.
* **Redundancy and buried behaviours.** Cluster centroids that sit close together describe
  the same theme, so the cluster count is a resolution setting rather than a count of
  behaviours; and a small distinctive theme gets absorbed into a large bland one, so the
  keyword probes read the raw feature strings independently of the clustering.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from scratch.llm_feature_discovery import centroids, cluster as cluster_mod
from scratch.llm_feature_discovery.cluster import ClusterParams
from scratch.llm_feature_discovery.rundir import RunDir

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
UNCLUSTERED_LABEL = "(unclustered noise)"
NEIGHBOURS_COMPARED = 10
FEATURES_SAMPLED = 500
SAMPLED_ROWS_PER_CHUNK = 64


# ---------------------------------------------------------------- stage 5: audit --------

def find_near_duplicate_clusters(run: RunDir, cluster_by_id: dict[int, dict],
                                 threshold: float = NEAR_DUPLICATE_COSINE_THRESHOLD
                                 ) -> list[dict]:
    """Cluster pairs whose centroids are close enough to describe the same theme.

    Args:
        run: The run directory.
        cluster_by_id: Cluster id -> its record from clusters.json.
        threshold: Centroid cosine at or above which a pair counts as near-duplicate.

    Returns:
        Pairs, most similar first.
    """
    n_clusters = len(cluster_by_id)
    centroid_matrix = centroids.compute(run.read_embeddings(), run.read_unique_features(),
                                        run.read_feature_cluster_map(), n_clusters)
    cosine = centroid_matrix @ centroid_matrix.T
    np.fill_diagonal(cosine, 0.0)
    pairs = [{"a": int(i), "b": int(j), "cosine": float(cosine[i, j]),
              "label_a": cluster_by_id[int(i)]["label"],
              "label_b": cluster_by_id[int(j)]["label"]}
             for i, j in zip(*np.triu_indices(n_clusters, k=1))
             if cosine[i, j] >= threshold]
    pairs.sort(key=lambda p: -p["cosine"])
    return pairs


def run_keyword_probes(trace_records: list[dict], unique_features: list[str],
                       feature_to_cluster: dict[str, int], cluster_by_id: dict[int, dict],
                       probes: dict[str, str] = KEYWORD_PROBES) -> dict[str, dict]:
    """Count how often each probe's behaviour appears, independent of the clustering.

    Args:
        trace_records: {scenario_id, trait_id, features} per labelled trace.
        unique_features: The feature vocabulary.
        feature_to_cluster: Feature -> cluster id; a miss means the feature is noise.
        cluster_by_id: Cluster id -> its record, for naming where matches landed.
        probes: Probe name -> regex.

    Returns:
        Probe name -> counts, examples, and which clusters its matches landed in.
    """
    instance_counts = Counter(f for record in trace_records for f in record["features"])
    results = {}
    for probe_name, pattern in probes.items():
        probe_re = re.compile(pattern, re.I)
        matching = [f for f in unique_features if probe_re.search(f)]
        matching_traces = {record["scenario_id"] for record in trace_records
                           if any(probe_re.search(f) for f in record["features"])}
        landed_in = Counter(
            cluster_by_id[feature_to_cluster[f]]["label"] if f in feature_to_cluster
            else UNCLUSTERED_LABEL
            for f in matching)
        results[probe_name] = {
            "unique_features": len(matching),
            "instances": sum(instance_counts[f] for f in matching),
            "traces": len(matching_traces),
            "prevalence": len(matching_traces) / len(trace_records),
            "top_examples": sorted(matching, key=lambda f: -instance_counts[f])[:8],
            "clusters_landed_in": landed_in.most_common(5),
        }
    return results


def audit_run(run: RunDir) -> dict:
    """Run both audit checks over a finished clustering.

    Args:
        run: The run directory holding clusters.json and embeddings.npy.

    Returns:
        {"near_duplicate_clusters", "probes", "dup_threshold", "n_clusters"}.
    """
    clusters = run.read_clusters()["clusters"]
    cluster_by_id = {c["cluster"]: c for c in clusters}
    return {"near_duplicate_clusters": find_near_duplicate_clusters(run, cluster_by_id),
            "probes": run_keyword_probes(run.read_trace_features(),
                                         run.read_unique_features(),
                                         run.read_feature_cluster_map(), cluster_by_id),
            "dup_threshold": NEAR_DUPLICATE_COSINE_THRESHOLD,
            "n_clusters": len(cluster_by_id)}


# ---------------------------------------------------------------- stage 6: gate ---------

def neighbour_overlap(embeddings: np.ndarray, reduced: np.ndarray, seed: int,
                      n_neighbors: int = NEIGHBOURS_COMPARED,
                      n_sampled: int = FEATURES_SAMPLED) -> float:
    """Fraction of each sampled feature's nearest neighbours that survive the reduction.

    The full n x n similarity matrix is far too large to hold, so this samples rows and
    scores each against the whole corpus in chunks. Cosine in the original space and
    Euclidean in the reduced one, because Euclidean is what HDBSCAN then uses.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        reduced: (n x m) coordinates for the same rows.
        seed: Seed for the row sample.
        n_neighbors: Neighbourhood size to compare.
        n_sampled: How many rows to score.

    Returns:
        Mean overlap in [0, 1]; 1.0 means every neighbourhood survived intact.
    """
    rng = np.random.default_rng(seed)
    sampled_rows = rng.choice(embeddings.shape[0],
                              size=min(n_sampled, embeddings.shape[0]), replace=False)
    overlaps = []
    for start in range(0, len(sampled_rows), SAMPLED_ROWS_PER_CHUNK):
        rows = sampled_rows[start:start + SAMPLED_ROWS_PER_CHUNK]
        # Higher cosine = nearer; negate so both spaces sort ascending on "far".
        full_distance = -(embeddings[rows] @ embeddings.T)
        reduced_distance = np.linalg.norm(reduced[rows][:, None, :] - reduced[None, :, :],
                                          axis=2)
        for offset, row in enumerate(rows):
            full_distance[offset, row] = np.inf       # exclude self
            reduced_distance[offset, row] = np.inf
            nearest_full = set(np.argpartition(full_distance[offset],
                                               n_neighbors)[:n_neighbors].tolist())
            nearest_reduced = set(np.argpartition(reduced_distance[offset],
                                                  n_neighbors)[:n_neighbors].tolist())
            overlaps.append(len(nearest_full & nearest_reduced) / n_neighbors)
    return float(np.mean(overlaps))


def probe_survives_reduction(probe_embeddings: np.ndarray | None, reducer,
                             embed_meta: dict) -> dict:
    """Push the stored embedding sanity probes through the fitted reducer.

    Args:
        probe_embeddings: (3 x d) probe vectors, or None for a run that did not keep them.
        reducer: A fitted UMAP object.
        embed_meta: Parsed embed_meta.json, for the full-dimensional baseline.

    Returns:
        Before/after cosines, or {"available": False} with a note.
    """
    if probe_embeddings is None:
        return {"available": False,
                "note": "run predates probe_embeddings.npy; rely on neighbour overlap"}
    reduced = np.asarray(reducer.transform(probe_embeddings), dtype=np.float32)
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
    synonym, unrelated = float(reduced[0] @ reduced[1]), float(reduced[0] @ reduced[2])
    return {"available": True,
            "probe": embed_meta.get("probe"),
            "full_dim_synonym": embed_meta.get("sanity_synonym"),
            "full_dim_unrelated": embed_meta.get("sanity_unrelated"),
            "reduced_synonym": synonym,
            "reduced_unrelated": unrelated,
            "gap_survives": bool(synonym > unrelated)}


def label_agreement(baseline_map: dict[str, int], candidate_map: dict[str, int],
                    features: list[str]) -> dict:
    """Score two clusterings of the same feature list against each other.

    Only features that BOTH clusterings placed are scored; anything either one left as
    noise is dropped and counted, because ARI is undefined for a point with no label.

    Args:
        baseline_map: Feature -> cluster id from the baseline run.
        candidate_map: Feature -> cluster id from the run under test.
        features: The shared feature list.

    Returns:
        ARI, AMI, and the counts they were computed over.

    Raises:
        ValueError: If the two clusterings share no clustered features.
    """
    shared = [f for f in features if f in baseline_map and f in candidate_map]
    if not shared:
        raise ValueError("the two clusterings share no clustered features")
    baseline = np.array([baseline_map[f] for f in shared], dtype=np.int32)
    candidate = np.array([candidate_map[f] for f in shared], dtype=np.int32)
    return {"ari": float(adjusted_rand_score(baseline, candidate)),
            "ami": float(adjusted_mutual_info_score(baseline, candidate)),
            "features_compared": len(shared),
            "features_dropped_as_noise": len(features) - len(shared),
            "baseline_clusters": int(len(set(baseline.tolist()))),
            "candidate_clusters": int(len(set(candidate.tolist())))}


def stability_sweep(embeddings: np.ndarray, params: ClusterParams,
                    reference_labels: np.ndarray, neighbors: tuple[int, ...],
                    seeds: tuple[int, ...]) -> tuple[list[dict], list[list[float]]]:
    """Re-fit across n_neighbors and seeds and score every fit against the others.

    Each fit is scored against the reference labelling AND against every other fit, so one
    lucky seed cannot pass as agreement.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: The reference knobs; n_neighbors is overridden per sweep point.
        reference_labels: Labels from the reference fit.
        neighbors: n_neighbors values to sweep.
        seeds: Seeds to sweep.

    Returns:
        (per-fit records, pairwise ARI matrix in the same order).
    """
    sweep, labelings = [], []
    for n_neighbors in neighbors:
        for seed in seeds:
            swept = ClusterParams(n_neighbors=n_neighbors, min_dist=params.min_dist,
                                  n_components=params.n_components,
                                  min_cluster_size=params.min_cluster_size)
            coords = cluster_mod.reduce_embeddings(embeddings, swept, seed)
            labels = cluster_mod.cluster_coords(coords, swept)
            n_noise = int((labels == cluster_mod.NOISE_LABEL).sum())
            record = {"n_neighbors": n_neighbors, "seed": seed,
                      "n_clusters": int(labels.max()) + 1,
                      "noise_fraction": n_noise / len(labels),
                      "ari_vs_reference": float(adjusted_rand_score(reference_labels, labels))}
            sweep.append(record)
            labelings.append(labels)
            print(f"  n_neighbors={n_neighbors} seed={seed}: {record['n_clusters']} "
                  f"clusters, {record['noise_fraction']:.1%} noise, "
                  f"ARI vs reference {record['ari_vs_reference']:.3f}")
    pairwise = [[float(adjusted_rand_score(a, b)) for b in labelings] for a in labelings]
    return sweep, pairwise


# ---------------------------------------------------------------- markdown --------------

def render_audit_section(audit: dict) -> str:
    """The redundancy + probe section, appended to the clustering report.

    Args:
        audit: The dict from audit_run.

    Returns:
        Markdown, starting with a blank line so it appends cleanly.
    """
    pairs, probes = audit["near_duplicate_clusters"], audit["probes"]
    lines = ["", "## Cluster redundancy audit", "",
             f"Cluster centroids with cosine >= {audit['dup_threshold']} describe "
             f"substantially the same theme. **{len(pairs)} such pairs** among "
             f"{audit['n_clusters']} clusters — a corpus with one dominant house style "
             "splits it across many labels, so treat the cluster count as a resolution "
             "setting, not a count of distinct behaviours.", ""]
    if pairs:
        lines += ["| cosine | cluster A | cluster B |", "|---:|---|---|"]
        lines += [f"| {p['cosine']:.3f} | {p['label_a']} | {p['label_b']} |"
                  for p in pairs[:25]]
    lines += ["", "## Keyword probes", "",
              "A behaviour can be real and still have no cluster of its own, because a "
              "small distinctive theme gets absorbed into a large bland one. These counts "
              "come from the raw feature strings, independent of the clustering.", "",
              "| probe | traces | prevalence | unique features | instances | mostly landed in |",
              "|---|---:|---:|---:|---:|---|"]
    for name, probe in probes.items():
        landed_in = probe["clusters_landed_in"][0][0] if probe["clusters_landed_in"] else "-"
        lines.append(f"| {name} | {probe['traces']} | {probe['prevalence']:.1%} | "
                     f"{probe['unique_features']} | {probe['instances']} | {landed_in} |")
    lines += [""]
    for name, probe in probes.items():
        if probe["top_examples"]:
            lines += [f"**{name}** examples: "
                      + "; ".join(f"`{e}`" for e in probe["top_examples"]), ""]
    return "\n".join(lines)


def render_gate_report(report: dict, baseline_name: str, candidate_name: str) -> str:
    """The gate's markdown mirror.

    Args:
        report: The dict written to clustering_comparison.json.
        baseline_name: Basename of the baseline run.
        candidate_name: Basename of the run under test.

    Returns:
        Markdown.
    """
    geometry, agree, stability, meta = (report["geometry"], report["agreement"],
                                        report["stability"], report["meta"])
    lines = [f"# Clustering gate — {candidate_name} vs {baseline_name}", "",
             f"{meta['unique_features']} features, `{meta['embedding_model']}`, "
             f"{meta['clustering']}.", "",
             "## Did the reduction keep the geometry?", "",
             f"Nearest-neighbour overlap @{geometry['neighbours_compared']} between the "
             f"{meta['embedding_dim']}d and {meta['cluster_params']['n_components']}d "
             f"spaces: **{geometry['neighbour_overlap']:.3f}**.", ""]
    probe = geometry["probe"]
    if probe["available"]:
        lines += [f"Embedding sanity probe: synonym pair {probe['full_dim_synonym']:.3f} "
                  f"-> {probe['reduced_synonym']:.3f}, unrelated pair "
                  f"{probe['full_dim_unrelated']:.3f} -> {probe['reduced_unrelated']:.3f}. "
                  f"{'Gap survives.' if probe['gap_survives'] else '**GAP LOST.**'} "
                  "Only the ordering is comparable — UMAP compresses the space, so every "
                  "reduced cosine runs higher.", ""]
    else:
        lines += [f"Probe check unavailable: {probe['note']}.", ""]
    lines += ["## Do the two clusterings agree?", "",
              "| metric | value |", "|---|---:|",
              f"| ARI | {agree['ari']:.3f} |", f"| AMI | {agree['ami']:.3f} |",
              f"| features compared | {agree['features_compared']} |",
              f"| dropped as noise | {agree['features_dropped_as_noise']} |",
              f"| baseline clusters | {agree['baseline_clusters']} |",
              f"| candidate clusters | {agree['candidate_clusters']} |", ""]
    if stability["sweep"]:
        lines += ["## Is the clustering stable?", "",
                  "| n_neighbors | seed | clusters | noise | ARI vs reference |",
                  "|---:|---:|---:|---:|---:|"]
        lines += [f"| {e['n_neighbors']} | {e['seed']} | {e['n_clusters']} | "
                  f"{e['noise_fraction']:.1%} | {e['ari_vs_reference']:.3f} |"
                  for e in stability["sweep"]]
        lines += ["", f"Lowest pairwise ARI across the sweep: "
                      f"**{stability['min_pairwise_ari']:.3f}**.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- dashboard -------------

DASHBOARD_STYLE = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1200px;
padding:28px;color:#1c1c1e;background:#fafafa}
h2{margin-top:34px;border-bottom:2px solid #e3e3e6;padding-bottom:6px}
table{border-collapse:collapse;width:100%;background:#fff;margin:10px 0}
th,td{border:1px solid #e3e3e6;padding:6px 9px;text-align:left;font-size:14px;vertical-align:top}
th{background:#f0f0f3;position:sticky;top:0} .mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;min-width:150px}
.big{font-size:24px;font-weight:650} .lab{color:#666;font-size:13px}
ul{margin:6px 0 6px 18px;padding:0} summary{cursor:pointer}
"""


def _esc(value) -> str:
    """HTML-escape any value.

    Args:
        value: Anything stringifiable.

    Returns:
        The escaped string.
    """
    return html.escape(str(value))


def _card(value, label: str) -> str:
    """One headline-number card.

    Args:
        value: The number.
        label: Its caption.

    Returns:
        HTML.
    """
    return f"<div class=card><div class=big>{_esc(value)}</div><div class=lab>{label}</div></div>"


def render_dashboard(meta: dict, clusters: list[dict], audit: dict, run_name: str) -> str:
    """Build the browsable HTML mirror of the clustering and its audit.

    Args:
        meta: clusters.json meta block.
        clusters: Per-cluster records, most prevalent first.
        audit: The dict from audit_run.
        run_name: The run directory's basename.

    Returns:
        A complete, self-contained HTML document.
    """
    pairs, probes = audit["near_duplicate_clusters"], audit["probes"]
    cluster_rows = "".join(
        f"<tr><td>{c['cluster']}</td><td><b>{_esc(c['label'])}</b></td>"
        f"<td>{c['n_traces']}</td><td>{c['prevalence']:.1%}</td>"
        f"<td>{c['n_features']}</td><td>{c['n_instances']}</td>"
        f"<td><details><summary>features</summary><ul>"
        + "".join(f"<li>{_esc(f)}</li>" for f in c["example_features"])
        + f"</ul><p class=mono>traits: {_esc(json.dumps(c['trait_mix']))}</p></details></td></tr>"
        for c in clusters)
    duplicate_rows = "".join(
        f"<tr><td>{p['cosine']:.3f}</td><td>{_esc(p['label_a'])}</td>"
        f"<td>{_esc(p['label_b'])}</td></tr>" for p in pairs[:30])
    probe_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{p['traces']}</td><td>{p['prevalence']:.1%}</td>"
        f"<td>{p['unique_features']}</td>"
        f"<td class=mono>{_esc('; '.join(p['top_examples'][:4]))}</td></tr>"
        for name, p in probes.items())
    cards = "".join([_card(meta["traces"], "traces"),
                     _card(meta["unique_features"], "unique features"),
                     _card(meta["n_clusters"], "clusters"),
                     _card(meta["n_noise_features"], "unclustered (noise)"),
                     _card(len(pairs), "near-duplicate cluster pairs"),
                     _card(f"{meta.get('sanity_synonym', float('nan')):.2f}/"
                           f"{meta.get('sanity_unrelated', float('nan')):.2f}",
                           "embedding sanity syn/unrel")])
    return f"""<!doctype html><meta charset=utf-8>
<title>Feature discovery — {_esc(run_name)}</title><style>{DASHBOARD_STYLE}</style>
<h1>LLM-driven feature discovery</h1>
<p class=mono>{_esc(meta['traces'])} reasoning traces · {_esc(meta['feature_instances'])} feature
instances · {_esc(meta['unique_features'])} unique · embeddings {_esc(meta['embedding_model'])}
({_esc(meta['embedding_dim'])}d) · naming {_esc(meta['naming_model'])}
· {_esc(meta['clustering'])} · {_esc(meta['timestamp_utc'])}</p>
<div class=cards>{cards}</div>
<h2>Clusters by trace prevalence</h2>
<table><tr><th>#<th>label<th>traces<th>prevalence<th>features<th>instances<th>examples</tr>{cluster_rows}</table>
<h2>Near-duplicate clusters (centroid cosine &ge; {audit['dup_threshold']})</h2>
<table><tr><th>cosine<th>A<th>B</tr>{duplicate_rows or '<tr><td colspan=3>none</td></tr>'}</table>
<h2>Keyword probes</h2>
<table><tr><th>probe<th>traces<th>prevalence<th>unique features<th>examples</tr>{probe_rows}</table>
"""
