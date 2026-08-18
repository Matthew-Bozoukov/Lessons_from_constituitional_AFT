# ABOUTME: Render the markdown mirrors — clustering report, audit section, and the
# ABOUTME: clustering gate — so every number is greppable without opening a JSON file.

"""Markdown renderers.

Pure functions from parsed JSON to text. They hold no I/O and no analysis, which is what
keeps the analysis modules free of formatting and lets the wording be edited without
touching anything that computes a number.
"""

from __future__ import annotations


def clustering_report(meta: dict, clusters: list[dict], run_name: str) -> str:
    """The main report for a finished clustering.

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
             f"Naming: `{meta['naming_model']}`, 100 random features per cluster, prompt "
             f"verbatim from the post.", "",
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


def audit_section(audit: dict) -> str:
    """The redundancy + probe section, appended to the clustering report.

    Args:
        audit: The dict from audit.audit_run.

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


def comparison_report(report: dict, baseline_name: str, candidate_name: str) -> str:
    """The clustering gate's markdown mirror.

    Args:
        report: The dict written to clustering_comparison.json.
        baseline_name: Basename of the baseline run.
        candidate_name: Basename of the run under test.

    Returns:
        Markdown.
    """
    geometry, agree = report["geometry"], report["agreement"]
    stability = report["stability"]
    meta = report["meta"]
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
