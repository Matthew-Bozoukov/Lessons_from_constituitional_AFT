# ABOUTME: Export this module's named clusters as property rows in the shared interchange
# ABOUTME: format, so they can be merged with TurF's and LESS's into one List of Properties.

"""The hand-off out of this module.

This module is one of several producers that feed a single **List of Properties** — the
others being TurF, the LESS ranking fed through an LLM/ML step, and the trace-level
UMAP+clustering. That list is what the ablation stage consumes, so every producer has to
emit rows a merger can put side by side without knowing how they were made.

One property per line in `properties.jsonl`, so files from different producers concatenate:

    property_id   "<source>:<run>:c<NN>" — stable across reruns of the same run directory
    source        which producer emitted it; "feature_discovery" here
    label         the property, as a short phrase
    prevalence    fraction of the corpus's traces exhibiting it — the comparable number
    n_traces      traces exhibiting it
    n_instances   times it was observed, repeats included
    support       producer-specific detail (feature counts, trait mix) — not merged on
    evidence      example feature strings, so a reader can judge the label
    provenance    run directory, git sha, models and params that produced it

`prevalence` is the field that has to mean the same thing across producers: the share of
traces in the SAME corpus exhibiting the property. Everything else is advisory.
"""

from __future__ import annotations

import json

from scratch.llm_feature_discovery.rundir import RunDir

SOURCE = "feature_discovery"
PROPERTIES_FILE = "properties.jsonl"
PROPERTIES_META_FILE = "properties_meta.json"


def build_rows(run: RunDir, source: str = SOURCE) -> list[dict]:
    """Turn a finished clustering into property rows.

    Args:
        run: The run directory holding clusters.json.
        source: Producer name to stamp on every row.

    Returns:
        One row per cluster, most prevalent first.
    """
    payload = run.read_clusters()
    meta, clusters = payload["meta"], payload["clusters"]
    provenance = {"run_dir": meta["run_dir"], "git_sha": meta.get("git_sha"),
                  "timestamp_utc": meta.get("timestamp_utc"),
                  "embedding_model": meta.get("embedding_model"),
                  "naming_model": meta.get("naming_model"),
                  "cluster_params": meta.get("cluster_params")}
    return [{"property_id": f"{source}:{run.name}:c{c['cluster']:03d}",
             "source": source,
             "label": c["label"],
             "prevalence": c["prevalence"],
             "n_traces": c["n_traces"],
             "n_instances": c["n_instances"],
             "support": {"n_features": c["n_features"], "trait_mix": c["trait_mix"]},
             "evidence": {"example_features": c["example_features"]},
             "provenance": provenance}
            for c in clusters]


def export(run: RunDir, source: str = SOURCE) -> tuple[int, str]:
    """Write properties.jsonl and its coverage metadata into the run directory.

    Args:
        run: The run directory.
        source: Producer name to stamp on every row.

    Returns:
        (rows written, path to properties.jsonl).
    """
    rows = build_rows(run, source)
    path = run.write_text(PROPERTIES_FILE,
                          "".join(json.dumps(row) + "\n" for row in rows))
    meta = run.read_clusters()["meta"]
    # Coverage belongs with the export, not inside the rows: a merger needs to know what
    # share of the corpus these properties DO NOT account for before it trusts the list.
    unique_features = meta.get("unique_features")
    total_instances = meta.get("feature_instances")
    run.write_json(PROPERTIES_META_FILE, {
        "source": source,
        "run_dir": meta["run_dir"],
        "properties": len(rows),
        "traces": meta.get("traces"),
        "unique_features": unique_features,
        "feature_instances": total_instances,
        "unclustered_features": meta.get("n_noise_features"),
        "unclustered_feature_share": (meta["n_noise_features"] / unique_features
                                      if meta.get("n_noise_features") is not None
                                      and unique_features else None),
        "unclustered_instances": meta.get("noise_instances"),
        "unclustered_instance_share": (meta["noise_instances"] / total_instances
                                       if meta.get("noise_instances") is not None
                                       and total_instances else None),
        "git_sha": meta.get("git_sha"),
        "timestamp_utc": meta.get("timestamp_utc"),
    })
    return len(rows), str(path)
