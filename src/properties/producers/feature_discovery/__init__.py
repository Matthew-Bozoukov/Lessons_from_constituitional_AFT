# ABOUTME: feature_discovery — an autorater invents its own vocabulary per trace, the
# ABOUTME: vocabulary is embedded and clustered, and each cluster becomes a property.

"""LLM-driven feature discovery (LessWrong post `WAZWA6FPQvH8okouJ`).

An autorater is shown one trace at a time, with no metadata and no schema, and asked what
its "features" are. Because it is not scoring against axes we chose in advance, it can
surface behaviours nothing anticipated — which is the whole reason to run it alongside the
other three producers rather than instead of them.

    trace -> 10-20 free-text features -> dedupe to a vocabulary -> embed -> cluster -> name

**Port status: the producer code still lives in `scratch/llm_feature_discovery/`.** What is
here is the boundary: `produce()` reads that module's run directory and turns its clusters
into Property rows. After the port it will read the same artifacts from the same schema,
produced next door — the ARTIFACTS are the interface, which is why porting the producer
changes nothing downstream. Six files move in beside this one:

    prompts    the post's two verbatim prompts
    extract    trace -> features -> the unique vocabulary
    cluster    grouping + naming  (its clustering becomes shared/grouping.py)
    centroids  the noise rule, in one place
    audit      redundancy pairs, keyword probes, the clustering gate, the dashboard
    rundir     what a run directory holds

and three of its current files do NOT move, because `shared/` already holds them:
`embed.py` becomes `shared/embed.py`, `properties.py` becomes `registry.py`, and the
grouping half of `cluster.py` becomes `shared/grouping.py`.

One thing this adds that the scratch module never had: a DETECTOR. That module's export
writes a label, a prevalence and some example features; none of those can select the rows
an ablation should edit. So each cluster's features are fed through `shared/interpret.py`,
which returns the label AND the yes/no test — and the test is what lets `ablation/` act on
the cluster and `verify.py` show that acting worked. The interpreter's label is kept
ALONGSIDE the scratch module's own naming, not instead of it: they come from different
prompts (the post's ~5-word naming prompt versus the ablation-oriented one), and a
disagreement between them is a signal about the cluster, not a bug to paper over.

Two caveats from the existing runs, worth re-reading before quoting a number from this
producer (the full list is in that module's README):

* The cluster count is a RESOLUTION SETTING, not a count of behaviours. 84 of 11,175
  cluster pairs sat at centroid cosine >= 0.90 in the k=150 run.
* A cluster label is not evidence a behaviour is absent. `Displays evaluations awareness`
  (89 occurrences) landed inside a generic cluster; only a keyword probe surfaced it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import interpret as interpret_mod
from src.utils import git_sha, timestamp

SOURCE = "feature_discovery"
SCRATCH_MODULE = "scratch.llm_feature_discovery"


def read_run_dir(run_dir: str | Path) -> dict:
    """Read a feature-discovery run's clusters.json.

    Args:
        run_dir: The run directory (e.g. output/feature_discovery/20260812_092119).

    Returns:
        The parsed {"meta", "clusters"} payload.

    Raises:
        FileNotFoundError: With the command that produces the missing file.
    """
    path = Path(run_dir) / "clusters.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. feature_discovery is not ported yet — run it under "
            f"scratch and point this producer at its run directory:\n"
            f"  uv run python -m {SCRATCH_MODULE} extract --input <sft.jsonl>\n"
            f"  uv run python -m {SCRATCH_MODULE} dedupe  --run-dir {run_dir}\n"
            f"  uv run python -m {SCRATCH_MODULE} embed   ...\n"
            f"  uv run python -m {SCRATCH_MODULE} cluster --run-dir {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def produce(records, cfg, out_dir: str | Path, target=None) -> list[Property]:
    """Turn a feature-discovery run's clusters into Property rows.

    Args:
        records: The corpus, used only to re-measure prevalence with the detectors when
            `measure_with_detector` is set. The cluster prevalences themselves come from
            the run directory, which measured them over the traces it labelled.
        cfg: The producer's config block. Keys: `run_dir` (required), `min_prevalence`,
            `interpret {model, n_shown, workers}`, `measure_with_detector`,
            `detector {model, workers, sample}`.
        out_dir: Where to write this producer's preview artifacts.
        target: Unused; feature discovery describes a corpus.

    Returns:
        Property rows, most prevalent first.

    Raises:
        KeyError: If the config does not name a `run_dir`.
    """
    import dataclasses
    import random

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)
    run_dir = Path(str(cfg["run_dir"]))
    payload = read_run_dir(run_dir)
    meta, clusters = payload["meta"], payload["clusters"]
    minimum = float(cfg.get("min_prevalence", 0.0))
    clusters = [c for c in clusters if c["prevalence"] >= minimum]
    # The run's OWN recorded name, not the directory it happens to be read from: the
    # published 2026-08-12 run is checked in under docs/feature_discovery/, and a
    # property_id that changed when someone copied the artifacts would break every
    # reference to it in an ablation config, a train config name and a LOG entry.
    run_name = Path(str(meta.get("run_dir") or run_dir)).name

    interpretations = interpret_mod.interpret_many(
        {c["cluster"]: c["example_features"] for c in clusters},
        channel="reasoning",
        **block(cfg, "interpret"))

    # Coverage belongs beside the rows, not inside them: a merger needs to know what share
    # of the corpus these properties do NOT account for before it trusts the list.
    unique_features = meta.get("unique_features")
    provenance = {"run_dir": str(run_dir), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(),
                  "producer_git_sha": meta.get("git_sha"),
                  "producer_timestamp_utc": meta.get("timestamp_utc"),
                  "embedding_model": meta.get("embedding_model"),
                  "naming_model": meta.get("naming_model"),
                  "cluster_params": meta.get("cluster_params"),
                  "traces": meta.get("traces"),
                  "unclustered_features": meta.get("n_noise_features"),
                  "unclustered_feature_share": (
                      meta["n_noise_features"] / unique_features
                      if meta.get("n_noise_features") is not None and unique_features
                      else None),
                  "unclustered_instances": meta.get("noise_instances")}

    properties = []
    for cluster in clusters:
        interpretation = interpretations.get(cluster["cluster"])
        if interpretation is None:
            continue
        fields = interpretation.to_dict()
        properties.append(Property.make(
            SOURCE, run_name, f"c{cluster['cluster']:03d}",
            prevalence=round(float(cluster["prevalence"]), 4),
            n_records=int(cluster["n_traces"]), n_instances=int(cluster["n_instances"]),
            corpus={"path": meta.get("input") or meta.get("run_dir")},
            support={"cluster": int(cluster["cluster"]),
                     "producer_label": cluster["label"],
                     "n_features": cluster["n_features"],
                     "trait_mix": cluster.get("trait_mix"),
                     "prevalence_kind": "cluster_membership"},
            evidence={"example_features": cluster["example_features"]},
            provenance=provenance, **fields))

    if bool(cfg.get("measure_with_detector", False)) and records:
        detector_cfg = block(cfg, "detector")
        sample_n = int(detector_cfg.pop("sample", 200))
        sample = (records if len(records) <= sample_n
                  else random.Random(0).sample(list(records), sample_n))
        corpus = (records[0].metadata.get("corpus") or {})
        remeasured = []
        for prop in properties:
            verdicts = interpret_mod.detect(sample, prop.label, prop.detector,
                                            channel=prop.channel, **detector_cfg)
            updated = prop.with_prevalence(interpret_mod.prevalence(verdicts), corpus)
            remeasured.append(dataclasses.replace(updated, support={
                **updated.support, "cluster_membership_prevalence": prop.prevalence,
                "prevalence_kind": "detector_measured",
                "detector_sample_n": len(sample)}))
        properties = remeasured

    properties.sort(key=lambda p: -(p.prevalence or 0))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    return properties
