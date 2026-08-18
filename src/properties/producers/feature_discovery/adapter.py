# ABOUTME: feature_discovery's produce(): read a feature-discovery run directory's
# ABOUTME: clusters.json, write each cluster a detector, and emit Property rows.

"""The boundary between feature discovery and the shared List of Properties.

While the producer still lives under `scratch/llm_feature_discovery/`, this reads the run
directory it writes. After the port it will read the same artifacts from the same schema,
produced next door — the artifacts ARE the interface, which is why porting the producer
changes nothing here.

One thing this adapter adds that the scratch module never had: a DETECTOR. That module's
export writes a label, a prevalence and some example features; none of those can select the
rows an ablation should edit. So each cluster's features are fed through
`shared/interpret.py`, which returns the label AND the yes/no test — and the test is what
lets `ablation/` act on the cluster and `verify.py` show that acting worked.

The label the interpreter writes is kept alongside the scratch module's own naming, not
instead of it. They come from different prompts (the post's ~5-word naming prompt versus
the ablation-oriented one here) and a disagreement between them is a signal about the
cluster, not a bug to paper over.
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
        out_dir: Where to write this adapter's preview artifacts.
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
