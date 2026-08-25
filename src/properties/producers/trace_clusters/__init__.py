# ABOUTME: trace_clusters — embed WHOLE records, cluster them, name the clusters. The
# ABOUTME: simplest producer, and the reference implementation of the shared layer.

"""Cluster the traces themselves, not descriptions of them.

    embed -> group -> interpret -> Property rows

The other producers put a model between the record and the vector: an autorater writes
features or attributes, and those strings get embedded. That buys abstraction (the
autorater says what a trace DOES) at the cost of an extra model's opinion in the loop.
This producer does the direct thing — embed the record's text, cluster the embeddings,
name each cluster from its members — which is cheap, has no autorater in it, and answers a
question the others cannot: do two scenario formats occupy DIFFERENT regions of trace
space at all?

That question is the 2026-08-17 action item ("UMAP + clustering on good traces; compare DA
vs Courtroom / Peer Critique"), and it needs the direct version: an autorater's vocabulary
would smooth over exactly the surface differences the comparison is about.

Every stage of it is a call into `shared/`, which is the point: if adding a producer takes
more than one file, something belongs in `shared/` that is not there yet.

Prevalence here is MEMBERSHIP: the share of records whose embedding landed in the group.
That is a different quantity from what the detector would say if run over the corpus, and
the two disagree at the edges of a cluster — so the config can ask for the detector to be
run (`measure_with_detector: true`), which replaces the membership number with the measured
one and records both. Cross-producer comparisons should use the measured one; it is the
only number all four producers can produce the same way.

The arm comparison (`group_by:`) is what the corpus-vs-corpus question needs: point this at
a mixture holding two scenario formats, group by the metadata field that separates them,
and every property row carries the per-arm share alongside the corpus-wide one. A property
at 40% in difficult advice and 3% in courtroom is a candidate explanation for a difference
between the two arms; a property at 30% in both is not, however interesting it reads.

Its known weakness, and why the other three producers exist: whole-text embeddings track
topic and register at least as strongly as behaviour, so a cluster here can easily be "all
the medical scenarios" rather than a move the model makes. `interpret.py` is prompted to
reject topic labels, and every property carries a detector that must be applied to a
single record — but read a trace_clusters label with more suspicion than a
feature_discovery one, and check the detector before spending a training run on it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import embed as embed_mod
from src.properties.shared import grouping as grouping_mod
from src.properties.shared import interpret as interpret_mod
from src.properties.sources.base import Record
from src.utils import git_sha, timestamp

SOURCE = "trace_clusters"
# A cluster this small is a handful of near-duplicates, not a property of the corpus.
MIN_GROUP_RECORDS = 5


def _excerpt(text: str, limit: int = 1200) -> str:
    """Trim a record's text for use as evidence in an interpretation prompt.

    Args:
        text: The channel's full text.
        limit: Characters to keep.

    Returns:
        The excerpt, ellipsised when trimmed.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


def _arm_shares(records: list[Record], member_idx: np.ndarray,
                group_by: str) -> dict | None:
    """Per-arm share of a group's members, and each arm's own base rate.

    Args:
        records: Every record in the corpus, in embedding order.
        member_idx: Indices of this group's members.
        group_by: Metadata key naming the arm (e.g. "source_label", "pipeline").

    Returns:
        arm -> {"n_in_group", "n_in_corpus", "share_of_arm"}, or None when no record
        carries the key.
    """
    arms = [str(r.metadata.get(group_by)) for r in records]
    if all(a == "None" for a in arms):
        return None
    totals: dict[str, int] = {}
    for arm in arms:
        totals[arm] = totals.get(arm, 0) + 1
    hits: dict[str, int] = {}
    for i in member_idx:
        hits[arms[i]] = hits.get(arms[i], 0) + 1
    return {arm: {"n_in_group": hits.get(arm, 0), "n_in_corpus": total,
                  "share_of_arm": round(hits.get(arm, 0) / total, 4)}
            for arm, total in sorted(totals.items())}


def produce(records: list[Record], cfg, out_dir: str | Path,
            target=None) -> list[Property]:
    """Cluster records and emit one Property per cluster.

    Args:
        records: The corpus.
        cfg: The producer's config block. Keys:
            channel (default "reasoning"), embed {backend, model, batch, workers},
            grouping {reduce, cluster, k, min_cluster_size, ...},
            interpret {model, n_shown, workers},
            group_by (metadata key separating arms, optional),
            measure_with_detector (bool, default False),
            detector {model, workers, sample} — how many records to re-measure on.
        out_dir: Run directory for this producer's artifacts.
        target: Unused; trace_clusters describes a corpus rather than explaining an
            outcome. Accepted so every producer has one signature.

    Returns:
        Property rows, most prevalent first.

    Raises:
        ValueError: If no record carries text in the chosen channel.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(cfg), resolve=True))
    run = Path(out_dir)
    run.mkdir(parents=True, exist_ok=True)
    channel = str(cfg.get("channel", "reasoning"))

    kept = [r for r in records if r.channel(channel).strip()]
    if not kept:
        raise ValueError(
            f"no record carries text in the {channel!r} channel. A corpus of "
            "non-thinking rows has nothing to cluster in `reasoning`; point the config "
            "at `response`, or at a corpus with traces.")
    if len(kept) < len(records):
        print(f">>> {len(records) - len(kept)} of {len(records)} records have an empty "
              f"{channel} channel and are excluded from the denominator")

    texts = [_excerpt(r.channel(channel), int(cfg.get("excerpt_chars", 4000)))
             for r in kept]
    vectors, embed_meta = embed_mod.embed(texts, **block(cfg, "embed"))
    embed_mod.save(run / "embeddings.npy", vectors, embed_meta)

    params = grouping_mod.GroupingParams(**block(cfg, "grouping"))
    result = grouping_mod.group(vectors, params)
    np.save(run / "labels.npy", result.labels)
    if result.coords is not None:
        np.save(run / "coords.npy", result.coords)
    np.save(run / "centroids.npy", result.centroids)
    print(f">>> {result.n_groups} groups, {result.n_noise} of {len(kept)} records "
          f"unclustered ({result.meta['noise_share']:.1%})")

    groups = {g: result.members(g) for g in range(result.n_groups)}
    groups = {g: idx for g, idx in groups.items() if len(idx) >= MIN_GROUP_RECORDS}
    evidence = {g: [texts[i] for i in idx[:200]] for g, idx in groups.items()}
    interpretations = interpret_mod.interpret_many(
        evidence, channel=channel, **block(cfg, "interpret"))

    provenance = {"run_dir": str(run), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(), "embedding": embed_meta.to_dict(),
                  "grouping": result.meta, "channel": channel,
                  "n_records": len(kept), "n_records_excluded": len(records) - len(kept)}
    corpus = (kept[0].metadata.get("corpus") or {}) if kept else {}
    group_by = cfg.get("group_by")

    properties = []
    for group_id, interpretation in interpretations.items():
        member_idx = groups[group_id]
        properties.append(Property.make(
            SOURCE, run.name, f"g{group_id:03d}",
            corpus=corpus,
            prevalence=round(len(member_idx) / len(kept), 4),
            n_records=int(len(member_idx)), n_instances=int(len(member_idx)),
            support={"group": int(group_id), "n_members": int(len(member_idx)),
                     "arms": _arm_shares(kept, member_idx, str(group_by))
                     if group_by else None,
                     "prevalence_kind": "cluster_membership"},
            evidence={"example_records": [kept[i].record_id for i in member_idx[:10]],
                      "example_excerpts": [_excerpt(texts[i], 300)
                                           for i in member_idx[:3]]},
            provenance=provenance,
            **interpretation.to_dict()))

    if bool(cfg.get("measure_with_detector", False)):
        properties = _remeasure(properties, kept, cfg, corpus)

    properties.sort(key=lambda p: -(p.prevalence or 0))
    (run / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    return properties


def _remeasure(properties: list[Property], records: list[Record], cfg,
               corpus: dict) -> list[Property]:
    """Replace cluster-membership prevalence with detector-measured prevalence.

    Membership says "this record's embedding landed here"; the detector says "this record
    does this thing". They differ at cluster edges, and only the second is a number the
    other producers can also produce, so it is the one a merged list should compare on.

    Args:
        properties: The rows to re-measure.
        records: The corpus.
        cfg: The producer config; reads `detector.{model, workers, sample}`.
        corpus: The corpus stamp.

    Returns:
        The rows, each carrying the measured prevalence and the membership number kept in
        `support` so the disagreement stays visible.
    """
    import random

    detector_cfg = block(cfg, "detector")
    sample_n = int(detector_cfg.pop("sample", 200))
    sample = (records if len(records) <= sample_n
              else random.Random(0).sample(records, sample_n))
    print(f">>> re-measuring {len(properties)} properties with their detectors over "
          f"{len(sample)} of {len(records)} records")
    out = []
    for prop in properties:
        verdicts = interpret_mod.detect(sample, prop.label, prop.detector,
                                        channel=prop.channel, **detector_cfg)
        measured = interpret_mod.prevalence(verdicts)
        remeasured = prop.with_prevalence(measured, corpus)
        out.append(dataclasses.replace(remeasured, support={
            **remeasured.support,
            "cluster_membership_prevalence": prop.prevalence,
            "prevalence_kind": "detector_measured",
            "detector_sample_n": len(sample)}))
    return out
