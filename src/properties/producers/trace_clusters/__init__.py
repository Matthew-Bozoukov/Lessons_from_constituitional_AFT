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

## Two corpora, two questions

Point this at a TRAINING MIXTURE and it says what the data contains. Point it at ROLLOUTS
from several trained models and it says what those models actually do — and because
rollouts are judged, every group it finds comes back with a number attached.

That difference matters more than it sounds. A corpus-side cluster list is unranked: every
group is "here is a thing the data does", nothing says which is worth a training run, and
choosing an ablation target is guesswork. Rollout-side, `shared/outcomes.py` crosses each
group against the violation flag WITHIN each arm, and the list arrives ordered. The two
directions the corpus cannot show — mass spent on properties the model never picked up,
and properties the model exhibits that were never in the data — are exactly what the two
config blocks below are for.

    group_by: arm            per-group prevalence split by which model produced the trace
    outcomes: {...}          per-group violation rate, WITHIN arm, BH-corrected
    compare_to: {run_dir}    every trace's cosine to the nearest TRAINING-corpus centroid

## Refit versus assign, and why this runs both

Refitting on rollouts finds groups on the rollouts' own terms, so a behaviour the training
corpus does not contain can surface as its own group. Assigning to the training run's
centroids instead keeps the numbers comparable with corpus prevalence — but nearest-centroid
NEVER ABSTAINS, so a property with no home in the corpus is silently absorbed into whatever
is closest and disappears.

So `compare_to:` does not replace the refit; it annotates it. The same pooled vectors are
scored against the training centroids, and each refit group carries the distance profile of
its members. A group whose members ALL sit below `min_cosine` from every training centroid
is the elicited-but-not-taught candidate — visible only because both views ran over one
embedding pass.

Prevalence here is MEMBERSHIP: the share of records whose embedding landed in the group.
That is a different quantity from what the detector would say if run over the corpus, and
the two disagree at the edges of a cluster — so the config can ask for the detector to be
run (`measure_with_detector: true`), which replaces the membership number with the measured
one and records both. Cross-producer comparisons should use the measured one; it is the
only number all four producers can produce the same way.

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
import random
from pathlib import Path

import numpy as np

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import embed as embed_mod
from src.properties.shared import grouping as grouping_mod
from src.properties.shared import interpret as interpret_mod
from src.properties.shared import outcomes as outcomes_mod
from src.properties.sources.base import Record
from src.utils import git_sha, timestamp

SOURCE = "trace_clusters"
# A cluster this small is a handful of near-duplicates, not a property of the corpus.
MIN_GROUP_RECORDS = 5
# Cosine to the nearest training centroid below which a trace has no home cluster in the
# training corpus. Carried over from the 2026-08-15 assignment run, where it separated
# reasoning the corpus has a group for from reasoning it does not.
DEFAULT_MIN_COSINE = 0.60


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
        group_by: Metadata key naming the arm (e.g. "arm", "source_label", "pipeline").

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


# --- the training-corpus view -----------------------------------------------------------

def prior_centroids(run_dir: str | Path, embed_meta: embed_mod.EmbedMeta
                    ) -> tuple[np.ndarray, list[str], dict]:
    """Full-dimensional centroids of a previous trace_clusters run, plus its labels.

    Centroids are recomputed from that run's `embeddings.npy` rather than read from its
    `centroids.npy`, and the difference is load-bearing. A run that clustered under
    `reduce: umap` wrote centroids in UMAP space, and no new point can be placed in that
    space without the fitted reducer, which is not a saved artifact. Averaging the members'
    ORIGINAL embeddings gives a centroid in the space both runs share — the embedding
    model's — which is the only space in which a cosine measured here means what a cosine
    measured there meant.

    Args:
        run_dir: A previous trace_clusters run directory (holds embeddings.npy,
            labels.npy, and properties_preview.json).
        embed_meta: This run's embedding metadata, checked against that run's.

    Returns:
        ((g x d) L2-normalised centroids, one label per centroid, that run's meta).

    Raises:
        FileNotFoundError: If the run directory is missing an artifact this needs.
        ValueError: If the two runs used different embedding models — their cosines are
            different quantities and comparing them is the mistake this check exists for.
    """
    run = Path(run_dir)
    missing = [name for name in ("embeddings.npy", "labels.npy")
               if not (run / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run} is missing {missing}: `compare_to` needs a previous trace_clusters "
            "run directory (the one whose properties this run is being compared against), "
            "not a properties.jsonl")

    meta_path = run / "embeddings_meta.json"
    prior_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if prior_meta.get("model") and prior_meta["model"] != embed_meta.model:
        raise ValueError(
            f"{run} was embedded with {prior_meta['model']!r} and this run with "
            f"{embed_meta.model!r}. Cosines from two embedding spaces are not comparable; "
            "re-embed one side with the other's model before comparing them.")

    vectors = embed_mod.normalise(np.load(run / "embeddings.npy").astype(np.float32))
    labels = np.load(run / "labels.npy")
    groups = sorted(int(g) for g in set(labels.tolist()) if g >= 0)
    centroids = embed_mod.normalise(
        np.stack([vectors[labels == g].mean(axis=0) for g in groups]).astype(np.float32))

    names = {}
    preview = run / "properties_preview.json"
    if preview.exists():
        for row in json.loads(preview.read_text()):
            group = (row.get("support") or {}).get("group")
            if group is not None:
                names[int(group)] = row.get("label", "")
    return centroids, [names.get(g, f"g{g:03d}") for g in groups], prior_meta


def _novelty(vectors: np.ndarray, centroids: np.ndarray, labels: list[str],
             min_cosine: float) -> dict:
    """Score every record against the training centroids and find the homeless ones.

    Args:
        vectors: (n x d) this run's L2-normalised embeddings.
        centroids: (g x d) L2-normalised training centroids.
        labels: One label per centroid.
        min_cosine: Below this, a record has no home cluster in the training corpus.

    Returns:
        {"best_cosine" (n,), "nearest" (n,), "nearest_label", "unhoused" (n,) bool,
         "summary"}.
    """
    similarity = np.asarray(vectors, np.float32) @ centroids.T
    nearest = similarity.argmax(axis=1)
    best = similarity.max(axis=1)
    unhoused = best < min_cosine
    return {
        "best_cosine": best, "nearest": nearest, "unhoused": unhoused,
        "nearest_label": [labels[i] for i in nearest],
        "summary": {"n_training_groups": int(centroids.shape[0]),
                    "min_cosine": min_cosine,
                    "median_best_cosine": round(float(np.median(best)), 4),
                    "n_unhoused": int(unhoused.sum()),
                    "share_unhoused": round(float(unhoused.mean()), 4)},
    }


def _group_novelty(novelty: dict, member_idx: np.ndarray) -> dict:
    """One group's distance profile against the training corpus.

    A group whose members ALL sit below the cosine floor is behaviour the model produces
    and the training corpus has no group for. That is the finding an assign-only analysis
    structurally cannot make: nearest-centroid never abstains, so those records would have
    been absorbed into whichever training group happened to be closest.

    Args:
        novelty: The output of `_novelty`.
        member_idx: This group's member indices.

    Returns:
        The profile, including the modal nearest training group.
    """
    cosines = novelty["best_cosine"][member_idx]
    unhoused = novelty["unhoused"][member_idx]
    nearest = [novelty["nearest_label"][i] for i in member_idx]
    modal = max(set(nearest), key=nearest.count) if nearest else None
    return {"median_cosine_to_training": round(float(np.median(cosines)), 4),
            "share_unhoused": round(float(unhoused.mean()), 4),
            "nearest_training_group": modal,
            # Every member below the floor: nothing in the training corpus resembles this.
            "elicited_not_taught": bool(unhoused.all())}


# --- the producer -----------------------------------------------------------------------

def produce(records: list[Record], cfg, out_dir: str | Path,
            target=None) -> list[Property]:
    """Cluster records and emit one Property per cluster.

    Args:
        records: The corpus.
        cfg: The producer's config block. Keys:
            channel (default "reasoning"), embed {backend, model, batch, workers},
            grouping {reduce, cluster, k, min_cluster_size, ...},
            baseline_grouping {...} — a second grouping of the SAME vectors, reported as
                an agreement check rather than exported (see `_gate_grouping`),
            interpret {model, n_shown, workers},
            group_by (metadata key separating arms, optional),
            outcomes {field, fdr, min_arm_records} — cross groups with judged outcomes,
            compare_to {run_dir, min_cosine} — score against a previous run's centroids,
            measure_with_detector (bool, default False),
            detector {model, workers, sample} — how many records to re-measure on.
        out_dir: Run directory for this producer's artifacts.
        target: Unused; trace_clusters describes a corpus rather than explaining an
            outcome. Accepted so every producer has one signature.

    Returns:
        Property rows. Ordered by ablation priority (most protective within-arm lift
        first) when outcomes were crossed, and by prevalence otherwise.

    Raises:
        ValueError: If no record carries text in the chosen channel, or if `outcomes:` is
            configured over records that carry none.
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
    if len(vectors) != len(texts):
        # Row i of the matrix IS record i throughout: cluster members are indices into
        # both. A backend that dropped or reordered a row would silently attach every
        # label, arm and outcome to the wrong record.
        raise ValueError(f"embedding returned {len(vectors)} vectors for {len(texts)} "
                         "records; rows must correspond 1:1 and in order")
    embed_mod.save(run / "embeddings.npy", vectors, embed_meta)

    params = grouping_mod.GroupingParams(**block(cfg, "grouping"))
    result = grouping_mod.group(vectors, params)
    np.save(run / "labels.npy", result.labels)
    if result.coords is not None:
        np.save(run / "coords.npy", result.coords)
    np.save(run / "centroids.npy", result.centroids)
    print(f">>> {result.n_groups} groups, {result.n_noise} of {len(kept)} records "
          f"unclustered ({result.meta['noise_share']:.1%})")
    _gate_grouping(vectors, result, cfg, run)

    floor = int(cfg.get("min_group_records", MIN_GROUP_RECORDS))
    groups = {g: result.members(g) for g in range(result.n_groups)}
    dropped = {g: len(idx) for g, idx in groups.items() if len(idx) < floor}
    groups = {g: idx for g, idx in groups.items() if len(idx) >= floor}
    if dropped:
        # CLAUDE.md: no silent caps. A run that quietly drops every group reads as "found
        # nothing" when what happened is that the floor was set above the group sizes.
        print(f">>> {len(dropped)} of {result.n_groups} groups below "
              f"min_group_records={floor} and not exported "
              f"({sum(dropped.values())} records): sizes {sorted(dropped.values())}")
    if not groups:
        raise ValueError(
            f"every one of {result.n_groups} groups is smaller than "
            f"min_group_records={floor}, so this run would export nothing. Lower the "
            "floor, lower the clustering resolution, or point it at more records.")
    evidence = {g: [texts[i] for i in idx[:200]] for g, idx in groups.items()}
    interpretations = interpret_mod.interpret_many(
        evidence, channel=channel, **block(cfg, "interpret"))

    novelty = _run_novelty(vectors, cfg, run, embed_meta, kept)
    provenance = {"run_dir": str(run), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(), "embedding": embed_meta.to_dict(),
                  "grouping": result.meta, "channel": channel,
                  "n_records": len(kept), "n_records_excluded": len(records) - len(kept)}
    if novelty:
        provenance["compare_to"] = novelty["summary"]
    corpus = (kept[0].metadata.get("corpus") or {}) if kept else {}
    group_by = cfg.get("group_by")

    properties = []
    for group_id, interpretation in interpretations.items():
        member_idx = groups[group_id]
        support = {"group": int(group_id), "n_members": int(len(member_idx)),
                   "arms": _arm_shares(kept, member_idx, str(group_by))
                   if group_by else None,
                   "prevalence_kind": "cluster_membership"}
        if novelty:
            support["novelty"] = _group_novelty(novelty, member_idx)
        properties.append(Property.make(
            SOURCE, run.name, f"g{group_id:03d}",
            corpus=corpus,
            prevalence=round(len(member_idx) / len(kept), 4),
            n_records=int(len(member_idx)), n_instances=int(len(member_idx)),
            support=support,
            evidence={"example_records": [kept[i].record_id for i in member_idx[:10]],
                      "example_excerpts": [_excerpt(texts[i], 300)
                                           for i in member_idx[:3]]},
            provenance=provenance,
            **interpretation.to_dict()))

    if bool(cfg.get("measure_with_detector", False)):
        properties = _remeasure(properties, kept, cfg, corpus, str(group_by or ""))

    properties = _cross_outcomes(properties, kept, groups, cfg, run)
    (run / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    (run / "report.md").write_text(
        _report(properties, kept, result, cfg, novelty, texts), encoding="utf-8")
    return properties


def _gate_grouping(vectors: np.ndarray, result, cfg, run: Path) -> None:
    """Run a second grouping of the same vectors and report how far the two agree.

    Callum's note from the 2026-08-17 whiteboard, and the reason this is not optional
    reading: "with the smaller UMAP it's really important to validate whether it's actually
    doing anything — compare the clusters you get without using UMAP. If they're very
    similar by a suitably chosen metric, UMAP's not helping." The comparison is written to
    disk rather than gated on, because what counts as "helping" depends on the corpus; the
    point is that the number exists next to the result instead of being assumed.

    Args:
        vectors: The full-dimensional matrix.
        result: The Grouping that will be exported.
        cfg: The producer config; reads `baseline_grouping:`.
        run: The run directory.
    """
    spec = block(cfg, "baseline_grouping")
    if not spec:
        return
    baseline = grouping_mod.group(vectors, grouping_mod.GroupingParams(**spec))
    comparison = grouping_mod.compare(baseline, result, vectors=vectors)
    (run / "grouping_comparison.json").write_text(json.dumps(comparison, indent=1))
    overlap = (comparison.get("geometry") or {}).get("neighbour_overlap")
    print(f">>> grouping gate: ARI {comparison['agreement']['ari']:.3f} vs "
          f"{spec.get('reduce', 'none')}+{spec.get('cluster', 'kmeans')} "
          f"({baseline.n_groups} groups)"
          + (f", neighbour overlap {overlap:.3f}" if overlap is not None else ""))


def _run_novelty(vectors: np.ndarray, cfg, run: Path, embed_meta, records) -> dict | None:
    """Score this run's vectors against a previous run's centroids, if asked to.

    Args:
        vectors: This run's embeddings.
        cfg: The producer config; reads `compare_to: {run_dir, min_cosine}`.
        run: The run directory.
        embed_meta: This run's embedding metadata.
        records: The records, in embedding order, for the per-record dump.

    Returns:
        The `_novelty` result, or None when no comparison was configured.
    """
    spec = block(cfg, "compare_to")
    if not spec:
        return None
    min_cosine = float(spec.get("min_cosine", DEFAULT_MIN_COSINE))
    centroids, labels, prior_meta = prior_centroids(spec["run_dir"], embed_meta)
    novelty = _novelty(vectors, centroids, labels, min_cosine)
    novelty["summary"]["run_dir"] = str(spec["run_dir"])
    novelty["summary"]["prior_embedding"] = prior_meta.get("model")
    (run / "novelty.json").write_text(json.dumps({
        "summary": novelty["summary"],
        "records": [{"record_id": r.record_id,
                     "best_cosine": round(float(novelty["best_cosine"][i]), 4),
                     "nearest_training_group": novelty["nearest_label"][i],
                     "unhoused": bool(novelty["unhoused"][i])}
                    for i, r in enumerate(records)],
    }, indent=1), encoding="utf-8")
    print(f">>> vs {spec['run_dir']}: median cosine to nearest training group "
          f"{novelty['summary']['median_best_cosine']:.3f}, "
          f"{novelty['summary']['n_unhoused']} of {len(records)} records unhoused "
          f"({novelty['summary']['share_unhoused']:.1%}) at cosine < {min_cosine}")
    return novelty


def _cross_outcomes(properties: list[Property], records: list[Record],
                    groups: dict[int, np.ndarray], cfg, run: Path) -> list[Property]:
    """Attach each group's within-arm outcome rates and reorder by ablation priority.

    This is what turns a cluster list into a ranking. It is deliberately the LAST step:
    the rates are read off the same membership the labels were written from, so no property
    is named with its outcome in view — an interpreter told which clusters violate would
    write detectors for violating, which is a different property than the one clustered.

    Args:
        properties: The rows, each carrying `support["group"]`.
        records: The corpus, in embedding order.
        groups: group id -> member indices.
        cfg: The producer config; reads `outcomes: {field, fdr, min_arm_records}` and
            `group_by`.
        run: The run directory.

    Returns:
        The rows with `support["outcomes"]` attached, most protective first. Unchanged
        when no `outcomes:` block is configured.

    Raises:
        ValueError: If outcomes are configured but no record carries one.
    """
    spec = block(cfg, "outcomes")
    if not spec:
        return sorted(properties, key=lambda p: -(p.prevalence or 0))
    if all(r.outcome is None for r in records):
        raise ValueError(
            "`outcomes:` is configured but no record carries one. Outcomes come from a "
            "judged rollout source (odcv_rollouts, agentic_rollouts); a training mixture "
            "has none, and crossing a corpus with outcomes it does not have would report "
            "an empty table as a null result.")

    field = str(spec.get("field", "violation"))
    arm_key = str(cfg.get("group_by") or "arm")
    min_arm = int(spec.get("min_arm_records", outcomes_mod.MIN_ARM_RECORDS))
    crosstabs = {}
    for prop in properties:
        member_idx = groups[prop.support["group"]]
        member_ids = {records[i].record_id for i in member_idx}
        crosstabs[prop.property_id] = outcomes_mod.by_arm(
            records, member_ids, arm_key=arm_key, outcome_key=field)

    ranking = outcomes_mod.rank(crosstabs, fdr=float(spec.get("fdr", 0.10)),
                                min_arm_records=min_arm)
    order = {row["group"]: i for i, row in enumerate(ranking)}
    by_id = {row["group"]: row for row in ranking}

    out = []
    for prop in properties:
        row = by_id[prop.property_id]
        out.append(dataclasses.replace(prop, support={
            **prop.support,
            "outcomes": {"field": field, "arm_key": arm_key,
                         "within_arm_lift": row["lift"], "p": row["p"], "q": row["q"],
                         "significant": row["significant"], "n_arms": row["n_arms"],
                         "n_arms_dropped": row["n_arms_dropped"],
                         "pooled_lift_confounded": row["pooled_lift"],
                         "by_arm": crosstabs[prop.property_id]["arms"],
                         "n_unjudged": crosstabs[prop.property_id]["n_unjudged"]}}))
    out.sort(key=lambda p: order[p.property_id])

    (run / "ranking.json").write_text(json.dumps(ranking, indent=1), encoding="utf-8")
    n_sig = sum(1 for row in ranking if row["significant"])
    n_measurable = sum(1 for row in ranking if row["lift"] is not None)
    print(f">>> crossed {len(ranking)} groups with `{field}` within {arm_key}: "
          f"{n_measurable} measurable, {n_sig} survive BH at q<={spec.get('fdr', 0.10)}. "
          "This is a RANKING OF ABLATION CANDIDATES, not a causal result.")
    if not n_measurable:
        # Every group is perfectly confounded with an arm: it has no same-arm non-members
        # to compare against, so there is no within-arm contrast to measure. Saying so is
        # the point — the pooled number would have supplied a large and entirely spurious
        # effect for each of them.
        print("!!! no group has a measurable within-arm lift: every group is confined to "
              f"one {arm_key}, so members have no same-arm non-members to compare with. "
              "The pooled column is NOT a fallback — it is the confound. Cluster at a "
              "lower resolution, or accept that these groups are arm markers.")
    return out


def _stratified(records: list[Record], n: int, arm_key: str) -> list[Record]:
    """Sample records for detector re-measurement, keeping each arm's share.

    An unstratified sample over a pooled rollout set draws from whichever arm has the most
    rollouts, and the detector-measured prevalence then describes that arm rather than the
    corpus. Stratifying costs nothing and keeps the measured number comparable with the
    membership number it replaces.

    Args:
        records: The corpus.
        n: Sample size.
        arm_key: Metadata key naming the arm; falls back to a plain sample when absent.

    Returns:
        The sample.
    """
    rng = random.Random(0)
    if len(records) <= n:
        return list(records)
    by_arm: dict[str, list[Record]] = {}
    for record in records:
        by_arm.setdefault(str(record.metadata.get(arm_key, "all")), []).append(record)
    if len(by_arm) < 2:
        return rng.sample(list(records), n)
    out: list[Record] = []
    for arm, rows in sorted(by_arm.items()):
        take = max(1, round(n * len(rows) / len(records)))
        out += rng.sample(rows, min(take, len(rows)))
    return out[:n]


def _remeasure(properties: list[Property], records: list[Record], cfg,
               corpus: dict, arm_key: str) -> list[Property]:
    """Replace cluster-membership prevalence with detector-measured prevalence.

    Membership says "this record's embedding landed here"; the detector says "this record
    does this thing". They differ at cluster edges, and only the second is a number the
    other producers can also produce, so it is the one a merged list should compare on.

    Args:
        properties: The rows to re-measure.
        records: The corpus.
        cfg: The producer config; reads `detector.{model, workers, sample}`.
        corpus: The corpus stamp.
        arm_key: Metadata key naming the arm, for stratified sampling.

    Returns:
        The rows, each carrying the measured prevalence and the membership number kept in
        `support` so the disagreement stays visible.
    """
    detector_cfg = block(cfg, "detector")
    sample_n = int(detector_cfg.pop("sample", 200))
    sample = _stratified(records, sample_n, arm_key)
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


# --- the markdown mirror ----------------------------------------------------------------

def _pct(value: float | None) -> str:
    """A signed percentage cell, or an em dash when there was nothing to measure.

    Args:
        value: The value, or None.

    Returns:
        The cell text.
    """
    return "—" if value is None else f"{value:+.1%}"


def _report(properties: list[Property], records: list[Record], result, cfg,
            novelty: dict | None, texts: list[str]) -> str:
    """A greppable mirror of the run: prevalence by arm, outcome rates, novelty.

    CLAUDE.md's rule — numbers must be readable without opening a json file — and the
    practical one: the three tables below are what a reader needs to pick an ablation
    target, and they only mean anything side by side.

    Args:
        properties: The rows, in export order.
        records: The corpus, in embedding order.
        result: The Grouping.
        cfg: The producer config.
        novelty: The `_novelty` result, or None.
        texts: The embedded excerpts, for the unhoused listing.

    Returns:
        The markdown.
    """
    arm_key = str(cfg.get("group_by") or "arm")
    arms = sorted({str(r.metadata.get(arm_key)) for r in records
                   if r.metadata.get(arm_key) is not None})
    crossed = any("outcomes" in p.support for p in properties)

    lines = [f"# trace_clusters — {len(properties)} properties over {len(records)} records",
             "",
             f"Grouping: `{result.meta['params']}`. {result.n_groups} groups, "
             f"{result.n_noise} records unclustered ({result.meta['noise_share']:.1%}).",
             ""]
    if crossed:
        lines += ["Rows are ordered by ABLATION PRIORITY: the within-arm difference in "
                  "outcome rate between records in the group and records in the same arm "
                  "outside it, most protective first. This is correlational — read it as "
                  "a shortlist, not a result.", ""]

    header = "| property | prevalence |" + "".join(f" {a} |" for a in arms)
    lines += ["## Prevalence by arm", "",
              header, "|---|--:|" + "--:|" * len(arms)]
    for prop in properties:
        shares = prop.support.get("arms") or {}
        cells = "".join(
            f" {shares.get(a, {}).get('share_of_arm', 0):.1%} |" for a in arms)
        lines.append(f"| {prop.label} | {(prop.prevalence or 0):.1%} |{cells}")

    if crossed:
        lines += ["", "## Outcome rate, within arm", "",
                  "`lift` is members minus non-members OF THE SAME ARM. `pooled` is the "
                  "same difference computed across arms and is CONFOUNDED by their "
                  "different base rates — it is printed only so the gap is visible.", "",
                  "| property | lift | pooled | q | arms | significant |",
                  "|---|--:|--:|--:|--:|:--|"]
        for prop in properties:
            o = prop.support.get("outcomes") or {}
            lift = _pct(o.get("within_arm_lift"))
            pooled = _pct(o.get("pooled_lift_confounded"))
            q = "—" if o.get("q") is None else f"{o['q']:.3f}"
            lines.append(f"| {prop.label} | {lift} | {pooled} | {q} | "
                         f"{o.get('n_arms', 0)} | "
                         f"{'yes' if o.get('significant') else ''} |")

    if novelty:
        lines += ["", "## Distance to the training corpus", "",
                  f"Against `{novelty['summary']['run_dir']}` "
                  f"({novelty['summary']['n_training_groups']} training groups). "
                  f"Median cosine {novelty['summary']['median_best_cosine']:.3f}; "
                  f"{novelty['summary']['n_unhoused']} of {len(records)} records "
                  f"({novelty['summary']['share_unhoused']:.1%}) sit below "
                  f"{novelty['summary']['min_cosine']} from every training group.", "",
                  "A group marked ELICITED is one where EVERY member is below the floor: "
                  "behaviour these models produce that the training corpus has no group "
                  "for. Nearest-centroid assignment alone cannot show this, because it "
                  "never abstains.", "",
                  "| property | median cosine | share unhoused | nearest training group |"
                  " |", "|---|--:|--:|---|:--|"]
        for prop in properties:
            n = prop.support.get("novelty") or {}
            lines.append(
                f"| {prop.label} | {n.get('median_cosine_to_training', 0):.3f} | "
                f"{n.get('share_unhoused', 0):.1%} | "
                f"{n.get('nearest_training_group') or '—'} | "
                f"{'**ELICITED**' if n.get('elicited_not_taught') else ''} |")

        worst = np.argsort(novelty["best_cosine"])[:15]
        lines += ["", "### The 15 records furthest from every training group", ""]
        for i in worst:
            lines.append(f"- `{novelty['best_cosine'][i]:.3f}` "
                         f"({records[i].record_id}) {_excerpt(texts[i], 220)}")

    return "\n".join(lines) + "\n"
