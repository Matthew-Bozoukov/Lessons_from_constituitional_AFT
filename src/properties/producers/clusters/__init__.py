# ABOUTME: clusters — the one clustering producer: embed evidence about records, group it,
# ABOUTME: name each group. `evidence: features | traces` picks WHAT gets embedded.

"""Group records into named properties, two ways, one code path.

    (extract) -> embed -> group -> interpret -> Property rows

There are two established ways to find what a corpus of reasoning does, and they differ in
exactly one place: what you turn into a vector.

    evidence: features   an autorater reads each record ALONE and writes 10-20 free-text
                         descriptions of what it does. Those DESCRIPTIONS are deduped into
                         a vocabulary, embedded and clustered. (The LessWrong method,
                         post `WAZWA6FPQvH8okouJ`.)
    evidence: traces     the record's own text is embedded and clustered directly. No
                         autorater, no extra opinion in the loop.

Everything after the vectors is identical — same grouping, same naming, same outcome
crossing, same property rows — so they live in one module rather than two, and a run
switches between them with one config key. That is the point: it makes them COMPARABLE.
Two modules would have drifted into two embedders and two notions of prevalence, and the
question "does the abstraction step buy anything?" would have been unanswerable.

## Which to use

`features` is the better property finder, and the default. Descriptions have already
thrown the subject matter away, so a group is a recurring MOVE ("weighs likelihood against
severity") rather than a recurring topic. It also gives a record MANY properties, which is
what records are actually like — one trace checks authorisation AND weighs harm AND
hedges.

`traces` is the better occupancy comparer, and it answers a question `features` cannot: do
two scenario formats sit in DIFFERENT regions of trace space at all? (The 2026-08-17 action
item.) An autorater's vocabulary would smooth away precisely the surface differences that
question is about. Its known weakness is the mirror of the above — raw text similarity
tracks topic and register at least as hard as behaviour, so a group can easily be "the
medical scenarios". The interpreter is warned about this per-mode (`EVIDENCE_FRAMING` in
shared/interpret.py) and told to say so in the caveat rather than invent a behavioural
label for a topical cluster. Read a `traces` label with more suspicion than a `features`
one.

The two also cost differently, in opposite directions: `features` pays one autorater call
per record up front and then embeds short strings; `traces` pays nothing up front and
embeds long ones.

## What prevalence means in each

    features   share of records with AT LEAST ONE feature in this group. Groups OVERLAP,
               so these do not sum to 1.
    traces     share of records whose own vector landed in this group. Each record is in
               exactly one group (or in noise), so these DO sum to 1.

`support.prevalence_kind` records which, because comparing the two numbers as if they were
the same quantity is the mistake this note exists to prevent. Either can be replaced by a
detector-measured prevalence (`measure_with_detector: true`), and that measured number is
the one to use across producers — it means the same thing everywhere.

## Two corpora, two questions

Point this at a TRAINING MIXTURE and it says what the data contains. Point it at ROLLOUTS
from several trained models and it says what those models actually do — and because
rollouts are judged, every group comes back with a number attached.

That difference is the reason the rollout side exists. A corpus-side cluster list is
unranked: every group is "here is a thing the data does", nothing says which is worth a
training run, and choosing an ablation target is guesswork. Rollout-side,
`shared/outcomes.py` crosses each group against the violation flag WITHIN each arm, and the
list arrives ordered.

    group_by: arm            per-group prevalence split by which model produced the record
    outcomes: {...}          per-group violation rate, WITHIN arm, BH-corrected
    compare_to: {run_dir}    cosine to the nearest TRAINING-corpus centroid

## Refit versus assign, and why this runs both

Refitting on rollouts finds groups on the rollouts' own terms, so a behaviour the training
corpus does not contain can surface as its own group. Assigning to the training run's
centroids instead keeps the numbers comparable with corpus prevalence — but nearest-centroid
NEVER ABSTAINS, so a property with no home in the corpus is silently absorbed into whatever
is closest and disappears.

So `compare_to:` does not replace the refit; it annotates it. The same vectors are scored
against the training centroids, and each refit group carries the distance profile of its
members. A group whose members ALL sit below `min_cosine` from every training centroid is
the elicited-but-not-taught candidate — visible only because both views ran over one
embedding pass.

Every stage is a call into `shared/`, which is the point: if adding a producer takes more
than one file, something belongs in `shared/` that is not there yet.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np

from src.properties import block
from src.properties.registry import Property
from src.properties.shared import attributes as attributes_mod
from src.properties.shared import audit as audit_mod
from src.properties.shared import embed as embed_mod
from src.properties.shared import grouping as grouping_mod
from src.properties.shared import interpret as interpret_mod
from src.properties.shared import outcomes as outcomes_mod
from src.properties.sources.base import Record
from src.utils import git_sha, timestamp

SOURCE = "clusters"
EVIDENCE_KINDS = ("features", "traces")
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


def _corpus_stamp(records: list[Record]) -> dict:
    """What every prevalence on this run is a share OF.

    `prevalence` is the one field that means the same thing across producers, and it is
    only interpretable next to the corpus it was measured on. Taking the first record's
    stamp works for a single-corpus run and is WRONG for a pooled one: it would name one
    arm's run directory on rows describing all five. So a pooled run is stamped as the set
    it actually is.

    Args:
        records: The records, in embedding order.

    Returns:
        The stamp: the single corpus's own when they all share one, else the list of them.
    """
    stamps = []
    for record in records:
        stamp = record.metadata.get("corpus") or {}
        if stamp and stamp not in stamps:
            stamps.append(stamp)
    if not stamps:
        return {}
    if len(stamps) == 1:
        return stamps[0]
    return {"pooled": stamps, "n_corpora": len(stamps)}


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


# --- what gets embedded -----------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Units:
    """The things this run embeds, and how they map back to records.

    The whole point of the module having one code path is that everything downstream of
    the vectors reads THIS and never asks which mode produced it. `features` and `traces`
    differ only in how these lists are built.

    Attributes:
        texts: One string per embedded unit, in embedding-row order.
        records: unit -> the record indices it came from. A feature string extracted from
            forty traces has forty; a trace excerpt has one.
        instances: unit -> how many times it occurred across the corpus. Repeats matter:
            a feature in 400 traces and a feature said 400 times in 50 traces are
            different findings, and only counting both distinguishes them.
        by_record: record -> its unit indices. The inverse of `records`, kept because the
            per-record joins (novelty, members.jsonl) need it and rebuilding it per group
            is quadratic.
        kind: "features" or "traces".
        meta: Counts and provenance for the run's `run_meta`/report.
    """

    texts: list[str]
    records: list[list[int]]
    instances: list[int]
    by_record: list[list[int]]
    kind: str
    meta: dict = dataclasses.field(default_factory=dict)


def _trace_units(records: list[Record], channel: str, excerpt_chars: int) -> Units:
    """One unit per record: the record's own text.

    Args:
        records: The corpus.
        channel: Which channel to embed.
        excerpt_chars: Truncation length.

    Returns:
        The units, one-to-one with records.
    """
    texts = [_excerpt(r.channel(channel), excerpt_chars) for r in records]
    index = [[i] for i in range(len(records))]
    return Units(texts=texts, records=index, instances=[1] * len(texts),
                 by_record=[[i] for i in range(len(records))], kind="traces",
                 meta={"n_units": len(texts), "excerpt_chars": excerpt_chars})


def _feature_units(records: list[Record], channel: str, cfg, run: Path) -> Units:
    """One unit per DISTINCT feature string, extracted one record at a time.

    The autorater sees a single record and nothing else — no metadata, no trait, no other
    records — which is what lets it name behaviours nobody chose in advance. Its output is
    free text rather than a schema, so identical strings recurring across records are the
    signal that a behaviour is common.

    Deduplication is not just thrift, though it is that too (embedding the same string 400
    times wastes the pod). Clustering the vocabulary rather than the occurrences means a
    stock phrase cannot pull a cluster toward itself by sheer repetition; the occurrence
    counts travel alongside instead, and both get reported.

    Extraction is the expensive half of this mode and the stage most likely to be
    interrupted, so each record's features are appended to `features.jsonl` as they land
    and a rerun labels only what the file does not already hold. A run killed at 95% keeps
    its 95%.

    Args:
        records: The corpus.
        channel: Which channel the autorater describes.
        cfg: The producer config; reads `extract: {model, n, n_min, workers, reuse}`.
        run: The run directory, where features.jsonl lives.

    Returns:
        The units, one per distinct feature string.

    Raises:
        ValueError: If extraction produced no features at all.
    """
    extract_cfg = block(cfg, "extract")
    workers = int(extract_cfg.pop("workers", 16))
    extract_cfg.pop("reuse", None)  # resuming is now unconditional; the file IS the cache
    spec = attributes_mod.AttributeSpec(style="freeform", channel=channel, **extract_cfg)

    path = run / "features.jsonl"
    rows = attributes_mod.extract_to(records, spec, path, workers=workers)
    failed = [row for row in rows if row.get("error")]
    counts: dict[str, int] = {}
    unit_records: dict[str, list[int]] = {}
    by_record: list[list[int]] = []
    for i, row in enumerate(rows):
        for feature in row["attributes"]:
            counts[feature] = counts.get(feature, 0) + 1
            holders = unit_records.setdefault(feature, [])
            if not holders or holders[-1] != i:
                holders.append(i)
    if not counts:
        raise ValueError(
            f"the autorater returned no features for any of {len(records)} records "
            f"({len(failed)} calls errored). There is nothing to cluster; check the "
            "extract model and the channel before spending an embedding pass.")

    texts = sorted(counts)
    position = {feature: u for u, feature in enumerate(texts)}
    by_record = [sorted({position[f] for f in row["attributes"]}) for row in rows]
    print(f">>> {len(rows) - len(failed)} of {len(records)} records labelled, "
          f"{sum(counts.values())} feature instances, {len(texts)} distinct")
    if failed:
        # Not fatal and not hidden: a failed record contributes to no group, so the damage
        # is reduced coverage, and a reader has to be able to see how much.
        print(f"!!! {len(failed)} records produced no features "
              f"(e.g. {failed[0].get('error', '')[:120]})")
    return Units(
        texts=texts,
        records=[unit_records[f] for f in texts],
        instances=[counts[f] for f in texts],
        by_record=by_record, kind="features",
        meta={"n_units": len(texts), "feature_instances": sum(counts.values()),
              "records_labelled": len(rows) - len(failed),
              "records_failed": len(failed), "extract": spec.to_dict()})


def build_units(records: list[Record], channel: str, cfg, run: Path) -> Units:
    """Build whichever evidence this run's `evidence:` asks for.

    Args:
        records: The corpus.
        channel: Which channel to read.
        cfg: The producer config.
        run: The run directory.

    Returns:
        The units.

    Raises:
        ValueError: On an unknown evidence kind.
    """
    kind = str(cfg.get("evidence", "features"))
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"evidence must be one of {EVIDENCE_KINDS}, got {kind!r}")
    if kind == "traces":
        return _trace_units(records, channel, int(cfg.get("excerpt_chars", 4000)))
    return _feature_units(records, channel, cfg, run)


# --- the training-corpus view -----------------------------------------------------------

def prior_centroids(run_dir: str | Path, embed_meta: embed_mod.EmbedMeta
                    ) -> tuple[np.ndarray, list[str], dict]:
    """Full-dimensional centroids of a previous `clusters` run, plus its labels.

    Centroids are recomputed from that run's `embeddings.npy` rather than read from its
    `centroids.npy`, and the difference is load-bearing. A run that clustered under
    `reduce: umap` wrote centroids in UMAP space, and no new point can be placed in that
    space without the fitted reducer, which is not a saved artifact. Averaging the members'
    ORIGINAL embeddings gives a centroid in the space both runs share — the embedding
    model's — which is the only space in which a cosine measured here means what a cosine
    measured there meant.

    Args:
        run_dir: A previous `clusters` run directory (holds embeddings.npy,
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
            f"{run} is missing {missing}: `compare_to` needs a previous `clusters` "
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
    """Group a corpus into named properties, and emit one Property per group.

    Args:
        records: The corpus — a training mixture's rows, or rollouts from trained models.
        cfg: The producer's config block. Keys:
            evidence ("features" | "traces") — WHAT gets embedded; the one switch that
                chooses between the two methods,
            channel (default "reasoning"),
            extract {model, n, n_min, workers, reuse} — the autorater, features mode only,
            excerpt_chars — truncation, traces mode only,
            embed {backend, model, batch, workers},
            grouping {reduce, cluster, k, min_cluster_size, ...},
            baseline_grouping {...} — a second grouping of the SAME vectors, reported as
                an agreement check rather than exported (see `_gate_grouping`),
            interpret {model, n_shown, workers},
            min_group_records — smallest group worth exporting,
            group_by (metadata key separating arms, optional),
            outcomes {field, fdr, min_arm_records} — cross groups with judged outcomes,
            compare_to {run_dir, min_cosine} — score against a previous run's centroids,
            audit {stability, neighbours, seeds, threshold} or false to skip — the
                redundancy / buried-behaviour / seed-stability checks,
            measure_with_detector (bool, default False),
            detector {model, workers, sample}.
        out_dir: Run directory for this producer's artifacts.
        target: Unused; this producer describes a corpus rather than explaining an
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

    units = build_units(kept, channel, cfg, run)
    print(f">>> evidence: {units.kind} — {len(units.texts)} units over {len(kept)} records")

    vectors, embed_meta = _embed(units, cfg, run)
    if len(vectors) != len(units.texts):
        # Row u of the matrix IS unit u throughout: cluster members are indices into both,
        # and units index back to records. A backend that dropped or reordered a row would
        # silently attach every label, arm and outcome to the wrong records.
        raise ValueError(f"embedding returned {len(vectors)} vectors for "
                         f"{len(units.texts)} units; rows must correspond 1:1 and in order")

    params = grouping_mod.GroupingParams(**block(cfg, "grouping"))
    result = grouping_mod.group(vectors, params)
    np.save(run / "labels.npy", result.labels)
    if result.coords is not None:
        np.save(run / "coords.npy", result.coords)
    np.save(run / "centroids.npy", result.centroids)
    print(f">>> {result.n_groups} groups, {result.n_noise} of {len(units.texts)} "
          f"{units.kind} unclustered ({result.meta['noise_share']:.1%})")
    _gate_grouping(vectors, result, cfg, run)

    groups, below_floor = _group_members(result, units, cfg)
    evidence = {g: [units.texts[u] for u in m["units"][:400]] for g, m in groups.items()}
    interpret_cfg = block(cfg, "interpret")
    # The post samples 100 features per cluster and the published runs did too, so a
    # features run keeps that number. A traces run cannot: 100 excerpts of 4000 characters
    # is a 100k-token naming prompt, and the excerpts are far more redundant than feature
    # strings are, so fewer of them carry the same information.
    interpret_cfg.setdefault("n_shown", 100 if units.kind == "features" else 30)
    interpretations = interpret_mod.interpret_many(
        evidence, channel=channel, evidence_kind=units.kind, **interpret_cfg)

    novelty = _run_novelty(vectors, cfg, run, embed_meta, units)
    provenance = {"run_dir": str(run), "git_sha": git_sha(),
                  "timestamp_utc": timestamp(), "embedding": embed_meta.to_dict(),
                  "grouping": result.meta, "channel": channel, "evidence": units.kind,
                  "units": units.meta,
                  "n_records": len(kept), "n_records_excluded": len(records) - len(kept)}
    if novelty:
        provenance["compare_to"] = novelty["summary"]
    corpus = _corpus_stamp(kept)
    group_by = cfg.get("group_by")

    properties = []
    for group_id, interpretation in interpretations.items():
        member = groups[group_id]
        idx = member["records"]
        support = {
            "group": int(group_id),
            "n_members": int(len(idx)),
            "arms": _arm_shares(kept, idx, str(group_by)) if group_by else None,
            # features: groups overlap, so these do not sum to 1. traces: they do.
            "prevalence_kind": ("feature_membership" if units.kind == "features"
                                else "record_membership"),
            "n_units": int(len(member["units"])),
            "trait_mix": _trait_mix(kept, idx),
        }
        if novelty:
            support["novelty"] = _group_novelty(novelty, member["units"])
        properties.append(Property.make(
            SOURCE, run.name, f"g{group_id:03d}",
            corpus=corpus,
            prevalence=round(len(idx) / len(kept), 4),
            n_records=int(len(idx)), n_instances=int(member["instances"]),
            support=support,
            evidence={"example_records": [kept[i].record_id for i in idx[:10]],
                      "example_units": sorted(units.texts[u]
                                              for u in member["units"])[:12],
                      "example_excerpts": [_excerpt(kept[i].channel(channel), 300)
                                           for i in idx[:3]]},
            provenance=provenance,
            **interpretation.to_dict()))

    if bool(cfg.get("measure_with_detector", False)):
        properties = _remeasure(properties, kept, cfg, corpus, str(group_by or ""))

    properties = _cross_outcomes(properties, kept, groups, cfg, run)
    _write_members(run, properties, kept, groups, units, result,
                   below_floor, novelty)
    audit = (audit_mod.write(run, vectors, result, units, properties, len(kept),
                             block(cfg, "audit"))
             if cfg.get("audit", True) is not False else None)
    _write_coverage(run, groups, kept, units, result)
    (run / "properties_preview.json").write_text(
        json.dumps([p.to_dict() for p in properties], indent=1), encoding="utf-8")
    (run / "report.md").write_text(
        _report(properties, kept, result, cfg, novelty, units)
        + ("\n" + audit_mod.report(audit) if audit else ""), encoding="utf-8")
    return properties


def _embed(units: Units, cfg, run: Path) -> tuple[np.ndarray, embed_mod.EmbedMeta]:
    """Embed the units, reusing this run directory's vectors when they still apply.

    Retuning the clustering — a different `min_cluster_size`, `reduce: none` to check
    whether UMAP is helping, another seed — is the loop you run ten times, and it does not
    change a single vector. Re-embedding tens of thousands of strings on a rented GPU each
    time round is the single most expensive way to answer a cheap question, and it is why
    the original module made clustering its own stage against an existing run directory.

    The saved vectors are reused only when the unit list is IDENTICAL, in order, and the
    embedding model matches. Anything else — one more record, a different channel, another
    embedder — and the cached matrix describes different text, so it is discarded rather
    than silently mismatched against the current units.

    Args:
        units: This run's units.
        cfg: The producer config; reads `embed:` and `reuse_embeddings` (default True).
        run: The run directory.

    Returns:
        (vectors, their EmbedMeta).
    """
    path = run / "embeddings.npy"
    meta_path = run / "embeddings_meta.json"
    units_path = run / "units.json"
    embed_cfg = block(cfg, "embed")
    wanted = str(embed_cfg.get("model") or "")

    if bool(cfg.get("reuse_embeddings", True)) and path.exists() and units_path.exists():
        cached_units = json.loads(units_path.read_text())
        cached_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        same_model = not wanted or cached_meta.get("model") == wanted
        if cached_units == units.texts and same_model:
            vectors = embed_mod.normalise(np.load(path).astype(np.float32))
            print(f">>> reusing {len(vectors)} embeddings from {path} "
                  f"({cached_meta.get('model')}) — nothing to re-embed")
            return vectors, embed_mod.EmbedMeta(
                backend=cached_meta.get("backend", "cached"),
                model=cached_meta.get("model", ""), dim=int(vectors.shape[1]),
                n=len(vectors), probe_cosines=cached_meta.get("probe_cosines", {}))
        print(f">>> {path} holds {len(cached_units)} units for a different "
              f"{'model' if not same_model else 'unit list'}; re-embedding")

    vectors, embed_meta = embed_mod.embed(units.texts, **embed_cfg)
    embed_mod.save(path, vectors, embed_meta)
    units_path.write_text(json.dumps(units.texts, ensure_ascii=False), encoding="utf-8")
    return vectors, embed_meta


def _group_members(result, units: Units, cfg) -> dict[int, dict]:
    """Resolve each cluster of UNITS into the records it covers.

    In traces mode this is the identity — one unit is one record. In features mode it is
    the step that makes a record able to hold several properties at once, which is what
    records are actually like: one trace checks authorisation AND weighs harm AND hedges.

    The size floor is applied to RECORDS, not units, in both modes. A cluster of 300
    feature strings that only forty traces ever said is a forty-trace finding.

    Args:
        result: The Grouping over units.
        units: The units.
        cfg: The producer config; reads `min_group_records`.

    Returns:
        (group id -> {"units", "records", "instances"}, the group ids the floor removed).
        The dropped ids are returned rather than discarded because `members.jsonl` has to
        tell a record excluded by the FLOOR apart from one excluded as NOISE, and only
        this function knows which was which.

    Raises:
        ValueError: If the floor removes every group.
    """
    floor = int(cfg.get("min_group_records", MIN_GROUP_RECORDS))
    groups: dict[int, dict] = {}
    dropped: dict[int, int] = {}
    for group_id in range(result.n_groups):
        unit_idx = result.members(group_id)
        covered: set[int] = set()
        for u in unit_idx:
            covered.update(units.records[u])
        if len(covered) < floor:
            dropped[group_id] = len(covered)
            continue
        groups[group_id] = {
            "units": unit_idx,
            "records": np.array(sorted(covered), dtype=np.int64),
            "instances": int(sum(units.instances[u] for u in unit_idx)),
        }
    if dropped:
        # CLAUDE.md: no silent caps. A run that quietly drops every group reads as "found
        # nothing" when what happened is that the floor was set above the group sizes.
        print(f">>> {len(dropped)} of {result.n_groups} groups cover fewer than "
              f"min_group_records={floor} records and are not exported: "
              f"sizes {sorted(dropped.values())}")
    if not groups:
        raise ValueError(
            f"every one of {result.n_groups} groups covers fewer than "
            f"min_group_records={floor} records, so this run would export nothing. Lower "
            "the floor, lower the clustering resolution, or point it at more records.")
    return groups, set(dropped)


def _trait_mix(records: list[Record], member_idx: np.ndarray) -> dict | None:
    """Which traits a group's records came from, most common first.

    Carried over from the feature-discovery runs, where it is the fastest way to see that
    a "property" is really one trait's house style rather than a behaviour.

    Args:
        records: The corpus, in record order.
        member_idx: This group's record indices.

    Returns:
        trait_id -> count, or None when no record carries one.
    """
    counts: dict[str, int] = {}
    for i in member_idx:
        trait = records[i].metadata.get("trait_id")
        if trait is not None:
            counts[str(trait)] = counts.get(str(trait), 0) + 1
    if not counts:
        return None
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _write_coverage(run: Path, groups: dict[int, dict], records: list[Record],
                    units: Units, result) -> Path:
    """Write what these properties do NOT account for.

    A list of properties reads as a description of the corpus, and it is only ever a
    description of the part that clustered. A merger has to know that share before
    trusting the list, so it is written beside the rows rather than left to be derived
    from three other files.

    Args:
        run: The run directory.
        groups: group id -> {"units", "records", "instances"}.
        records: The corpus.
        units: The units.
        result: The Grouping.

    Returns:
        The path written.
    """
    covered: set[int] = set()
    for member in groups.values():
        covered.update(int(i) for i in member["records"])
    noise_units = int(result.n_noise)
    noise_instances = sum(units.instances[u]
                          for u in np.flatnonzero(result.labels < 0).tolist())
    total_instances = sum(units.instances)
    payload = {
        "evidence": units.kind,
        "properties": len(groups),
        "records": len(records),
        "units": len(units.texts),
        "unclustered_units": noise_units,
        "unclustered_unit_share": round(noise_units / len(units.texts), 4)
        if units.texts else None,
        "unclustered_instances": noise_instances,
        "unclustered_instance_share": round(noise_instances / total_instances, 4)
        if total_instances else None,
        "records_with_no_property": len(records) - len(covered),
        "git_sha": git_sha(), "timestamp_utc": timestamp(),
        **units.meta,
    }
    path = run / "coverage.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def _write_members(run: Path, properties: list[Property], records: list[Record],
                   groups: dict[int, dict], units: Units, result,
                   below_floor: set[int], novelty: dict | None) -> Path:
    """Write the record -> property join table.

    A property row carries ten example ids, which is enough to sanity-check a label and
    nowhere near enough to work with. Full membership is otherwise only in `labels.npy`,
    which is POSITIONAL and — in features mode — indexes feature strings rather than
    records. Neither answers "show me the traces in this cluster" without a reconstruction
    nobody should have to do.

    One line per MEMBERSHIP EDGE, not per record, because in features mode a record
    belongs to several properties and one-line-per-record could not represent that. In
    traces mode each record has exactly one edge, so the file reads the same way.

    A record in no group at all still gets a line, with a null property and the reason.
    It is part of the denominator every prevalence is a share of, so dropping it here
    would make this file disagree with the numbers beside it.

    Args:
        run: The run directory.
        properties: The exported rows.
        records: The corpus, in record order.
        groups: group id -> {"units", "records", "instances"}.
        units: The units, for the per-record novelty roll-up.
        result: The Grouping, for the unit labels behind an exclusion reason.
        below_floor: Group ids the size floor removed.
        novelty: The `_novelty` result (over UNITS), or None.

    Returns:
        The path written.
    """
    cosines = _record_cosines(novelty, units, len(records)) if novelty else None
    lines, placed = [], set()
    for prop in properties:
        group = prop.support["group"]
        for i in groups[group]["records"].tolist():
            placed.add(i)
            row = {"record_id": records[i].record_id,
                   "property_id": prop.property_id, "label": prop.label,
                   "group": int(group), "excluded": None,
                   **_record_columns(records[i])}
            if cosines:
                row |= cosines[i]
            lines.append(json.dumps(row, ensure_ascii=False))

    for i, record in enumerate(records):
        if i in placed:
            continue
        row = {"record_id": record.record_id, "property_id": None, "label": None,
               "group": None,
               # Three different reasons, and they mean different things: the autorater
               # said nothing about this record, everything it said was low-density noise,
               # or what it said only ever landed in groups the floor removed. A reader
               # chasing coverage needs to tell them apart.
               "excluded": _exclusion_reason(units.by_record[i], result, below_floor),
               **_record_columns(record)}
        if cosines:
            row |= cosines[i]
        lines.append(json.dumps(row, ensure_ascii=False))

    path = run / "members.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> {len(lines)} membership rows over {len(records)} records -> {path.name} "
          f"({len(records) - len(placed)} records carry no property)")
    return path


def _exclusion_reason(unit_idx: list[int], result, below_floor: set[int]) -> str:
    """Why one record ended up with no property.

    Args:
        unit_idx: The record's unit indices.
        result: The Grouping.
        below_floor: Group ids the size floor removed.

    Returns:
        "no_units", "unclustered", or "below_floor".
    """
    if not unit_idx:
        return "no_units"
    reached = {int(result.labels[u]) for u in unit_idx}
    if reached & below_floor:
        return "below_floor"
    return "unclustered"


def _record_columns(record: Record) -> dict:
    """The record-identifying columns every members.jsonl row carries.

    Args:
        record: The record.

    Returns:
        The columns, including the path to the rollout so a reader can open it.
    """
    return {"arm": record.metadata.get("arm"),
            "trait_id": record.metadata.get("trait_id"),
            "outcome": record.outcome,
            "rollout_path": record.metadata.get("rollout_path"),
            "reasoning_chars": len(record.reasoning)}


def _record_cosines(novelty: dict, units: Units, n_records: int) -> list[dict]:
    """Roll unit-level distances up to one number per record.

    Novelty is measured on whatever was embedded. In traces mode that is already
    per-record; in features mode a record has many features, and the honest summary is its
    CLOSEST one — a record is only "unhoused" when nothing it said resembles anything the
    training corpus contains.

    Args:
        novelty: The `_novelty` result over units.
        units: The units.
        n_records: How many records.

    Returns:
        One dict of novelty columns per record.
    """
    out = []
    for i in range(n_records):
        mine = units.by_record[i]
        if not mine:
            out.append({"cosine_to_training": None, "nearest_training_group": None,
                        "unhoused": None})
            continue
        best = max(mine, key=lambda u: float(novelty["best_cosine"][u]))
        out.append({"cosine_to_training": round(float(novelty["best_cosine"][best]), 4),
                    "nearest_training_group": novelty["nearest_label"][best],
                    "unhoused": bool(novelty["unhoused"][best])})
    return out


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


def _run_novelty(vectors: np.ndarray, cfg, run: Path, embed_meta,
                 units: Units) -> dict | None:
    """Score this run's vectors against a previous run's centroids, if asked to.

    Measured on the EMBEDDED UNITS, whatever those are — a cosine is only meaningful
    between things in the same space, and in features mode the prior run's centroids are
    centroids of feature strings.

    Args:
        vectors: This run's embeddings.
        cfg: The producer config; reads `compare_to: {run_dir, min_cosine}`.
        run: The run directory.
        embed_meta: This run's embedding metadata.
        units: The units, in embedding order, for the per-unit dump.

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
    novelty["summary"]["measured_on"] = units.kind
    (run / "novelty.json").write_text(json.dumps({
        "summary": novelty["summary"],
        "units": [{"unit": text,
                   "best_cosine": round(float(novelty["best_cosine"][u]), 4),
                   "nearest_training_group": novelty["nearest_label"][u],
                   "unhoused": bool(novelty["unhoused"][u])}
                  for u, text in enumerate(units.texts)],
    }, indent=1), encoding="utf-8")
    print(f">>> vs {spec['run_dir']}: median cosine to nearest training group "
          f"{novelty['summary']['median_best_cosine']:.3f}, "
          f"{novelty['summary']['n_unhoused']} of {len(units.texts)} {units.kind} "
          f"unhoused ({novelty['summary']['share_unhoused']:.1%}) at cosine < {min_cosine}")
    return novelty


def _cross_outcomes(properties: list[Property], records: list[Record],
                    groups: dict[int, dict], cfg, run: Path) -> list[Property]:
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
        member_ids = {records[i].record_id
                      for i in groups[prop.support["group"]]["records"]}
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
                         "n_arms_underpowered": row["n_arms_underpowered"],
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
        # No group has same-arm non-members, so there is no within-arm contrast to
        # measure. Two ways to get here and they call for opposite fixes, so the message
        # names both rather than guessing which one happened. Either way the pooled column
        # is not a fallback: it is exactly the confound the within-arm split removes.
        print(f"!!! no group has a measurable within-arm lift: no group has "
              f"same-{arm_key} non-members to compare against. Either a group covers "
              f"EVERY record in its {arm_key} (cluster at a finer resolution), or each "
              f"group sits entirely inside one {arm_key} (they are {arm_key} markers, not "
              "behaviours). The pooled column is NOT a fallback — it is the confound.")
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
            novelty: dict | None, units: Units) -> str:
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
        units: The units, for the unhoused listing.

    Returns:
        The markdown.
    """
    arm_key = str(cfg.get("group_by") or "arm")
    arms = sorted({str(r.metadata.get(arm_key)) for r in records
                   if r.metadata.get(arm_key) is not None})
    crossed = any("outcomes" in p.support for p in properties)

    lines = [f"# clusters — {len(properties)} properties over {len(records)} records", "",
             f"Evidence: **{units.kind}** — {len(units.texts)} embedded units. "
             f"Grouping: `{result.meta['params']}`. {result.n_groups} groups, "
             f"{result.n_noise} units unclustered ({result.meta['noise_share']:.1%}).", "",
             "Prevalence is the share of records with at least one unit in the group; "
             + ("groups OVERLAP, so these do not sum to 100%."
                if units.kind == "features"
                else "each record is in one group, so these sum to 100%."), ""]
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
        lines += ["", f"### The 15 {units.kind} furthest from every training group", ""]
        for u in worst:
            lines.append(f"- `{novelty['best_cosine'][u]:.3f}` "
                         f"{_excerpt(units.texts[u], 220)}")

    return "\n".join(lines) + "\n"
