# ABOUTME: Disjoint row pools inside one corpus: every row labelled (A = the rows an earlier mixture took,
# ABOUTME: B, C, ... = stratified shares of the rest), published beside the corpus, selected at mix time.
"""Splitting a corpus into pools that mixtures draw from without sharing a row.

Three da-15 mixtures that are "3 independent 15%s" need three sets of difficult-advice rows
that share nothing: the first was built before anyone thought of the other two, so its 650
rows are pool A, read off the published mixture itself; what is left is dealt into B and C,
stratified so each pool holds every trait x AI label x run in the same proportions. The labels
are published with the corpus (`splits/<name>.json`, at the revision a mixture pins), and a
mixture source names one pool (`split: {file: ..., keep: B}`) — `build_mixture` then fills its
share from those rows exactly as it fills from a whole corpus.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def row_key(messages: list[dict]) -> str:
    """A row's identity across a corpus and the mixtures built from it: its conversation.

    A mixture keeps `messages` and `source` and drops a synth row's metadata, so the
    conversation (role, content, reasoning) is the only thing both sides still hold.
    """
    turns = [{k: m.get(k) or "" for k in ("role", "content", "reasoning_content")}
             for m in messages]
    return hashlib.sha256(json.dumps(turns, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def rows_taken_by(corpus: list[dict], mixture: list[dict], source: str) -> set[str]:
    """The scenario ids of the corpus rows a published mixture took for `source`.

    Raises:
        ValueError: a mixture row of that source matches no corpus row, or more than one,
            or the mixture has no rows of that source.
    """
    by_key: dict[str, list[str]] = {}
    for r in corpus:
        by_key.setdefault(row_key(r["messages"]), []).append(r["metadata"]["scenario_id"])
    taken = [r for r in mixture if r.get("source") == source]
    if not taken:
        raise ValueError(f"the mixture holds no `{source}` rows")
    out: set[str] = set()
    for r in taken:
        hits = by_key.get(row_key(r["messages"]), [])
        if len(hits) != 1:
            raise ValueError(f"a `{source}` mixture row matches {len(hits)} corpus rows; "
                             "the mixture was not built from this corpus")
        out.add(hits[0])
    if len(out) != len(taken):
        raise ValueError("two mixture rows are the same corpus row")
    return out


def stratum(row: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
    """A corpus row's stratum: its metadata fields, `run` being its id prefix."""
    from src.data.synth.ours.extend import run_prefix

    md = row["metadata"]
    return tuple(run_prefix(md["scenario_id"], md["trait_id"]) if f == "run" else str(md.get(f, ""))
                 for f in fields)


def assign_pools(corpus: list[dict], taken: set[str], pools: tuple[str, ...] = ("B", "C"), *,
                 taken_label: str = "A", strata: tuple[str, ...] = ("trait_id", "ai", "run"),
                 seed: int = 0) -> dict[str, str]:
    """Label every corpus row: `taken_label` for `taken`, one of `pools` for the rest.

    The rest is dealt round-robin over the pools, stratum by stratum in sorted order
    (each stratum's rows in a seeded shuffle), with ONE cursor carried across strata. Any
    contiguous stretch of a round-robin gives every pool the same count to within one, so
    every stratum, every trait and the whole corpus are each split evenly to within one row.

    Returns:
        `scenario_id -> label` for every row.
    """
    assert pools and taken_label not in pools, (pools, taken_label)
    ids = {r["metadata"]["scenario_id"] for r in corpus}
    missing = sorted(taken - ids)
    assert not missing, f"{len(missing)} taken ids are not in the corpus ({missing[0]} ...)"
    labels = {sid: taken_label for sid in taken}
    groups: dict[tuple, list[str]] = {}
    for r in corpus:
        sid = r["metadata"]["scenario_id"]
        if sid not in taken:
            groups.setdefault(stratum(r, strata), []).append(sid)
    cursor = 0
    for key in sorted(groups):
        members = sorted(groups[key])
        random.Random(f"{seed}:{':'.join(key)}").shuffle(members)
        for sid in members:
            labels[sid] = pools[cursor % len(pools)]
            cursor += 1
    return labels


def split_counts(corpus: list[dict], labels: dict[str, str],
                 strata: tuple[str, ...] = ("trait_id", "ai", "run")) -> dict:
    """Rows per label, per label x trait, and per label x stratum — the split's own record."""
    by_label = Counter(labels[r["metadata"]["scenario_id"]] for r in corpus)
    by_trait: dict[str, Counter] = {}
    by_stratum: dict[str, Counter] = {}
    for r in corpus:
        lab = labels[r["metadata"]["scenario_id"]]
        by_trait.setdefault(lab, Counter())[r["metadata"]["trait_id"]] += 1
        by_stratum.setdefault(lab, Counter())["|".join(stratum(r, strata))] += 1
    return {"rows": dict(sorted(by_label.items())),
            "by_trait": {k: dict(sorted(v.items())) for k, v in sorted(by_trait.items())},
            "by_stratum": {k: dict(sorted(v.items())) for k, v in sorted(by_stratum.items())}}


def write_split(path: Path, labels: dict[str, str], meta: dict) -> Path:
    """`{"meta": ..., "labels": {scenario_id: label}}`, the file a mixture source names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "labels": dict(sorted(labels.items()))},
                               indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_split(path: str | Path) -> tuple[dict[str, str], dict]:
    """(labels, meta) from a split file."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(doc["labels"]), dict(doc.get("meta") or {})


def select(rows, labels: dict[str, str], keep: str, *, name: str) -> list:
    """The rows of one pool. Every row must carry a label: a split that does not cover the
    corpus it is applied to was made for another revision of it.

    Raises:
        ValueError: a row has no label, or no row has `keep`.
    """
    out, unlabelled = [], 0
    for raw in rows:
        sid = str((raw.get("metadata") or {}).get("scenario_id") or "")
        lab = labels.get(sid)
        if lab is None:
            unlabelled += 1
        elif lab == keep:
            out.append(raw)
    if unlabelled:
        raise ValueError(f"source {name!r}: {unlabelled} rows carry no split label; the split "
                         "was made for a different revision of this corpus")
    if not out:
        raise ValueError(f"source {name!r}: no row is in pool {keep!r} (labels: "
                         f"{sorted(set(labels.values()))})")
    return out
