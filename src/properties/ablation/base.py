# ABOUTME: The ablation contract: applicable(prop, records, adapter) says whether a kind of
# ABOUTME: intervention CAN run, apply(prop, records, cfg) runs it and reports what it did.

"""What an ablation is, and what it is allowed to do.

The paper's claim is that a property is load-bearing. The only way to earn that claim is
to remove the property, retrain, and see the eval move. Four ways to remove it, in
increasing order of how large an intervention they are — which is also increasing order of
how many things they change besides the property:

    mask        the property's tokens stay in the sequence but carry NO LOSS. Weakest.
                The model still reads them; it is just never trained to produce them.
                Tokenisation is identical to the control, so nothing is confounded by it.
    filter      rows carrying the property are dropped, or the corpus is SPLIT into
                has-X and lacks-X halves and both are trained. Changes the corpus size or
                its composition, and X correlates with other things.
    rewrite     one LLM call per affected row rewrites the property out (optionally
                substituting a named replacement). Callum's recommendation, 2026-08-17:
                "taking an existing dataset and doing an ad hoc, specific rewrite to vary
                a targeted property is a very good kind of experiment... it gets you two
                datasets that will be very similar in most ways."
    regenerate  re-run the generation pipeline with the property suppressed at a named
                stage. Largest: a stage ablation changes many things at once, which is the
                confound the other three exist to avoid.

Two rules every kind obeys.

**Only a training corpus can be ablated.** `sources/base.py` marks a source `ablatable`;
rollouts are not. Ablating rollouts would produce a dataset nothing was ever trained on.

**An ablation reports what it changed, and the report is not optional.** `AblationResult`
carries the changed ids and the counts, `verify.py` measures the prevalence drop, and
`scripts/properties/ablate.py` refuses to hand a corpus to training when nothing changed —
an arm identical to its control is a wasted pod and, worse, reads as a null result.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from src.properties.registry import Property
from src.properties.sources.base import Record, SourceAdapter

# Ordered weakest to strongest intervention; `scripts/properties/ablate.py` prints this so
# a config author sees what else they could have run.
KINDS = ("mask", "filter", "rewrite", "regenerate")


@dataclass
class AblationResult:
    """What one ablation produced.

    Attributes:
        kind: Which ablation ran.
        property_id: The property it targeted.
        rows: The ablated corpus, ready to write as jsonl. Rows carry every field of the
            original except what the ablation deliberately changed.
        arms: Named extra corpora when the ablation produces more than one — `filter` in
            split mode returns {"has": rows, "lacks": rows}. Empty for the usual case.
        changed_ids: record_ids the ablation touched.
        detected_ids: record_ids the detector flagged, whether or not the ablation could
            act on them. `detected - changed` is the intervention's miss rate and belongs
            in the report rather than in someone's head.
        report: Counts, judge usage, and anything else a reader needs to trust the arm.
    """

    kind: str
    property_id: str
    rows: list[dict]
    arms: dict[str, list[dict]] = field(default_factory=dict)
    changed_ids: list[str] = field(default_factory=list)
    detected_ids: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """The headline numbers, for a run_meta and for stdout.

        Returns:
            The counts, including the share of detected rows the ablation actually acted
            on — a low number means the intervention is weaker than the property list
            implies.
        """
        detected, changed = len(self.detected_ids), len(self.changed_ids)
        return {"kind": self.kind, "property_id": self.property_id,
                "n_rows": len(self.rows), "n_detected": detected, "n_changed": changed,
                "acted_on_share": round(changed / detected, 4) if detected else None,
                "arms": sorted(self.arms) or None, **self.report}


@dataclass(frozen=True)
class AblationSpec:
    """One kind of ablation.

    Attributes:
        name: The kind, as a config names it.
        module: Module under src.properties.ablation defining `applicable` and `apply`.
        strength: Rank in KINDS — how large an intervention this is.
        needs_judge: True when it spends OpenRouter calls per affected row.
    """

    name: str
    module: str
    strength: int
    needs_judge: bool = True


ABLATIONS: dict[str, AblationSpec] = {
    "mask": AblationSpec("mask", "mask", 0),
    "filter": AblationSpec("filter", "filter", 1),
    "rewrite": AblationSpec("rewrite", "rewrite", 2),
    "regenerate": AblationSpec("regenerate", "regenerate", 3, needs_judge=False),
}


def resolve(kind: str):
    """Import one ablation's module (the only place they are imported).

    Args:
        kind: A key of ABLATIONS.

    Returns:
        The module, exposing `applicable(prop, records, adapter, cfg)` and
        `apply(prop, records, cfg)`.

    Raises:
        KeyError: If the kind is not registered.
    """
    if kind not in ABLATIONS:
        raise KeyError(f"unknown ablation {kind!r}; registered: {sorted(ABLATIONS)}")
    return importlib.import_module(
        f"src.properties.ablation.{ABLATIONS[kind].module}")


def check_corpus(adapter: SourceAdapter) -> tuple[bool, str]:
    """The check every ablation shares: is this source a training corpus at all?

    Args:
        adapter: The source adapter the records came from.

    Returns:
        (ok, reason).
    """
    if not adapter.ablatable:
        return False, (
            f"source {adapter.name!r} is evidence, not training data: an ablated copy of "
            "it is a dataset nothing was ever trained on. Point the ablation at the "
            "corpus the model was trained on (mixture_rows).")
    return True, ""


def candidates(records: list[Record], cfg) -> tuple[list[Record], set[str]]:
    """Split the corpus into the rows an ablation may touch and the ids of the rest.

    A training mixture is mostly replay data. The property was discovered in one share of
    it, and only that share should be judged (a detector pass over 10,000 rows to act on
    716 is money spent to learn nothing) — but the ABLATED CORPUS MUST STILL BE THE WHOLE
    MIXTURE. An arm that trains on the 716 rows alone is not the control's experiment with
    one property removed; it is a different experiment.

    That is why the restriction lives here rather than as a filter on the source: the
    source loads everything, `restrict:` narrows what gets judged and edited, and every
    row is written back.

    Args:
        records: The whole corpus.
        cfg: The ablation config block; reads `restrict:`, a mapping of metadata key ->
            required value (e.g. {source_label: synthdoc_difficult_advice}).

    Returns:
        (the rows the ablation may touch, ids of the rows it must leave alone).

    Raises:
        ValueError: If the restriction matches nothing — a typo in a metadata key is
            otherwise indistinguishable from a property that is not in the corpus.
    """
    spec = (cfg.get("restrict") if cfg is not None and hasattr(cfg, "get") else None)
    if not spec:
        return list(records), set()
    wanted = {k: str(v) for k, v in dict(spec).items()}
    chosen = [r for r in records
              if all(str(r.metadata.get(k)) == v for k, v in wanted.items())]
    if not chosen:
        raise ValueError(
            f"restrict: {wanted} matches none of the {len(records)} records. Check the "
            f"metadata keys — one record carries {sorted(records[0].metadata)[:8]}"
            if records else f"restrict: {wanted} matches nothing in an empty corpus")
    print(f">>> restrict: {wanted} -> {len(chosen)}/{len(records)} rows are candidates; "
          f"the other {len(records) - len(chosen)} are written back untouched")
    return chosen, {r.record_id for r in records} - {r.record_id for r in chosen}


def check_channel(prop: Property, wanted: tuple[str, ...]) -> tuple[bool, str]:
    """Whether a property lives in a channel this ablation can act on.

    Args:
        prop: The property.
        wanted: Channels this ablation handles.

    Returns:
        (ok, reason).
    """
    if prop.channel not in wanted:
        return False, (f"property is in the {prop.channel!r} channel; this ablation acts "
                       f"on {wanted}")
    return True, ""


def applicable(kind: str, prop: Property, records: list[Record],
               adapter: SourceAdapter, cfg=None) -> tuple[bool, str]:
    """Whether one ablation kind can run against this property and corpus.

    Args:
        kind: The ablation kind.
        prop: The property to ablate.
        records: The corpus.
        adapter: The source adapter the records came from.
        cfg: The ablation's config block, when it has one.

    Returns:
        (ok, reason). The reason is written for a config author, so it says what to change.
    """
    return resolve(kind).applicable(prop, records, adapter, cfg)


def applicable_kinds(prop: Property, records: list[Record],
                     adapter: SourceAdapter, cfg=None) -> dict[str, tuple[bool, str]]:
    """Every kind's verdict on this property, weakest intervention first.

    Args:
        prop: The property.
        records: The corpus.
        adapter: The source adapter.
        cfg: The ablation config block.

    Returns:
        kind -> (ok, reason), in KINDS order. Printed by the driver so a config author can
        see that a stronger intervention was available and chose the weaker one on purpose.
    """
    return {kind: applicable(kind, prop, records, adapter, cfg) for kind in KINDS}


def apply(kind: str, prop: Property, records: list[Record], cfg) -> AblationResult:
    """Run one ablation.

    Args:
        kind: The ablation kind.
        prop: The property to ablate.
        records: The corpus.
        cfg: The ablation's config block — this is where per-property specifics live
            (span descriptions, rewrite instructions, the stage to regenerate without).

    Returns:
        The AblationResult.
    """
    return resolve(kind).apply(prop, records, cfg)
