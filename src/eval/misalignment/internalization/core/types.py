# ABOUTME: The data model: Clause, FakeClause, ClauseSet, Item, Completion, Verdict, ScoreRow.
# ABOUTME: Item is the load-bearing abstraction - one prompt shown to the model = one Item.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from .hashing import stable_hash

# Item families. A family decides what the prompt asks for and which judges apply;
# it is never branched on inside a judge.
FAMILIES = (
    "retrieval",
    "fake_clause",
    "application",
    "conflict",
    "over_refusal",
    "persona_drift",
)

# Scenario difficulty for application items. "clear" exists to catch a model that has
# lost the behaviour entirely; "ambiguous" exists so the ceiling is not 100%.
DIFFICULTIES = ("clear", "edge", "ambiguous", "na")

CLEAN = "clean"


def condition_label(pressure: str = "", ood_axis: str = "", ood_value: str = "") -> str:
    """Build the canonical condition string for an item.

    One string so the results store has a single grouping key, while `pressure`,
    `ood_axis`, and `ood_value` stay as separate columns for per-axis faceting.

    Args:
        pressure: Robustness wrapper name, if any.
        ood_axis: OOD distance axis name, if any.
        ood_value: Value along that axis.

    Returns:
        "clean", "pressure:<name>", or "ood:<axis>=<value>".
    """
    if pressure:
        return f"pressure:{pressure}"
    if ood_axis:
        return f"ood:{ood_axis}={ood_value}"
    return CLEAN


@dataclass(frozen=True)
class Clause:
    """One addressable normative clause of the constitution.

    Attributes:
        clause_id: Stable id; the join key for every plot and table.
        title: Short human label used on plot axes.
        text: The normative statement. Always placed in the judge's context, so a
            judge never grades against its own memory of the constitution.
        rationale: The constitution's *stated* reason for the clause. The
            justification judge compares the model's reason against this, which is
            what separates a recalled rationale from a plausible post-hoc one.
        principle: Parent principle id, for grouping clauses on plots.
        priority_tier: 1-4 position in the spec's conflict-resolution ordering.
            Lower dominates. Used only by the conflict judge.
        entailments: Clause ids whose behaviour should move together with this one.
            Unused in Tier A; recorded now so the Tier B spillover matrix has a
            ground truth to compare against without re-authoring the clause set.
        held_out: True if this clause is excluded from data generation, making it
            the memorisation control. Decided before generation, never after.
        tags: Free-form labels for slicing.
    """

    clause_id: str
    title: str
    text: str
    rationale: str
    principle: str = ""
    priority_tier: int = 3
    entailments: tuple[str, ...] = ()
    held_out: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "clause_id": self.clause_id,
            "title": self.title,
            "text": self.text,
            "rationale": self.rationale,
            "principle": self.principle,
            "priority_tier": self.priority_tier,
            "entailments": list(self.entailments),
            "held_out": self.held_out,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Clause:
        """Rebuild a Clause from to_dict output."""
        return cls(
            clause_id=d["clause_id"],
            title=d["title"],
            text=d["text"],
            rationale=d["rationale"],
            principle=d.get("principle", ""),
            priority_tier=int(d.get("priority_tier", 3)),
            entailments=tuple(d.get("entailments") or ()),
            held_out=bool(d.get("held_out", False)),
            tags=tuple(d.get("tags") or ()),
        )


@dataclass(frozen=True)
class FakeClause:
    """A fabricated clause that is not in the constitution.

    Attributes:
        fake_id: Stable id.
        text: The fabricated normative statement.
        near_clause_id: The real clause it is designed to be confusable with.
            Discrimination is only informative against a plausible neighbour; a
            fake clause about an unrelated topic measures nothing.
        why_fake: One line on what makes it wrong. Shown to the judge, never to the
            model under test.
    """

    fake_id: str
    text: str
    near_clause_id: str
    why_fake: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "fake_id": self.fake_id,
            "text": self.text,
            "near_clause_id": self.near_clause_id,
            "why_fake": self.why_fake,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FakeClause:
        """Rebuild a FakeClause from to_dict output."""
        return cls(
            fake_id=d["fake_id"],
            text=d["text"],
            near_clause_id=d["near_clause_id"],
            why_fake=d.get("why_fake", ""),
        )


@dataclass(frozen=True)
class ClauseSet:
    """A frozen, versioned set of clauses plus their distractors.

    Attributes:
        spec_id: Identifier of the constitution version this was cut from.
        clauses: The clauses, in spec order.
        fakes: The fabricated distractors.
        priority_order: Human-readable statement of the conflict ordering, handed
            verbatim to the conflict judge.
        priority_note: How strictly the ordering applies (the spec's ordering is
            holistic, not lexical - a judge told otherwise grades the wrong thing).
    """

    spec_id: str
    clauses: tuple[Clause, ...]
    fakes: tuple[FakeClause, ...] = ()
    priority_order: str = ""
    priority_note: str = ""

    def __iter__(self) -> Iterator[Clause]:
        """Iterate clauses in spec order."""
        return iter(self.clauses)

    def __len__(self) -> int:
        """Return the number of clauses."""
        return len(self.clauses)

    def get(self, clause_id: str) -> Clause:
        """Look up a clause by id.

        Args:
            clause_id: The id to find.

        Returns:
            The Clause.

        Raises:
            KeyError: If the id is not in the set.
        """
        for c in self.clauses:
            if c.clause_id == clause_id:
                return c
        raise KeyError(f"Unknown clause {clause_id!r}. Known: {[c.clause_id for c in self.clauses]}")

    def find(self, clause_id: str) -> Clause | None:
        """Look up a clause, returning None instead of raising.

        Families that belong to no single clause (the persona probes, the derived
        capability rows) carry a synthetic id. Their rubrics never reference `clause`,
        and Jinja's strict undefined still fails loudly if one ever starts to.

        Args:
            clause_id: The id to find.

        Returns:
            The Clause, or None if the id is not in the set.
        """
        for c in self.clauses:
            if c.clause_id == clause_id:
                return c
        return None

    def fakes_for(self, clause_id: str) -> tuple[FakeClause, ...]:
        """Return the distractors targeted at one clause."""
        return tuple(f for f in self.fakes if f.near_clause_id == clause_id)

    @property
    def trained(self) -> tuple[Clause, ...]:
        """Clauses available to data generation."""
        return tuple(c for c in self.clauses if not c.held_out)

    @property
    def held_out(self) -> tuple[Clause, ...]:
        """Clauses excluded from data generation."""
        return tuple(c for c in self.clauses if c.held_out)

    @property
    def fingerprint(self) -> str:
        """Content hash of the clause set; changes whenever any clause text changes."""
        return stable_hash(
            {
                "spec_id": self.spec_id,
                "clauses": [c.to_dict() for c in self.clauses],
                "fakes": [f.to_dict() for f in self.fakes],
                "priority_order": self.priority_order,
            }
        )


@dataclass(frozen=True)
class Item:
    """One prompt shown to the model under test.

    Base items and derived items are the same type. A robustness or OOD item is a
    base item with `parent_item_id` set, so every stressed score has a clean
    counterpart to difference against - that pairing is what the delta bars and
    decay curves are made of.

    Attributes:
        item_id: Stable content hash. Identical across runs and recipes, which is
            what lets two runs join row for row.
        clause_id: The governing clause.
        family: One of FAMILIES.
        difficulty: One of DIFFICULTIES.
        prompt: The user message.
        system: System prompt; "" means the harness default.
        history: Prior turns for multi-turn items, as OpenAI-style messages.
        pressure: Robustness wrapper name, "" when clean.
        ood_axis: OOD distance axis, "" when in-distribution.
        ood_value: Value along that axis.
        parent_item_id: The clean item this was derived from; "" for base items.
        variant: Index distinguishing sibling items with the same coordinates.
        meta: Family-specific fields. `norm_violating_path` and `legitimate_goal`
            for application items; `secondary_clause_id` and `expected_winner` for
            conflict items; `fake_id` for fake-clause items.
    """

    item_id: str
    clause_id: str
    family: str
    difficulty: str
    prompt: str
    system: str = ""
    history: tuple[dict[str, str], ...] = ()
    pressure: str = ""
    ood_axis: str = ""
    ood_value: str = ""
    parent_item_id: str = ""
    variant: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def condition(self) -> str:
        """Canonical condition string for grouping in the results store."""
        return condition_label(self.pressure, self.ood_axis, self.ood_value)

    @property
    def condition_kind(self) -> str:
        """"clean", "pressure", or "ood" - the condition without its parameters.

        Rubrics scope themselves with this: an axis measuring rationale recall has no
        reason to be re-run under adversarial pressure, and paying to do so is the
        largest avoidable cost in the suite.
        """
        return self.condition.split(":")[0]

    @property
    def is_derived(self) -> bool:
        """True if this item was transformed from a clean parent."""
        return bool(self.parent_item_id)

    def messages(self, default_system: str = "") -> list[dict[str, str]]:
        """Render the item as an OpenAI-style message list.

        Args:
            default_system: System prompt used when the item declares none.

        Returns:
            The message list to send to the model under test.
        """
        msgs: list[dict[str, str]] = []
        system = self.system or default_system
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(dict(t) for t in self.history)
        msgs.append({"role": "user", "content": self.prompt})
        return msgs

    def derive(self, **changes: Any) -> Item:
        """Return a derived item with a freshly computed id and this item as parent.

        Args:
            **changes: Fields to override on the copy.

        Returns:
            The derived Item.
        """
        child = replace(self, parent_item_id=self.item_id, **changes)
        return replace(child, item_id=make_item_id(child))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "item_id": self.item_id,
            "clause_id": self.clause_id,
            "family": self.family,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "system": self.system,
            "history": [dict(t) for t in self.history],
            "pressure": self.pressure,
            "ood_axis": self.ood_axis,
            "ood_value": self.ood_value,
            "parent_item_id": self.parent_item_id,
            "variant": self.variant,
            "condition": self.condition,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Item:
        """Rebuild an Item from to_dict output (the `condition` key is derived)."""
        return cls(
            item_id=d["item_id"],
            clause_id=d["clause_id"],
            family=d["family"],
            difficulty=d.get("difficulty", "na"),
            prompt=d["prompt"],
            system=d.get("system", ""),
            history=tuple(dict(t) for t in d.get("history") or ()),
            pressure=d.get("pressure", ""),
            ood_axis=d.get("ood_axis", ""),
            ood_value=d.get("ood_value", ""),
            parent_item_id=d.get("parent_item_id", ""),
            variant=int(d.get("variant", 0)),
            meta=dict(d.get("meta") or {}),
        )


def make_item_id(item: Item) -> str:
    """Compute an item's stable content id.

    Derived from the item's coordinates and its rendered prompt, and *not* from
    the run or the recipe, so the same item carries the same id in every run. Two
    item sets built from the same clauses and config produce identical ids, which
    is what makes results joinable across item-set rebuilds.

    Args:
        item: The item (its current item_id is ignored).

    Returns:
        A 16-hex-character id prefixed with the family.
    """
    digest = stable_hash(
        {
            "clause_id": item.clause_id,
            "family": item.family,
            "difficulty": item.difficulty,
            "prompt": item.prompt,
            "system": item.system,
            "history": [dict(t) for t in item.history],
            "pressure": item.pressure,
            "ood_axis": item.ood_axis,
            "ood_value": item.ood_value,
            "variant": item.variant,
            "meta": item.meta,
        }
    )
    return f"{item.family}_{digest}"


@dataclass(frozen=True)
class Completion:
    """One model response to one item.

    Attributes:
        item_id: The item answered.
        text: The answer, with any reasoning trace stripped out.
        thinking: The reasoning trace, when the model exposes one.
        model: Model id that produced it.
        finish_reason: Provider finish reason; a truncated answer must not be
            silently scored as a refusal.
        error: Non-empty when generation failed for this item.
    """

    item_id: str
    text: str
    thinking: str = ""
    model: str = ""
    finish_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the completion is usable for judging."""
        return not self.error and bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "item_id": self.item_id,
            "text": self.text,
            "thinking": self.thinking,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Completion:
        """Rebuild a Completion from to_dict output."""
        return cls(
            item_id=d["item_id"],
            text=d.get("text", ""),
            thinking=d.get("thinking", ""),
            model=d.get("model", ""),
            finish_reason=d.get("finish_reason", ""),
            error=d.get("error", ""),
        )


@dataclass(frozen=True)
class Verdict:
    """One judge's score on one axis for one completion.

    Attributes:
        item_id: The item judged.
        axis: Metric name, e.g. "compliance" or "tension_recognition".
        score: Normalised to [0, 1] so axes on different rubric scales can share a
            heatmap and a results table.
        raw_score: The rubric-native score, kept so a rubric change is auditable.
        passed: Binarised at the axis's declared threshold. Proportion plots use
            this; graded plots use `score`.
        rationale: The judge's stated reason. Not scored - kept so a surprising
            aggregate can be traced back to the individual judgements behind it.
        judge_model: Model that produced the verdict.
        error: Non-empty when judging failed.
    """

    item_id: str
    axis: str
    score: float
    raw_score: float = 0.0
    passed: bool = False
    rationale: str = ""
    judge_model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "item_id": self.item_id,
            "axis": self.axis,
            "score": self.score,
            "raw_score": self.raw_score,
            "passed": self.passed,
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verdict:
        """Rebuild a Verdict from to_dict output."""
        return cls(
            item_id=d["item_id"],
            axis=d["axis"],
            score=float(d.get("score", 0.0)),
            raw_score=float(d.get("raw_score", 0.0)),
            passed=bool(d.get("passed", False)),
            rationale=d.get("rationale", ""),
            judge_model=d.get("judge_model", ""),
            error=d.get("error", ""),
        )
