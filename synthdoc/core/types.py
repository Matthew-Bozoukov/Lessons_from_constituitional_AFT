# ABOUTME: The data model: SpecChunk, ScenarioSpec, Turn, StageRecord, Document.
# ABOUTME: ScenarioSpec is the load-bearing abstraction - one experimental condition = one document.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .hashing import stable_hash

# Roles permitted in a Turn. Fixed so the snapshot schema stays stable.
ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class SpecChunk:
    """One addressable piece of a model spec.

    Attributes:
        spec_id: Which spec this came from (constitution_v3, rules_only, ...).
        chunk_id: Stable ID; the join key for coverage reporting.
        text: The chunk text as it will be shown to the generator.
        granularity: bullet | trait | section.
        parent_id: The section this chunk was cut from (used by adjacent grouping).
        order_idx: Position within parent (used by adjacent grouping).
        meta: Free-form annotations, e.g. register, prescriptive, has_tradeoffs.
    """

    spec_id: str
    chunk_id: str
    text: str
    granularity: str
    parent_id: str
    order_idx: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by identity fields only (meta is a dict and not hashable)."""
        return hash((self.spec_id, self.chunk_id))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "spec_id": self.spec_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "granularity": self.granularity,
            "parent_id": self.parent_id,
            "order_idx": self.order_idx,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpecChunk:
        """Rebuild a SpecChunk from to_dict output."""
        return cls(
            spec_id=d["spec_id"],
            chunk_id=d["chunk_id"],
            text=d["text"],
            granularity=d["granularity"],
            parent_id=d.get("parent_id", ""),
            order_idx=int(d.get("order_idx", 0)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass(frozen=True)
class ScenarioSpec:
    """One experimental condition. The sampler emits these; generators only render them.

    Single-chunk and many-chunk conditions are the same type: `chunks` is always a
    tuple, and `grouping_strategy` is recorded as "single" when len(chunks) == 1 so
    those rows stay joinable with the rest of the corpus.

    Attributes:
        chunks: The spec chunks this document must demonstrate.
        grouping_strategy: single | random | adjacent | semantic.
        doc_type: Registered doc_type plugin name.
        axes: Declared axis values (tools, reasoning, explicitness, ...). Every
            scenario in a run carries the same axis keys so the snapshot schema
            is stable and groupbys are total.
        seed: Per-example seed, derived from the run seed and the example index.
    """

    chunks: tuple[SpecChunk, ...]
    grouping_strategy: str
    doc_type: str
    axes: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    @property
    def spec_id(self) -> str:
        """The spec these chunks came from."""
        return self.chunks[0].spec_id if self.chunks else ""

    @property
    def chunk_ids(self) -> list[str]:
        """Chunk IDs in group order."""
        return [c.chunk_id for c in self.chunks]

    @property
    def scenario_hash(self) -> str:
        """Stable hash of the experimental condition.

        Constant across sweep arms, which is what makes paired comparisons work.
        Chunk *text* is included so that editing the spec invalidates scenarios
        rather than silently reusing stale IDs.
        """
        return stable_hash(
            {
                "spec_id": self.spec_id,
                "chunks": [(c.chunk_id, stable_hash(c.text)) for c in self.chunks],
                "grouping_strategy": self.grouping_strategy,
                "doc_type": self.doc_type,
                "axes": dict(sorted(self.axes.items())),
                "seed": self.seed,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "grouping_strategy": self.grouping_strategy,
            "doc_type": self.doc_type,
            "axes": dict(self.axes),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioSpec:
        """Rebuild a ScenarioSpec from to_dict output."""
        return cls(
            chunks=tuple(SpecChunk.from_dict(c) for c in d["chunks"]),
            grouping_strategy=d["grouping_strategy"],
            doc_type=d["doc_type"],
            axes=dict(d.get("axes") or {}),
            seed=int(d.get("seed", 0)),
        )


@dataclass
class Turn:
    """One message in a document.

    Attributes:
        role: One of ROLES.
        content: Message text.
        thinking: Reasoning trace kept separate from content, so that
            reasoning_location can be varied without reparsing the document.
        tool_calls: JSON string of tool calls, or "" when none.
    """

    role: str
    content: str
    thinking: str = ""
    tool_calls: str = ""

    def __post_init__(self) -> None:
        """Validate the role against the fixed schema."""
        if self.role not in ROLES:
            raise ValueError(f"Bad role {self.role!r}; expected one of {ROLES}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "role": self.role,
            "content": self.content,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        """Rebuild a Turn from to_dict output."""
        return cls(
            role=d["role"],
            content=d.get("content") or "",
            thinking=d.get("thinking") or "",
            tool_calls=d.get("tool_calls") or "",
        )


@dataclass
class StageRecord:
    """Provenance for a single model call. Mandatory, not optional.

    Attributes:
        stage_idx: Index of the stage that produced this record.
        stage_name: e.g. stage_01_revised.
        kind: Plugin that made the call, e.g. "generate" or "critique_rewrite".
        model: Model id used.
        prompt_hash: Hash of the exact rendered prompt.
        params: Sampling params as a canonical JSON string.
        prompt_tokens: Prompt tokens reported by the provider.
        completion_tokens: Completion tokens reported by the provider.
        cost_usd: Cost from the run's price table.
        cached: True if the call was served from the local cache.
    """

    stage_idx: int
    stage_name: str
    kind: str
    model: str
    prompt_hash: str
    params: str = "{}"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "stage_idx": self.stage_idx,
            "stage_name": self.stage_name,
            "kind": self.kind,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "params": self.params,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageRecord:
        """Rebuild a StageRecord from to_dict output."""
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class Document:
    """A generated document plus everything needed to re-derive and trace it.

    doc_id is hash(scenario_hash, run_id) and is CONSTANT across stages, so
    stage N and stage N+1 join row-for-row.

    Attributes:
        doc_id: Stable document identity within a run.
        scenario: The condition this document realizes.
        turns: The document itself.
        lineage: Every model call that touched it, oldest first.
        stage_idx: Stage that produced this version.
        stage_name: Name of that stage.
        input_doc_id: doc_id of the input row (equal to doc_id; kept explicit so
            snapshot joins are self-describing and future fan-out stages can differ).
        filter_scores: Filter name -> score. Empty until the filter stage.
        filter_verdict: keep | drop | None. None until the filter stage.
        dropped_by: Name of the first filter that voted drop, else "".
        error: Non-empty when a stage failed for this document.
    """

    doc_id: str
    scenario: ScenarioSpec
    turns: list[Turn] = field(default_factory=list)
    lineage: list[StageRecord] = field(default_factory=list)
    stage_idx: int = 0
    stage_name: str = "stage_00_generated"
    input_doc_id: str | None = None
    filter_scores: dict[str, float] = field(default_factory=dict)
    filter_verdict: str | None = None
    dropped_by: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True if the document has content and no recorded error."""
        return not self.error and bool(self.turns)

    @property
    def cost_usd_total(self) -> float:
        """Cumulative cost of every model call in this document's lineage."""
        return round(sum(r.cost_usd for r in self.lineage), 6)

    @property
    def n_tokens(self) -> int:
        """Total tokens across this document's lineage."""
        return sum(r.prompt_tokens + r.completion_tokens for r in self.lineage)

    def text(self) -> str:
        """Flattened text of all turns, used for dedup and length statistics."""
        return "\n\n".join(f"{t.role}: {t.thinking}\n{t.content}".strip() for t in self.turns)

    def advanced(self, stage_idx: int, stage_name: str) -> Document:
        """Return a shallow copy marked as belonging to a later stage.

        Args:
            stage_idx: New stage index.
            stage_name: New stage name.

        Returns:
            A copy with stage fields updated and input_doc_id set to this doc_id.
        """
        return replace(
            self,
            stage_idx=stage_idx,
            stage_name=stage_name,
            input_doc_id=self.doc_id,
            turns=list(self.turns),
            lineage=list(self.lineage),
            filter_scores=dict(self.filter_scores),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (full fidelity, used by local JSONL)."""
        return {
            "doc_id": self.doc_id,
            "scenario": self.scenario.to_dict(),
            "turns": [t.to_dict() for t in self.turns],
            "lineage": [r.to_dict() for r in self.lineage],
            "stage_idx": self.stage_idx,
            "stage_name": self.stage_name,
            "input_doc_id": self.input_doc_id,
            "filter_scores": dict(self.filter_scores),
            "filter_verdict": self.filter_verdict,
            "dropped_by": self.dropped_by,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        """Rebuild a Document from to_dict output."""
        return cls(
            doc_id=d["doc_id"],
            scenario=ScenarioSpec.from_dict(d["scenario"]),
            turns=[Turn.from_dict(t) for t in d.get("turns") or []],
            lineage=[StageRecord.from_dict(r) for r in d.get("lineage") or []],
            stage_idx=int(d.get("stage_idx", 0)),
            stage_name=d.get("stage_name", "stage_00_generated"),
            input_doc_id=d.get("input_doc_id"),
            filter_scores=dict(d.get("filter_scores") or {}),
            filter_verdict=d.get("filter_verdict"),
            dropped_by=d.get("dropped_by") or "",
            error=d.get("error") or "",
        )


def make_doc_id(scenario_hash: str, run_id: str) -> str:
    """Compute the stage-invariant document id.

    Args:
        scenario_hash: ScenarioSpec.scenario_hash.
        run_id: The run identifier.

    Returns:
        A short hex id, constant for this (scenario, run) across all stages.
    """
    return stable_hash({"scenario_hash": scenario_hash, "run_id": run_id})
