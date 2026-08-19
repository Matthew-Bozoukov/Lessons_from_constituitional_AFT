# ABOUTME: The three types every producer and every ablation is written against —
# ABOUTME: Record, Target, SourceAdapter — and the shared channel/message normaliser.

"""What a source hands the rest of this module.

A producer must not know whether its evidence came from a training mixture, an ODCV
container log, or an agentic-misalignment sample directory. It reads `Record`s, which
carry the same three text CHANNELS in every case:

    query      what the model was asked (first user turn / the scenario prompt)
    reasoning  its private chain of thought, "" when there is none
    response   what it actually said or did

Three fields exist because the producers disagree about which channel is the cause and
which is the effect. TURF's trigger side is query+reasoning and its behaviour side is the
response; feature discovery reads reasoning alone; trace_clusters can be pointed at any
of them. Keeping all three on every record means that choice stays the producer's, made
in a config, rather than baked into a loader.

Two more fields carry what the channels cannot:

    outcome    the source's own label for how the record turned out — a violation flag, a
               judge score. Only rollout sources have one; a training corpus does not, and
               `has_outcomes` on the adapter says so rather than leaving a caller to
               discover it from a None.
    raw        the ORIGINAL row, untouched. Ablation edits a corpus and writes it back,
               and it must preserve every field it did not deliberately change — a mixture
               row carries `source`, `supervise`, metadata that training reads. Rebuilding
               a row from the channels would silently drop them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The channels, in causal order: what was asked, what was thought, what was said.
CHANNELS = ("query", "reasoning", "response")


@dataclass(frozen=True)
class Record:
    """One unit of evidence: a query, an optional reasoning trace, and a response.

    Attributes:
        record_id: Stable id, unique within the source. Joins a property back to the rows
            that produced it and an ablation back to the rows it changed.
        query: The first user turn's content.
        response: The first assistant turn's content.
        reasoning: That turn's private reasoning, "" when the record has none.
        outcome: The source's own outcome labels (e.g. {"violation": True, "score": 4}),
            or None for sources that have none.
        metadata: Whatever the source knows about the record (trait_id, scenario_id,
            condition, arm). Producers group and stratify on this; nothing requires a key.
        raw: The original row, so an ablation can write the corpus back unharmed.
    """

    record_id: str
    query: str
    response: str
    reasoning: str = ""
    outcome: dict | None = None
    metadata: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def channel(self, name: str) -> str:
        """The text of one channel.

        Args:
            name: One of CHANNELS.

        Returns:
            That channel's text.

        Raises:
            ValueError: If `name` is not a channel.
        """
        if name not in CHANNELS:
            raise ValueError(f"unknown channel {name!r}; known: {CHANNELS}")
        return getattr(self, name)


@dataclass(frozen=True)
class Target:
    """The behaviour a property is being traced back FROM.

    TURF needs one ("find the training properties behind this refusal"); LESS needs one
    (its validation set Dval is exactly this); feature_discovery and trace_clusters do
    not, since they describe a corpus rather than explain an outcome. A producer declares
    which it is via `ProducerSpec.needs_target`.

    Attributes:
        target_id: Stable id, used in property ids so two targets never collide.
        behaviour: The behaviour in prose — a TURF rubric's `principle_specific_details`.
            Describes the BEHAVIOUR only: naming candidate properties here would hand the
            discovery its answer, which is why the rubrics forbid it.
        polarity: "satisfy" (attribute good behaviour) or "violate" (the paper's original).
        cases: Exemplar records exhibiting the behaviour. One for a TURF trace, a whole
            subtask's worth for a LESS validation split.
        subtask: Optional grouping label when several targets share a validation set —
            LESS's `max_j` runs over these.
        provenance: Where this came from (rubric path, export file, eval run).
    """

    target_id: str
    behaviour: str
    polarity: str = "satisfy"
    cases: list[Record] = field(default_factory=list)
    subtask: str | None = None
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceAdapter:
    """Where one kind of evidence lives and how it becomes Records.

    Attributes:
        name: Registry key; recorded on every property row this source's evidence feeds.
        load: kwargs (straight from a config's `source:` block) -> Records.
        has_outcomes: True when every Record carries an `outcome`. Producers that split
            evidence by outcome (good traces vs bad) refuse a source without one rather
            than silently treating the whole corpus as "good".
        ablatable: True when this source is a TRAINING CORPUS an ablation may rewrite and
            hand to `uv run train`. Rollouts are evidence, not training data: ablating
            them would produce a dataset nothing was ever trained on.
    """

    name: str
    load: Callable[..., list[Record]]
    has_outcomes: bool = False
    ablatable: bool = False


def first_turns(messages: list[dict]) -> tuple[str, str, str] | None:
    """Reduce a chat transcript to (query, response, reasoning), or None if unusable.

    The FIRST user turn and the FIRST assistant turn, matching SURF's `_extract_first_turn`
    — system prompts and later turns are ignored. Keeping this identical to SURF matters
    for one producer only (TURF's index must live in the paper's distribution), but every
    producer using the same reduction is what makes their prevalences comparable.

    Args:
        messages: Interchange-format messages.

    Returns:
        (query, response, reasoning), or None when there is no user or no assistant turn.
    """
    if not isinstance(messages, list):
        return None
    query = next((m.get("content") or ""
                  for m in messages if m.get("role") == "user"), None)
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if query is None or assistant is None:
        return None
    return (query, assistant.get("content") or "",
            (assistant.get("reasoning_content") or "").strip())
