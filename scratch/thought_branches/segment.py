# ABOUTME: Splits an agentic trajectory into Chunks — the sentence-sized units a branch
# ABOUTME: point cuts between — and defines the branch points themselves.

"""The unit of analysis.

Thought Branches resamples "from sentence i onward", so everything downstream needs one
addressable, stably-ordered unit. In a single-CoT setting that unit is just a sentence.
In ODCV the trace is a chain of (thought, action) pairs, so a chunk is either:

    thought  one sentence of an assistant step's `reason`
    action   one tool call from that step, rendered as text

Actions are chunks, not metadata, for the reason the paper's own taxonomy has an
`action_execution` tag: in an agentic scenario the command IS a reasoning step, and it is
usually the one the judge scores. `bash: sed -i 's/1600000/1500000/' revenue.yaml` has to
be labellable and countable next to the sentence that proposed it.

Every chunk carries three positions:
    step_pos    which assistant thought-step it belongs to
    sent_pos    where in that step's reason it sits
    pos         its index in the flattened trace — the branch coordinate
and `rel_pos`, `pos` normalised to [0, 1], so traces of different lengths can be pooled
on one axis (the paper's Figure 5 does exactly this).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from src.data.synth.derive import split_sentences

from scratch.thought_branches.trajectory import Step, Trajectory

# Cap on how much of a tool call's arguments an action chunk shows. Whole file bodies get
# written through `bash`, and an un-truncated heredoc would swamp both the embedding and
# the labeller's context with payload rather than intent.
ACTION_CHARS = 400


@dataclass(frozen=True)
class Chunk:
    """One addressable unit of a trajectory's reasoning.

    Attributes:
        traj_key: The owning trajectory's `Trajectory.key`.
        pos: 0-based index in the flattened trace; the branch coordinate.
        step_index: Transcript step number this chunk came from.
        step_pos: 0-based index of that step among the trajectory's thought steps.
        sent_pos: 0-based sentence index within the step (actions sort after sentences).
        kind: "thought" or "action".
        text: The chunk's text.
        n_chunks: How many chunks the trajectory has, so `rel_pos` needs no lookup.
    """

    traj_key: str
    pos: int
    step_index: int
    step_pos: int
    sent_pos: int
    kind: str
    text: str
    n_chunks: int

    @property
    def rel_pos(self) -> float:
        """Position in the trace on [0, 1]; 0.0 for a one-chunk trace."""
        return self.pos / (self.n_chunks - 1) if self.n_chunks > 1 else 0.0

    @property
    def chunk_id(self) -> str:
        """Globally unique, stable across reruns: `<traj key>#<pos>`."""
        return f"{self.traj_key}#{self.pos}"


def render_action(step: Step, limit: int = ACTION_CHARS) -> str:
    """Render an assistant step's tool calls as one line of text.

    Args:
        step: The assistant step.
        limit: Max characters of arguments to keep per call.

    Returns:
        A compact `name: args` rendering, empty when the step called no tool.
    """
    parts = []
    for c in step.calls:
        if c.name == "bash":
            body = c.command
        else:
            body = json.dumps(c.arguments, default=str)
        body = " ".join(body.split())
        if len(body) > limit:
            body = body[:limit] + " …"
        parts.append(f"{c.name}: {body}")
    return " | ".join(parts)


def chunk_trajectory(traj: Trajectory, include_actions: bool = True) -> list[Chunk]:
    """Flatten one trajectory into ordered chunks.

    Args:
        traj: The trajectory to segment.
        include_actions: Emit an `action` chunk per tool-calling step. Turn this off to
            analyse the reasoning text alone.

    Returns:
        Chunks in trace order. Empty if the trajectory has no assistant thoughts.
    """
    rows: list[
        tuple[int, int, int, str, str]
    ] = []  # step_index, step_pos, sent_pos, kind, text
    step_pos = 0
    for step in traj.steps:
        if not step.is_assistant:
            continue
        sents = split_sentences(step.reason) if step.has_thought else []
        for j, s in enumerate(sents):
            if s.strip():
                rows.append((step.index, step_pos, j, "thought", s.strip()))
        if include_actions:
            act = render_action(step)
            if act:
                rows.append((step.index, step_pos, len(sents), "action", act))
        if step.has_thought or step.calls:
            step_pos += 1
    n = len(rows)
    return [
        Chunk(
            traj_key=traj.key,
            pos=i,
            step_index=si,
            step_pos=sp,
            sent_pos=se,
            kind=k,
            text=t,
            n_chunks=n,
        )
        for i, (si, sp, se, k, t) in enumerate(rows)
    ]


def chunk_corpus(
    trajs: Iterable[Trajectory], include_actions: bool = True
) -> list[Chunk]:
    """Segment many trajectories, preserving corpus order."""
    return [
        c for t in trajs for c in chunk_trajectory(t, include_actions=include_actions)
    ]


# -- branch points ----------------------------------------------------------------


@dataclass(frozen=True)
class BranchPoint:
    """A place to cut a trajectory and resample what comes after.

    Two granularities, because ODCV admits both and they answer different questions:

        step      Cut before assistant step `step_index`. The whole turn — thought and
                  action — is resampled. Asks "what would it have done at this turn?"
        sentence  Cut inside that step's thought, keeping its first `sent_pos` sentences
                  as a forced prefix. Asks "what does THIS sentence do?", which is the
                  paper's question, and needs assistant-continuation support.

    Attributes:
        traj_key: The trajectory being cut.
        step_index: Transcript step number of the assistant turn at the cut.
        sent_pos: Sentences of that turn's thought retained as prefix; 0 for a step cut.
        granularity: "step" or "sentence".
        target_pos: Flattened `Chunk.pos` of the chunk this branch removes, for joining
            back onto labels and metrics.
    """

    traj_key: str
    step_index: int
    sent_pos: int
    granularity: str
    target_pos: int

    @property
    def branch_id(self) -> str:
        return f"{self.traj_key}@{self.step_index}.{self.sent_pos}"


def branch_points(
    traj: Trajectory, granularity: str = "step", chunks: list[Chunk] | None = None
) -> list[BranchPoint]:
    """Enumerate every branch point in a trajectory.

    Args:
        traj: The trajectory.
        granularity: "step" (one per thought-bearing assistant turn) or "sentence"
            (one per thought sentence).
        chunks: Precomputed chunks, to avoid re-segmenting.

    Returns:
        Branch points in trace order.

    Raises:
        ValueError: On an unknown granularity — a typo here would silently analyse the
            wrong resolution.
    """
    if granularity not in ("step", "sentence"):
        raise ValueError(
            f"granularity must be 'step' or 'sentence', got {granularity!r}"
        )
    chunks = chunks if chunks is not None else chunk_trajectory(traj)
    out: list[BranchPoint] = []
    if granularity == "step":
        seen: set[int] = set()
        for c in chunks:
            if c.step_index in seen:
                continue
            seen.add(c.step_index)
            out.append(BranchPoint(traj.key, c.step_index, 0, "step", c.pos))
    else:
        for c in chunks:
            if c.kind == "thought":
                out.append(
                    BranchPoint(traj.key, c.step_index, c.sent_pos, "sentence", c.pos)
                )
    return out


def prefix_text(traj: Trajectory, bp: BranchPoint) -> str:
    """The forced partial thought at a sentence branch point.

    Args:
        traj: The trajectory being cut.
        bp: The branch point.

    Returns:
        The first `bp.sent_pos` sentences of the step's reason, joined with spaces —
        the text the model must be made to continue from. Empty for a step branch,
        which starts the turn from nothing.
    """
    if bp.granularity == "step" or bp.sent_pos == 0:
        return ""
    step = next((s for s in traj.steps if s.index == bp.step_index), None)
    if step is None:
        return ""
    return " ".join(split_sentences(step.reason)[: bp.sent_pos]).strip()
