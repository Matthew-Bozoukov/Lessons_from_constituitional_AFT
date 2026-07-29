# ABOUTME: Planners decide WHICH situation is worth writing, before any document exists.
# ABOUTME: A separate stage, so "did planning help?" is a stage diff rather than a guess.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control import loader
from ..core.hashing import stable_hash
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, extract_json
from ..core.prompting import scenario_vars
from ..core.registry import has, resolve
from ..core.types import Document, StageRecord


@dataclass
class PlanningContext:
    """Configuration for the planning stage.

    Attributes:
        llm: Cache-wrapped LLM client.
        model: Planner model (independently ablatable from the generator's).
        params: Sampling params.
        template: Entry in control/prompts/planning.yaml.
        stage_idx: Stage index for provenance and cache keying.
        stage_name: Stage name for provenance.
    """

    llm: CachedLLM
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    template: str = "what_how_why"
    stage_idx: int = 0
    stage_name: str = "stage_00_planned"


class PromptedPlanner:
    """Generic planner driven by a control/prompts/planning.yaml entry.

    Splits "decide what to write about" from "write it". Doing both in one call lets
    the generator settle on the first obvious situation and then justify it; planning
    first makes the choice of situation an explicit, inspectable artifact that shows up
    in its own stage snapshot.
    """

    def __init__(self, ctx: PlanningContext) -> None:
        """Initialize with a planning context."""
        self.ctx = ctx
        self.entry = loader.entry("planning", ctx.template)
        self.fields: list[str] = list(self.entry.get("fields") or [])

    def build_messages(self, doc: Document) -> list[dict]:
        """Render the planning prompt for one document."""
        variables = scenario_vars(doc.scenario)
        return [
            {"role": "system", "content": loader.render(self.entry["system"], **variables)},
            {"role": "user", "content": loader.render(self.entry["user"], **variables)},
        ]

    def plan(self, doc: Document) -> Document:
        """Attach a scenario plan to a document.

        A planning failure is recorded and leaves the plan empty rather than raising:
        generation falls back to planning-free behaviour for that row, so one bad plan
        costs one document's worth of structure instead of the run.

        Args:
            doc: A seeded document (scenario set, no turns yet).

        Returns:
            The document with `plan` and `plan_kind` populated where possible.
        """
        ctx = self.ctx
        messages = self.build_messages(doc)
        try:
            resp, prompt_hash = ctx.llm.call(
                stage_idx=ctx.stage_idx,
                input_hash=doc.scenario.scenario_hash,
                model=ctx.model,
                messages=messages,
                params=ctx.params,
                scope="plan",
            )
        except Exception as e:
            doc.error = f"plan: {type(e).__name__}: {e}"
            return doc

        doc.lineage.append(
            StageRecord(
                stage_idx=ctx.stage_idx,
                stage_name=ctx.stage_name,
                kind=f"plan:{ctx.template}",
                model=ctx.model,
                prompt_hash=prompt_hash,
                params=stable_hash(ctx.params, 12),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=ctx.llm.cost(resp),
                cached=resp.cached,
            )
        )

        try:
            payload = extract_json(resp.content)
        except ParseError as e:
            doc.error = f"plan: ParseError: {e}"
            return doc
        if not isinstance(payload, dict):
            doc.error = f"plan: expected an object, got {type(payload).__name__}"
            return doc

        wanted = self.fields or sorted(payload)
        doc.plan = {k: str(payload.get(k, "")).strip() for k in wanted if payload.get(k)}
        doc.plan_kind = ctx.template
        if not doc.plan:
            doc.error = f"plan: none of the expected fields {wanted} were present"
        return doc


def build_planner(ctx: PlanningContext):
    """Resolve the planner for a template.

    A template registered under kind "planner" wins; otherwise any template declared in
    control/prompts/planning.yaml is served by PromptedPlanner.

    Args:
        ctx: The planning context, carrying the template name.

    Returns:
        A planner instance exposing .plan(document).
    """
    if has("planner", ctx.template):
        return resolve("planner", ctx.template)(ctx)
    loader.entry("planning", ctx.template)  # raises PromptError if undeclared
    return PromptedPlanner(ctx)
