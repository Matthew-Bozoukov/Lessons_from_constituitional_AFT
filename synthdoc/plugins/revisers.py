# ABOUTME: Revisers are composable stages that rewrite a Document. The revision list
# ABOUTME: in the config IS the dose, so per-pass effects are measurable by construction.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control import loader
from ..core.hashing import stable_hash
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, parse_turns, render_document
from ..core.prompting import scenario_vars
from ..core.registry import has, resolve
from ..core.types import Document, StageRecord


@dataclass
class RevisionContext:
    """Configuration for one revision pass.

    Attributes:
        llm: Cache-wrapped LLM client.
        kind: Entry name in control/prompts/revision.yaml.
        model: Model for this pass (may differ from the generator's).
        params: Sampling params.
        context: "fresh" shows the reviser only the document and the spec excerpt;
            "same" also shows the original generation instructions. This is itself
            an ablation axis, which is why it is a field and not a hardcoded choice.
        gen_template: Generation template version, needed when context == "same".
        stage_idx: Stage index for provenance and cache keying.
        stage_name: Stage name for provenance.
        keep_on_failure: When True, a pass that fails leaves the document unchanged
            instead of marking it errored.
    """

    llm: CachedLLM
    kind: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    context: str = "fresh"
    gen_template: str = "v2"
    stage_idx: int = 1
    stage_name: str = "stage_01_revised"
    keep_on_failure: bool = True


class PromptedReviser:
    """Generic reviser driven entirely by a control/prompts/revision.yaml entry."""

    def __init__(self, ctx: RevisionContext) -> None:
        """Initialize with a revision context."""
        self.ctx = ctx

    def build_messages(self, doc: Document) -> list[dict]:
        """Render the revision prompt for one document.

        Args:
            doc: The document to revise.

        Returns:
            OpenAI-style messages.
        """
        tpl = loader.entry("revision", self.ctx.kind)
        variables = scenario_vars(doc.scenario, document=render_document(doc.turns))
        user = loader.render(tpl["user"], **variables)

        if self.ctx.context == "same":
            gen_tpl = loader.entry("generation", self.ctx.gen_template)
            original = loader.render(gen_tpl["user"], **scenario_vars(doc.scenario))
            user = (
                "## Original generation instructions\n\n"
                f"{original}\n\n"
                "---\n\n"
                f"{user}"
            )
        return [
            {"role": "system", "content": loader.render(tpl["system"], **variables)},
            {"role": "user", "content": user},
        ]

    def revise(self, doc: Document) -> Document:
        """Run one revision pass.

        Args:
            doc: Input document (already advanced to this stage).

        Returns:
            The document with revised turns and an appended lineage record.
        """
        ctx = self.ctx
        if not doc.ok:
            return doc

        messages = self.build_messages(doc)
        # Keying on the rendered input text means an unchanged upstream document
        # re-uses this pass's cached output on re-run.
        input_hash = stable_hash(render_document(doc.turns))
        try:
            resp, prompt_hash = ctx.llm.call(
                stage_idx=ctx.stage_idx,
                input_hash=input_hash,
                model=ctx.model,
                messages=messages,
                params=ctx.params,
            )
        except Exception as e:
            if not ctx.keep_on_failure:
                doc.error = f"{ctx.kind}: {type(e).__name__}: {e}"
            return doc

        doc.lineage.append(
            StageRecord(
                stage_idx=ctx.stage_idx,
                stage_name=ctx.stage_name,
                kind=ctx.kind,
                model=ctx.model,
                prompt_hash=prompt_hash,
                params=stable_hash({**ctx.params, "context": ctx.context}, 12),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=ctx.llm.cost(resp),
                cached=resp.cached,
            )
        )
        try:
            doc.turns = parse_turns(resp.content)
        except ParseError as e:
            # A pass that returns garbage must not destroy a good document.
            if not ctx.keep_on_failure:
                doc.error = f"{ctx.kind}: ParseError: {e}"
        return doc


def build_reviser(ctx: RevisionContext):
    """Resolve the reviser for a kind.

    A kind registered under "reviser" wins; otherwise any kind declared in
    control/prompts/revision.yaml is served by PromptedReviser.

    Args:
        ctx: The revision context, carrying the kind.

    Returns:
        A reviser instance exposing .revise(document).
    """
    if has("reviser", ctx.kind):
        return resolve("reviser", ctx.kind)(ctx)
    loader.entry("revision", ctx.kind)  # raises PromptError if undeclared
    return PromptedReviser(ctx)
