# ABOUTME: Generators render a planned ScenarioSpec into a Document, under a chosen
# ABOUTME: generation strategy. Doc-type variation lives in prompts, never in code paths.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control import loader
from ..core.hashing import stable_hash
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, parse_turns
from ..core.prompting import scenario_vars
from ..core.registry import has, resolve
from ..core.types import Document, ScenarioSpec, StageRecord, make_doc_id


@dataclass
class GenerationContext:
    """Everything a generator needs that is not the document itself.

    Attributes:
        llm: Cache-wrapped LLM client.
        model: Generator model id (the model ablation axis).
        params: Sampling params.
        template: Template version from control/prompts/generation.yaml.
        strategy: Registered generation strategy name.
        strategy_params: Strategy-specific settings, e.g. {"n": 4} for best_of_n.
        run_id: Run identifier, part of doc_id.
        stage_idx: Stage index for provenance and cache keying.
        stage_name: Stage name for provenance.
        max_parse_retries: Repair attempts when output is not valid JSON.
    """

    llm: CachedLLM
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    template: str = "v2"
    strategy: str = "single_pass"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    run_id: str = "run"
    stage_idx: int = 0
    stage_name: str = "stage_00_generated"
    max_parse_retries: int = 1

    def record(self, kind: str, resp, prompt_hash: str, params: dict[str, Any]) -> StageRecord:
        """Build a lineage record for one call made under this context."""
        return StageRecord(
            stage_idx=self.stage_idx,
            stage_name=self.stage_name,
            kind=kind,
            model=self.model,
            prompt_hash=prompt_hash,
            params=stable_hash(params, 12),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=self.llm.cost(resp),
            cached=resp.cached,
        )


def seed_document(scenario: ScenarioSpec, run_id: str, stage_name: str = "stage_00_generated",
                  stage_idx: int = 0) -> Document:
    """Create the empty Document for a scenario.

    Identity is fixed here, before any model call, so a document keeps the same doc_id
    whether or not planning runs and whichever strategy fills it in.

    Args:
        scenario: The condition to realize.
        run_id: Run identifier.
        stage_name: Stage the document starts at.
        stage_idx: Index of that stage.

    Returns:
        A Document with identity set and no content.
    """
    doc_id = make_doc_id(scenario.scenario_hash, run_id)
    return Document(
        doc_id=doc_id,
        scenario=scenario,
        stage_idx=stage_idx,
        stage_name=stage_name,
        input_doc_id=doc_id,
    )


class PromptedGenerator:
    """The only generator most doc types need.

    Renders the configured generation template with the scenario's chunks, doc-type
    instructions, axis fragments, and the scenario plan when one exists, then parses the
    response into Turns. Doc types differ only by the text they contribute.
    """

    def __init__(self, ctx: GenerationContext) -> None:
        """Initialize with a generation context."""
        self.ctx = ctx

    def build_messages(self, doc: Document, repair: str = "") -> list[dict]:
        """Render the prompt for one document.

        Args:
            doc: The document being generated.
            repair: Optional appended instruction used on a parse retry.

        Returns:
            OpenAI-style messages.
        """
        tpl = loader.entry("generation", self.ctx.template)
        variables = scenario_vars(doc.scenario, plan=doc.plan or {})
        user = loader.render(tpl["user"], **variables)
        if repair:
            user = f"{user}\n\n{repair}"
        return [
            {"role": "system", "content": loader.render(tpl["system"], **variables)},
            {"role": "user", "content": user},
        ]

    def generate(self, doc: Document) -> Document:
        """Generate one document.

        A parse failure is recorded on the Document rather than raised, so the row
        survives into the snapshot and the failure rate is inspectable instead of
        being a hole in the corpus.

        Args:
            doc: A seeded (and optionally planned) document.

        Returns:
            The document with turns, or carrying an error.
        """
        ctx = self.ctx
        repair = ""
        last_error = ""
        for attempt in range(ctx.max_parse_retries + 1):
            messages = self.build_messages(doc, repair)
            params = dict(ctx.params)
            if attempt:
                # Perturb the cache key so a retry is not served the failed response.
                params["seed"] = doc.scenario.seed + attempt
            try:
                resp, prompt_hash = ctx.llm.call(
                    stage_idx=ctx.stage_idx,
                    input_hash=self.input_hash(doc),
                    model=ctx.model,
                    messages=messages,
                    params=params,
                    scope="generate",
                )
            except Exception as e:  # provider failure after its own retries
                last_error = f"{type(e).__name__}: {e}"
                break

            doc.lineage.append(ctx.record("generate", resp, prompt_hash, params))
            try:
                doc.turns = parse_turns(resp.content)
                doc.error = ""
                return doc
            except ParseError as e:
                last_error = f"ParseError: {e}"
                repair = (
                    "Your previous response could not be parsed as JSON "
                    f"({e}). Return ONLY the JSON object, with no prose before or "
                    "after it, and with all newlines inside strings escaped."
                )

        doc.error = last_error or "generation failed"
        return doc

    def input_hash(self, doc: Document) -> str:
        """Cache input key: the condition plus the plan, since the plan drives output."""
        return stable_hash({"scenario": doc.scenario.scenario_hash, "plan": doc.plan})


def build_generator(doc: Document, ctx: GenerationContext):
    """Resolve the generator for a document's doc type.

    A doc type registered under kind "doc_type" wins; otherwise any doc type declared in
    control/prompts/doc_types.yaml is served by PromptedGenerator.

    Args:
        doc: The document whose doc_type is being resolved.
        ctx: Generation context passed to the generator constructor.

    Returns:
        A generator instance exposing .generate(document).
    """
    doc_type = doc.scenario.doc_type
    if has("doc_type", doc_type):
        return resolve("doc_type", doc_type)(ctx)
    loader.doc_type_instructions(doc_type)  # raises PromptError if undeclared
    return PromptedGenerator(ctx)


def build_strategy(ctx: GenerationContext):
    """Resolve the configured generation strategy.

    Args:
        ctx: Generation context carrying the strategy name.

    Returns:
        A strategy instance exposing .run(document).
    """
    from .strategies import SinglePass  # noqa: F401  - ensures strategies are registered

    return resolve("strategy", ctx.strategy)(ctx)
