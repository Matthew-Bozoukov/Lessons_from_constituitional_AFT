# ABOUTME: Generators render a ScenarioSpec into a Document. All doc-type variation
# ABOUTME: lives in control/prompts/doc_types.yaml, so a new doc type needs no Python.

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
    """Everything a generator needs that is not the scenario itself.

    Attributes:
        llm: Cache-wrapped LLM client.
        model: Generator model id (the model ablation axis).
        params: Sampling params.
        template: Template version from control/prompts/generation.yaml.
        run_id: Run identifier, part of doc_id.
        stage_idx: Stage index for provenance and cache keying.
        stage_name: Stage name for provenance.
        max_parse_retries: Repair attempts when output is not valid JSON.
    """

    llm: CachedLLM
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    template: str = "v2"
    run_id: str = "run"
    stage_idx: int = 0
    stage_name: str = "stage_00_generated"
    max_parse_retries: int = 1


class PromptedGenerator:
    """The only generator most doc types need.

    Renders the configured generation template with the scenario's chunks, doc-type
    instructions, and axis fragments, then parses the response into Turns. Doc types
    differ only by the text they contribute, never by a code path here.
    """

    def __init__(self, ctx: GenerationContext) -> None:
        """Initialize with a generation context."""
        self.ctx = ctx

    def build_messages(self, scenario: ScenarioSpec, repair: str = "") -> list[dict]:
        """Render the prompt for one scenario.

        Args:
            scenario: The condition to render.
            repair: Optional appended instruction used on a parse retry.

        Returns:
            OpenAI-style messages.
        """
        tpl = loader.entry("generation", self.ctx.template)
        variables = scenario_vars(scenario)
        user = loader.render(tpl["user"], **variables)
        if repair:
            user = f"{user}\n\n{repair}"
        return [
            {"role": "system", "content": loader.render(tpl["system"], **variables)},
            {"role": "user", "content": user},
        ]

    def generate(self, scenario: ScenarioSpec) -> Document:
        """Generate one document.

        A parse failure is recorded on the Document rather than raised, so the row
        survives into the snapshot and the failure rate is inspectable instead of
        being a hole in the corpus.

        Args:
            scenario: The condition to realize.

        Returns:
            A Document, possibly carrying an error and no turns.
        """
        ctx = self.ctx
        doc = Document(
            doc_id=make_doc_id(scenario.scenario_hash, ctx.run_id),
            scenario=scenario,
            stage_idx=ctx.stage_idx,
            stage_name=ctx.stage_name,
        )
        doc.input_doc_id = doc.doc_id

        repair = ""
        last_error = ""
        for attempt in range(ctx.max_parse_retries + 1):
            messages = self.build_messages(scenario, repair)
            params = dict(ctx.params)
            if attempt:
                # Perturb the cache key so a retry is not served the failed response.
                params["seed"] = scenario.seed + attempt
            try:
                resp, prompt_hash = ctx.llm.call(
                    stage_idx=ctx.stage_idx,
                    input_hash=scenario.scenario_hash,
                    model=ctx.model,
                    messages=messages,
                    params=params,
                )
            except Exception as e:  # provider failure after its own retries
                last_error = f"{type(e).__name__}: {e}"
                break

            doc.lineage.append(
                StageRecord(
                    stage_idx=ctx.stage_idx,
                    stage_name=ctx.stage_name,
                    kind="generate",
                    model=ctx.model,
                    prompt_hash=prompt_hash,
                    params=stable_hash(params, 12),
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    cost_usd=ctx.llm.cost(resp),
                    cached=resp.cached,
                )
            )
            try:
                doc.turns = parse_turns(resp.content)
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


def build_generator(scenario: ScenarioSpec, ctx: GenerationContext):
    """Resolve the generator for a scenario's doc type.

    A doc type registered under kind "doc_type" wins; otherwise any doc type
    declared in control/prompts/doc_types.yaml is served by PromptedGenerator.

    Args:
        scenario: The scenario whose doc_type is being resolved.
        ctx: Generation context passed to the generator constructor.

    Returns:
        A generator instance exposing .generate(scenario).
    """
    if has("doc_type", scenario.doc_type):
        return resolve("doc_type", scenario.doc_type)(ctx)
    loader.doc_type_instructions(scenario.doc_type)  # raises PromptError if undeclared
    return PromptedGenerator(ctx)
