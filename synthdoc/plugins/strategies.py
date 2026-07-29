# ABOUTME: Generation strategies: how many model calls produce one document, and in what
# ABOUTME: order. Registered plugins, so `generation.strategy` is a one-line ablation.

from __future__ import annotations

from typing import Any

from ..control import loader
from ..core.hashing import stable_hash
from ..core.parsing import ParseError, extract_json, parse_turns, render_document
from ..core.prompting import scenario_vars
from ..core.registry import register
from ..core.types import Document
from .generators import GenerationContext, build_generator


class BaseStrategy:
    """Turns a seeded document into a finished one, using one or more model calls."""

    name = "base"

    def __init__(self, ctx: GenerationContext) -> None:
        """Initialize with a generation context."""
        self.ctx = ctx

    def run(self, doc: Document) -> Document:  # pragma: no cover - abstract
        """Fill in the document's turns."""
        raise NotImplementedError


@register("strategy", "single_pass")
class SinglePass(BaseStrategy):
    """One call per document. The control arm, and what the pipeline did originally."""

    name = "single_pass"

    def run(self, doc: Document) -> Document:
        """Generate the document in a single call."""
        return build_generator(doc, self.ctx).generate(doc)


@register("strategy", "draft_then_align")
class DraftThenAlign(BaseStrategy):
    """Draft an answer, then refine it toward the excerpt in a separate context.

    GDM's two-phase response generation. What the DRAFT sees is `draft_context`:

    - "spec_in_system" (default) is faithful to their description - the trait sits in
      the drafting model's system prompt, and the refinement pass moves the answer
      closer to the chunk "in a realistic, non-performative way".
    - "no_spec" drafts blind. Not what GDM describe; the hypothesis is that a draft
      written without the spec carries a more natural voice for the align pass to keep.

    Those two are the point of the strategy being a config field rather than a fixed
    recipe: the difference between them is measurable.

    Params:
        draft_context: spec_in_system | no_spec.
        draft_model: Model for the draft; defaults to the generator's.
        draft_temperature: Sampling temperature for the draft.
    """

    name = "draft_then_align"
    DRAFT_CONTEXTS = {"spec_in_system": "draft_spec_in_system", "no_spec": "draft_no_spec"}

    def run(self, doc: Document) -> Document:
        """Draft, then align, then parse into turns."""
        ctx = self.ctx
        entry = loader.entry("strategies", "draft_then_align")
        draft_context = str(ctx.strategy_params.get("draft_context", "spec_in_system"))
        if draft_context not in self.DRAFT_CONTEXTS:
            doc.error = (
                f"draft_then_align: unknown draft_context {draft_context!r}; "
                f"expected one of {sorted(self.DRAFT_CONTEXTS)}"
            )
            return doc
        draft_entry = entry[self.DRAFT_CONTEXTS[draft_context]]

        user_prompt = (doc.plan or {}).get("user_prompt", "").strip()
        if not user_prompt:
            # Without a plan there is no user turn to draft against. Fall back to a
            # single pass rather than inventing one, and say so in the lineage.
            doc = build_generator(doc, ctx).generate(doc)
            if doc.lineage:
                doc.lineage[-1].kind = "generate(no-plan-fallback)"
            return doc

        params = dict(ctx.params)
        params["temperature"] = float(
            ctx.strategy_params.get("draft_temperature", params.get("temperature", 1.0))
        )
        draft_model = ctx.strategy_params.get("draft_model") or ctx.model

        draft_vars = scenario_vars(doc.scenario, plan=doc.plan or {}, user_prompt=user_prompt)
        draft_messages = [
            {"role": "system", "content": loader.render(draft_entry["system"], **draft_vars)},
            {"role": "user", "content": loader.render(draft_entry["user"], **draft_vars)},
        ]
        try:
            resp, prompt_hash = ctx.llm.call(
                stage_idx=ctx.stage_idx,
                input_hash=stable_hash(
                    {"prompt": user_prompt, "phase": "draft", "context": draft_context}
                ),
                model=draft_model,
                messages=draft_messages,
                params=params,
                scope="generate",
            )
        except Exception as e:
            doc.error = f"draft: {type(e).__name__}: {e}"
            return doc
        doc.lineage.append(ctx.record(f"draft:{draft_context}", resp, prompt_hash, params))
        draft = resp.content

        variables = scenario_vars(doc.scenario, plan=doc.plan or {}, draft=draft,
                                  user_prompt=user_prompt)
        align_messages = [
            {"role": "system", "content": loader.render(entry["align"]["system"], **variables)},
            {"role": "user", "content": loader.render(entry["align"]["user"], **variables)},
        ]
        try:
            resp, prompt_hash = ctx.llm.call(
                stage_idx=ctx.stage_idx,
                input_hash=stable_hash({"draft": draft, "scenario": doc.scenario.scenario_hash}),
                model=ctx.model,
                messages=align_messages,
                params=ctx.params,
                scope="generate",
            )
        except Exception as e:
            doc.error = f"align: {type(e).__name__}: {e}"
            return doc
        doc.lineage.append(ctx.record("align", resp, prompt_hash, ctx.params))

        try:
            doc.turns = parse_turns(resp.content)
        except ParseError as e:
            doc.error = f"align: ParseError: {e}"
        return doc


@register("strategy", "best_of_n")
class BestOfN(BaseStrategy):
    """Sample n documents for the same condition and keep the best.

    Anthropic's sample-and-filter, generalised: they sampled responses and kept the ones
    where the assistant behaved correctly. Selection is on spec fidelity rather than
    polish, because picking the smoothest candidate optimises for the wrong thing.

    Params:
        n: Candidates to generate.
        selector: "judge" (an LLM picks) or "first_ok" (cheap control arm).
        selector_model: Model for the judge; defaults to the generator's.
    """

    name = "best_of_n"

    def run(self, doc: Document) -> Document:
        """Generate n candidates, then select one."""
        ctx = self.ctx
        n = max(1, int(ctx.strategy_params.get("n", 4)))
        generator = build_generator(doc, ctx)

        base_lineage = list(doc.lineage)
        candidates: list[Document] = []
        for i in range(n):
            candidate = doc.advanced(ctx.stage_idx, ctx.stage_name)
            candidate.lineage = []
            # Distinct params per candidate, or the cache would return one document n times.
            candidate_ctx_params = dict(ctx.params)
            candidate_ctx_params["seed"] = doc.scenario.seed + 1000 * (i + 1)
            saved, generator.ctx.params = generator.ctx.params, candidate_ctx_params
            try:
                candidates.append(generator.generate(candidate))
            finally:
                generator.ctx.params = saved

        usable = [c for c in candidates if c.ok]
        if not usable:
            failed = candidates[0] if candidates else doc
            failed.lineage = base_lineage + failed.lineage
            failed.error = failed.error or "best_of_n: no candidate parsed"
            return failed

        chosen = usable[0]
        select_record = None
        if len(usable) > 1 and ctx.strategy_params.get("selector", "judge") == "judge":
            chosen, select_record = self._judge(doc, usable)

        # Every candidate's records are kept, tagged by whether it was selected.
        # Dropping the discarded ones would under-report this strategy's cost by a
        # factor of n - exactly the number the strategy sweep exists to weigh.
        doc.lineage = list(base_lineage)
        for i, candidate in enumerate(candidates):
            selected = candidate is chosen
            for record in candidate.lineage:
                if record.kind == "generate":
                    record.kind = f"generate:cand{i}" + ("" if selected else "(discarded)")
                doc.lineage.append(record)
        if select_record is not None:
            doc.lineage.append(select_record)

        doc.turns = chosen.turns
        doc.error = ""
        return doc

    def _judge(self, doc: Document, candidates: list[Document]):
        """Ask a model which candidate best follows the excerpt.

        Returns:
            Tuple of (chosen document, selection lineage record or None).
        """
        ctx = self.ctx
        entry = loader.entry("strategies", "best_of_n")
        rendered = [render_document(c.turns) for c in candidates]
        variables = scenario_vars(doc.scenario, candidates=rendered)
        messages = [
            {"role": "system", "content": loader.render(entry["selector"]["system"], **variables)},
            {"role": "user", "content": loader.render(entry["selector"]["user"], **variables)},
        ]
        params = {"temperature": 0.0, "max_tokens": 300}
        try:
            resp, prompt_hash = ctx.llm.call(
                stage_idx=ctx.stage_idx,
                input_hash=stable_hash([stable_hash(r) for r in rendered]),
                model=ctx.strategy_params.get("selector_model") or ctx.model,
                messages=messages,
                params=params,
                scope="generate",
            )
        except Exception:
            return candidates[0], None

        chosen = candidates[0]
        try:
            payload = extract_json(resp.content)
            index = int(payload["choice"]) if isinstance(payload, dict) else 0
            if 0 <= index < len(candidates):
                chosen = candidates[index]
        except (ParseError, KeyError, TypeError, ValueError):
            pass
        return chosen, ctx.record("best_of_n:select", resp, prompt_hash, params)


def strategy_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract strategy-specific params from a `generation:` config block.

    Args:
        cfg: The generation config block.

    Returns:
        The `strategy_params` mapping, defaulting to empty.
    """
    return dict(cfg.get("strategy_params") or {})
