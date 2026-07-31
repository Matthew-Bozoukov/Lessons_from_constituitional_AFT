# ABOUTME: The binary rubric judge. Everything that differs between axes is data in rubrics.yaml.
# ABOUTME: Blinded by construction: recipe, step and model id are never passed in, so they cannot leak.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..control import loader
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, coerce_score, parse_verdict
from ..core.types import ClauseSet, Completion, Item, Verdict


@dataclass
class JudgeConfig:
    """Judge-model settings.

    Attributes:
        model: Judge model id.
        temperature: 0 by default so verdicts are stable.
        max_tokens: Completion cap; binary verdicts need very little.
        max_parse_retries: Re-asks after malformed JSON before giving up on the item.
    """

    model: str = "google/gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 500
    max_parse_retries: int = 2

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> JudgeConfig:
        """Build a JudgeConfig from the `judge` block of a resolved config."""
        block = dict(cfg.get("judge") or {})
        return cls(
            model=str(block.get("model", "google/gemini-2.5-flash")),
            temperature=float(block.get("temperature", 0.0)),
            max_tokens=int(block.get("max_tokens", 500)),
            max_parse_retries=int(block.get("max_parse_retries", 2)),
        )


def _knows_context(item: Item, clauses: ClauseSet) -> dict[str, Any]:
    """Give the judge the FULL clause list and the target's position in it.

    This is the fix for the worst metric in v1. Scoring `knows` as similarity to one clause made
    judges disagree 41% of the time, always about how close was close enough. Presenting every
    principle turns it into a matching task with exactly one right answer.
    """
    ordered = list(clauses.clauses)
    target = clauses.find(item.clause_id)
    return {
        "all_clauses": ordered,
        "target_index": ordered.index(target) + 1 if target in ordered else 0,
    }


def _discriminates_context(item: Item, clauses: ClauseSet) -> dict[str, Any]:
    """Pass the probe's ground truth through; it comes off the item, never the response."""
    return {
        "is_real": bool(item.meta.get("is_real", False)),
        "candidate_text": str(item.meta.get("candidate_text", "")),
        "why_fake": str(item.meta.get("why_fake", "")),
    }


# Axis -> callable(item, clauses) -> extra template variables. Only the two axes that genuinely
# need more than (clause, item, response) appear here.
EXTRA_CONTEXT = {"knows": _knows_context, "discriminates": _discriminates_context}


class RubricJudge:
    """Grades one binary axis from its declared rubric.

    One class rather than one subclass per axis: v1 had eight subclasses of which six contained
    nothing but `axis = "name"`. The two axes needing extra template context declare it in
    EXTRA_CONTEXT instead.

    Blinded by construction - recipe, checkpoint step and model id are not parameters of this class,
    so they cannot reach the prompt.
    """

    def __init__(self, axis: str, clauses: ClauseSet) -> None:
        """Initialize for one axis.

        Args:
            axis: Rubric axis name, declared in control/prompts/rubrics.yaml.
            clauses: The clause set the rubric is graded against.

        Raises:
            ValueError: If the rubric is not binary. Every axis is a rate now, and a graded one
                would silently break the analysis layer's rate arithmetic.
        """
        self.axis = axis
        self.clauses = clauses
        self.rubric = loader.rubric(axis)
        if float(self.rubric["scale_max"]) != 1.0:
            raise ValueError(
                f"Rubric {axis!r} has scale_max {self.rubric['scale_max']}; every axis must be "
                f"binary (scale_max: 1). Graded scales were unreliable and unused in v1."
            )

    def context(self, item: Item, response: str) -> dict[str, Any]:
        """Build the template context for one grading call.

        Args:
            item: The item graded.
            response: The response text.

        Returns:
            Template variables.
        """
        base: dict[str, Any] = {
            "clause": self.clauses.find(item.clause_id),
            "item": item,
            "response": response,
        }
        extra = EXTRA_CONTEXT.get(self.axis)
        if extra is not None:
            base.update(extra(item, self.clauses))
        return base

    def __call__(
        self, item: Item, completion: Completion, llm: CachedLLM, config: JudgeConfig
    ) -> Verdict:
        """Grade one completion on this axis.

        A generation failure becomes an errored verdict rather than a zero: a model that timed out
        did not fail the axis, and scoring it as a failure biases every rate that includes it.

        Args:
            item: The item graded.
            completion: The model's response.
            llm: Cached judge client.
            config: Judge model settings.

        Returns:
            The Verdict. `error` is non-empty when grading could not be completed.
        """
        if not completion.ok:
            return Verdict(
                item_id=item.item_id,
                axis=self.axis,
                score=0.0,
                judge_model=config.model,
                error=completion.error or "empty completion",
            )

        pack = loader.pack("rubrics")
        prompt = loader.render(
            self.rubric["template"], **self.context(item, completion.text.strip())
        )
        messages = [
            {"role": "system", "content": pack["system"]},
            {"role": "user", "content": prompt},
        ]
        params = {"temperature": config.temperature, "max_tokens": config.max_tokens}

        last: Exception | None = None
        for attempt in range(max(1, config.max_parse_retries)):
            msgs = messages if attempt == 0 else [
                *messages,
                {"role": "user", "content": "Return only the JSON object described above."},
            ]
            try:
                resp = llm.call(scope="judge", model=config.model, messages=msgs, params=params)
                payload = parse_verdict(resp.content, list(self.rubric["fields"]))
                raw = coerce_score(payload["score"], 1.0)
            except (ParseError, RuntimeError) as e:
                last = e
                continue
            return Verdict(
                item_id=item.item_id,
                axis=self.axis,
                score=raw,
                raw_score=raw,
                passed=raw >= 1.0,
                rationale=str(payload.get("rationale", ""))[:400],
                judge_model=config.model,
            )
        return Verdict(
            item_id=item.item_id,
            axis=self.axis,
            score=0.0,
            judge_model=config.model,
            error=f"judge failed after {config.max_parse_retries} attempts: {last}",
        )


def build_judges(clauses: ClauseSet, axes: list[str] | None = None) -> dict[str, RubricJudge]:
    """Instantiate one judge per declared axis.

    Args:
        clauses: The clause set the rubrics grade against.
        axes: Axes to build; defaults to every axis declared in rubrics.yaml.

    Returns:
        Mapping of axis name to judge.
    """
    return {axis: RubricJudge(axis, clauses) for axis in (axes or loader.declared_axes())}
