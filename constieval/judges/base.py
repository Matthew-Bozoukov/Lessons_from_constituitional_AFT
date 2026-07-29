# ABOUTME: The rubric judge: renders a rubric with the clause text always in context,
# ABOUTME: calls a blinded judge model, and normalises the verdict onto a common scale.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..control import loader
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, coerce_score, parse_verdict
from ..core.types import ClauseSet, Completion, Item, Verdict

# Marker for a follow-up probe appended to the graded response. Kept out of the rubric
# text so the judge sees the same framing whether or not a follow-up pass ran.
FOLLOWUP_HEADER = "[Follow-up - the assistant was then asked why that constraint exists:]"


@dataclass
class JudgeConfig:
    """Judge-model settings.

    Attributes:
        model: Judge model id.
        temperature: Sampling temperature; 0 by default so verdicts are stable.
        max_tokens: Completion cap.
        max_parse_retries: Re-asks after malformed JSON before giving up on the item.
    """

    model: str = "anthropic/claude-sonnet-4.5"
    temperature: float = 0.0
    max_tokens: int = 900
    max_parse_retries: int = 2

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> JudgeConfig:
        """Build a JudgeConfig from the `judge` block of a resolved config."""
        block = dict(cfg.get("judge") or {})
        return cls(
            model=str(block.get("model", "anthropic/claude-sonnet-4.5")),
            temperature=float(block.get("temperature", 0.0)),
            max_tokens=int(block.get("max_tokens", 900)),
            max_parse_retries=int(block.get("max_parse_retries", 2)),
        )


class RubricJudge:
    """Grades one axis from its declared rubric.

    Everything that differs between axes - the prompt, the scale, the pass threshold,
    which families it applies to - is declared in `control/prompts/rubrics.yaml`. A
    subclass exists only when an axis needs extra template context, never to change how
    grading works.

    The judge is blinded by construction: the only things placed in its context are the
    clause, the item, and the response text. Recipe, checkpoint step, and model id are
    not passed in and cannot be, because this class never receives them.
    """

    axis: str = ""

    def __init__(self, clauses: ClauseSet) -> None:
        """Initialize with the clause set the rubric is graded against."""
        self.clauses = clauses
        self.rubric = loader.rubric(self.axis)

    @property
    def scale_max(self) -> float:
        """Top of this axis's rubric scale."""
        return float(self.rubric["scale_max"])

    @property
    def direction(self) -> str:
        """"higher_better" or "lower_better"; consumed only by the analysis layer."""
        return str(self.rubric["direction"])

    def applies_to(self, item: Item) -> bool:
        """Return True if this axis is declared for the item's family."""
        return item.family in (self.rubric.get("applies_to") or [])

    def context(self, item: Item, response: str) -> dict[str, Any]:
        """Build the template context for one grading call.

        Subclasses extend this; they never replace the clause or response entries.

        Args:
            item: The item graded.
            response: The response text.

        Returns:
            Template variables.
        """
        return {
            "clause": self.clauses.find(item.clause_id),
            "item": item,
            "response": response,
            "priority_order": self.clauses.priority_order,
            "priority_note": self.clauses.priority_note,
        }

    def passed(self, raw: float) -> bool:
        """Binarise a raw score at the rubric's threshold, honouring direction."""
        threshold = float(self.rubric["pass_at"])
        return raw >= threshold if self.direction == "higher_better" else raw <= threshold

    def build_response(self, completion: Completion, followup: Completion | None = None) -> str:
        """Assemble the text to grade.

        Args:
            completion: The main completion.
            followup: An optional follow-up turn, appended under a fixed header.

        Returns:
            The response text placed in the judge's context.
        """
        text = completion.text.strip()
        if followup is not None and followup.ok:
            return f"{text}\n\n{FOLLOWUP_HEADER}\n{followup.text.strip()}"
        return text

    def __call__(
        self,
        item: Item,
        completion: Completion,
        llm: CachedLLM,
        config: JudgeConfig,
        followup: Completion | None = None,
    ) -> Verdict:
        """Grade one completion on this axis.

        A generation failure is recorded as an errored verdict rather than a zero: a
        model that timed out did not fail the axis, and scoring it as a failure would
        quietly bias every aggregate that includes it.

        Args:
            item: The item graded.
            completion: The model's response.
            llm: Cached judge client.
            config: Judge model settings.
            followup: Optional follow-up turn to append.

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
            self.rubric["template"], **self.context(item, self.build_response(completion, followup))
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
                raw = coerce_score(payload["score"], self.scale_max)
            except (ParseError, RuntimeError) as e:
                last = e
                continue
            return Verdict(
                item_id=item.item_id,
                axis=self.axis,
                score=raw / self.scale_max if self.scale_max else 0.0,
                raw_score=raw,
                passed=self.passed(raw),
                rationale=str(payload.get("rationale", ""))[:600],
                judge_model=config.model,
            )
        return Verdict(
            item_id=item.item_id,
            axis=self.axis,
            score=0.0,
            judge_model=config.model,
            error=f"judge failed after {config.max_parse_retries} attempts: {last}",
        )
