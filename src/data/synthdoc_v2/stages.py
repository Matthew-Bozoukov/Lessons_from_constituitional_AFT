# ABOUTME: The five LLM stages of the difficult-advice pipeline, one function each.
# ABOUTME: Every stage takes the previous stage's records and returns the next stage's.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openrouter import ChatResult, OpenRouterClient, map_threaded  # noqa: E402
from utils import extract_json  # noqa: E402

from . import prompts  # noqa: E402
from .constitution import Trait  # noqa: E402

# USD per 1M tokens, OpenRouter list prices.
PRICES: dict[str, dict[str, float]] = {
    "openai/gpt-5.6-luna": {"in": 0.10, "out": 0.60},
    "openai/gpt-5.6-terra": {"in": 1.00, "out": 6.00},
    "openai/gpt-5.6-sol": {"in": 5.00, "out": 30.00},
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the USD cost of one call, or 0.0 for an unpriced model."""
    p = PRICES.get(model)
    if not p:
        return 0.0
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


class Usage:
    """Running token and cost totals, tallied per model and per stage.

    Per-stage tallies matter for cost estimation: the stages differ by an order of
    magnitude in tokens per call, so a per-model average would misprice most of them.
    """

    def __init__(self) -> None:
        """Start an empty tally."""
        self.by_model: dict[str, dict[str, float]] = {}
        self.by_stage: dict[str, dict[str, float]] = {}

    def add(self, model: str, res: ChatResult, stage: str = "") -> None:
        """Record one completion against its model and stage."""
        usd = cost_of(model, res.prompt_tokens, res.completion_tokens)
        for key, bucket in ((model, self.by_model), (stage or "unknown", self.by_stage)):
            b = bucket.setdefault(
                key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
            )
            b["calls"] += 1
            b["prompt_tokens"] += res.prompt_tokens
            b["completion_tokens"] += res.completion_tokens
            b["usd"] += usd

    @property
    def usd(self) -> float:
        """Total spend so far."""
        return sum(b["usd"] for b in self.by_model.values())

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {"by_model": self.by_model, "by_stage": self.by_stage,
                "total_usd": round(self.usd, 4)}


def _call(client: OpenRouterClient, usage: Usage, model: str, system: str, user: str,
          temperature: float, max_tokens: int, stage: str) -> tuple[Any, ChatResult]:
    """Run one chat completion and parse its JSON body.

    Args:
        client: OpenRouter client.
        usage: Tally to record the call against.
        model: OpenRouter model id.
        system: System message.
        user: User message.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        stage: Stage name, for per-stage accounting.

    Returns:
        (parsed JSON, raw ChatResult).

    Raises:
        AssertionError: If the model hit the token cap, which truncates the JSON body
            and would otherwise surface as a confusing parse error.
    """
    res = client.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage.add(model, res, stage)
    assert res.finish_reason != "length", (
        f"{stage}: {model} hit max_tokens={max_tokens} and truncated its JSON. "
        f"Raise max_tokens for this stage, or lower scenarios_per_call."
    )
    return extract_json(res.content), res


# --- stage 2 -----------------------------------------------------------------------


def generate_scenarios(traits: list[Trait], client: OpenRouterClient, usage: Usage,
                       model: str, per_trait: int, per_call: int, temperature: float,
                       max_tokens: int, workers: int) -> list[dict]:
    """Generate difficult situations for each trait.

    Scenarios are requested in batches rather than all at once: a single call asking for
    40 situations would exceed any sane `max_tokens` and truncate its JSON. Batching also
    improves diversity, since each call is told to vary domains only within its own batch.

    Args:
        traits: The segmented constitution.
        client: OpenRouter client.
        usage: Tally.
        model: Model id.
        per_trait: Scenarios requested per trait, in total.
        per_call: Scenarios requested per API call.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        workers: Thread pool size.

    Returns:
        One record per scenario.
    """
    # (trait index, batch index, how many this batch asks for)
    batches: list[tuple[int, int, int]] = []
    for ti in range(len(traits)):
        remaining = per_trait
        bi = 0
        while remaining > 0:
            n = min(per_call, remaining)
            batches.append((ti, bi, n))
            remaining -= n
            bi += 1

    def one(k: int) -> list[dict]:
        ti, bi, n = batches[k]
        t = traits[ti]
        parsed, _ = _call(
            client, usage, model,
            prompts.SCENARIO_SYSTEM,
            prompts.SCENARIO_USER.format(trait_name=t.name, trait_text=t.text, n=n),
            temperature, max_tokens, stage="scenarios",
        )
        assert isinstance(parsed, list), f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
        return [{
            "scenario_id": f"{t.trait_id}_b{bi:02d}_s{j:03d}",
            "trait_id": t.trait_id,
            "trait_name": t.name,
            "trait_text": t.text,
            "domain": s.get("domain", ""),
            "situation": s["situation"],
            "shortcut": s.get("shortcut", ""),
        } for j, s in enumerate(parsed)]

    nested = map_threaded(one, len(batches), max_workers=workers, desc="stage2:scenarios")
    return [r for group in nested for r in group]


# --- stage 3 -----------------------------------------------------------------------


def draft_prompts(scenarios: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                  temperature: float, max_tokens: int, workers: int) -> list[dict]:
    """Write a first-attempt system and user prompt for each scenario."""
    def one(i: int) -> dict:
        s = scenarios[i]
        parsed, _ = _call(
            client, usage, model,
            prompts.DRAFT_SYSTEM,
            prompts.DRAFT_USER.format(situation=s["situation"], shortcut=s["shortcut"]),
            temperature, max_tokens, stage="draft",
        )
        return {**s, "draft_system": parsed["system"], "draft_user": parsed["user"]}

    return map_threaded(one, len(scenarios), max_workers=workers, desc="stage3:draft")


# --- stage 4 -----------------------------------------------------------------------


def refine_prompts(drafts: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                   constitution: str, temperature: float, max_tokens: int,
                   workers: int) -> list[dict]:
    """Rewrite each draft prompt into a sharper test of its target trait.

    The full constitution and the specific target trait are both injected, so the model
    can tell which principle the prompt is supposed to stress.
    """
    def one(i: int) -> dict:
        d = drafts[i]
        parsed, _ = _call(
            client, usage, model,
            prompts.REFINE_SYSTEM,
            prompts.REFINE_USER.format(
                constitution=constitution, trait_name=d["trait_name"],
                trait_text=d["trait_text"], draft_system=d["draft_system"],
                draft_user=d["draft_user"],
            ),
            temperature, max_tokens, stage="refine",
        )
        return {**d, "system": parsed["system"], "user": parsed["user"],
                "refine_changes": parsed.get("changes", "")}

    return map_threaded(one, len(drafts), max_workers=workers, desc="stage4:refine")


# --- stage 5 -----------------------------------------------------------------------


def generate_responses(refined: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                       style_guidance: str, temperature: float, max_tokens: int,
                       workers: int) -> list[dict]:
    """Answer each refined prompt with explicit reasoning, steered by the target trait."""
    def one(i: int) -> dict:
        r = refined[i]
        parsed, _ = _call(
            client, usage, model,
            prompts.RESPONSE_SYSTEM.format(
                system=r["system"], trait_name=r["trait_name"], trait_text=r["trait_text"],
                style_guidance=style_guidance,
            ),
            prompts.RESPONSE_USER.format(user=r["user"]),
            temperature, max_tokens, stage="respond",
        )
        return {**r, "draft_reasoning": parsed["reasoning"], "draft_response": parsed["response"]}

    return map_threaded(one, len(refined), max_workers=workers, desc="stage5:respond")


# --- stage 6 -----------------------------------------------------------------------


def rewrite_responses(responses: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                      constitution: str, temperature: float, max_tokens: int,
                      workers: int) -> list[dict]:
    """Rewrite each response to maximally exhibit its target trait.

    The blog calls this the critical step: the reviewer sees the whole transcript with the
    relevant constitution section in context, then rewrites rather than scores.
    """
    def one(i: int) -> dict:
        r = responses[i]
        parsed, _ = _call(
            client, usage, model,
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER.format(
                constitution=constitution, trait_name=r["trait_name"],
                trait_text=r["trait_text"], system=r["system"], user=r["user"],
                reasoning=r["draft_reasoning"], response=r["draft_response"],
            ),
            temperature, max_tokens, stage="rewrite",
        )
        return {**r, "reasoning": parsed["reasoning"], "response": parsed["response"],
                "rewrite_changes": parsed.get("changes", "")}

    return map_threaded(one, len(responses), max_workers=workers, desc="stage6:rewrite")


def to_sft(records: list[dict]) -> list[dict]:
    """Convert final records into chat form with the trait carried in metadata.

    Args:
        records: Stage-6 output.

    Returns:
        One `{messages, metadata}` record each, assistant turn carrying `reasoning_content`.
    """
    out = []
    for r in records:
        out.append({
            "messages": [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["response"],
                 "reasoning_content": r["reasoning"]},
            ],
            "metadata": {
                "scenario_id": r["scenario_id"],
                "trait_id": r["trait_id"],
                "trait_name": r["trait_name"],
                "trait_text": r["trait_text"],
                "domain": r.get("domain", ""),
                "shortcut": r.get("shortcut", ""),
                "situation": r["situation"],
            },
        })
    return out
