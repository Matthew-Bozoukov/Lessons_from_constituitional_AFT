# ABOUTME: The five LLM stages of the difficult-advice pipeline, one function each.
# ABOUTME: Every stage takes the previous stage's records and returns the next stage's.

from __future__ import annotations

from src.endpoints.openrouter import OpenRouterClient

from ..constitution import Trait
from ..core import Checkpoint, Usage, call_json, call_tagged, resilient, run_items
from . import prompts

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
        parsed, _ = call_json(
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

    nested = resilient(one, len(batches), workers, "stage2:scenarios")
    return [r for group in nested for r in group]


# --- stage 3 -----------------------------------------------------------------------


def draft_prompts(scenarios: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                  temperature: float, max_tokens: int, workers: int) -> list[dict]:
    """Write a first-attempt system and user prompt for each scenario."""
    def one(i: int) -> dict:
        s = scenarios[i]
        parsed, _ = call_json(
            client, usage, model,
            prompts.DRAFT_SYSTEM,
            prompts.DRAFT_USER.format(situation=s["situation"], shortcut=s["shortcut"]),
            temperature, max_tokens, stage="draft",
        )
        return {**s, "draft_system": parsed["system"], "draft_user": parsed["user"]}

    return resilient(one, len(scenarios), workers, "stage3:draft")


# --- stage 4 -----------------------------------------------------------------------


def refine_prompts(drafts: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                   constitution: str, temperature: float, max_tokens: int,
                   workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each draft prompt into a sharper test of its target trait.

    The full constitution and the specific target trait are both injected, so the model
    can tell which principle the prompt is supposed to stress.
    """
    def one(d: dict) -> dict:
        parsed, _ = call_json(
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

    return run_items(drafts, one, workers, "stage4:refine", ckpt)


# --- stage 5 -----------------------------------------------------------------------


def generate_responses(refined: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                       style_guidance: str, temperature: float, max_tokens: int,
                       workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Answer each refined prompt with explicit reasoning, steered by the target trait."""
    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": prompts.RESPONSE_SYSTEM.format(
                system=r["system"], trait_name=r["trait_name"], trait_text=r["trait_text"],
                style_guidance=style_guidance)},
             {"role": "user", "content": prompts.RESPONSE_USER.format(user=r["user"])}],
            temperature, max_tokens, "respond", ("reasoning", "response"),
        )
        return {**r, "draft_reasoning": parsed["reasoning"], "draft_response": parsed["response"]}

    return run_items(refined, one, workers, "stage5:respond", ckpt)


# --- stage 6 -----------------------------------------------------------------------


def rewrite_responses(responses: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                      constitution: str, temperature: float, max_tokens: int,
                      workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each response to maximally exhibit its target trait.

    The blog calls this the critical step: the reviewer sees the whole transcript with the
    relevant constitution section in context, then rewrites rather than scores.
    """
    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": prompts.REWRITE_SYSTEM},
             {"role": "user", "content": prompts.REWRITE_USER.format(
                 constitution=constitution, trait_name=r["trait_name"],
                 trait_text=r["trait_text"], system=r["system"], user=r["user"],
                 reasoning=r["draft_reasoning"], response=r["draft_response"])}],
            temperature, max_tokens, "rewrite", ("reasoning", "response", "changes"),
        )
        return {**r, "reasoning": parsed["reasoning"], "response": parsed["response"],
                "rewrite_changes": parsed.get("changes", "")}

    return run_items(responses, one, workers, "stage6:rewrite", ckpt)


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
