# ABOUTME: The single generation pass. Every Tier A judge scores these completions, so the
# ABOUTME: model is queried once per item no matter how many axes are being measured.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..control import loader
from ..core.cache import CallCache, CacheConfig
from ..core.llm import CachedLLM, PriceTable, build_client, map_threaded
from ..core.parsing import split_thinking
from ..core.types import Completion, Item


@dataclass
class TargetConfig:
    """Settings for the model under test.

    Attributes:
        provider: Registered llm provider name.
        model: Model id served by that provider.
        base_url: Endpoint override, for a locally served checkpoint.
        temperature: Sampling temperature.
        max_tokens: Completion cap. Must be generous for a thinking model: a cap that
            truncates the reasoning trace produces an empty answer that would otherwise
            be scored as a refusal.
        system: Default system prompt for items that declare none.
        enable_thinking: Toggles the chat template's thinking mode on vLLM.
        justification_followup: Ask application items a second turn about why the
            constraint exists. Off by default so the suite stays a single pass.
        extra_body: Extra provider body fields, merged after enable_thinking.
        client: Provider constructor kwargs. The `hf` provider takes `adapter`,
            `device`, `dtype`, and `trust_remote_code` here.
    """

    provider: str = "vllm"
    model: str = "qwen3"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    system: str = ""
    enable_thinking: bool = True
    justification_followup: bool = False
    extra_body: dict[str, Any] | None = None
    client: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TargetConfig:
        """Build a TargetConfig from the `target` block of a resolved config."""
        block = dict(cfg.get("target") or {})
        return cls(
            provider=str(block.get("provider", "vllm")),
            model=str(block.get("model", "qwen3")),
            base_url=block.get("base_url") or None,
            temperature=float(block.get("temperature", 0.7)),
            max_tokens=int(block.get("max_tokens", 2048)),
            system=str(block.get("system", "")),
            enable_thinking=bool(block.get("enable_thinking", True)),
            justification_followup=bool(block.get("justification_followup", False)),
            extra_body=dict(block.get("extra_body") or {}),
            client=dict(block.get("client") or {}),
        )

    def params(self) -> dict[str, Any]:
        """Return the sampling params for one call.

        Thinking is toggled differently depending on who renders the chat template.
        vLLM and in-process transformers take `enable_thinking` as a chat-template kwarg;
        a hosted API renders the template itself and expects its own reasoning parameter,
        so sending the kwarg there is at best ignored and at worst a 400. Providers that
        do not consume it therefore never receive it, and set their own switch through
        `extra_body` in the config.
        """
        extra: dict[str, Any] = {}
        if self.provider in ("vllm", "hf"):
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        extra.update(self.extra_body or {})
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": extra,
        }


def build_target(cfg: dict[str, Any], cache: CallCache | None = None) -> CachedLLM:
    """Instantiate the cached client for the model under test.

    Args:
        cfg: Resolved run config.
        cache: Cache to share; a fresh one is built from the config when omitted.

    Returns:
        A CachedLLM.
    """
    target = TargetConfig.from_config(cfg)
    kwargs: dict[str, Any] = dict(target.client or {})
    if target.base_url:
        kwargs["base_url"] = target.base_url
    # In-process providers need the weights named at construction, not per call.
    kwargs.setdefault("model", target.model)
    if target.provider not in ("hf",):
        kwargs.pop("model", None)
    return CachedLLM(
        inner=build_client(target.provider, **kwargs),
        cache=cache or CallCache(CacheConfig.from_config(cfg)),
        prices=PriceTable(cfg.get("pricing") or {}),
    )


def generate(
    items: Sequence[Item],
    llm: CachedLLM,
    target: TargetConfig,
    max_workers: int = 16,
    desc: str = "generate",
) -> dict[str, Completion]:
    """Run one completion per item.

    A per-item failure is captured on the Completion rather than raised: one flaky item
    should not discard a pass over a thousand. Judges refuse to score an errored
    completion, and the analysis layer reports the error rate alongside every aggregate.

    Args:
        items: Items to answer.
        llm: Cached target client.
        target: Target settings.
        max_workers: Concurrency.
        desc: Progress bar description.

    Returns:
        item_id -> Completion.
    """
    params = target.params()

    def answer(item: Item) -> Completion:
        """Answer one item, folding any failure onto the Completion."""
        try:
            resp = llm.call(
                scope="generate",
                model=target.model,
                messages=item.messages(default_system=target.system),
                params=params,
            )
        except Exception as e:  # noqa: BLE001 - any provider failure is per-item data
            return Completion(item_id=item.item_id, text="", model=target.model, error=repr(e))
        # Reasoning arrives either in a provider field or inline in <think> tags,
        # depending on how the checkpoint is served. Both are normalised here so the
        # judge only ever sees the answer.
        inline_thinking, answer_text = split_thinking(resp.content)
        return Completion(
            item_id=item.item_id,
            text=answer_text,
            thinking=resp.reasoning or inline_thinking,
            model=resp.model or target.model,
            finish_reason=resp.finish_reason,
        )

    completions = map_threaded(answer, list(items), max_workers=max_workers, desc=desc)
    return {c.item_id: c for c in completions}


def generate_followups(
    items: Sequence[Item],
    completions: dict[str, Completion],
    llm: CachedLLM,
    target: TargetConfig,
    max_workers: int = 16,
) -> dict[str, Completion]:
    """Ask a second turn about why the constraint applied exists.

    Optional. It measures justification more directly than reading a reason out of an
    answer that was not asked for one, at the cost of a second pass over application
    items - which is why the suite does not do it by default.

    Args:
        items: Application items to follow up on.
        completions: The first-pass completions, used as conversation history.
        llm: Cached target client.
        target: Target settings.
        max_workers: Concurrency.

    Returns:
        item_id -> follow-up Completion, for items whose first pass succeeded.
    """
    followup_prompt = loader.pack("items")["templates"]["justification_followup"]
    params = target.params()
    eligible = [i for i in items if (completions.get(i.item_id) or Completion("", "")).ok]

    def ask(item: Item) -> Completion:
        """Ask the follow-up as a third turn on the item's own conversation."""
        first = completions[item.item_id]
        messages = [
            *item.messages(default_system=target.system),
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": followup_prompt},
        ]
        try:
            resp = llm.call(
                scope="generate", model=target.model, messages=messages, params=params
            )
        except Exception as e:  # noqa: BLE001 - a missing follow-up degrades, never fails
            return Completion(item_id=item.item_id, text="", model=target.model, error=repr(e))
        _, answer_text = split_thinking(resp.content)
        return Completion(
            item_id=item.item_id,
            text=answer_text,
            model=resp.model or target.model,
            finish_reason=resp.finish_reason,
        )

    results = map_threaded(ask, eligible, max_workers=max_workers, desc="followup")
    return {c.item_id: c for c in results if c.ok}
