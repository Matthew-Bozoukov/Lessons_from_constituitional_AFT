# ABOUTME: The injected LLM interface plus OpenRouter and offline implementations.
# ABOUTME: Generator-model ablation is a config line because nothing imports a concrete client.

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .hashing import stable_hash, text_hash
from .registry import register

# Default price table in USD per 1M tokens. Override or extend under `pricing:` in
# the run config; unknown models cost 0.0 and are reported in the manifest.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4.5": {"in": 3.0, "out": 15.0},
    "anthropic/claude-opus-4.1": {"in": 15.0, "out": 75.0},
    "anthropic/claude-haiku-4.5": {"in": 1.0, "out": 5.0},
    "openai/gpt-4.1": {"in": 2.0, "out": 8.0},
    "openai/gpt-4.1-mini": {"in": 0.4, "out": 1.6},
    "google/gemini-2.5-pro": {"in": 1.25, "out": 10.0},
    "qwen/qwen3-235b-a22b": {"in": 0.2, "out": 0.6},
}


@dataclass
class LLMResponse:
    """A single completion plus the accounting the lineage needs.

    Attributes:
        content: Assistant text.
        prompt_tokens: Prompt tokens reported by the provider.
        completion_tokens: Completion tokens reported by the provider.
        finish_reason: Provider finish reason.
        model: Model actually used.
        cached: True when served from the local cache.
        reasoning: Provider reasoning trace when exposed, else "".
    """

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    model: str = ""
    cached: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (used as the cache payload)."""
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        """Rebuild an LLMResponse from a cache payload."""
        return cls(
            content=d.get("content", ""),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            finish_reason=d.get("finish_reason", ""),
            model=d.get("model", ""),
            reasoning=d.get("reasoning", ""),
        )


@runtime_checkable
class LLMClient(Protocol):
    """The only surface the pipeline knows about."""

    def complete(
        self, model: str, messages: list[dict], **params: Any
    ) -> LLMResponse:  # pragma: no cover - protocol
        """Run one chat completion."""
        ...


class PriceTable:
    """Maps model ids to USD cost, tracking which models had no listed price."""

    def __init__(self, overrides: dict[str, dict[str, float]] | None = None) -> None:
        """Initialize with optional per-model {in, out} USD-per-1M overrides."""
        self.prices = dict(DEFAULT_PRICES)
        for model, p in (overrides or {}).items():
            self.prices[model] = {"in": float(p["in"]), "out": float(p["out"])}
        self.unpriced: set[str] = set()

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the USD cost of one call, or 0.0 for an unpriced model."""
        p = self.prices.get(model)
        if p is None:
            self.unpriced.add(model)
            return 0.0
        return round(
            (prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000, 8
        )


@register("llm", "openrouter")
class OpenRouterLLM:
    """OpenAI-compatible OpenRouter client with bounded retry on transient errors."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 6,
        timeout: float = 600.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: API key; falls back to OPENROUTER_API_KEY.
            base_url: Override the API base URL (e.g. to point at a local vLLM).
            max_attempts: Retry attempts for transient failures.
            timeout: Per-request timeout in seconds.

        Raises:
            RuntimeError: If no API key is available.
        """
        from dotenv import load_dotenv

        load_dotenv()
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it, put it in .env, or run "
                "with llm.provider=echo for an offline dry run."
            )
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url or self.BASE_URL, api_key=key, timeout=timeout
        )
        self._max_attempts = max_attempts

    def complete(self, model: str, messages: list[dict], **params: Any) -> LLMResponse:
        """Run one chat completion, retrying only transient provider errors.

        Args:
            model: OpenRouter model id.
            messages: OpenAI-style message list.
            **params: Sampling params passed through (temperature, max_tokens, ...).

        Returns:
            An LLMResponse.

        Raises:
            RuntimeError: If the provider returns no content after all attempts.
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_random_exponential,
        )

        transient = (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)

        @retry(
            retry=retry_if_exception_type(transient),
            wait=wait_random_exponential(min=2, max=60),
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
        )
        def _call():
            return self._client.chat.completions.create(
                model=model, messages=messages, **params
            )

        resp = _call()
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning", None) or ""
        if not content.strip():
            raise RuntimeError(f"Empty content from {model}: finish={choice.finish_reason}")
        usage = resp.usage
        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
            model=model,
            reasoning=reasoning,
        )


@register("llm", "echo")
class EchoLLM:
    """Deterministic offline client. Used by tests and `--dry-run`.

    Returns schema-valid JSON derived from a hash of the prompt, so the whole
    pipeline - including parsing, dedup, and rating - exercises end to end with
    no network and no spend.
    """

    def __init__(self, **_: Any) -> None:
        """Accept and ignore provider kwargs so config shape is identical."""
        self._lock = threading.Lock()
        self.calls = 0

    def complete(self, model: str, messages: list[dict], **params: Any) -> LLMResponse:
        """Return a deterministic fake completion shaped like a real one."""
        with self._lock:
            self.calls += 1
        prompt = "\n".join(m.get("content", "") for m in messages)
        h = text_hash(prompt, 8)
        if "SCORES" in prompt or "rubric" in prompt.lower():
            body = (
                '{"scores": {"spec_fidelity": 4, "realism": 4, "non_preachiness": 4},'
                f' "overall": 4, "justification": "echo-{h}"}}'
            )
        else:
            body = (
                '{"turns": [{"role": "user", "content": "echo user turn '
                f'{h}"}}, {{"role": "assistant", "thinking": "echo reasoning {h}",'
                f' "content": "echo assistant turn {h}"}}]}}'
            )
        return LLMResponse(
            content=body,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(body) // 4),
            finish_reason="stop",
            model=model,
            reasoning=f"echo-reasoning-{h}",
        )


@dataclass
class CachedLLM:
    """Wraps an LLMClient with the content-addressed cache and cost accounting.

    Attributes:
        inner: The underlying client.
        cache: The on-disk cache.
        prices: Price table used to attribute cost_usd.
    """

    inner: LLMClient
    cache: Any
    prices: PriceTable = field(default_factory=PriceTable)

    def call(
        self,
        *,
        stage_idx: int,
        input_hash: str,
        model: str,
        messages: list[dict],
        params: dict[str, Any],
    ) -> tuple[LLMResponse, str]:
        """Run a completion through the cache.

        Args:
            stage_idx: Stage making the call.
            input_hash: Hash of the stage input, so a re-run with changed input misses.
            model: Model id.
            messages: Rendered prompt.
            params: Sampling params.

        Returns:
            Tuple of (response, prompt_hash). response.cached indicates a cache hit.
        """
        prompt_hash = stable_hash(messages)
        key = self.cache.key(stage_idx, input_hash, prompt_hash, model, params)
        hit = self.cache.get(key)
        if hit is not None:
            resp = LLMResponse.from_dict(hit)
            resp.cached = True
            return resp, prompt_hash
        resp = self.inner.complete(model, messages, **params)
        self.cache.put(key, resp.to_dict())
        return resp, prompt_hash

    def cost(self, resp: LLMResponse) -> float:
        """Return the USD cost of a response (cache hits still report their price)."""
        return self.prices.cost(resp.model, resp.prompt_tokens, resp.completion_tokens)


def build_client(provider: str, **kwargs: Any) -> LLMClient:
    """Instantiate a registered LLM provider by name.

    Args:
        provider: Registered name under kind "llm" (openrouter | echo).
        **kwargs: Passed to the provider constructor.

    Returns:
        An LLMClient.
    """
    from .registry import resolve

    return resolve("llm", provider)(**kwargs)
