# ABOUTME: Thin OpenRouter (OpenAI-compatible) chat client with bounded retry and
# ABOUTME: threaded concurrency, used for data generation and grading.

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_PINS_PATH = Path(__file__).resolve().parents[2] / "configs/endpoints/providers.yaml"

_pins: dict | None = None
_warned_unpinned: set[str] = set()


def provider_pin(model: str) -> dict | None:
    """The OpenRouter `provider` routing object pinned for `model`, or None.

    Pins live in configs/endpoints/providers.yaml (per-model `order` merged over
    `defaults`). Providers serving one model id differ in quantization/backend, so
    every call routes through a pin; an unpinned model still runs but warns once per
    process — add an entry rather than silencing it.
    """
    global _pins
    if _pins is None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(PROVIDER_PINS_PATH))
        _pins = {mid: {**cfg["defaults"], **spec}
                 for mid, spec in (cfg["models"] or {}).items()}
    pin = _pins.get(model)
    if pin is None and model not in _warned_unpinned:
        _warned_unpinned.add(model)
        print(f"!!! no provider pin for {model!r} — routing is provider-random; "
              f"add it to {PROVIDER_PINS_PATH.name} (configs/endpoints/)")
    return pin


class EmptyCompletionError(RuntimeError):
    """A completion came back with `content=None` despite finish_reason=stop.

    OpenRouter load-balances a model across many upstream providers and re-routes on
    every call; a provider intermittently returns a blank body (observed on
    deepseek-chat-v3.1, 2026-08-07 — 4/20 concurrent calls, unreproducible minutes
    later across 18 calls / 7 providers). Treating it as transient re-routes to
    another provider instead of failing the whole item on one bad backend.
    """


# Only transient failures are retried; everything else fails fast and surfaces. An empty
# completion is transient BY ROUTING for unpinned models (the retry lands on a different
# upstream provider); under a provider pin the retry re-asks the same provider, which
# still clears intermittent blanks.
_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, EmptyCompletionError)


@dataclass
class ChatResult:
    """A single chat completion result.

    Attributes:
        content: The assistant message text.
        prompt_tokens: Prompt token count reported by the API.
        completion_tokens: Completion token count reported by the API.
        finish_reason: The provider-reported finish reason.
        provider: Which upstream provider actually served the call — record it in
            run artifacts the way temperature is recorded (see providers.yaml).
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    provider: str = ""


class OpenRouterClient:
    """Minimal client for OpenRouter chat completions."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        """
        key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    @retry(
        retry=retry_if_exception_type(_TRANSIENT),
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ChatResult:
        """Run a single chat completion, retrying only on transient errors.

        Args:
            model: OpenRouter model id (e.g. "anthropic/claude-sonnet-4.5").
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.
            **kwargs: Passed through to the completions API.

        Returns:
            A ChatResult with content, token usage and the serving provider.
        """
        # Route through the model's provider pin (configs/endpoints/providers.yaml) —
        # unpinned models warn once inside provider_pin and route provider-random.
        pin = provider_pin(model)
        if pin is not None:
            extra = dict(kwargs.pop("extra_body", None) or {})
            extra.setdefault("provider", pin)
            kwargs["extra_body"] = extra
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            # Retryable: the decorator re-routes to another upstream provider. A model
            # that blanks on EVERY provider (e.g. a hard content filter) exhausts the
            # attempts and this surfaces — the caller still sees a clear failure.
            raise EmptyCompletionError(
                f"Model {model} returned empty content (provider "
                f"{getattr(resp, 'provider', '?')}): {resp}")
        usage = resp.usage
        return ChatResult(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
            provider=getattr(resp, "provider", "") or "",
        )


def map_threaded(
    fn: Callable[[int], object],
    n: int,
    max_workers: int = 16,
    desc: str = "",
) -> list:
    """Apply fn to each index 0..n-1 concurrently, preserving input order.

    Exceptions from fn propagate (fail-fast); results keep their original index.

    Args:
        fn: Callable taking an index and returning a result.
        n: Number of items.
        max_workers: Thread pool size.
        desc: tqdm description.

    Returns:
        List of results in index order.
    """
    results: list = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc=desc):
            idx = futures[fut]
            results[idx] = fut.result()
    return results
