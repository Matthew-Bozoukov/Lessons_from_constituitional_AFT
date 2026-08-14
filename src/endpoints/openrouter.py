# ABOUTME: Thin OpenRouter (OpenAI-compatible) chat client with bounded retry and
# ABOUTME: threaded concurrency, used for data generation and grading.

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


class EmptyCompletionError(RuntimeError):
    """A completion came back with `content=None` despite finish_reason=stop.

    Providers intermittently return a blank body (observed on deepseek-chat-v3.1,
    2026-08-07 — 4/20 concurrent calls, unreproducible minutes later). Treating it as
    transient retries the call — against the SAME provider, since every model is pinned
    (PROVIDER_PINS below) — instead of failing the whole item on one bad response.
    """


# Only transient failures are retried; everything else fails fast and surfaces. With
# every model pinned to one provider, a retry always lands on the SAME host — it covers
# transient upstream blips, and a hard refusal surfaces after the attempts. That is the
# right failure mode: a "successful" reroute to a host that filters differently is a
# silent data change.
_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, EmptyCompletionError)


# THE provider registry: every model id that goes through OpenRouterClient MUST match an
# entry here (longest matching prefix wins), or supply its own extra_body["provider"]
# (that always wins). A model with neither is a hard error — no request is ever left to
# OpenRouter's free routing, so a given model id is served by the same provider on every
# call of every run, open-weight models included.
#
# Why: upstream hosts of the same weights are NOT interchangeable — third-party clouds
# wrap the model in their own content filters (2026-08-14, difficult-advice
# revise_prompts: Bedrock refused 2.6% of calls with finish_reason=content_filter, and
# after excluding Bedrock, Google Vertex refused the same prompts — all served fine by
# Anthropic itself), serve different quantizations, and only first-party endpoints honor
# extensions like `cache_control` reliably. Free routing is therefore a silent
# data-composition change, not a convenience.
#
# Every entry names the model creator's own endpoint (slugs verified against
# https://openrouter.ai/api/v1/models/<id>/endpoints on 2026-08-14; Gemini has two
# Google-operated hosts and we pin the direct API, google-ai-studio, not the Vertex
# cloud wrapper). When a new family comes into use, its call fails fast with
# instructions — extend this map deliberately; never widen a pin to multiple providers.
PROVIDER_PINS: dict[str, dict] = {
    "anthropic/": {"order": ["anthropic"], "allow_fallbacks": False},
    "openai/": {"order": ["openai"], "allow_fallbacks": False},
    "google/": {"order": ["google-ai-studio"], "allow_fallbacks": False},
    # xAI is grok's only OpenRouter host today (verified 2026-08-14), so this pin is
    # future-proofing against resellers appearing rather than a live re-route.
    "x-ai/": {"order": ["xai"], "allow_fallbacks": False},
    "qwen/": {"order": ["alibaba"], "allow_fallbacks": False},
    "moonshotai/": {"order": ["moonshotai"], "allow_fallbacks": False},
}


def pin_provider(model: str, extra_body: dict | None) -> dict:
    """Resolve the one provider this model is served by and merge it into `extra_body`.

    Args:
        model: OpenRouter model id.
        extra_body: Caller-supplied extra request body, if any.

    Returns:
        extra_body with a `provider` key — the caller's own block if it sent one,
        else the model's PROVIDER_PINS entry (longest matching prefix).

    Raises:
        ValueError: The model has no pin and the caller sent no provider block. Every
            model must be served by the same provider on every call; there is no
            free-routing fallback.
    """
    out = dict(extra_body or {})
    if "provider" in out:
        return out
    matches = [k for k in PROVIDER_PINS if model.startswith(k)]
    if not matches:
        raise ValueError(
            f"no provider pin for {model!r}: every model routed through OpenRouter "
            "must be served by the same provider on every call (open-weight models "
            "included). Add a PROVIDER_PINS entry for it in src/endpoints/openrouter.py "
            "naming its ONE provider (check https://openrouter.ai/api/v1/models/"
            f"{model}/endpoints), or pass an explicit extra_body['provider'] block."
        )
    out["provider"] = dict(PROVIDER_PINS[max(matches, key=len)])
    return out


@dataclass
class ChatResult:
    """A single chat completion result.

    Attributes:
        content: The assistant message text.
        prompt_tokens: Prompt token count reported by the API.
        completion_tokens: Completion token count reported by the API.
        finish_reason: The provider-reported finish reason.
        cached_tokens: Prompt tokens served from cache, when the provider reports it.
            0 means "no cache hit OR the provider said nothing", so it is a floor on
            savings rather than a measurement -- see `CACHE_MARK`.
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    cached_tokens: int = 0


# Everything BEFORE this marker in a message becomes a separately cacheable block.
#
# Data generation re-sends one long invariant prefix on every call: the difficult-advice
# refine and rewrite stages each inject the whole constitution, which measured 5,763
# tokens -- 65% and 56% of their input respectively, across 4,000 calls in a 2,000-record
# run. Anthropic bills a cache read at 0.1x and a write at 1.25x, so caching that prefix
# saves ~$62 of a ~$234 run, and cached reads are faster besides.
#
# The marker is placed in the CONFIG's prompt text, not here, because only the prompt
# knows where its invariant part ends -- for those two stages that is the closing
# </constitution> tag, after which the target trait varies. It is stripped before the
# request goes out, so the model sees byte-identical text either way; `test_openrouter.py`
# pins that. A prompt with no marker is sent unchanged, so this is inert until opted into.
#
# Caveat worth knowing: Anthropic ignores a cached prefix below ~1024 tokens, silently.
# Marking a short prefix is not an error, it just does nothing.
CACHE_MARK = "<<<cache>>>"


def _split_cached(content: str) -> list[dict] | str:
    """Turn marked text into content blocks, or return it unchanged when unmarked.

    Returns a plain string in the no-marker case rather than a one-element block list:
    an unmarked call must produce a byte-identical request to the one it produced before
    this function existed.
    """
    if CACHE_MARK not in content:
        return content
    prefix, _, rest = content.partition(CACHE_MARK)
    return [{"type": "text", "text": prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": rest}]


def apply_cache_control(messages: list[dict], model: str) -> list[dict]:
    """Convert `CACHE_MARK` in any message into an Anthropic cache breakpoint.

    Gated on the model being an Anthropic one: `cache_control` is a provider extension,
    and passing it to a model that does not understand it risks a 400 rather than a
    silent no-op. For every other provider the marker is simply stripped, so the same
    config runs anywhere and only the billing differs.
    """
    if not any(CACHE_MARK in str(m.get("content") or "") for m in messages):
        return messages
    anthropic = model.startswith("anthropic/")
    out = []
    for m in messages:
        content = str(m.get("content") or "")
        if CACHE_MARK not in content:
            out.append(m)
            continue
        out.append({**m, "content": _split_cached(content) if anthropic
                    else content.replace(CACHE_MARK, "")})
    return out


class OpenRouterClient:
    """Minimal client for OpenRouter chat completions."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        """
        key = api_key or os.environ["OPENROUTER_API_KEY"]
        # Both bounds are explicit because the defaults COMPOUND with the `chat` retry
        # below. Left implicit, one stuck request costs 6 tenacity attempts x 3 SDK
        # attempts x the SDK's 600s timeout ~ 3 hours, during which it holds a worker.
        # Measured 2026-08-13: a 197-record run took 10.7h instead of ~25min because two
        # such requests drained a 16-worker pool; the 67 records queued behind the second
        # one completed in 3 minutes once it cleared.
        #
        # timeout 420s, not lower: the rewrite stage generates up to 12,288 tokens, which
        # legitimately runs into the low hundreds of seconds. This bounds a HANG, and must
        # stay above the slowest honest call or it will truncate real work.
        # max_retries=0: retry policy lives in `chat` (tenacity), which alone knows which
        # errors are transient. Two retry layers multiply rather than add.
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key,
                             timeout=420.0, max_retries=0)

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
            A ChatResult with content and token usage.
        """
        extra_body = pin_provider(model, kwargs.pop("extra_body", None))
        resp = self.client.chat.completions.create(
            model=model,
            messages=apply_cache_control(messages, model),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
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
        # Providers report cache hits in different places and some not at all, so this is
        # read defensively and defaults to 0. It exists so a run can PROVE caching worked
        # rather than assume it: a stage whose cached_tokens stays 0 is paying full price.
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = getattr(details, "cached_tokens", 0) or 0
        return ChatResult(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
            cached_tokens=int(cached),
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
