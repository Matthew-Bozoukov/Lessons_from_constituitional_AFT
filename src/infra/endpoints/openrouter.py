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
PROVIDER_PINS_PATH = Path(__file__).resolve().parents[3] / "configs/endpoints/providers.yaml"

_pins: dict | None = None


def provider_pin(model: str) -> dict:
    """The OpenRouter `provider` routing object pinned for `model`.

    THE single source of truth is configs/endpoints/providers.yaml: one provider per
    model id (per-model entry merged over `defaults`), no prefix inference, no
    free-routing fallback. Hosts of the same weights are NOT interchangeable: they
    differ in quantization/backend, wrap models in their own content filters
    (2026-08-14, difficult-advice revise_prompts: Bedrock refused 2.6% of calls as
    content_filter and Google Vertex refused the same prompts — all served fine by
    Anthropic), and only the vendor's endpoint honors extensions like `cache_control`
    reliably.

    Raises:
        ValueError: `model` has no entry. Every model routed through OpenRouter must
            be served by the same provider on every call — add its ONE provider to
            the yaml (check https://openrouter.ai/api/v1/models/<id>/endpoints).
    """
    global _pins
    if _pins is None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(PROVIDER_PINS_PATH))
        _pins = {mid: {**cfg["defaults"], **spec}
                 for mid, spec in (cfg["models"] or {}).items()}
    pin = _pins.get(model)
    if pin is None:
        raise ValueError(
            f"no provider pin for {model!r}: every model routed through OpenRouter "
            "must be served by the same provider on every call. Add its ONE provider "
            f"to {PROVIDER_PINS_PATH} (check https://openrouter.ai/api/v1/models/"
            f"{model}/endpoints), or pass an explicit extra_body['provider'] block.")
    # `price` is provenance/accounting, not routing — it never goes in the request body.
    return {k: v for k, v in pin.items() if k != "price"}


def provider_price(model: str) -> dict | None:
    """The USD-per-1M `{in, out}` price pinned for `model`, or None if unpriced.

    THE single source of truth is the `price:` field beside each pin in
    configs/endpoints/providers.yaml — the price belongs with the provider/tier it is
    the price OF, so a tier change moves both together. cost accounting in
    src/data/synth reads this rather than hardcoding a table that silently drifts from
    the pin.
    """
    provider_pin(model)  # populates _pins and raises on an unpinned model
    price = (_pins or {}).get(model, {}).get("price")
    if price is None:
        return None
    return {"in": float(price["in"]), "out": float(price["out"])}


class _CompletionFailure(RuntimeError):
    """Base for HTTP-200 responses that carry no completion; not raised directly.

    Both subclasses expose the provider's in-body error payload (when present) so
    callers can persist a typed record — {provider, code, message} — rather than a
    bare stack trace.
    """

    def __init__(self, message: str, provider_error: dict | None = None,
                 provider: str = "") -> None:
        super().__init__(message)
        self.provider_error = provider_error or {}
        self.provider = provider


class EmptyCompletionError(_CompletionFailure):
    """The response carried no completion and no deterministic explanation.

    Blank body (`content=None` despite finish_reason=stop — observed on
    deepseek-chat-v3.1, 2026-08-07, 4/20 concurrent calls, unreproducible minutes
    later), `choices=None` with no in-body error, or an in-body error whose code is
    transient by HTTP semantics (429/5xx). Treated as transient: retried against the
    SAME provider (every model is pinned, configs/endpoints/providers.yaml) instead
    of failing the whole item on one bad response.
    """


class ProviderRejectionError(_CompletionFailure):
    """The provider REJECTED the request: HTTP-200 envelope, structured in-body
    error with a deterministic 4xx code (e.g. Gemini's PROHIBITED_CONTENT filter
    block, 2026-08-17). NOT retried — retrying a deterministic rejection re-bills
    the identical failure six times — so it surfaces on the FIRST attempt with the
    payload attached. Distinct from a model refusing in text, which is a normal
    completion and raises nothing.
    """


# Only transient failures are retried; everything else fails fast and surfaces. With
# every model pinned to one provider, a retry always lands on the SAME host — it covers
# transient upstream blips, and a hard refusal surfaces after the attempts. That is the
# right failure mode: a "successful" reroute to a host that filters differently is a
# silent data change.
_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, EmptyCompletionError)


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
        provider: Which upstream provider actually served the call — record it in
            run artifacts the way temperature is recorded (see providers.yaml).
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    cached_tokens: int = 0
    provider: str = ""


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


def build_request_body(model: str, messages: list[dict], temperature: float,
                       max_tokens: int, extra_body: dict | None = None) -> dict:
    """The JSON body `chat` would send for these arguments, minus the model id.

    Exists for the async batch path (`stage_runtime.run_batch`), which posts raw
    request bodies instead of going through the SDK: cache markers and provider pins
    must be applied by the SAME code either way, or a batched request would silently
    differ from its interactive twin. The model id is omitted because the batch API
    takes it once per job, not per request.
    """
    extra = dict(extra_body or {})
    if "provider" not in extra:
        extra["provider"] = provider_pin(model)
    return {"messages": apply_cache_control(messages, model),
            "temperature": temperature, "max_tokens": max_tokens, **extra}


def result_from_payload(model: str, data: dict) -> ChatResult:
    """A ChatResult from a raw chat-completion JSON dict (a batch result's `body`).

    Mirrors the post-processing in `chat` — no-choices classification, the
    content_filter and empty-content guards, defensive cache-token reads — so a
    completion is judged by one set of rules whether it arrived over the SDK or out
    of a batch. Raises the same typed errors; the batch caller catches them and
    routes the request to the interactive mop-up path instead of retrying blindly.
    """
    provider = str(data.get("provider") or "")
    choices = data.get("choices") or []
    if not choices:
        err = data.get("error")
        err_d = (err if isinstance(err, dict)
                 else {"message": str(err)} if err else None)
        msg = f"Model {model} returned no choices (provider {provider or '?'}): {data}"
        code = (err_d or {}).get("code")
        if isinstance(code, int) and 400 <= code < 500 and code != 429:
            raise ProviderRejectionError(msg, provider_error=err_d, provider=provider)
        raise EmptyCompletionError(msg, provider_error=err_d, provider=provider)
    choice = choices[0]
    content = (choice.get("message") or {}).get("content")
    finish = choice.get("finish_reason") or ""
    if finish == "content_filter":
        # Same classification as `chat`: an output-sample filter, retryable — which
        # for a batch result means the interactive mop-up (a fresh sample) owns it.
        raise EmptyCompletionError(
            f"Model {model} blocked by content filter (provider {provider or '?'}): "
            "finish_reason=content_filter",
            provider_error={"code": "content_filter",
                            "message": "finish_reason=content_filter "
                                       f"({len(content or '')} chars of partial "
                                       "content dropped)"},
            provider=provider)
    if not content:
        raise EmptyCompletionError(
            f"Model {model} returned empty content (provider {provider or '?'}): "
            f"{data}", provider=provider)
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return ChatResult(
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        finish_reason=finish,
        cached_tokens=int(details.get("cached_tokens") or 0),
        provider=provider,
    )


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
            A ChatResult with content, token usage and the serving provider.
        """
        # Route through the model's provider pin (configs/endpoints/providers.yaml).
        # A caller-supplied extra_body["provider"] wins; otherwise an unpinned model
        # is a hard error inside provider_pin — free routing is never the fallback.
        extra = dict(kwargs.pop("extra_body", None) or {})
        if "provider" not in extra:
            extra["provider"] = provider_pin(model)
        kwargs["extra_body"] = extra
        resp = self.client.chat.completions.create(
            model=model,
            messages=apply_cache_control(messages, model),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if not resp.choices:
            # choices=None: either a blank (like content=None below) or an in-body
            # error the SDK parsed into model_extra. Classification is by payload
            # shape + the code's HTTP class ONLY — never provider-specific message
            # text, which varies per model and would rot.
            err = getattr(resp, "error", None) or (
                getattr(resp, "model_extra", None) or {}).get("error")
            err_d = (err if isinstance(err, dict)
                     else {"message": str(err)} if err else None)
            provider = getattr(resp, "provider", "") or ""
            msg = (f"Model {model} returned no choices (provider "
                   f"{getattr(resp, 'provider', '?')}): {resp}")
            code = (err_d or {}).get("code")
            if isinstance(code, int) and 400 <= code < 500 and code != 429:
                raise ProviderRejectionError(msg, provider_error=err_d,
                                             provider=provider)
            raise EmptyCompletionError(msg, provider_error=err_d, provider=provider)
        choice = resp.choices[0]
        content = choice.message.content
        if choice.finish_reason == "content_filter":
            # OpenAI-protocol hard filter, with or without partial text. Partial
            # output is DROPPED rather than returned: a silently truncated
            # completion poisons downstream parses worse than a loud rejection.
            # RETRYABLE, unlike an in-body 4xx: a finish_reason filter fired on what
            # the model SAMPLED, not on the request — measured 2026-08-20 on the
            # gemini difficult-advice arm, the same write_scenarios prompt passed
            # 11/12 calls at temperature 1.1. An input-level block arrives as an
            # in-body 4xx (e.g. Gemini's PROHIBITED_CONTENT) and stays deterministic
            # above; a prompt the filter always trips on exhausts the retries and
            # surfaces here with the same typed payload.
            raise EmptyCompletionError(
                f"Model {model} blocked by content filter (provider "
                f"{getattr(resp, 'provider', '?')}): finish_reason=content_filter",
                provider_error={"code": "content_filter",
                                "message": "finish_reason=content_filter "
                                           f"({len(content or '')} chars of partial "
                                           "content dropped)"},
                provider=getattr(resp, "provider", "") or "")
        if not content:
            # None OR empty string: an undiagnosable blank either way — the empty
            # string previously slipped through as a "successful" ChatResult and
            # died later at the caller's parse gate with no retry.
            # Retryable: the decorator re-hits the SAME pinned provider. A model
            # that blanks on every attempt exhausts the retries and this surfaces —
            # the caller still sees a clear failure.
            raise EmptyCompletionError(
                f"Model {model} returned empty content (provider "
                f"{getattr(resp, 'provider', '?')}): {resp}",
                provider=getattr(resp, "provider", "") or "")
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
