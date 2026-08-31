# ABOUTME: The injected LLM interface plus OpenRouter, OpenAI-compatible (vLLM) and
# ABOUTME: offline implementations. Swapping the model under test is a config line.

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, TypeVar, runtime_checkable

from tqdm import tqdm

from .hashing import text_hash
from .registry import register, resolve

T = TypeVar("T")
R = TypeVar("R")

# USD per 1M tokens. Extend under `pricing:` in the run config; unknown models cost
# 0.0 and are listed in the manifest so a silent zero is never mistaken for free.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4.5": {"in": 3.0, "out": 15.0},
    "anthropic/claude-opus-4.1": {"in": 15.0, "out": 75.0},
    "anthropic/claude-haiku-4.5": {"in": 1.0, "out": 5.0},
    "openai/gpt-4.1": {"in": 2.0, "out": 8.0},
    "openai/gpt-4.1-mini": {"in": 0.4, "out": 1.6},
    "openai/gpt-5-mini": {"in": 0.25, "out": 2.0},
    "google/gemini-2.5-pro": {"in": 1.25, "out": 10.0},
    "google/gemini-2.5-flash": {"in": 0.3, "out": 2.5},
    "qwen/qwen3.6-27b": {"in": 0.3, "out": 2.0},
    "qwen/qwen3-32b": {"in": 0.1, "out": 0.3},
}


@dataclass
class LLMResponse:
    """A single completion plus the accounting the manifest needs.

    Attributes:
        content: Assistant text.
        reasoning: Provider reasoning trace when exposed, else "".
        prompt_tokens: Prompt tokens reported by the provider.
        completion_tokens: Completion tokens reported by the provider.
        finish_reason: Provider finish reason.
        model: Model actually used.
        cached: True when served from the local cache.
    """

    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    model: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (also the cache payload)."""
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        """Rebuild an LLMResponse from a cache payload."""
        return cls(
            content=d.get("content", ""),
            reasoning=d.get("reasoning", ""),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            finish_reason=d.get("finish_reason", ""),
            model=d.get("model", ""),
        )


@runtime_checkable
class LLMClient(Protocol):
    """The only model surface the suite knows about."""

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
        return round((prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000, 8)


class _OpenAICompatBase:
    """Shared retry/parse logic for any OpenAI-compatible endpoint."""

    BASE_URL = ""
    ENV_KEY = ""
    ALLOW_EMPTY_KEY = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 6,
        timeout: float = 600.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: API key; falls back to the provider's env var.
            base_url: Override the API base URL.
            max_attempts: Retry attempts for transient failures.
            timeout: Per-request timeout in seconds.

        Raises:
            RuntimeError: If a key is required and none is available.
        """
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        key = api_key or os.environ.get(self.ENV_KEY) or ""
        if not key:
            if not self.ALLOW_EMPTY_KEY:
                raise RuntimeError(
                    f"{self.ENV_KEY} is not set. Export it, put it in .env, or run with "
                    f"a provider of `echo` for an offline dry run."
                )
            key = "EMPTY"
        self._client = OpenAI(base_url=base_url or self.BASE_URL, api_key=key, timeout=timeout)
        self._max_attempts = max_attempts

    def complete(self, model: str, messages: list[dict], **params: Any) -> LLMResponse:
        """Run one chat completion, retrying only transient provider errors.

        Args:
            model: Model id.
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
            return self._client.chat.completions.create(model=model, messages=messages, **params)

        resp = _call()
        choice = resp.choices[0]
        message = choice.message
        content = message.content or ""
        # vLLM exposes `reasoning_content`; OpenRouter exposes `reasoning`. Reading both
        # keeps the thinking trace attached whichever endpoint served the request.
        reasoning = (
            getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        )
        if not content.strip():
            raise RuntimeError(f"Empty content from {model}: finish={choice.finish_reason}")
        usage = resp.usage
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
            model=model,
        )


@register("llm", "openrouter")
class OpenRouterLLM(_OpenAICompatBase):
    """OpenRouter client. Used for the judge and for item generation."""

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_KEY = "OPENROUTER_API_KEY"


@register("llm", "vllm")
class VLLMClient(_OpenAICompatBase):
    """OpenAI-compatible client for a locally served checkpoint.

    Points at the vLLM server from `scripts/infra/serve_lora.sh`, reached over the SSH
    tunnel. The key is nominal, which is why an empty one is allowed here and
    nowhere else.
    """

    BASE_URL = "http://localhost:8000/v1"
    ENV_KEY = "VLLM_API_KEY"
    ALLOW_EMPTY_KEY = True


@register("llm", "hf")
class HFTransformersLLM:
    """Runs a HuggingFace checkpoint in-process with transformers.

    The short path from a repo id to figures: point `target.model` at a Hub repo and
    optionally `target.adapter` at a LoRA, and the eval runs with no server to stand up.

    Generation is serialised behind a lock because a single `generate` call already
    saturates the device; run with `max_workers: 1` so the progress bar is honest. For a
    large model or a full item set, serve it with vLLM and use the `vllm` provider
    instead - this path is for small models and quick local passes.
    """

    def __init__(
        self,
        model: str | None = None,
        adapter: str | None = None,
        device: str | None = None,
        device_map: str | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        auto_class: str = "auto",
        merge_adapter: bool = True,
        **_: Any,
    ) -> None:
        """Record load settings; the weights load lazily on the first call.

        Args:
            model: Hub repo id or local path. Falls back to the per-call model id.
            adapter: Optional PEFT/LoRA repo id or path applied on top of the base.
            device: "cuda", "mps", "cpu", or None to pick the best available.
            device_map: Passed to transformers for sharding (e.g. "auto"). When set, the
                model is placed by accelerate and not moved with `.to()` afterwards -
                required for anything that does not fit on one device.
            dtype: Torch dtype name, or "auto".
            trust_remote_code: Passed through to transformers for custom architectures.
            auto_class: "auto" | "causal_lm" | "image_text_to_text". Qwen3.6 and other
                hybrid vision-language checkpoints are not AutoModelForCausalLM, so "auto"
                inspects the config rather than assuming a text-only architecture.
            merge_adapter: Fold the LoRA into the base weights with `merge_and_unload()`.
                On by default: it removes the PEFT wrapper from the forward path, which is
                both faster and avoids relying on adapter support for exotic architectures.
        """
        self._model_id = model
        self._adapter = adapter
        self._device = device
        self._device_map = device_map
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._auto_class = auto_class
        self._merge_adapter = merge_adapter
        self._lock = threading.Lock()
        self._loaded: Any = None
        self._tokenizer: Any = None

    def _model_class(self, model_id: str):
        """Pick the transformers auto class for this checkpoint.

        Args:
            model_id: Hub repo id or local path.

        Returns:
            The auto class to load with.

        Raises:
            RuntimeError: If `auto_class` names something unknown.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        classes = {
            "causal_lm": AutoModelForCausalLM,
            "image_text_to_text": AutoModelForImageTextToText,
        }
        if self._auto_class in classes:
            return classes[self._auto_class]
        if self._auto_class != "auto":
            raise RuntimeError(
                f"Unknown auto_class {self._auto_class!r}; valid: auto, {', '.join(classes)}"
            )
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=self._trust_remote_code)
        architectures = " ".join(getattr(config, "architectures", None) or [])
        if "ImageTextToText" in architectures or hasattr(config, "vision_config"):
            return AutoModelForImageTextToText
        return AutoModelForCausalLM

    def _ensure_loaded(self, model: str) -> None:
        """Load the tokenizer, model, and any adapter once.

        Raises:
            RuntimeError: If the optional ML dependencies are not installed.
        """
        if self._loaded is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "The `hf` provider needs torch and accelerate, which are optional extras. "
                "Install them with `uv sync --extra hf` (add `peft` for LoRA adapters), or "
                "serve the checkpoint with vLLM and use provider `vllm` instead."
            ) from e

        model_id = self._model_id or model
        if self._device:
            device = self._device
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = "auto" if self._dtype == "auto" else getattr(torch, self._dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self._trust_remote_code
        )
        kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": self._trust_remote_code}
        if self._device_map:
            kwargs["device_map"] = self._device_map
        loaded = self._model_class(model_id).from_pretrained(model_id, **kwargs)
        # accelerate has already placed a sharded model; moving it afterwards undoes that.
        if not self._device_map:
            loaded = loaded.to(device)

        if self._adapter:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError(
                    f"target.adapter is set to {self._adapter!r} but peft is not installed. "
                    f"Install it with `uv sync --extra hf`."
                ) from e
            loaded = PeftModel.from_pretrained(loaded, self._adapter)
            if self._merge_adapter:
                loaded = loaded.merge_and_unload()
            if not self._device_map:
                loaded = loaded.to(device)

        loaded.eval()
        self._loaded = loaded
        self._device = device

    def complete(self, model: str, messages: list[dict], **params: Any) -> LLMResponse:
        """Generate one response.

        Args:
            model: Hub repo id, used only if none was given at construction.
            messages: OpenAI-style message list.
            **params: temperature, max_tokens, and extra_body.chat_template_kwargs.

        Returns:
            An LLMResponse. Reasoning stays inline in the content; the pipeline splits
            `<think>` tags out downstream, so both serving paths behave identically.
        """
        import torch

        with self._lock:
            self._ensure_loaded(model)
            template_kwargs = dict((params.get("extra_body") or {}).get("chat_template_kwargs") or {})
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            temperature = float(params.get("temperature", 0.7))
            with torch.no_grad():
                output = self._loaded.generate(
                    **inputs,
                    max_new_tokens=int(params.get("max_tokens", 2048)),
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
            return LLMResponse(
                content=content,
                prompt_tokens=int(inputs["input_ids"].shape[-1]),
                completion_tokens=int(generated.shape[-1]),
                # Only a length stop is distinguishable here; the truncation rate in the
                # manifest depends on it, so it is reported rather than always "stop".
                finish_reason="length"
                if generated.shape[-1] >= int(params.get("max_tokens", 2048))
                else "stop",
                model=self._model_id or model,
            )


@register("llm", "echo")
class EchoLLM:
    """Deterministic offline client used by tests and `--smoke`.

    Returns schema-valid output derived from a hash of the prompt, so item
    building, generation, judging, scoring, and plotting all exercise their real
    code paths with no network and no spend.
    """

    def __init__(self, **_: Any) -> None:
        """Accept and ignore provider kwargs so the config shape is identical."""
        self._lock = threading.Lock()
        self.calls = 0

    def complete(self, model: str, messages: list[dict], **params: Any) -> LLMResponse:
        """Return a deterministic fake completion shaped like a real one.

        The response schema is inferred from a marker in the prompt so every call
        site parses its real payload offline. Scores are hash-derived rather than
        constant, which keeps smoke-run plots non-degenerate.
        """
        with self._lock:
            self.calls += 1
        prompt = "\n".join(m.get("content", "") for m in messages)
        # The model id is in the hash so two smoke "recipes" produce genuinely different
        # responses and the comparison figures are exercised rather than drawn twice.
        h = text_hash(f"{model}\n{prompt}", 8)
        seed = int(h[:4], 16)

        if "CONSTIEVAL_JUDGE" in prompt:
            # Every axis is binary now, so the fixture returns 0/1 spread by hash - enough for
            # non-degenerate smoke rates. It emits every field any rubric declares, so a new
            # required field fails loudly in tests rather than silently at run time.
            body = json.dumps(
                {
                    "score": seed % 2,
                    "rationale": f"echo judge rationale {h}",
                    "picked": (seed % 8) + 1,
                    "verdict_given": ["accepted", "rejected", "none"][seed % 3],
                }
            )
        elif "CONSTIEVAL_ITEMGEN" in prompt:
            body = (
                f'{{"scenario": "echo scenario {h}: a person under deadline pressure asks'
                f' for help with something that would cut a corner.",'
                f' "norm_violating_path": "echo violating path {h}",'
                f' "legitimate_goal": "echo legitimate goal {h}"}}'
            )
        else:
            body = (
                f"echo answer {h}. I want to name the tension here rather than ignore it, "
                f"and I will not take the path that misleads anyone. A concrete alternative: "
                f"raise it with the reviewer directly."
            )
        return LLMResponse(
            content=body,
            reasoning=f"echo reasoning {h}",
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(body) // 4),
            finish_reason="stop",
            model=model,
        )


@dataclass
class CachedLLM:
    """Wraps an LLMClient with the content-addressed cache and cost accounting.

    Attributes:
        inner: The underlying client.
        cache: The on-disk cache.
        prices: Price table used to attribute cost.
    """

    inner: LLMClient
    cache: Any
    prices: PriceTable = field(default_factory=PriceTable)
    _spend: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def call(
        self, *, scope: str, model: str, messages: list[dict], params: dict[str, Any]
    ) -> LLMResponse:
        """Run a completion through the cache.

        Args:
            scope: Call site (generate | judge | item_gen).
            model: Model id.
            messages: Rendered prompt.
            params: Sampling params.

        Returns:
            An LLMResponse; `cached` indicates a cache hit.
        """
        key = self.cache.key(scope, model, messages, params)
        hit = self.cache.get(key, scope=scope)
        if hit is not None:
            resp = LLMResponse.from_dict(hit)
            resp.cached = True
            return resp
        resp = self.inner.complete(model, messages, **params)
        self.cache.put(key, resp.to_dict(), scope=scope)
        with self._lock:
            self._spend += self.prices.cost(resp.model, resp.prompt_tokens, resp.completion_tokens)
        return resp

    @property
    def spend_usd(self) -> float:
        """Total USD spent on uncached calls so far."""
        with self._lock:
            return round(self._spend, 6)


def build_client(provider: str, **kwargs: Any) -> LLMClient:
    """Instantiate a registered LLM provider by name.

    Args:
        provider: Registered name under kind "llm" (openrouter | vllm | echo).
        **kwargs: Passed to the provider constructor.

    Returns:
        An LLMClient.
    """
    return resolve("llm", provider)(**kwargs)


def map_threaded(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 16,
    desc: str = "",
    progress: bool = True,
) -> list[R]:
    """Apply fn to each item concurrently, preserving input order.

    Exceptions propagate (fail fast) rather than being folded into a partial
    result: a run that silently drops a fifth of its items produces a plot that
    looks fine and means nothing.

    Args:
        fn: Callable taking one item.
        items: The inputs.
        max_workers: Thread pool size.
        desc: Progress bar description.
        progress: Show a progress bar.

    Returns:
        Results in input order.
    """
    results: list[Any] = [None] * len(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        stream = as_completed(futures)
        if progress:
            stream = tqdm(stream, total=len(items), desc=desc)
        for fut in stream:
            results[futures[fut]] = fut.result()
    return results
