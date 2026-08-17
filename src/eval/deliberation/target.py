# ABOUTME: Shared target-calling for the deliberation evals: one chat turn against the served
# ABOUTME: endpoint, trace split across every vLLM shape, and trace stats every eval reports.

"""The one place a deliberation eval talks to the model under test.

Three things are centralised here because getting any of them wrong is a silent
mis-measurement rather than a crash:

- **The out-of-band reasoning field is not stable across vLLM versions** (`reasoning_content`
  on 0.8.x, `reasoning` on 0.26). Reading only one reports every trace as empty and trips
  the CLAUDE.md gotcha-2 alarm on a model that is reasoning normally. `resolve_trace` owns
  the three shapes; this module owns finding the field.
- **A dropped request must not sink the arm.** `map_threaded` is fail-fast, so one timeout
  would discard a whole finished arm. A failed call is recorded as an empty answer, which
  scores unparseable and surfaces in `parse_rate` rather than vanishing.
- **Trace length is a reported metric, not a diagnostic.** docs/in_domain_evals.md makes it
  a headline: courtroom and peer critique reason measurably less than difficult advice, so
  every score here is reported next to the trace length that produced it. Without it a
  variant that merely got terser (or wordier) is invisible.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from openai import OpenAI

from src.model_profile import resolve_trace


@dataclass(frozen=True)
class Reply:
    """One completion from the target, already split into trace and visible answer.

    Attributes:
        think: The reasoning trace, empty when the model did not reason (or thinking is off).
        answer: The visible reply, with any inline `<think>` block removed.
        raw: `message.content` exactly as returned, kept so a rollout stays auditable.
        finish_reason: Provider-reported finish reason; `"error"` when the call failed.
        error: Exception class name when the call failed, else "".
    """

    think: str
    answer: str
    raw: str
    finish_reason: str
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the call completed (a completion that stopped early still counts)."""
        return not self.error


@dataclass
class Generation:
    """Target sampling settings, read from an eval config's `generation:` block."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    request_timeout: float = 180.0
    max_retries: int = 2
    parallel: int = 16
    # Qwen-family templates take this to switch the <think> prefill. Left None for an API
    # target, whose template is not ours to pin (CLAUDE.md, "The eval framework").
    enable_thinking: bool | None = None

    @classmethod
    def from_cfg(cls, node) -> "Generation":
        """Build from an OmegaConf node (or dict).

        An unknown key is a hard error rather than a silent drop: a typo'd
        `max_token: 4096` that quietly kept the 2048 default would truncate every trace
        and be indistinguishable from a model that stopped reasoning (CLAUDE.md gotcha 4).
        """
        known = set(cls.__dataclass_fields__)
        given = {str(k): v for k, v in dict(node or {}).items()}
        unknown = set(given) - known
        assert not unknown, (f"unknown generation key(s): {sorted(unknown)} "
                             f"(known: {sorted(known)})")
        return cls(**given)


def client_for(served, gen: Generation) -> OpenAI:
    """An OpenAI client pointed at the served target (vLLM or a public API endpoint).

    `served.base_url` boots vLLM lazily on first access, so this is also the moment an HF
    target's server starts. The key comes from the target, never from a config — secrets
    stay out of the scientific record.
    """
    return OpenAI(base_url=served.base_url, api_key=served.api_key,
                  timeout=float(gen.request_timeout), max_retries=int(gen.max_retries))


def ask(client: OpenAI, served, messages: list[dict], gen: Generation) -> Reply:
    """Run one chat turn against the target and split the trace out of the completion.

    Args:
        client: From `client_for`.
        served: The ServedTarget (for `model_name` and whether template kwargs apply).
        messages: OpenAI-style message list.
        gen: Sampling settings.

    Returns:
        A `Reply`. A failed call returns `error` set and empty text rather than raising —
        see the module docstring.
    """
    extra: dict = {}
    # A served arm's thinking mode is pinned into the chat template at serve time, so this
    # is a belt-and-braces request, not the mechanism. An API target has no template of
    # ours; sending the kwarg there would be a silent no-op at best and a 400 at worst.
    if gen.enable_thinking is not None and not served.is_api:
        extra["chat_template_kwargs"] = {"enable_thinking": bool(gen.enable_thinking)}
    try:
        resp = client.chat.completions.create(
            model=served.model_name,
            messages=messages,
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            max_tokens=int(gen.max_tokens),
            **({"extra_body": extra} if extra else {}),
        )
    except Exception as exc:  # noqa: BLE001 — see module docstring
        return Reply(think="", answer="", raw="", finish_reason="error",
                     error=type(exc).__name__)
    choice = resp.choices[0]
    raw = choice.message.content or ""
    reasoning = (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None))
    think, answer = resolve_trace(raw, reasoning)
    return Reply(think=think, answer=answer, raw=raw,
                 finish_reason=choice.finish_reason or "")


def trace_stats(replies: list[Reply]) -> dict:
    """Trace-health metrics every deliberation eval reports beside its score.

    `empty_think_rate` is CLAUDE.md gotcha 4's alarm (a ~0-length trace means the arm
    stopped reasoning); the length percentiles are the length control that keeps "scored
    higher" separable from "wrote more".
    """
    ok = [r for r in replies if r.ok]
    lengths = [len(r.think) for r in ok]
    return {
        "n_calls": len(replies),
        "error_rate": round(sum(not r.ok for r in replies) / max(len(replies), 1), 4),
        "truncation_rate": round(
            sum(r.finish_reason == "length" for r in ok) / max(len(ok), 1), 4),
        "empty_think_rate": round(sum(n == 0 for n in lengths) / max(len(lengths), 1), 4),
        "think_chars_mean": round(statistics.fmean(lengths), 1) if lengths else 0.0,
        "think_chars_median": round(statistics.median(lengths), 1) if lengths else 0.0,
        "answer_chars_mean": round(
            statistics.fmean([len(r.answer) for r in ok]), 1) if ok else 0.0,
    }
