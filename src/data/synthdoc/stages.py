# ABOUTME: The five LLM stages of the synthdoc pipeline, one function each. Stage behaviour
# ABOUTME: that differs between corpora lives in flavors/; this module owns only the calling,
# ABOUTME: retrying, checkpointing and accounting that every flavor shares.

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.endpoints.openrouter import ChatResult, OpenRouterClient, map_threaded  # noqa: E402
from utils import extract_json  # noqa: E402

from .constitution import Trait  # noqa: E402

# USD per 1M tokens, OpenRouter list prices.
PRICES: dict[str, dict[str, float]] = {
    "openai/gpt-5.6-luna": {"in": 0.10, "out": 0.60},
    "openai/gpt-5.6-terra": {"in": 1.00, "out": 6.00},
    "openai/gpt-5.6-sol": {"in": 5.00, "out": 30.00},
    "anthropic/claude-sonnet-5": {"in": 2.00, "out": 10.00},
    "anthropic/claude-sonnet-4.5": {"in": 3.00, "out": 15.00},
    "anthropic/claude-opus-5": {"in": 5.00, "out": 25.00},
    "anthropic/claude-haiku-4.5": {"in": 1.00, "out": 5.00},
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


def _parse_json(text: str) -> Any:
    """Parse a model's JSON body, tolerating unescaped control characters.

    Models routinely emit literal newlines inside JSON string values. `json.loads` rejects
    those in strict mode, so a single such response would otherwise fail the whole call.
    `strict=False` accepts them and is strictly more permissive -- anything that parsed
    before still parses identically.

    Args:
        text: Raw completion text.

    Returns:
        The parsed JSON value.
    """
    try:
        return extract_json(text)
    except Exception:  # noqa: BLE001 - fall back to the lenient parser, then re-raise below
        pass
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        candidate = candidate[4:] if candidate.lower().startswith("json") else candidate
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
    if start != -1:
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        candidate = candidate[start : end + 1]
    return json.loads(candidate, strict=False)


def _parse_tagged(text: str, keys: tuple[str, ...]) -> dict[str, str]:
    """Pull <key>...</key> blocks out of a completion.

    Long prose in JSON is fragile: an unescaped quote inside a string value ends it early
    and the whole object fails to parse. Every stage-6 failure was exactly that. Tags carry
    arbitrary text -- quotes, apostrophes, newlines -- with nothing to escape.

    Args:
        text: Raw completion.
        keys: Required tag names.

    Returns:
        Mapping tag name to its stripped contents.

    Raises:
        ValueError: If a required tag is missing or unclosed.
    """
    out: dict[str, str] = {}
    for k in keys:
        m = re.search(rf"<{k}>(.*?)</{k}>", text, re.DOTALL)
        if not m:
            raise ValueError(f"missing <{k}> block")
        out[k] = m.group(1).strip()
    return out


def _call_tagged(client: OpenRouterClient, usage: Usage, model: str, system: str, user: str,
                 temperature: float, max_tokens: int, stage: str, keys: tuple[str, ...],
                 attempts: int = 3, validate=None,
                 extra_body: dict | None = None) -> dict[str, str]:
    """Run a completion expecting tagged blocks, retrying if a tag is missing or rejected.

    Args:
        client: OpenRouter client.
        usage: Tally to record the call against.
        model: OpenRouter model id.
        system: System message.
        user: User message.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        stage: Stage name, for per-stage accounting.
        keys: Required tag names.
        attempts: How many times to try before raising.
        validate: Optional callable taking the parsed blocks and raising ValueError to
            reject them. Its message is fed back to the model on the retry, so a flavor can
            enforce properties of the generated text itself and not merely its shape.
        extra_body: Provider-specific options passed through to OpenRouter, e.g.
            `{"reasoning": {"enabled": False}}`.

    Returns:
        Mapping tag name to its contents.

    Raises:
        ValueError: If every attempt was missing a tag or was rejected by `validate`.
    """
    last = ""
    for attempt in range(attempts):
        nudge = "" if attempt == 0 else (
            f"\n\nYour previous reply was rejected: {last}\nReturn ONLY the "
            f"{' and '.join('<' + k + '>...</' + k + '>' for k in keys)} blocks, and fix "
            f"the problem above."
        )
        res = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user + nudge}],
            temperature=temperature, max_tokens=max_tokens,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        usage.add(model, res, stage)
        assert res.finish_reason != "length", (
            f"{stage}: {model} hit max_tokens={max_tokens} and truncated. Raise max_tokens.")
        try:
            parsed = _parse_tagged(res.content, keys)
            if validate is not None:
                validate(parsed)
            return parsed
        except ValueError as exc:
            last = str(exc)
    raise ValueError(f"{stage}: no valid tagged output after {attempts} attempts. {last}")


def _call(client: OpenRouterClient, usage: Usage, model: str, system: str, user: str,
          temperature: float, max_tokens: int, stage: str, attempts: int = 3,
          extra_body: dict | None = None) -> tuple[Any, ChatResult]:
    """Run one chat completion and parse its JSON body, retrying a malformed reply.

    A model occasionally returns prose or truncated JSON. Retrying that single call is far
    cheaper than losing the stage, so parse failures are retried with an explicit nudge
    before giving up.

    Args:
        client: OpenRouter client.
        usage: Tally to record the call against.
        model: OpenRouter model id.
        system: System message.
        user: User message.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        stage: Stage name, for per-stage accounting.
        attempts: How many times to try before raising.
        extra_body: Provider-specific options passed through to OpenRouter, e.g.
            `{"reasoning": {"enabled": False}}`.

    Returns:
        (parsed JSON, raw ChatResult).

    Raises:
        ValueError: If every attempt returned unparseable content.
        AssertionError: If the model hit the token cap, which truncates the JSON body.
    """
    last = ""
    for attempt in range(attempts):
        nudge = "" if attempt == 0 else (
            "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object, with "
            "all newlines inside string values escaped as \\n."
        )
        res = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user + nudge}],
            temperature=temperature,
            max_tokens=max_tokens,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        usage.add(model, res, stage)
        assert res.finish_reason != "length", (
            f"{stage}: {model} hit max_tokens={max_tokens} and truncated its JSON. "
            f"Raise max_tokens for this stage, or lower scenarios_per_call."
        )
        try:
            return _parse_json(res.content), res
        except Exception as exc:  # noqa: BLE001 - retried below, raised on the last attempt
            last = f"{type(exc).__name__}: {exc} | content[:200]={res.content[:200]!r}"
    raise ValueError(f"{stage}: unparseable JSON after {attempts} attempts. {last}")


def _dispatch(spec: tuple, client: OpenRouterClient, usage: Usage, model: str,
              temperature: float, max_tokens: int, stage: str,
              extra_body: dict | None = None) -> Any:
    """Run one stage call, in JSON or tagged-block form as the flavor's spec asks.

    A flavor returns `(system, user)` for a stage whose output is small enough to survive
    JSON, and `(system, user, keys)` for one whose output is long prose -- an inbox dump, a
    deliberation -- where a single unescaped quote would otherwise fail the whole call.

    Args:
        spec: What the flavor's `*_call` returned.
        client: OpenRouter client.
        usage: Tally.
        model: Model id.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        stage: Stage name, for per-stage accounting.

    Returns:
        The parsed JSON value, or the mapping of tag name to contents.
    """
    system, user = spec[0], spec[1]
    keys = spec[2] if len(spec) > 2 else None
    if keys:
        return _call_tagged(client, usage, model, system, user, temperature, max_tokens,
                            stage, keys, extra_body=extra_body)
    parsed, _ = _call(client, usage, model, system, user, temperature, max_tokens, stage,
                      extra_body=extra_body)
    return parsed


def _resilient(fn, n: int, workers: int, desc: str, max_fail_pct: float = 2.0,
               denom: int | None = None) -> list:
    """Map `fn` over indices, keeping successes when individual items fail.

    A single malformed reply must not discard a stage's other results -- those cost real
    money and only exist in memory until the stage completes. Failures are counted and
    reported loudly, and the run still aborts if too many fail, which would indicate a
    systematic problem rather than a one-off.

    Args:
        fn: Callable taking an index.
        n: Number of items.
        workers: Thread pool size.
        desc: Progress label.
        max_fail_pct: Abort above this failure percentage.
        denom: What to measure the failure rate against, when that is not `n`. On a resume
            `n` counts only the items still outstanding -- which are precisely the ones that
            already failed -- so measuring against it makes any resume look catastrophic and
            guarantees an abort. The question the guard is asking is what fraction of the
            whole stage is missing, so the whole stage is the denominator.

    Returns:
        Successful results, in order, with failures dropped.

    Raises:
        RuntimeError: If the failure rate exceeds `max_fail_pct`.
    """
    total = denom if denom is not None else n
    errors: list[str] = []

    def guarded(i: int):
        try:
            return fn(i)
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced below
            errors.append(f"[{i}] {type(exc).__name__}: {exc}")
            return None

    out = map_threaded(guarded, n, max_workers=workers, desc=desc)
    ok = [r for r in out if r is not None]
    if errors:
        pct = 100 * len(errors) / max(total, 1)
        print(f"!!! {desc}: {len(errors)}/{n} items failed "
              f"({pct:.1f}% of {total} in this stage). First 3:")
        for e in errors[:3]:
            print("   ", e)
        if pct > max_fail_pct:
            raise RuntimeError(
                f"{desc}: {pct:.1f}% of this stage's {total} items failed, above "
                f"max_fail_pct={max_fail_pct}. This looks systematic rather than incidental."
            )
    return ok


class Checkpoint:
    """Append-only partial results for one stage, flushed after every completed item.

    Without this, a stage's results live only in memory until it finishes, so any crash
    discards every call already paid for. Records land on disk as they complete and are
    reloaded on resume, so a restart re-runs only what is genuinely missing.

    Attributes:
        path: The partial jsonl.
        key: Field identifying a record, used to skip work already done.
        done: Records already on disk, keyed by `key`.
    """

    def __init__(self, path: Path | str, key: str = "scenario_id") -> None:
        """Load any existing partial results for this stage."""
        self.path = Path(path)
        self.key = key
        self._lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial line from a hard kill; the item simply re-runs
                self.done[r[self.key]] = r

    def record(self, r: dict) -> None:
        """Append one completed record and flush it to disk immediately."""
        with self._lock:
            self.done[r[self.key]] = r
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()


def _run_items(items: list[dict], fn, workers: int, desc: str,
               ckpt: Checkpoint | None = None, max_fail_pct: float = 2.0) -> list[dict]:
    """Process items concurrently, skipping and recording via a checkpoint when given.

    Args:
        items: Input records.
        fn: Callable taking one input record and returning the output record.
        workers: Thread pool size.
        desc: Progress label.
        ckpt: Optional checkpoint for resume and incremental save.
        max_fail_pct: Abort above this failure percentage.

    Returns:
        Output records in input order, with failures dropped.
    """
    if ckpt is None:
        return _resilient(lambda i: fn(items[i]), len(items), workers, desc, max_fail_pct)

    todo = [it for it in items if it[ckpt.key] not in ckpt.done]
    if len(todo) < len(items):
        print(f">>> {desc}: resuming -- {len(items) - len(todo)} already saved, "
              f"{len(todo)} remaining")
    if todo:
        def one(i: int) -> dict:
            r = fn(todo[i])
            ckpt.record(r)
            return r

        _resilient(one, len(todo), workers, desc, max_fail_pct, denom=len(items))
    return [ckpt.done[it[ckpt.key]] for it in items if it[ckpt.key] in ckpt.done]


# --- stage 2 -----------------------------------------------------------------------


def generate_scenarios(traits: list[Trait], client: OpenRouterClient, usage: Usage,
                       flavor: ModuleType, batches: list[dict], model: str,
                       temperature: float, max_tokens: int, workers: int,
                       extra_body: dict | None = None,
                       max_fail_pct: float = 2.0) -> list[dict]:
    """Generate the situations each later stage builds on, one call per planned batch.

    Scenarios are requested in batches rather than all at once: a single call asking for
    40 situations would exceed any sane `max_tokens` and truncate its JSON. Batching also
    improves diversity, since each call is told to vary domains only within its own batch.

    Args:
        traits: The segmented constitution.
        client: OpenRouter client.
        usage: Tally.
        flavor: The flavor module supplying prompts and record construction.
        batches: Batch specs from `flavor.plan`.
        model: Model id.
        temperature: Sampling temperature.
        max_tokens: Completion cap.
        workers: Thread pool size.

    Returns:
        One record per scenario.
    """
    def one(k: int) -> list[dict]:
        b = batches[k]
        t = traits[b["trait_index"]]
        system, user = flavor.scenario_call(b, t)
        parsed, _ = _call(client, usage, model, system, user, temperature, max_tokens,
                          stage="scenarios", extra_body=extra_body)
        assert isinstance(parsed, list), f"{t.trait_id}: expected a JSON array, got {type(parsed)}"
        return flavor.scenario_records(b, t, parsed)

    nested = _resilient(one, len(batches), workers, "stage2:scenarios", max_fail_pct)
    return [r for group in nested for r in group]


# --- stage 3 -----------------------------------------------------------------------


def draft_prompts(scenarios: list[dict], client: OpenRouterClient, usage: Usage,
                  flavor: ModuleType, model: str, temperature: float, max_tokens: int,
                  workers: int, ckpt: Checkpoint | None = None,
                  extra_body: dict | None = None, max_fail_pct: float = 2.0) -> list[dict]:
    """Write a first-attempt system prompt and first message for each scenario."""
    def one(s: dict) -> dict:
        parsed = _dispatch(flavor.draft_call(s), client, usage, model, temperature,
                           max_tokens, "draft", extra_body)
        return flavor.apply_draft(s, parsed)

    return _run_items(scenarios, one, workers, "stage3:draft", ckpt, max_fail_pct)


# --- stage 4 -----------------------------------------------------------------------


def refine_prompts(drafts: list[dict], client: OpenRouterClient, usage: Usage,
                   flavor: ModuleType, model: str, constitution: str, temperature: float,
                   max_tokens: int, workers: int, ckpt: Checkpoint | None = None,
                   extra_body: dict | None = None, max_fail_pct: float = 2.0) -> list[dict]:
    """Rewrite each draft into a sharper test of its target trait.

    The full constitution and the specific target trait are both injected, so the model
    can tell which principle the prompt is supposed to stress.
    """
    def one(d: dict) -> dict:
        parsed = _dispatch(flavor.refine_call(d, constitution), client, usage, model,
                           temperature, max_tokens, "refine", extra_body)
        return flavor.apply_refine(d, parsed)

    return _run_items(drafts, one, workers, "stage4:refine", ckpt, max_fail_pct)


# --- stage 5 -----------------------------------------------------------------------


def generate_responses(refined: list[dict], client: OpenRouterClient, usage: Usage,
                       flavor: ModuleType, model: str, style_guidance: str,
                       temperature: float, max_tokens: int, workers: int,
                       ckpt: Checkpoint | None = None,
                       extra_body: dict | None = None,
                       max_fail_pct: float = 2.0) -> list[dict]:
    """Answer each refined prompt with explicit reasoning, steered by the target trait."""
    def one(r: dict) -> dict:
        system, user, keys = flavor.respond_call(r, style_guidance)
        parsed = _call_tagged(client, usage, model, system, user, temperature, max_tokens,
                              "respond", keys, extra_body=extra_body)
        return flavor.apply_respond(r, parsed)

    return _run_items(refined, one, workers, "stage5:respond", ckpt, max_fail_pct)


# --- stage 6 -----------------------------------------------------------------------


def rewrite_responses(responses: list[dict], client: OpenRouterClient, usage: Usage,
                      flavor: ModuleType, model: str, constitution: str, temperature: float,
                      max_tokens: int, workers: int, ckpt: Checkpoint | None = None,
                      extra_body: dict | None = None, max_fail_pct: float = 2.0) -> list[dict]:
    """Rewrite each response to maximally exhibit its target trait.

    The blog calls this the critical step: the reviewer sees the whole transcript with the
    relevant constitution section in context, then rewrites rather than scores. A flavor may
    also supply `validate_rewrite`, which rejects and retries a completion whose text breaks
    that corpus's contract rather than merely its JSON shape.
    """
    check = getattr(flavor, "validate_rewrite", None)

    def one(r: dict) -> dict:
        system, user, keys = flavor.rewrite_call(r, constitution)
        parsed = _call_tagged(
            client, usage, model, system, user, temperature, max_tokens, "rewrite", keys,
            validate=(lambda p: check(r, p)) if check else None, extra_body=extra_body,
        )
        return flavor.apply_rewrite(r, parsed)

    return _run_items(responses, one, workers, "stage6:rewrite", ckpt, max_fail_pct)
