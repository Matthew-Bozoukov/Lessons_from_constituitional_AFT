# ABOUTME: Pipeline-agnostic machinery every synth generation pipeline builds on:
# ABOUTME: priced usage tallies, parse-retrying LLM calls, resilient fan-out, checkpoints.

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from src.endpoints.openrouter import ChatResult, OpenRouterClient, map_threaded
from src.utils import extract_json

# USD per 1M tokens, OpenRouter list prices.
PRICES: dict[str, dict[str, float]] = {
    "openai/gpt-5.6-luna": {"in": 0.10, "out": 0.60},
    "openai/gpt-5.6-terra": {"in": 1.00, "out": 6.00},
    "openai/gpt-5.6-sol": {"in": 5.00, "out": 30.00},
    "anthropic/claude-sonnet-5": {"in": 2.00, "out": 10.00},
    "anthropic/claude-sonnet-4.5": {"in": 3.00, "out": 15.00},
    "anthropic/claude-opus-5": {"in": 5.00, "out": 25.00},
    "anthropic/claude-haiku-4.5": {"in": 1.00, "out": 5.00},
    # Non-Anthropic drafting models in the natural-turn configs; OpenRouter rates as
    # of 2026-08-14.
    "x-ai/grok-4.3": {"in": 1.25, "out": 2.50},
    "qwen/qwen3-32b": {"in": 0.08, "out": 0.28},
    # Courtroom's drafter/judge lineages. An unpriced model is silently billed at $0
    # by cost_of AND the estimator, which also blinds the budget_usd guard to that
    # stage's spend -- price every model a config names.
    "google/gemini-2.5-pro": {"in": 1.25, "out": 10.00},
    # Priced at the tier the providers.yaml pin routes to (2026-08-20): flash rides
    # google-vertex/global/priority, 3.1-pro rides google-ai-studio/flex. If a pin
    # changes tier, change the price with it or the budget guard drifts.
    "google/gemini-3.7-flash": {"in": 0.675, "out": 3.375},
    "google/gemini-3.1-pro-preview": {"in": 1.00, "out": 6.00},
    "x-ai/grok-4.6": {"in": 2.00, "out": 6.00},
    "qwen/qwen3-max": {"in": 0.78, "out": 3.90},
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

    def add(self, model: str, res: ChatResult, stage: str = "",
            usd_scale: float = 1.0) -> None:
        """Record one completion against its model and stage.

        `cached_tokens` is tallied so a finished run can show whether prompt caching
        actually engaged. Without it a `<<<cache>>>` marker that silently stopped working
        -- a reworded prompt prefix, a prefix that fell under Anthropic's ~1024-token
        minimum, a non-Anthropic model -- looks exactly like one that is working, and the
        run just quietly costs more. `usd` is still computed at the full rate, so the
        number is a conservative floor on spend rather than a discount applied twice.

        `usd_scale` is the one deliberate exception: OpenRouter's batch API bills at a
        flat 50% of list price, and pricing a batched stage at full rate would make the
        budget guard halt runs that are actually inside budget. Batched calls pass 0.5;
        cache discounts stay uncounted either way, so the floor property survives.
        """
        usd = cost_of(model, res.prompt_tokens, res.completion_tokens) * usd_scale
        for key, bucket in ((model, self.by_model), (stage or "unknown", self.by_stage)):
            b = bucket.setdefault(
                key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "usd": 0.0}
            )
            b["calls"] += 1
            b["prompt_tokens"] += res.prompt_tokens
            b["completion_tokens"] += res.completion_tokens
            b["cached_tokens"] += getattr(res, "cached_tokens", 0)
            b["usd"] += usd

    @property
    def usd(self) -> float:
        """Total spend so far."""
        return sum(b["usd"] for b in self.by_model.values())

    def as_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {"by_model": self.by_model, "by_stage": self.by_stage,
                "total_usd": round(self.usd, 4)}


# Judges return a tag and one sentence inside a tight max_tokens; a model's hidden
# extended thinking otherwise eats that budget and returns EMPTY content with
# finish_reason=length (observed 2026-08-05: 500-token cap, 95+ reasoning tokens,
# content=None). Every judge call in this package passes it -- which is only worth
# anything if there is one of it.
JUDGE_NO_REASONING = {"reasoning": {"enabled": False}}


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
    and the whole object fails to parse. Tags carry arbitrary text -- quotes, apostrophes,
    newlines -- with nothing to escape.

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


def lint_problems(parsed: dict, spec: dict) -> list[str]:
    """Return the reasons tagged output fails a stage's lint contract (empty = pass).

    Lives here rather than in `operators.py` because two callers need it: the generic
    `llm_tagged` operator, and the model-eval-model rewrite stage, whose output IS the
    training target and so must be held to the same content contract.

    The pre-action deliberation voice contract is the archetype: reasoning that reaches for rule
    vocabulary, or that is too short to have done any weighing, is rejected so the call
    retries rather than the corpus absorbing it. `max_chars` is the opposite guard, and
    the one a generated USER turn needs: a follow-up that grows into a paragraph has
    started doing the analysis the assistant's turn is supposed to do.

    `allowed` is the constraint a one-word verdict needs: a tag that must be `held` or
    `revised` and comes back "mostly held" is not a formatting slip to normalise away, it
    is a model that did not answer the question, and the call should be retried.

    Args:
        parsed: Tag name -> text, as returned by a tagged call.
        spec: `{fields, ban_patterns, min_chars, max_chars, allowed}` from the stage entry,
            or a LIST of such contracts. A stage that returns tags of different kinds --
            paragraphs of prose beside a one-word verdict -- needs more than one, since a
            `min_chars` meant for the prose would reject the verdict outright.

    Returns:
        One human-readable problem string per violation.
    """
    if isinstance(spec, list):
        return [p for one in spec for p in lint_problems(parsed, one)]
    problems = []
    min_chars = int(spec.get("min_chars", 0))
    max_chars = int(spec.get("max_chars", 0))
    allowed = [str(v) for v in (spec.get("allowed") or [])]
    patterns = [(pat, re.compile(pat, re.IGNORECASE))
                for pat in spec.get("ban_patterns", [])]
    for tag in spec.get("fields", []):
        if tag not in parsed:
            continue
        text = parsed[tag]
        for pat, rx in patterns:
            m = rx.search(text)
            if m:
                problems.append(f"<{tag}> rule-vocabulary {m.group(0)!r} (matched {pat})")
        if min_chars and len(text) < min_chars:
            problems.append(f"<{tag}> is {len(text)} chars, under the {min_chars} minimum")
        if max_chars and len(text) > max_chars:
            problems.append(f"<{tag}> is {len(text)} chars, over the {max_chars} maximum")
        if allowed and text not in allowed:
            problems.append(f"<{tag}> is {text[:40]!r}, not one of {allowed}")
    return problems


def call_tagged(client: OpenRouterClient, usage: Usage, model: str,
                messages: list[dict], temperature: float, max_tokens: int, stage: str,
                keys: tuple[str, ...], attempts: int = 3,
                extra: dict | None = None) -> dict[str, str]:
    """Run a completion expecting tagged blocks, retrying if a tag is missing.

    Takes a full message list rather than (system, user) so callers can present real
    chat history -- model-eval-model's self-reflection cells put the response under evaluation in a
    genuine assistant turn. The retry nudge is appended to the last message, which must
    be the user turn.
    """
    assert messages[-1]["role"] == "user", "the last message must be the user turn"
    last = ""
    for attempt in range(attempts):
        nudge = "" if attempt == 0 else (
            f"\n\nYour previous reply was missing a required block. Return ONLY the "
            f"{' and '.join('<' + k + '>...</' + k + '>' for k in keys)} blocks."
        )
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] += nudge
        res = client.chat(
            model=model, messages=msgs,
            temperature=temperature, max_tokens=max_tokens,
            **({"extra_body": extra} if extra else {}),
        )
        usage.add(model, res, stage)
        assert res.finish_reason != "length", (
            f"{stage}: {model} hit max_tokens={max_tokens} and truncated. Raise max_tokens.")
        try:
            return _parse_tagged(res.content, keys)
        except ValueError as exc:
            last = f"{exc} | content[:200]={res.content[:200]!r}"
    raise ValueError(f"{stage}: no valid tagged output after {attempts} attempts. {last}")


def call_json(client: OpenRouterClient, usage: Usage, model: str, system: str, user: str,
              temperature: float, max_tokens: int, stage: str, attempts: int = 3,
              extra: dict | None = None) -> tuple[Any, ChatResult]:
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
            **({"extra_body": extra} if extra else {}),
        )
        usage.add(model, res, stage)
        assert res.finish_reason != "length", (
            f"{stage}: {model} hit max_tokens={max_tokens} and truncated its JSON. "
            f"Raise max_tokens for this stage, or lower the per-call batch size."
        )
        try:
            return _parse_json(res.content), res
        except Exception as exc:  # noqa: BLE001 - retried below, raised on the last attempt
            last = f"{type(exc).__name__}: {exc} | content[:200]={res.content[:200]!r}"
    raise ValueError(f"{stage}: unparseable JSON after {attempts} attempts. {last}")


def resilient(fn, n: int, workers: int, desc: str, max_fail_pct: float = 2.0,
              total: int | None = None) -> list:
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
        total: Denominator for the rate -- the WHOLE stage's item count. On a resume the
            still-outstanding items are precisely the ones that already failed once, so
            measuring against `n` alone would make any resume look catastrophic.

    Returns:
        Successful results, in order, with failures dropped.

    Raises:
        RuntimeError: If the failure rate exceeds `max_fail_pct`.
    """
    errors: list[str] = []
    tracebacks: list[str] = []

    def guarded(i: int):
        try:
            return fn(i)
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced below
            errors.append(f"[{i}] {type(exc).__name__}: {exc}")
            # A repr alone cannot say WHERE a library blew up (the 2026-08-14
            # PydanticUserError wave was undiagnosable from it); keep the first few
            # full tracebacks so a systematic failure names its origin.
            if len(tracebacks) < 3:
                import traceback

                tracebacks.append(traceback.format_exc())
            return None

    out = map_threaded(guarded, n, max_workers=workers, desc=desc)
    ok = [r for r in out if r is not None]
    if errors:
        pct = 100 * len(errors) / max(total or n, 1)
        print(f"!!! {desc}: {len(errors)}/{n} items failed ({pct:.1f}%). First 3:")
        for e in errors[:3]:
            print("   ", e)
        for tb in tracebacks[:1]:
            print("    first failure's traceback:")
            print("    " + tb.replace("\n", "\n    "))
        if pct > max_fail_pct:
            raise RuntimeError(
                f"{desc}: {pct:.1f}% of items failed, above max_fail_pct={max_fail_pct}. "
                f"This looks systematic rather than incidental."
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


def run_items(items: list[dict], fn, workers: int, desc: str,
              ckpt: Checkpoint | None = None, max_fail_pct: float = 2.0) -> list[dict]:
    """Process items concurrently, skipping and recording via a checkpoint when given.

    Args:
        items: Input records.
        fn: Callable taking one input record and returning the output record.
        workers: Thread pool size.
        desc: Progress label.
        ckpt: Optional checkpoint for resume and incremental save.

    Returns:
        Output records in input order, with failures dropped.
    """
    if ckpt is None:
        return resilient(lambda i: fn(items[i]), len(items), workers, desc,
                         max_fail_pct=max_fail_pct)

    todo = [it for it in items if it[ckpt.key] not in ckpt.done]
    if len(todo) < len(items):
        print(f">>> {desc}: resuming -- {len(items) - len(todo)} already saved, "
              f"{len(todo)} remaining")
    if todo:
        def one(i: int) -> dict:
            r = fn(todo[i])
            ckpt.record(r)
            return r

        resilient(one, len(todo), workers, desc, max_fail_pct=max_fail_pct,
                  total=len(items))
    return [ckpt.done[it[ckpt.key]] for it in items if it[ckpt.key] in ckpt.done]


# --- optional async batching (OpenRouter batch API, 50% token pricing) ----------------

BATCH_URL = "https://openrouter.ai/api/beta/batches"
# Below this many outstanding records the submit/poll overhead outweighs the discount
# and the stage silently stays interactive -- which also keeps `--smoke` interactive.
BATCH_MIN_ITEMS = 8
# Requests per batch job. Chunked because results are all-or-nothing PER JOB (`results`
# is null until the whole job completes): an expired/failed chunk loses only its slice.
BATCH_CHUNK = 500
BATCH_POLL_S = 30


def run_batch(model: str, requests: dict[str, dict], stage: str, state_path: Path,
              collect, chunk: int = BATCH_CHUNK, poll_s: int = BATCH_POLL_S) -> None:
    """Push `requests` (custom_id -> body) through OpenRouter's batch API.

    Submission state (batch ids + their custom_ids) persists at `state_path`, so a
    killed run resumes the SAME jobs instead of paying to resubmit them. Nothing is
    marked collected in the state: results stay retrievable by GET until the job
    expires, so `collect(custom_id, completion_payload)` -- the caller's parse+
    checkpoint hook -- is simply re-run on a resume, and the caller's checkpoint is
    what makes that idempotent. A job that ends failed/expired/cancelled is reported
    loudly and its requests are left uncollected; the caller's interactive mop-up owns
    them. The state file is removed once every job reaches a terminal status.
    """
    import time

    import requests as http

    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    # A resumed run's outstanding set may differ from what an old state was built
    # against (mopped-up records, a re-cut todo); jobs with no overlap are dropped
    # and only never-submitted ids go out again.
    state["batches"] = [b for b in state["batches"] if set(b["ids"]) & set(requests)]
    if state["batches"]:
        print(f">>> {stage}: resuming {len(state['batches'])} batch job(s)")
    submitted = {i for b in state["batches"] for i in b["ids"]}
    to_submit = [cid for cid in requests if cid not in submitted]
    for c0 in range(0, len(to_submit), chunk):
        ids = to_submit[c0:c0 + chunk]
        # Field ORDER matters: the API stream-parses and requires endpoint+model
        # before the requests array.
        r = http.post(BATCH_URL, headers=headers, json={
            "endpoint": "/v1/chat/completions", "model": model,
            "requests": [{"custom_id": cid, "body": requests[cid]} for cid in ids]})
        r.raise_for_status()
        state["batches"].append({"batch_id": r.json()["id"], "ids": ids})
        # State written after EVERY submit: a crash mid-submission strands nothing.
        state_path.write_text(json.dumps(state))
        print(f">>> {stage}: submitted batch {r.json()['id']} ({len(ids)} requests)")

    done: set[str] = set()
    dead: set[str] = set()
    while True:
        pending = [b for b in state["batches"]
                   if b["batch_id"] not in done | dead]
        if not pending:
            break
        for b in pending:
            s = http.get(f"{BATCH_URL}/{b['batch_id']}", headers=headers).json()
            status = s.get("status")
            if status == "completed":
                by_id = {res.get("custom_id"): res for res in s.get("results") or []}
                for cid in b["ids"]:
                    if cid not in requests:
                        continue
                    res = by_id.get(cid) or {}
                    resp = res.get("response") or {}
                    if not res.get("error") and resp.get("status_code") == 200:
                        collect(cid, resp.get("body") or {})
                done.add(b["batch_id"])
                print(f">>> {stage}: batch {b['batch_id']} collected", flush=True)
            elif status in ("failed", "expired", "cancelled"):
                dead.add(b["batch_id"])
                print(f"!!! {stage}: batch {b['batch_id']} ended {status} -- its "
                      f"{len(b['ids'])} request(s) fall to the interactive path",
                      flush=True)
            else:
                print(f">>> {stage}: batch {b['batch_id']} {status}: "
                      f"{s.get('request_counts') or {}}", flush=True)
        if any(b["batch_id"] not in done | dead for b in state["batches"]):
            time.sleep(poll_s)
    state_path.unlink(missing_ok=True)


def run_items_batched(items: list[dict], one, build_request, parse_result, *,
                      usage: Usage, model: str, stage: str, key: str, run_dir: Path,
                      workers: int, desc: str, ckpt: Checkpoint | None = None,
                      max_fail_pct: float = 2.0) -> list[dict]:
    """`run_items`, but with the bulk routed through the async batch API first.

    Semantics: one interactive warming call (the first record -- its result is used,
    and on Anthropic its cache write is what gives the batched fleet a prefix to hit),
    then every remaining record as ONE batched attempt-0 request, then the existing
    interactive path (`one`, with its full retry/nudge/lint budget) mopping up
    whatever the batch round could not deliver: transport errors, dead jobs, parse
    and lint rejects. Batch results are parsed by `parse_result(record, ChatResult)`
    -- a single attempt, no retries -- and checkpointed the moment their job lands,
    so a crash mid-collection re-reads the still-live batch results instead of
    re-paying for them. Batched completions are tallied at 0.5x list price, which is
    what the batch API bills.

    Falls back to plain `run_items` when fewer than BATCH_MIN_ITEMS records are
    outstanding (smoke runs, resumes near completion) or when records lack a unique
    `key` -- resume across restarts needs a stable custom_id per record.
    """
    from src.endpoints.openrouter import result_from_payload

    todo = [it for it in items if ckpt is None or it[ckpt.key] not in ckpt.done]
    keys = [str(it.get(key, "")) for it in todo]
    if len(todo) < BATCH_MIN_ITEMS or "" in keys or len(set(keys)) != len(keys):
        if len(todo) >= BATCH_MIN_ITEMS:
            print(f">>> {desc}: batch requested but records lack a unique "
                  f"{key!r}; running interactively")
        return run_items(items, one, workers, desc, ckpt, max_fail_pct=max_fail_pct)

    done_free: dict[str, dict] = {}  # collected outputs for a checkpoint-less stage

    def keep(r: dict, out: dict) -> None:
        if ckpt is not None:
            ckpt.record(out)
        else:
            done_free[str(r[key])] = out

    warm, rest = todo[0], todo[1:]
    try:
        keep(warm, one(warm))
    except Exception as exc:  # noqa: BLE001 - the record rejoins the mop-up below
        print(f"!!! {desc}: warming call failed ({type(exc).__name__}: {exc}); "
              "its record joins the mop-up")

    by_key = {str(r[key]): r for r in rest}
    bodies = {cid: build_request(r) for cid, r in by_key.items()}
    n_ok, n_fail = 0, 0

    def collect(cid: str, payload: dict) -> None:
        nonlocal n_ok, n_fail
        r = by_key[cid]
        try:
            res = result_from_payload(model, payload)
            usage.add(model, res, stage, usd_scale=0.5)
            keep(r, parse_result(r, res))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - the record falls to the mop-up
            n_fail += 1
            if n_fail <= 3:
                print(f"    {desc}: batch result {cid} rejected "
                      f"({type(exc).__name__}: {exc}); will mop up interactively")

    run_batch(model, bodies, desc, run_dir / f".batch_{desc}.json", collect)
    print(f">>> {desc}: batch delivered {n_ok}/{len(rest)}; mopping up the rest "
          "interactively")

    if ckpt is not None:
        return run_items(items, one, workers, desc, ckpt, max_fail_pct=max_fail_pct)
    remaining = [r for r in todo if str(r[key]) not in done_free]
    if remaining:
        def mop(i: int) -> dict:
            out = one(remaining[i])
            done_free[str(remaining[i][key])] = out
            return out

        resilient(mop, len(remaining), workers, desc, max_fail_pct=max_fail_pct,
                  total=len(items))
    return [done_free[str(it[key])] for it in items if str(it[key]) in done_free]


from dataclasses import dataclass, field


@dataclass
class Ctx:
    """Everything a stage function may need, one argument.

    `vars` holds template variables shared across stages (constitution, style guidance,
    ...); `manifest_extra` collects run-level metadata a stage wants in the manifest.
    The OpenRouter client is created lazily, so deterministic-only runs (and offline
    tests) never touch credentials.

    `cache` is the run's StageCache, so a stage producing a side artefact (a report) can
    mirror it to HF the way snapshots are. It is None when a Ctx is built outside
    `pipeline.run` (the `topup` verb, offline tests) -- fall back to a local write.
    """

    cfg: dict
    usage: Usage
    workers: int
    run_dir: Path
    smoke: bool
    vars: dict = field(default_factory=dict)
    manifest_extra: dict = field(default_factory=dict)
    cache: Any = None
    # Set by a stage to halt the run after it, keeping everything already produced.
    # The engine writes the manifest and stops; the CLI turns it into an exit code.
    stop: str | None = None
    _client: Any = None

    @property
    def client(self):
        """The shared OpenRouter client, created on first paid call."""
        if self._client is None:
            self._client = OpenRouterClient()
        return self._client

    @property
    def constitution(self) -> str:
        return self.vars["constitution"]


@dataclass(frozen=True)
class Stage:
    """One executable step, built from a config stage entry by its operator.

    Attributes:
        name: Snapshot name (`stage_<position>_<name>.jsonl`) and the ablation handle.
        fn: (ctx, records, ckpt) -> records. The whole stage.
        paid: Whether the stage spends money -- the budget guard runs before paid stages.
        checkpoint_key: Record field for per-item resume; None = stage-level cache only.
        ablate_fn: Null-operation (records -> records), built by the engine from the
            stage entry's `ablate_with` field-copy map. Absent = not ablatable.
        skip: (ctx, records) -> bool for structurally inapplicable stages.
        on_cached: Called when the snapshot is reused, to restore ctx.vars /
            manifest_extra a cache hit would otherwise lose.
        preview: Render one line of the first output record for the run log.
        observer: The stage inspects the records and returns them unchanged. It writes
            NO snapshot and takes NO position number, so inserting one anywhere in
            `stages:` renumbers nothing after it and existing run dirs stay resumable.
            It is never cached either: re-reading records it did not produce is cheap,
            and anything it pays for is protected by its own checkpoint.
    """

    name: str
    fn: Any
    paid: bool = False
    checkpoint_key: str | None = None
    ablate_fn: Any = None
    skip: Any = None
    on_cached: Any = None
    preview: Any = None
    observer: bool = False


def model_cfg(cfg: dict, key: str) -> dict[str, Any]:
    """Return the merged model settings for one stage.

    Args:
        cfg: Full run config.
        key: Stage key under `models`.

    Returns:
        Dict with model, temperature and max_tokens.
    """
    defaults = cfg.get("defaults", {})
    block = cfg["models"][key]
    out = {
        "model": block["model"],
        "temperature": float(block.get("temperature", defaults.get("temperature", 1.0))),
        "max_tokens": int(block.get("max_tokens", defaults.get("max_tokens", 4096))),
    }
    # OpenRouter's unified reasoning control. Extended thinking is billed as completion
    # tokens, so a stage that only assembles text rather than judging it can cost several
    # times its visible output. `reasoning: {enabled: false}` on such a stage is pure
    # saving; the stages that weigh the constitution keep it. (Measured: $81 off one run.)
    reasoning = block.get("reasoning", defaults.get("reasoning"))
    if reasoning is not None:
        out["extra_body"] = {"reasoning": dict(reasoning)}
    # Provider routing is NOT a synth concern: one provider per model, globally, in
    # configs/endpoints/providers.yaml (applied inside OpenRouterClient on every call).
    # Rejecting the key here keeps stale configs loud instead of silently ignored.
    if "provider" in block or "provider" in defaults:
        raise ValueError(
            f"models.{key}: provider routing moved to configs/endpoints/providers.yaml "
            "(one provider per model, applied by OpenRouterClient) — remove `provider:` "
            "from synth configs")
    return out


def measured_per_stage(manifest_path: str) -> tuple[dict[str, dict[str, float]], dict]:
    """Return per-stage per-call token averages from a completed run's manifest.

    Args:
        manifest_path: Path to manifest.json.

    Returns:
        (mapping stage -> {in_per_call, out_per_call}, the manifest itself).
    """
    m = json.loads(Path(manifest_path).read_text())
    by_stage = m["usage"].get("by_stage", {})
    out = {}
    for stage, b in by_stage.items():
        calls = max(b["calls"], 1)
        out[stage] = {"in_per_call": b["prompt_tokens"] / calls,
                      "out_per_call": b["completion_tokens"] / calls}
    return out, m
