# ABOUTME: The LLM stages of the difficult-advice pipeline and the MEM cells, one
# ABOUTME: function each. Every stage takes the previous stage's records and returns the next's.

from __future__ import annotations

import json
import random
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.endpoints.openrouter import ChatResult, OpenRouterClient, map_threaded  # noqa: E402
from utils import extract_json  # noqa: E402

from . import prompts  # noqa: E402
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
                 attempts: int = 3) -> dict[str, str]:
    """Run a completion expecting tagged blocks, retrying if a tag is missing."""
    last = ""
    for attempt in range(attempts):
        nudge = "" if attempt == 0 else (
            f"\n\nYour previous reply was missing a required block. Return ONLY the "
            f"{' and '.join('<' + k + '>...</' + k + '>' for k in keys)} blocks."
        )
        res = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user + nudge}],
            temperature=temperature, max_tokens=max_tokens,
        )
        usage.add(model, res, stage)
        assert res.finish_reason != "length", (
            f"{stage}: {model} hit max_tokens={max_tokens} and truncated. Raise max_tokens.")
        try:
            return _parse_tagged(res.content, keys)
        except ValueError as exc:
            last = f"{exc} | content[:200]={res.content[:200]!r}"
    raise ValueError(f"{stage}: no valid tagged output after {attempts} attempts. {last}")


def _call(client: OpenRouterClient, usage: Usage, model: str, system: str, user: str,
          temperature: float, max_tokens: int, stage: str, attempts: int = 3) -> tuple[Any, ChatResult]:
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


def _resilient(fn, n: int, workers: int, desc: str, max_fail_pct: float = 2.0) -> list:
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

    Returns:
        Successful results, in order, with failures dropped.

    Raises:
        RuntimeError: If the failure rate exceeds `max_fail_pct`.
    """
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
        pct = 100 * len(errors) / max(n, 1)
        print(f"!!! {desc}: {len(errors)}/{n} items failed ({pct:.1f}%). First 3:")
        for e in errors[:3]:
            print("   ", e)
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
            for line in self.path.open():
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
            with self.path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()


def _run_items(items: list[dict], fn, workers: int, desc: str,
               ckpt: Checkpoint | None = None) -> list[dict]:
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
        return _resilient(lambda i: fn(items[i]), len(items), workers, desc)

    todo = [it for it in items if it[ckpt.key] not in ckpt.done]
    if len(todo) < len(items):
        print(f">>> {desc}: resuming -- {len(items) - len(todo)} already saved, "
              f"{len(todo)} remaining")
    if todo:
        def one(i: int) -> dict:
            r = fn(todo[i])
            ckpt.record(r)
            return r

        _resilient(one, len(todo), workers, desc)
    return [ckpt.done[it[ckpt.key]] for it in items if it[ckpt.key] in ckpt.done]


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
        parsed, _ = _call(
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

    nested = _resilient(one, len(batches), workers, "stage2:scenarios")
    return [r for group in nested for r in group]


# --- stage 3 -----------------------------------------------------------------------


def draft_prompts(scenarios: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                  temperature: float, max_tokens: int, workers: int) -> list[dict]:
    """Write a first-attempt system and user prompt for each scenario."""
    def one(i: int) -> dict:
        s = scenarios[i]
        parsed, _ = _call(
            client, usage, model,
            prompts.DRAFT_SYSTEM,
            prompts.DRAFT_USER.format(situation=s["situation"], shortcut=s["shortcut"]),
            temperature, max_tokens, stage="draft",
        )
        return {**s, "draft_system": parsed["system"], "draft_user": parsed["user"]}

    return _resilient(one, len(scenarios), workers, "stage3:draft")


# --- stage 4 -----------------------------------------------------------------------


def refine_prompts(drafts: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                   constitution: str, temperature: float, max_tokens: int,
                   workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each draft prompt into a sharper test of its target trait.

    The full constitution and the specific target trait are both injected, so the model
    can tell which principle the prompt is supposed to stress.
    """
    def one(d: dict) -> dict:
        parsed, _ = _call(
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

    return _run_items(drafts, one, workers, "stage4:refine", ckpt)


# --- stage 5 -----------------------------------------------------------------------


def generate_responses(refined: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                       style_guidance: str, temperature: float, max_tokens: int,
                       workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Answer each refined prompt with explicit reasoning, steered by the target trait."""
    def one(r: dict) -> dict:
        parsed = _call_tagged(
            client, usage, model,
            prompts.RESPONSE_SYSTEM.format(
                system=r["system"], trait_name=r["trait_name"], trait_text=r["trait_text"],
                style_guidance=style_guidance,
            ),
            prompts.RESPONSE_USER.format(user=r["user"]),
            temperature, max_tokens, "respond", ("reasoning", "response"),
        )
        return {**r, "draft_reasoning": parsed["reasoning"], "draft_response": parsed["response"]}

    return _run_items(refined, one, workers, "stage5:respond", ckpt)


# --- stage 6 -----------------------------------------------------------------------


def rewrite_responses(responses: list[dict], client: OpenRouterClient, usage: Usage, model: str,
                      constitution: str, temperature: float, max_tokens: int,
                      workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Rewrite each response to maximally exhibit its target trait.

    The blog calls this the critical step: the reviewer sees the whole transcript with the
    relevant constitution section in context, then rewrites rather than scores.
    """
    def one(r: dict) -> dict:
        parsed = _call_tagged(
            client, usage, model,
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER.format(
                constitution=constitution, trait_name=r["trait_name"],
                trait_text=r["trait_text"], system=r["system"], user=r["user"],
                reasoning=r["draft_reasoning"], response=r["draft_response"],
            ),
            temperature, max_tokens, "rewrite", ("reasoning", "response", "changes"),
        )
        return {**r, "reasoning": parsed["reasoning"], "response": parsed["response"],
                "rewrite_changes": parsed.get("changes", "")}

    return _run_items(responses, one, workers, "stage6:rewrite", ckpt)


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


# --- MEM: model-evaluates-model ----------------------------------------------------
# Documents in which the model reasons about a response to a difficult-advice scenario
# and works out whether it was the right call. Cells run over a COMPLETED
# difficult-advice run (its scenarios and gold responses), so arm differences are
# attributable to format rather than content.


def _eval_response_text(p: dict) -> str:
    """Return the response a critique cell evaluates.

    The flawed cells store a perturbed response under `flawed_response`; the good cells
    have none and fall through to the gold response. Routing through one accessor is
    what keeps the critique generation blind: good and flawed twins build byte-identical
    prompts except for this text.
    """
    return p.get("flawed_response") or p["gold_response"]


def _control_messages(p: dict, constitution: str) -> tuple[str, str]:
    """Build the generation call for the reasoning-only control."""
    return (
        prompts.CONTROL_REASONING_SYSTEM,
        prompts.CONTROL_REASONING_USER.format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=prompts.EXPLICITNESS_STYLES[p["explicitness"]],
            system=p["system"], user=p["user"], response=p["gold_response"],
        ),
    )


def _wrap_transcript(p: dict) -> str:
    """Render the clean transcript-in-user-turn wrapper for a critique record."""
    return prompts.TRANSCRIPT_WRAP_VARIANTS[p["variant_ix"]].format(
        system=p["system"], user=p["user"], response=_eval_response_text(p))


def _critique_messages(p: dict, constitution: str) -> tuple[str, str]:
    """Build the generation call for the other-attribution critique cells (m3/m4).

    The user message is the exact wrapper the training record will carry plus the
    format scaffolding, which assembly strips (the stage-5 precedent). Nothing here may
    depend on whether the response is good or flawed.
    """
    return (
        prompts.MEM_CRITIQUE_SYSTEM.format(
            constitution=constitution, trait_name=p["trait_name"],
            trait_text=p["trait_text"],
            style_line=prompts.EXPLICITNESS_STYLES[p["explicitness"]],
        ),
        _wrap_transcript(p) + "\n\n---\n" + prompts.MEM_CRITIQUE_FORMAT,
    )


def _mem_metadata(r: dict, verdict: str | None = None) -> dict:
    """Metadata every MEM training record carries.

    `supervise` declares which assistant turns are training targets; the self cells will
    set "final" and thread it through the render/mask chain. Everything here today is
    single-final-assistant-turn, hence "all".
    """
    flaw = r.get("flaw") or {}
    return {
        "record_id": r["record_id"],
        "cell": r["cell"],
        "attribution": r["attribution"],
        "response_kind": r["response_kind"],
        "flaw_type": flaw.get("type"),
        "flaw_severity": flaw.get("severity"),
        "explicitness": r["explicitness"],
        "verdict": verdict,
        "scenario_id": r["scenario_id"],
        "trait_id": r["trait_id"],
        "trait_name": r["trait_name"],
        "domain": r.get("domain", ""),
        "situation": r["situation"],
        "shortcut": r.get("shortcut", ""),
        "source_run": r.get("source_run", ""),
        "supervise": "all",
    }


def _assemble_control(r: dict) -> dict:
    """Control record: the original exchange with only the reasoning trace replaced."""
    return {
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["gold_response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _mem_metadata(r),
    }


def _assemble_critique(r: dict) -> dict:
    """Critique record (m3/m4): transcript in the user turn, evaluation as the reply."""
    return {
        "messages": [
            {"role": "system", "content": prompts.MEM_EVAL_SYSTEM},
            {"role": "user", "content": _wrap_transcript(r)},
            {"role": "assistant", "content": r["response"],
             "reasoning_content": r["reasoning"]},
        ],
        "metadata": _mem_metadata(r, verdict=r["assessment"]),
    }


@dataclass(frozen=True)
class CellSpec:
    """One MEM cell: who the response is attributed to, whether it is good or flawed,
    and how its documents are generated and assembled.

    Attributes:
        cell: Registry key, also the `cell` field on every record.
        attribution: "self" | "other" | None (the control evaluates nothing).
        response_kind: "good" | "flawed" | None.
        model_key: Which `models:` block in the config prices and runs this cell; also
            the per-stage usage key, so measured estimates line up per cell family.
        tags: Required tagged blocks in the generation output.
        build_messages: (plan record, constitution) -> (system, user) for the call.
        assemble: generated record -> `{messages, metadata}` training record.
    """

    cell: str
    attribution: str | None
    response_kind: str | None
    model_key: str
    tags: tuple[str, ...]
    build_messages: Callable[[dict, str], tuple[str, str]]
    assemble: Callable[[dict], dict]


# M5 is a mixture of cells, not a cell. The flawed and self cells (m1-m3) join this
# registry in later passes; the perturbation and per-turn-masking machinery they need
# lands with them.
CELLS: dict[str, CellSpec] = {
    "control": CellSpec(
        cell="control", attribution=None, response_kind=None, model_key="control",
        tags=("reasoning",), build_messages=_control_messages,
        assemble=_assemble_control),
    "m4_other_good": CellSpec(
        cell="m4_other_good", attribution="other", response_kind="good",
        model_key="critique", tags=("reasoning", "response", "assessment"),
        build_messages=_critique_messages, assemble=_assemble_critique),
}

# Accepted <assessment> spellings -> canonical verdict.
_VERDICTS = {"sound": "sound", "issue_found": "issue_found", "issue found": "issue_found",
             "issue": "issue_found"}


def _norm_verdict(raw: str) -> str:
    """Canonicalise an <assessment> verdict, raising on anything unrecognised."""
    v = _VERDICTS.get(raw.strip().lower().replace("-", "_"))
    if v is None:
        raise ValueError(f"unrecognised <assessment> verdict: {raw!r}")
    return v


def _weighted_styles(n: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    """Return n style labels matching the weights as closely as rounding allows.

    Deterministic allocation rather than sampling, so coverage over explicitness is by
    construction and a smoke run's tiny n still gets a sensible split.
    """
    assert weights, "explicitness weights must be non-empty"
    total = sum(weights.values())
    keys = sorted(weights)
    counts = {k: int(n * weights[k] / total) for k in keys}
    remainder = sorted(keys, key=lambda k: (counts[k] - n * weights[k] / total, k))
    for k in remainder[: n - sum(counts.values())]:
        counts[k] += 1
    out = [k for k in keys for _ in range(counts[k])]
    rng.shuffle(out)
    return out


def plan_mem_records(source: list[dict], cells: dict[str, int],
                     explicitness: dict[str, float], seed: int,
                     source_run: str = "") -> list[dict]:
    """Allocate source scenarios to MEM cells. Deterministic, no LLM calls.

    Each cell draws its own trait-stratified, seeded sample from the source run, so
    trait coverage is by construction and cross-cell scenario reuse is deliberate
    (cells are separate training arms).

    Args:
        source: A completed run's stage-6 final records.
        cells: Cell name -> number of documents to plan. Zero-count cells are skipped.
        explicitness: Style label -> weight (see prompts.EXPLICITNESS_STYLES).
        seed: Base RNG seed; each cell derives its own stream from it.
        source_run: Provenance label (HF repo or run dir) carried into metadata.

    Returns:
        One plan record per document, `record_id = "<scenario_id>::<cell>"`.

    Raises:
        ValueError: An enabled cell is not registered, or asks for more documents than
            the source run holds.
    """
    enabled = {c: int(n) for c, n in cells.items() if int(n) > 0}
    unknown = sorted(set(enabled) - set(CELLS))
    if unknown:
        raise ValueError(
            f"unregistered cell(s) enabled: {unknown}. Registered: {sorted(CELLS)}. "
            f"The flawed/self cells land in later passes -- keep their counts at 0.")
    bad_style = sorted(set(explicitness) - set(prompts.EXPLICITNESS_STYLES))
    assert not bad_style, f"unknown explicitness style(s): {bad_style}"

    by_trait: dict[str, list[dict]] = {}
    for r in sorted(source, key=lambda r: r["scenario_id"]):
        by_trait.setdefault(r["trait_id"], []).append(r)

    plans: list[dict] = []
    for cell in sorted(enabled):
        want = enabled[cell]
        if want > len(source):
            raise ValueError(f"{cell}: wants {want} documents but the source run has "
                             f"only {len(source)}")
        spec = CELLS[cell]
        rng = random.Random(f"{seed}:{cell}")
        pools = {t: rng.sample(rows, len(rows)) for t, rows in sorted(by_trait.items())}
        order = sorted(pools)
        picked: list[dict] = []
        i = 0
        while len(picked) < want:
            pool = pools[order[i % len(order)]]
            if pool:
                picked.append(pool.pop())
            i += 1
        styles = _weighted_styles(want, explicitness, rng)
        for r, style in zip(picked, styles):
            plans.append({
                "record_id": f"{r['scenario_id']}::{cell}",
                "cell": cell,
                "attribution": spec.attribution,
                "response_kind": spec.response_kind,
                "scenario_id": r["scenario_id"],
                "trait_id": r["trait_id"],
                "trait_name": r["trait_name"],
                "trait_text": r["trait_text"],
                "domain": r.get("domain", ""),
                "situation": r["situation"],
                "shortcut": r.get("shortcut", ""),
                "system": r["system"],
                "user": r["user"],
                "gold_reasoning": r["reasoning"],
                "gold_response": r["response"],
                "flaw": None,
                "explicitness": style,
                "variant_ix": rng.randrange(len(prompts.TRANSCRIPT_WRAP_VARIANTS)),
                "source_run": source_run,
            })
    return plans


def generate_mem_documents(plans: list[dict], client: OpenRouterClient, usage: Usage,
                           model_cfgs: dict[str, dict], constitution: str,
                           workers: int, ckpt: Checkpoint | None = None) -> list[dict]:
    """Generate each planned MEM document via its cell's prompt builder.

    Args:
        plans: Stage-2 plan records (plus perturbations, once those exist).
        client: OpenRouter client.
        usage: Tally; calls are recorded under the cell's `model_key`, so a measured
            estimate can price each cell family separately.
        model_cfgs: model_key -> {model, temperature, max_tokens}.
        constitution: Full constitution text.
        workers: Thread pool size.
        ckpt: Optional checkpoint keyed by `record_id`.

    Returns:
        Plan records extended with the cell's generated fields, failures dropped.
    """
    def one(p: dict) -> dict:
        spec = CELLS[p["cell"]]
        m = model_cfgs[spec.model_key]
        system, user = spec.build_messages(p, constitution)
        parsed = _call_tagged(client, usage, m["model"], system, user,
                              m["temperature"], m["max_tokens"], spec.model_key,
                              spec.tags)
        out = {**p, **{k: parsed[k] for k in spec.tags}}
        if "assessment" in out:
            out["assessment"] = _norm_verdict(out["assessment"])
        return out

    return _run_items(plans, one, workers, "mem:generate", ckpt)


def to_mem_sft(records: list[dict]) -> list[dict]:
    """Assemble generated MEM records into training form, one assembler per cell."""
    return [CELLS[r["cell"]].assemble(r) for r in records]
