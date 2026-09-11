# ABOUTME: Deliberative SFT generation: sample Qwen with the constitution, judge, export without it.
# ABOUTME: Owns configuration, best-of-N generation, spec-aware filtering, resume, accounting, publication.

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import re
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from src.data.synth.ours.hf_cache import StageCache, read_jsonl
from src.infra.endpoints.openrouter import OpenRouterClient, provider_pin, provider_price
from src.infra.huggingface import hf_api, hf_repo_id, training_data_tags
from src.model_profile import resolve_trace
from src.naming import artifact_name, check_style, synth_name
from src.utils import git_sha, origin_url, timestamp

from .data import export_row, generation_messages, load_prompts
from .judge import (JUDGE_FIELDS, candidate_score, format_rejection, judge_messages, leak_pattern,
                    parse_scores, select)

FILTER_KEYS = {"candidates", "resample_rounds", "threshold", "min_rows", "judge"}
JUDGE_KEYS = {"model", "runs", "temperature", "max_tokens"}
JUDGE_OPTIONAL_KEYS = {"reasoning", "retries"}
# How hard the judge thinks: OpenRouter's unified `reasoning` block, passed through as-is.
# `effort: low|medium|high` (Anthropic maps it to a thinking budget) or `max_tokens: N`.
# Left unset, Sonnet chose its own budget and spent ~5k output tokens per call on the
# 2026-09-10 smoke, over half the judge bill; nothing about a 4-candidate comparison needs it.
JUDGE_REASONING_KEYS = {"effort", "max_tokens", "exclude"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _positive_int(name: str, value, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _priced(model: str) -> dict:
    pin = provider_pin(model)
    price = provider_price(model)
    if not price or any(not math.isfinite(float(price[k])) or float(price[k]) <= 0
                        for k in ("in", "out")):
        raise ValueError(f"The provider pin for {model} must have positive input/output prices")
    return pin


def generator_pin(cfg: dict) -> tuple[dict, dict]:
    """(provider routing object, price) for the generator.

    The registry (providers.yaml) pins ONE host per model id for every caller. A delib
    config may override it with `provider: {order: [<host>], price: {in, out}}` -- the
    teacher's serving host is part of THIS artifact's identity (it enters the resume
    signature, the manifest, the card and every row's provenance), and a different
    document type may need the registry's host (the reasoning-backfill probe needs the
    one host that honours `reasoning.max_tokens`). Phala was chosen 2026-09-10 because it is
    the only Qwen3.6 host that advertises prefix-cache pricing; the constitution is ~70% of
    every generation prompt.
    """
    override = cfg.get("provider")
    if override is None:
        return _priced(cfg["model"]), provider_price(cfg["model"])
    if (not isinstance(override, dict) or set(override) != {"order", "price"}
            or not isinstance(override["order"], list) or len(override["order"]) != 1
            or set(override["price"]) != {"in", "out"}
            or any(float(override["price"][k]) <= 0 for k in ("in", "out"))):
        raise ValueError("`provider:` must be {order: [<one host>], price: {in: $/M, out: $/M}}")
    return {"order": [str(override["order"][0])], "allow_fallbacks": False}, dict(override["price"])


def validate_config(cfg: dict) -> None:
    """Reject unsupported options before any generation or publication."""
    allowed = {"method", "pipeline", "source", "constitution", "model", "provider", "sampling",
               "generation_prompt", "retry_generation_prompt", "filter", "judge_prompt", "workers",
               "budget_usd", "limit", "output_dir", "hf_push", "hf_private", "smoke"}
    if unknown := set(cfg) - allowed:
        raise ValueError(f"Unsupported deliberative_alignment settings: {sorted(unknown)}")
    required = allowed - {"limit", "smoke", "provider", "retry_generation_prompt"}
    if missing := required - set(cfg):
        raise ValueError(f"Missing deliberative_alignment settings: {sorted(missing)}")
    if cfg["method"] != "deliberative_alignment":
        raise ValueError("method must be deliberative_alignment")
    check_style(cfg["pipeline"])
    if not isinstance(cfg["source"], dict) or set(cfg["source"]) - {"repo", "revision", "rows"}:
        raise ValueError("source accepts only repo, optional revision and optional rows; intake is always dataset.jsonl")
    if not cfg["source"].get("repo"):
        raise ValueError("source.repo is required")
    # The generator is whatever model the config names -- Qwen (self-generation, the original
    # recipe) or a stronger teacher such as Sonnet (the 2026-09-11 control: same prompts,
    # constitution, judge and training; only the teacher changes) -- as long as it is pinned
    # to one provider below and returns a native reasoning trace (checked per completion).
    if not isinstance(cfg["model"], str) or "/" not in cfg["model"]:
        raise ValueError("model must be an OpenRouter model id (<provider>/<model>)")
    pin, _ = generator_pin(cfg)
    if len(pin.get("order") or []) != 1 or pin.get("allow_fallbacks") is not False:
        raise ValueError("The generator must be pinned to exactly ONE provider with allow_fallbacks: false "
                         "(providers.yaml, or this config's `provider:` override)")
    sampling = cfg["sampling"]
    if not isinstance(sampling, dict) or set(sampling) - {"temperature", "max_tokens", "reasoning", "seed"}:
        raise ValueError("sampling accepts temperature, max_tokens, reasoning, and optional seed")
    if not {"temperature", "max_tokens", "reasoning"} <= sampling.keys():
        raise ValueError("sampling requires temperature, max_tokens, and reasoning")
    reasoning = sampling["reasoning"]
    if not isinstance(reasoning, dict):
        raise ValueError("sampling.reasoning must be a mapping")
    if templated_reasoning(cfg):
        # Anthropic's API returns a SUMMARY of the model's thinking and, on Sonnet 5, thinks
        # adaptively (no trace at all on some prompts; no budget forces one -- probed
        # 2026-09-11). Neither is training data. `templated: true` asks for no hidden
        # thinking and has the generation_prompt make the model write its deliberation
        # inside <think></think> ahead of the answer; the format gate splits it there.
        # Only for Anthropic models: where a provider returns the native trace (Qwen), the
        # native trace is authoritative and this would replace it with a weaker imitation.
        if reasoning != {"templated": True}:
            raise ValueError("sampling.reasoning with `templated: true` takes no other keys")
        if not cfg["model"].startswith("anthropic/"):
            raise ValueError("templated reasoning is for Anthropic generators only (summarised, adaptive "
                             "thinking); a model that returns its native trace must use it")
        if "<think>" not in cfg["generation_prompt"] or "</think>" not in cfg["generation_prompt"]:
            raise ValueError("templated reasoning needs the generation_prompt to instruct a "
                             "<think>...</think> block before the answer")
    else:
        if reasoning.get("enabled") is not True or reasoning.get("exclude") is not False:
            raise ValueError("sampling.reasoning must set enabled: true and exclude: false "
                             "(or, for an Anthropic generator, templated: true)")
        if set(reasoning) - {"enabled", "exclude", "max_tokens", "effort"}:
            raise ValueError("Only enabled, exclude, max_tokens and effort are supported in reasoning")
    if "effort" in reasoning and reasoning["effort"] not in ("low", "medium", "high"):
        raise ValueError("sampling.reasoning.effort must be low, medium or high")
    if "effort" in reasoning and "max_tokens" in reasoning:
        raise ValueError("sampling.reasoning takes effort OR max_tokens, not both")
    integer_settings = {"sampling.max_tokens": sampling["max_tokens"], "workers": cfg["workers"]}
    if "max_tokens" in reasoning:
        integer_settings["reasoning.max_tokens"] = reasoning["max_tokens"]
    if cfg.get("limit") is not None:
        integer_settings["limit"] = cfg["limit"]
    if any(type(value) is not int or value <= 0 for value in integer_settings.values()):
        raise ValueError("Token limits, workers and limit must be positive integers")
    if "seed" in sampling and (type(sampling["seed"]) is not int or sampling["seed"] < 0):
        raise ValueError("sampling.seed must be a nonnegative integer")
    if type(sampling["temperature"]) not in (int, float) or not 0 <= sampling["temperature"] <= 2:
        raise ValueError("Invalid sampling max_tokens or temperature")
    if "max_tokens" in reasoning and not 1024 <= int(reasoning["max_tokens"]) < int(sampling["max_tokens"]):
        raise ValueError("reasoning.max_tokens must be >=1024 and smaller than sampling.max_tokens")
    if type(cfg["budget_usd"]) not in (int, float) or not math.isfinite(cfg["budget_usd"]) or cfg["budget_usd"] <= 0:
        raise ValueError("budget_usd must be a positive finite number")
    if "{constitution}" not in cfg["generation_prompt"]:
        raise ValueError("generation_prompt must include the full {constitution}")
    if "retry_generation_prompt" in cfg:
        # The stronger wording, used only for a prompt whose candidate came back with no
        # <think> block (Sonnet skips the block under urgency: prompt 9 of the 2026-09-11
        # smoke, 0/7 blocks with the standard wording, 3/3 with this one). Every other
        # prompt keeps the plain instruction, so the wording never shapes traces that did
        # not need it.
        retry = cfg["retry_generation_prompt"]
        if not templated_reasoning(cfg):
            raise ValueError("retry_generation_prompt is only meaningful with templated reasoning")
        if not isinstance(retry, str) or "{constitution}" not in retry or "<think>" not in retry or "</think>" not in retry:
            raise ValueError("retry_generation_prompt must include the full {constitution} and the <think></think> instruction")
    for key in ("hf_push", "hf_private"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    _validate_filter(cfg)


def _validate_filter(cfg: dict) -> None:
    flt = cfg["filter"]
    if not isinstance(flt, dict) or set(flt) != FILTER_KEYS:
        raise ValueError(f"filter must set exactly {sorted(FILTER_KEYS)}")
    _positive_int("filter.candidates", flt["candidates"])
    _positive_int("filter.resample_rounds", flt["resample_rounds"], minimum=0)
    _positive_int("filter.min_rows", flt["min_rows"])
    if type(flt["threshold"]) is not int or not 1 <= flt["threshold"] <= 10:
        raise ValueError("filter.threshold must be an integer score from 1 to 10")
    judge = flt["judge"]
    if not isinstance(judge, dict) or not JUDGE_KEYS <= set(judge) <= JUDGE_KEYS | JUDGE_OPTIONAL_KEYS:
        raise ValueError(f"filter.judge must set {sorted(JUDGE_KEYS)} and may set {sorted(JUDGE_OPTIONAL_KEYS)}")
    if "reasoning" in judge:
        r = judge["reasoning"]
        if not isinstance(r, dict) or not r or set(r) - JUDGE_REASONING_KEYS:
            raise ValueError(f"filter.judge.reasoning accepts {sorted(JUDGE_REASONING_KEYS)}")
        if "effort" in r and r["effort"] not in ("low", "medium", "high"):
            raise ValueError("filter.judge.reasoning.effort must be low, medium or high")
        if "max_tokens" in r and not 1024 <= int(r["max_tokens"]) < int(judge["max_tokens"]):
            raise ValueError("filter.judge.reasoning.max_tokens must be >= 1024 and below judge.max_tokens")
    if "retries" in judge:
        _positive_int("filter.judge.retries", judge["retries"], minimum=0)
    if str(judge["model"]).startswith("qwen/"):
        raise ValueError("The judge must not be the model being taught")
    _priced(judge["model"])
    _positive_int("filter.judge.runs", judge["runs"])
    _positive_int("filter.judge.max_tokens", judge["max_tokens"])
    if type(judge["temperature"]) not in (int, float) or not 0 <= judge["temperature"] <= 2:
        raise ValueError("filter.judge.temperature must lie in [0, 2]")
    if not isinstance(cfg["judge_prompt"], str) or not all(f in cfg["judge_prompt"] for f in JUDGE_FIELDS):
        raise ValueError(f"judge_prompt must include all of {JUDGE_FIELDS}")


def _effective(cfg: dict, smoke: bool) -> dict:
    result = OmegaConf.to_container(OmegaConf.merge(cfg, cfg.get("smoke", {}) if smoke else {}), resolve=True)
    validate_config(result)
    return result


# Phrases that only the templated instructions (standard or retry) could have put in the
# user-facing answer: the block is "never shown to the user", so an answer that refers to it
# has leaked the harness, exactly as an answer naming the constitution has.
TEMPLATED_LEAK_RE = re.compile(
    r"\bthink block\b|\bworking notes\b|never shown to (?:the )?user|delays? nothing"
    r"|\bmy reasoning (?:above|block)\b|\bthe reasoning block\b|\bas I (?:reasoned|worked out|thought) (?:through )?above\b"
    r"|\bin my reasoning\b|\bthese instructions\b", re.I)
# Not in the pattern: "the reasoning above" / "as noted above" -- a structured answer refers to
# its OWN earlier sections that way (smoke 2026-09-11, prompt 7: "the architecture above").
NO_BLOCK = "Missing templated <think> reasoning; refusing answer-only SFT data"


def templated_reasoning(cfg: dict) -> bool:
    """True when the generator writes its trace inside <think></think> in the answer text
    (the Anthropic form) rather than returning it out of band (the native form)."""
    reasoning = (cfg.get("sampling") or {}).get("reasoning")
    return isinstance(reasoning, dict) and reasoning.get("templated") is True


def _completion(result, record: dict, leak=None, host: str = "alibaba", templated: bool = False) -> dict:
    """The format gate: one complete native reasoning response, or a ValueError naming why.

    A ValueError here is a FORMAT rejection of this candidate (the paper's first filter);
    it is recorded and never retried. Anything that points at the run rather than the
    sample -- the wrong serving provider -- is a RuntimeError and stops the run.
    """
    if result.provider.casefold() != host.casefold():
        raise RuntimeError(f"Unexpected serving provider: {result.provider!r} (pinned {host!r})")
    if result.finish_reason not in {"stop", "tool_calls"}:
        raise ValueError(f"Incomplete completion: finish_reason={result.finish_reason!r}")
    if templated:
        # No thinking was requested; the trace is the <think> block the prompt asked for.
        if result.reasoning_content:
            raise ValueError("Native reasoning returned alongside templated reasoning")
        trace, answer = resolve_trace(result.content, None)
        if not trace.strip():
            raise ValueError(NO_BLOCK)
        if "<think>" in answer or "</think>" in answer:
            raise ValueError("Stray <think> tag in the final answer")
        if m := TEMPLATED_LEAK_RE.search(answer):
            raise ValueError(f"Answer refers to the templated instructions: {m.group(0)!r}")
    else:
        # A native trace is authoritative; a final answer may legitimately quote <think>.
        trace, answer = ((result.reasoning_content, result.content) if result.reasoning_content
                         else resolve_trace(result.content, None))
        if not trace.strip():
            raise ValueError("Missing native reasoning; refusing answer-only SFT data")
    if not answer.strip() and not result.tool_calls:
        raise ValueError("Missing final answer or tool calls")
    if result.tool_calls and not record.get("tools"):
        raise ValueError("Generator returned tool calls for a prompt without tools")
    if result.finish_reason == "tool_calls" and not result.tool_calls:
        raise ValueError("tool_calls finish reason without tool calls")
    if reason := format_rejection(answer, leak):
        raise ValueError(reason)
    assistant = {"role": "assistant", "content": answer, "reasoning_content": trace}
    if result.tool_calls:
        assistant["tool_calls"] = result.tool_calls
    return assistant


def _sample(client, record: dict, candidate: int, cfg: dict, augmentation: str, leak=None,
            prompt_kind: str = "standard") -> dict:
    sampling = copy.deepcopy(cfg["sampling"])
    reasoning = sampling.pop("reasoning")
    templated = templated_reasoning(cfg)
    kwargs = {"tools": record["tools"]} if record.get("tools") else {}
    pin, _ = generator_pin(cfg)
    # Templated: hidden thinking explicitly OFF. Sonnet 5 thinks by default when no
    # `reasoning` body is sent and then follows the <think> instruction in that hidden
    # channel (16/16 candidates on the 2026-09-11 first attempt); `enabled: false` is what
    # makes it write the block in the answer instead.
    extra = {"reasoning": {"enabled": False}, "provider": pin} if templated \
        else {"reasoning": reasoning, "provider": pin}
    result = client.chat(model=cfg["model"], messages=generation_messages(record, augmentation),
                         extra_body=extra, **sampling, **kwargs)
    attempt = {"id": record["id"], "candidate": candidate, "prompt": prompt_kind, "response": asdict(result)}
    try:
        assistant = _completion(result, record, leak, host=pin["order"][0], templated=templated)
        # Validate export/tool schemas while the response is still checkpointed as an attempt.
        export_row(record, assistant, {})
        attempt["assistant"] = assistant
    except (ValueError, TypeError) as exc:
        attempt["rejected"] = str(exc)
    return attempt


def _judge(client, record: dict, group: list[dict], run: int, cfg: dict, constitution: str) -> dict:
    """Score one prompt's candidates side by side: one call, one score line per candidate."""
    judge = cfg["filter"]["judge"]
    candidates = [(a["candidate"], a["assistant"]) for a in group]
    extra = {"reasoning": judge["reasoning"]} if judge.get("reasoning") else {}
    result = client.chat(model=judge["model"],
                         messages=judge_messages(record, candidates, cfg["judge_prompt"], constitution),
                         temperature=judge["temperature"], max_tokens=judge["max_tokens"],
                         **({"extra_body": extra} if extra else {}))
    verdict = {"id": record["id"], "candidates": [c for c, _ in candidates], "run": run,
               "response": asdict(result)}
    try:
        if result.finish_reason != "stop":
            raise ValueError(f"judge truncated: finish_reason={result.finish_reason!r}")
        scores = parse_scores(result.content, [c for c, _ in candidates])
        verdict["scores"] = {str(c): score for c, score in scores.items()}
    except ValueError as exc:
        # A judge that fails to score is an error to retry on resume, never a rejection.
        verdict["error"] = str(exc)
    return verdict


def _judge_runs(client, record: dict, group: list[dict], runs: list[int], cfg: dict,
                constitution: str) -> list[dict]:
    """One prompt's judge runs, IN SEQUENCE: the message is identical across runs (only
    the sampling differs), so with a cache breakpoint at its end every run after the first
    reads the whole prompt from cache -- provided the first has finished being written,
    which concurrent submission of the runs would not guarantee. A run that raises is
    recorded as an error record (class name only: HTTP text may carry credentials) and
    the remaining runs still proceed."""
    # A judge that truncates or fails to score is a judging fault, not a fact about the
    # candidates: retried here (`filter.judge.retries`, default 2) before the round can read
    # the prompt as survivor-less and spend a Qwen resample on it. Each attempt is recorded,
    # so the manifest shows the retries.
    out = []
    retries = int(cfg["filter"]["judge"].get("retries", 2))
    for k in runs:
        for attempt in range(retries + 1):
            try:
                v = _judge(client, record, group, k, cfg, constitution)
            except Exception as exc:
                v = {"id": record["id"], "candidates": [a["candidate"] for a in group],
                     "run": k, "error": type(exc).__name__}
            if attempt:
                v["attempt"] = attempt
            out.append(v)
            if "scores" in v:
                break
    return out


class _MirrorThrottle:
    """Decide when a checkpoint may be mirrored to the Hub, and never let the mirror kill a run.

    The Hub allows 128 commits per repo per hour. Local checkpoints are the record; the
    mirror is a convenience for a dead driver, so it is attempted at most once per
    `interval_s` (always when `final`), and a failure (a 429, a network blip) is printed
    and skipped rather than raised -- the next checkpoint carries everything anyway.
    """

    def __init__(self, interval_s: float = 300.0, clock=time.monotonic):
        self.interval_s, self.clock, self.last, self.skipped = interval_s, clock, None, 0

    def due(self, final: bool = False) -> bool:
        return final or self.last is None or self.clock() - self.last >= self.interval_s

    def attempt(self, fn, final: bool = False) -> bool:
        if not self.due(final):
            return False
        try:
            fn()
        except Exception as exc:  # best-effort: the local checkpoint is authoritative
            self.skipped += 1
            print(f">>> Hub mirror skipped ({type(exc).__name__}); local checkpoint is the record",
                  flush=True)
            return False
        self.last = self.clock()
        return True


def _usage(attempts: list[dict], price: dict) -> dict:
    responses = [a["response"] for a in attempts if "response" in a]
    estimates = [r["prompt_tokens"] * float(price["in"]) / 1e6
                 + r["completion_tokens"] * float(price["out"]) / 1e6 for r in responses]
    costs = [r.get("cost") if r.get("cost") is not None else estimate
             for r, estimate in zip(responses, estimates)]
    return {"calls_with_usage": len(responses),
            "calls_without_usage": sum("response" not in a for a in attempts),
            "prompt_tokens": sum(r["prompt_tokens"] for r in responses),
            "completion_tokens": sum(r["completion_tokens"] for r in responses),
            "total_usd": sum(costs), "estimated_cost_calls": sum(r.get("cost") is None for r in responses)}


def _scores(verdicts: list[dict]) -> dict[tuple[str, int], list[int]]:
    scores: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for verdict in verdicts:
        for candidate, score in verdict.get("scores", {}).items():
            scores[(verdict["id"], int(candidate))].append(score)
    return scores


def _scored(verdicts: list[dict]) -> set[tuple[str, int, int]]:
    return {(v["id"], int(c), v["run"]) for v in verdicts for c in v.get("scores", {})}


def _survivors(records: list[dict], attempts: list[dict], verdicts: list[dict], flt: dict) -> dict:
    """Per prompt id: the (candidate, score) selected so far, or None. Pure, so resume agrees."""
    scores = _scores(verdicts)
    ok: dict[str, list[int]] = collections.defaultdict(list)
    for attempt in attempts:
        if "assistant" in attempt:
            ok[attempt["id"]].append(attempt["candidate"])
    chosen = {}
    for record in records:
        candidates = {c: candidate_score(scores[(record["id"], c)], flt["judge"]["runs"])
                      for c in ok[record["id"]]}
        chosen[record["id"]] = select(candidates, flt["threshold"])
    return chosen


def _judge_stage(records: list[dict], attempts: list[dict], verdicts: list[dict], flt: dict,
                 chosen: dict) -> list[dict]:
    """The stage-3 snapshot: every candidate's fate, per prompt, in source order."""
    scores = _scores(verdicts)
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for attempt in attempts:
        grouped[attempt["id"]].append(attempt)
    rows = []
    for record in records:
        candidates = []
        for attempt in sorted(grouped[record["id"]], key=lambda a: a["candidate"]):
            key = (record["id"], attempt["candidate"])
            entry = {"candidate": attempt["candidate"]}
            if "rejected" in attempt:
                entry["format_rejected"] = attempt["rejected"]
            elif "assistant" in attempt:
                entry["scores"] = scores[key]
                entry["score"] = candidate_score(scores[key], flt["judge"]["runs"])
            else:
                entry["error"] = attempt.get("error")
            candidates.append(entry)
        picked = chosen.get(record["id"])
        rows.append({"id": record["id"], "source_row": record["source_row"], "candidates": candidates,
                     "selected": picked[0] if picked else None,
                     "selected_score": picked[1] if picked else None})
    return rows


def run(cfg: dict, smoke: bool = False, resume: str | None = None) -> dict:
    """Generate N candidates per final-row prompt, judge each k times, export the best survivor.

    The paper's recipe (Guan et al. 2024, s2.3.2): a format gate drops malformed candidates
    (and, here, answers that leak the constitution), a spec-aware judge scores the survivors
    k times, a candidate's score is the MINIMUM across runs, and only high scorers are kept.
    The judge sees a prompt's candidates SIDE BY SIDE (2026-09-09: scored one at a time it
    gave 240/240 a 10 and missed real flaws; ranking forces it to discriminate). Further rounds of candidates are drawn
    for prompts with no survivor; a prompt still without one is REJECTED and listed in
    the manifest. Fewer than `filter.min_rows` survivors fails the run before publication.
    Resume retries transport/judge errors and reuses every checkpointed response and score.
    The budget is checked before each worker batch; in-flight calls can exceed it.
    """
    started = time.monotonic()
    cfg = _effective(cfg, smoke)
    flt = cfg["filter"]
    constitution = Path(cfg["constitution"]).read_text(encoding="utf-8")
    if not constitution.strip():
        raise ValueError("Constitution is empty")
    augmentation = cfg["generation_prompt"].format(constitution=constitution)
    retry_augmentation = (cfg["retry_generation_prompt"].format(constitution=constitution)
                          if cfg.get("retry_generation_prompt") else None)
    leak = leak_pattern(constitution)
    pin, price = generator_pin(cfg)
    judge_pin, judge_price = provider_pin(flt["judge"]["model"]), provider_price(flt["judge"]["model"])
    # Operational controls may change on resume; all data-affecting settings stay fixed.
    identity = {k: v for k, v in cfg.items() if k not in {"workers", "budget_usd", "output_dir", "smoke"}}
    signature = _digest({"config": identity, "augmentation": augmentation, "retry_augmentation": retry_augmentation,
                         "provider": pin, "price": price,
                         "judge_provider": judge_pin, "judge_price": judge_price})
    if resume:
        run_dir = Path(resume)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("signature") != signature or manifest.get("smoke") != smoke:
            raise ValueError("Resume config/constitution/provider changed; start a new run")
        records = read_jsonl(run_dir / "stage_1_prompts.jsonl")
        if _digest(records) != manifest["prompts_sha256"]:
            raise ValueError("The saved prompt snapshot changed")
        source = manifest["source"]
        ts, repo = manifest["run_id"], manifest["hf_repo"]
        previous_seconds = manifest.get("wall_clock_s", 0)
    else:
        records, source = load_prompts(cfg["source"])
        source_count = len(records)
        if cfg.get("limit") is not None:
            records = records[:int(cfg["limit"])]
        # Validate all API contexts before any paid call, not halfway through a run.
        for record in records:
            generation_messages(record, augmentation)
        ts = timestamp()
        run_dir = Path(cfg["output_dir"]) / (f"smoke_{ts}" if smoke else ts)
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}; use --resume")
        name = artifact_name(f"{cfg['pipeline']} synth smoke") if smoke else synth_name(cfg["pipeline"])
        repo = hf_repo_id(name) if cfg["hf_push"] else None
        if repo and hf_api().repo_exists(repo, repo_type="dataset"):
            raise FileExistsError(f"HF output already exists: {repo}; resume the original run or use a new config stem")
        manifest = {"run_id": ts, "method": "deliberative_alignment", "pipeline": cfg["pipeline"],
                    "signature": signature, "source": source, "source_rows": source_count,
                    "prompts_sha256": _digest(records), "constitution_sha256": hashlib.sha256(constitution.encode()).hexdigest(),
                    "provider_pin": pin, "price_per_million": price,
                    "judge_provider_pin": judge_pin, "judge_price_per_million": judge_price,
                    "hf_repo": repo, "smoke": smoke, "git_sha": git_sha(), "commands": [], "run_dir": str(run_dir)}
        previous_seconds = 0
    card = {"experiment": (f"Deliberative SFT: {cfg['pipeline']}; native Qwen reasoning, best-of-"
                           f"{flt['candidates']} filtered by a constitution-aware judge "
                           f"({flt['judge']['model']}, min of {flt['judge']['runs']} runs >= {flt['threshold']})"),
            "date_generated": ts, "constitution": cfg["constitution"],
            "source_repo": f"{origin_url()} @ {manifest['git_sha']}",
            "models": (f"{cfg['model']} through {pin['order'][0]}/OpenRouter (API revision not exposed); "
                       f"judge {flt['judge']['model']} through OpenRouter"),
            "generation_config": "manifest.json contains resolved config, provider pins, pricing, usage and filter stats",
            "schema": "dataset.jsonl: messages + metadata + optional tools; stages/ snapshots",
            "provenance": f"{shlex.join(sys.argv)}; prompts from {source['repo']} @ {source['revision']} / dataset.jsonl"}
    cache = StageCache(run_dir, repo, private=cfg["hf_private"], card_fields=card,
                       tags=training_data_tags("synth", cfg["pipeline"], cfg["constitution"], smoke=smoke))
    attempts_path = run_dir / "generations.partial.jsonl"
    verdicts_path = run_dir / "judgements.partial.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    verdicts = read_jsonl(verdicts_path) if verdicts_path.exists() else []
    known_ids = {r["id"] for r in records}
    if any(a["id"] not in known_ids or "candidate" not in a for a in attempts) or \
            any(v["id"] not in known_ids or "candidates" not in v for v in verdicts):
        raise ValueError("Checkpoint contains an unknown prompt id or a record from an older filter")
    by_id = {r["id"]: r for r in records}
    manifest["commands"].append({"command": shlex.join(sys.argv), "git_sha": git_sha(), "resume": resume})
    manifest.update(config=cfg, selected_rows=len(records), status="running", dataset=None, aborted=None)

    def spend() -> float:
        return _usage(attempts, price)["total_usd"] + _usage(verdicts, judge_price)["total_usd"]

    mirror = _MirrorThrottle()

    def save_state(chosen: dict | None = None, final: bool = False):
        chosen = chosen if chosen is not None else _survivors(records, attempts, verdicts, flt)
        manifest.update(usage={"generation": _usage(attempts, price), "judge": _usage(verdicts, judge_price),
                               "total_usd": spend()},
                        filter={"candidates": flt["candidates"], "resample_rounds": flt["resample_rounds"],
                                "threshold": flt["threshold"], "min_rows": flt["min_rows"],
                                "judge": flt["judge"],
                                "generated": sum("response" in a for a in attempts),
                                "format_rejected": sum("rejected" in a for a in attempts),
                                "judged": len(_scored(verdicts)),
                                "survivors": sum(v is not None for v in chosen.values()),
                                "rejected_ids": [i for i, v in chosen.items() if v is None]},
                        completed_rows=sum(v is not None for v in chosen.values()),
                        wall_clock_s=round(previous_seconds + time.monotonic() - started, 2))
        # Local first, every time: these files are the record. The Hub mirror is one
        # commit for all four, throttled and best-effort (see _MirrorThrottle).
        manifest_path, run_meta_path = run_dir / "manifest.json", run_dir / "run_meta.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        run_meta_path.write_text(json.dumps({"git_sha": manifest["git_sha"], "config": cfg,
                                             "timestamp": ts, "source": source, "commands": manifest["commands"],
                                             "constitution_sha256": manifest["constitution_sha256"]},
                                            indent=2, ensure_ascii=False), encoding="utf-8")
        mirror.attempt(lambda: cache.checkpoint(
            [manifest_path, run_meta_path, attempts_path, verdicts_path],
            f"checkpoint: {manifest['filter']['generated']} generated, {manifest['filter']['judged']} judged"),
            final=final)

    TRANSIENT = {"RateLimitError", "APITimeoutError", "APIConnectionError", "EmptyCompletionError"}

    def run_batches(todo: list, work, path: Path, store: list[dict], label: str, keep,
                    *, passes: int = 3, cooldown_s: int = 120) -> None:
        """Worker batches with a budget check before each; rate limits back off, anything else aborts.

        Every item is (record, candidate) or (record, group, runs); the error record it
        produces on an exception carries the same keys as a success so resume can retry it.
        An item whose failure is TRANSIENT (the client already retried it six times with
        backoff) is deferred, not fatal: after the pass it is retried at half the workers
        following a cooldown, up to `passes` times. Alibaba's rate limit tripped a batch about
        40 minutes into the 2026-09-11 full run at both 32 and 20 workers; aborting the run for
        that meant a manual resume each time. A non-transient error still aborts the pass.
        """
        workers = int(cfg["workers"])
        pending, errors = list(todo), []
        for pass_no in range(passes):
            deferred = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for offset in range(0, len(pending), workers):
                    if spend() >= float(cfg["budget_usd"]):
                        raise RuntimeError("Budget reached; raise budget_usd and resume")
                    batch = pending[offset:offset + workers]
                    futures = {executor.submit(work, *item): item for item in batch}
                    hard = []
                    for future in as_completed(futures):
                        item = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            # HTTP exception text may contain credentials; record only the class.
                            result = {"id": item[0]["id"], "error": type(exc).__name__}
                            if len(item) == 3:
                                result.update(candidates=[a["candidate"] for a in item[1]], run=item[2])
                            else:
                                result["candidate"] = item[1]
                        # A judge item yields one record per run; a generation item yields one.
                        recs = result if isinstance(result, list) else [result]
                        for rec in recs:
                            store.append(rec)
                            with path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                handle.flush()
                        # An error record superseded by a later success for the same run (the
                        # judge's inline retries, _judge_runs) is history, not a failure: on the
                        # 2026-09-11 full run prompt 146's first judge attempt failed to parse,
                        # its retry scored, and the run still aborted on the stale record.
                        resolved = {rec.get("run") for rec in recs if "error" not in rec}
                        failed_runs = []
                        for rec in recs:
                            if "error" in rec and rec.get("run") not in resolved:
                                which = rec.get("candidates", rec.get("candidate"))
                                errors.append(f"row {rec['id']} candidate {which}: {rec['error']}")
                                if rec["error"] in TRANSIENT:
                                    failed_runs.append(rec.get("run"))
                                else:
                                    hard.append(errors[-1])
                        if failed_runs and not hard:
                            # Retry only what failed: for a judge item, just the failed runs.
                            deferred.append((item[0], item[1], [k for k in failed_runs if k is not None])
                                            if len(item) == 3 else item)
                    save_state()
                    print(f">>> {label} {sum(keep(r) for r in store)}; ${manifest['usage']['total_usd']:.4f}",
                          flush=True)
                    if hard:
                        raise RuntimeError(f"{label} failed; progress checkpointed. " + "; ".join(hard))
            if not deferred:
                return
            workers = max(2, workers // 2)
            print(f">>> {label}: {len(deferred)} rate-limited item(s); cooling down {cooldown_s}s, then "
                  f"retrying at {workers} workers (pass {pass_no + 2}/{passes})", flush=True)
            time.sleep(cooldown_s)
            pending = deferred
        raise RuntimeError(f"{label} failed after {passes} passes; progress checkpointed. " + "; ".join(errors))

    try:
        # Replay publication on resume too: a local snapshot may have survived an upload failure.
        cache.save(1, "prompts", records)
        prompt_file = run_dir / "generation_prompt.txt"
        prompt_file.write_text(augmentation, encoding="utf-8")
        cache.mirror(prompt_file)
        if retry_augmentation:
            retry_file = run_dir / "retry_generation_prompt.txt"
            retry_file.write_text(retry_augmentation, encoding="utf-8")
            cache.mirror(retry_file)

        def sample(record, candidate):
            # The retry wording only after THIS prompt has produced a block-less candidate.
            retry = bool(retry_augmentation) and any(
                a["id"] == record["id"] and a.get("rejected") == NO_BLOCK for a in attempts)
            return _sample(client, record, candidate, cfg, retry_augmentation if retry else augmentation,
                           leak, prompt_kind="retry" if retry else "standard")
        save_state(final=True)
        client = None
        n, runs = flt["candidates"], flt["judge"]["runs"]
        chosen = _survivors(records, attempts, verdicts, flt)
        for round_no in range(flt["resample_rounds"] + 1):
            pending = [r for r in records if chosen[r["id"]] is None]
            if not pending:
                break
            lo, hi = round_no * n, (round_no + 1) * n
            settled = {(a["id"], a["candidate"]) for a in attempts if "error" not in a}
            gen_todo = [(r, c) for r in pending for c in range(lo, hi) if (r["id"], c) not in settled]
            scored = _scored(verdicts)
            if gen_todo or any((r["id"], c, k) not in scored for r in pending for c in range(lo, hi)
                               for k in range(runs) if (r["id"], c) in settled):
                client = client or OpenRouterClient()
            if gen_todo:
                print(f">>> round {round_no}: {len(pending)} prompts without a survivor, "
                      f"{len(gen_todo)} candidates to generate", flush=True)
                run_batches(gen_todo, sample, attempts_path, attempts, "generated", lambda a: "assistant" in a)
            ok = {(a["id"], a["candidate"]): a for a in attempts if "assistant" in a}
            # One comparative call per prompt per run over this round's surviving candidates;
            # a group with any unscored member is judged whole so every member shares a run.
            # A prompt's runs are one work item, executed in sequence (see _judge_runs).
            judge_todo = []
            for r in pending:
                group = [ok[(r["id"], c)] for c in range(lo, hi) if (r["id"], c) in ok]
                ks = [k for k in range(runs)
                      if group and any((r["id"], a["candidate"], k) not in scored for a in group)]
                if ks:
                    judge_todo.append((r, group, ks))
            if judge_todo:
                run_batches(judge_todo, lambda r, g, ks: _judge_runs(client, r, g, ks, cfg, constitution),
                            verdicts_path, verdicts, "judged", lambda v: "scores" in v)
            chosen = _survivors(records, attempts, verdicts, flt)
            print(f">>> round {round_no}: {sum(v is not None for v in chosen.values())}/{len(records)} "
                  f"prompts have a survivor", flush=True)
        save_state(chosen, final=True)
        rejected = [i for i, v in chosen.items() if v is None]
        if len(records) - len(rejected) < flt["min_rows"]:
            raise RuntimeError(f"Only {len(records) - len(rejected)} prompts survived the filter, "
                               f"below filter.min_rows={flt['min_rows']}; refusing to publish. "
                               f"Rejected: {rejected}")
        ordered = sorted(attempts, key=lambda a: (by_id[a["id"]]["source_row"], a["candidate"]))
        cache.save(2, "responses", ordered)
        cache.save(3, "judge", _judge_stage(records, attempts, verdicts, flt, chosen))
        ok = {(a["id"], a["candidate"]): a for a in attempts if "assistant" in a}
        rows = []
        for record in records:
            picked = chosen[record["id"]]
            if picked is None:
                continue
            candidate, score = picked
            judged = sum(1 for key in ok if key[0] == record["id"])
            provenance = {"source": source, "model": cfg["model"], "provider": pin["order"][0],
                          "constitution_sha256": manifest["constitution_sha256"],
                          "judge": {"model": flt["judge"]["model"], "runs": runs, "threshold": flt["threshold"],
                                    "score": score, "candidate": candidate, "candidates_judged": judged}}
            rows.append(export_row(record, ok[(record["id"], candidate)]["assistant"], provenance))
        cache.save(4, "export_sft", rows)
        cache.publish_final(rows)
        manifest.update(status="complete", dataset="dataset.jsonl")
        save_state(chosen, final=True)
    except BaseException as exc:
        manifest.update(status="aborted", aborted={"type": type(exc).__name__}, dataset=None)
        # Preserve local diagnosis even when the remote mirror itself is failing.
        try:
            save_state(final=True)
        except Exception:
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    return manifest
