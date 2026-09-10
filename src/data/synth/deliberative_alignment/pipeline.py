# ABOUTME: Deliberative SFT generation: sample Qwen with the constitution, judge, export without it.
# ABOUTME: Owns configuration, best-of-N generation, spec-aware filtering, resume, accounting, publication.

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
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
from .judge import JUDGE_FIELDS, candidate_score, format_rejection, judge_messages, parse_scores, select

FILTER_KEYS = {"candidates", "resample_rounds", "threshold", "min_rows", "judge"}
JUDGE_KEYS = {"model", "runs", "temperature", "max_tokens"}


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


def validate_config(cfg: dict) -> None:
    """Reject unsupported options before any generation or publication."""
    allowed = {"method", "pipeline", "source", "constitution", "model", "sampling",
               "generation_prompt", "filter", "judge_prompt", "workers", "budget_usd",
               "limit", "output_dir", "hf_push", "hf_private", "smoke"}
    if unknown := set(cfg) - allowed:
        raise ValueError(f"Unsupported deliberative_alignment settings: {sorted(unknown)}")
    required = allowed - {"limit", "smoke"}
    if missing := required - set(cfg):
        raise ValueError(f"Missing deliberative_alignment settings: {sorted(missing)}")
    if cfg["method"] != "deliberative_alignment":
        raise ValueError("method must be deliberative_alignment")
    check_style(cfg["pipeline"])
    if not isinstance(cfg["source"], dict) or set(cfg["source"]) - {"repo", "revision"}:
        raise ValueError("source accepts only repo and optional revision; intake is always dataset.jsonl")
    if not cfg["source"].get("repo"):
        raise ValueError("source.repo is required")
    if not str(cfg["model"]).startswith("qwen/"):
        raise ValueError("The deliberative SFT generator must be Qwen")
    pin = _priced(cfg["model"])
    if pin.get("order") != ["alibaba"] or pin.get("allow_fallbacks") is not False:
        raise ValueError("Qwen must be pinned to Alibaba with allow_fallbacks: false in providers.yaml")
    sampling = cfg["sampling"]
    if not isinstance(sampling, dict) or set(sampling) - {"temperature", "max_tokens", "reasoning", "seed"}:
        raise ValueError("sampling accepts temperature, max_tokens, reasoning, and optional seed")
    if not {"temperature", "max_tokens", "reasoning"} <= sampling.keys():
        raise ValueError("sampling requires temperature, max_tokens, and reasoning")
    reasoning = sampling["reasoning"]
    if not isinstance(reasoning, dict) or reasoning.get("enabled") is not True or reasoning.get("exclude") is not False:
        raise ValueError("sampling.reasoning must set enabled: true and exclude: false")
    if set(reasoning) - {"enabled", "exclude", "max_tokens"}:
        raise ValueError("Only enabled, exclude, and max_tokens are supported in reasoning")
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
    if not isinstance(judge, dict) or set(judge) != JUDGE_KEYS:
        raise ValueError(f"filter.judge must set exactly {sorted(JUDGE_KEYS)}")
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


def _completion(result, record: dict) -> dict:
    """The format gate: one complete native reasoning response, or a ValueError naming why.

    A ValueError here is a FORMAT rejection of this candidate (the paper's first filter);
    it is recorded and never retried. Anything that points at the run rather than the
    sample -- the wrong serving provider -- is a RuntimeError and stops the run.
    """
    if result.provider.casefold() != "alibaba":
        raise RuntimeError(f"Unexpected serving provider: {result.provider!r}")
    if result.finish_reason not in {"stop", "tool_calls"}:
        raise ValueError(f"Incomplete completion: finish_reason={result.finish_reason!r}")
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
    if reason := format_rejection(answer):
        raise ValueError(reason)
    assistant = {"role": "assistant", "content": answer, "reasoning_content": trace}
    if result.tool_calls:
        assistant["tool_calls"] = result.tool_calls
    return assistant


def _sample(client, record: dict, candidate: int, cfg: dict, augmentation: str) -> dict:
    sampling = copy.deepcopy(cfg["sampling"])
    reasoning = sampling.pop("reasoning")
    kwargs = {"tools": record["tools"]} if record.get("tools") else {}
    result = client.chat(model=cfg["model"], messages=generation_messages(record, augmentation),
                         extra_body={"reasoning": reasoning}, **sampling, **kwargs)
    attempt = {"id": record["id"], "candidate": candidate, "response": asdict(result)}
    try:
        assistant = _completion(result, record)
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
    result = client.chat(model=judge["model"],
                         messages=judge_messages(record, candidates, cfg["judge_prompt"], constitution),
                         temperature=judge["temperature"], max_tokens=judge["max_tokens"])
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
    pin, price = provider_pin(cfg["model"]), provider_price(cfg["model"])
    judge_pin, judge_price = provider_pin(flt["judge"]["model"]), provider_price(flt["judge"]["model"])
    # Operational controls may change on resume; all data-affecting settings stay fixed.
    identity = {k: v for k, v in cfg.items() if k not in {"workers", "budget_usd", "output_dir", "smoke"}}
    signature = _digest({"config": identity, "augmentation": augmentation, "provider": pin, "price": price,
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
            "models": (f"{cfg['model']} through Alibaba/OpenRouter (API revision not exposed); "
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

    def save_state(chosen: dict | None = None):
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
        cache.save_json("manifest.json", manifest)
        cache.save_json("run_meta.json", {"git_sha": manifest["git_sha"], "config": cfg,
                        "timestamp": ts, "source": source, "commands": manifest["commands"],
                        "constitution_sha256": manifest["constitution_sha256"]})
        cache.mirror(attempts_path)
        cache.mirror(verdicts_path)

    def run_batches(todo: list, work, path: Path, store: list[dict], label: str, keep) -> None:
        """Worker batches with a budget check before each; abort after a batch with errors.

        Every item is (record, candidate-or-attempt, [run]); the error record it produces
        on an exception carries the same keys as a success so resume can retry it.
        """
        workers = int(cfg["workers"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(todo), workers):
                if spend() >= float(cfg["budget_usd"]):
                    raise RuntimeError("Budget reached; raise budget_usd and resume")
                batch = todo[offset:offset + workers]
                futures = {executor.submit(work, *item): item for item in batch}
                errors = []
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
                    store.append(result)
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                    if "error" in result:
                        which = result.get("candidates", result.get("candidate"))
                        errors.append(f"row {result['id']} candidate {which}: {result['error']}")
                save_state()
                print(f">>> {label} {sum(keep(r) for r in store)}; ${manifest['usage']['total_usd']:.4f}",
                      flush=True)
                if errors:
                    raise RuntimeError(f"{label} failed; progress checkpointed. " + "; ".join(errors))

    try:
        # Replay publication on resume too: a local snapshot may have survived an upload failure.
        cache.save(1, "prompts", records)
        prompt_file = run_dir / "generation_prompt.txt"
        prompt_file.write_text(augmentation, encoding="utf-8")
        cache.mirror(prompt_file)
        save_state()
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
                run_batches(gen_todo, lambda r, c: _sample(client, r, c, cfg, augmentation),
                            attempts_path, attempts, "generated", lambda a: "assistant" in a)
            ok = {(a["id"], a["candidate"]): a for a in attempts if "assistant" in a}
            # One comparative call per prompt per run over this round's surviving candidates;
            # a group with any unscored member is judged whole so every member shares a run.
            judge_todo = []
            for r in pending:
                group = [ok[(r["id"], c)] for c in range(lo, hi) if (r["id"], c) in ok]
                for k in range(runs):
                    if group and any((r["id"], a["candidate"], k) not in scored for a in group):
                        judge_todo.append((r, group, k))
            if judge_todo:
                run_batches(judge_todo, lambda r, g, k: _judge(client, r, g, k, cfg, constitution),
                            verdicts_path, verdicts, "judged", lambda v: "scores" in v)
            chosen = _survivors(records, attempts, verdicts, flt)
            print(f">>> round {round_no}: {sum(v is not None for v in chosen.values())}/{len(records)} "
                  f"prompts have a survivor", flush=True)
        save_state(chosen)
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
            provenance = {"source": source, "model": cfg["model"], "provider": "Alibaba",
                          "constitution_sha256": manifest["constitution_sha256"],
                          "judge": {"model": flt["judge"]["model"], "runs": runs, "threshold": flt["threshold"],
                                    "score": score, "candidate": candidate, "candidates_judged": judged}}
            rows.append(export_row(record, ok[(record["id"], candidate)]["assistant"], provenance))
        cache.save(4, "export_sft", rows)
        cache.publish_final(rows)
        manifest.update(status="complete", dataset="dataset.jsonl")
        save_state(chosen)
    except BaseException as exc:
        manifest.update(status="aborted", aborted={"type": type(exc).__name__}, dataset=None)
        # Preserve local diagnosis even when the remote mirror itself is failing.
        try:
            save_state()
        except Exception:
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    return manifest
