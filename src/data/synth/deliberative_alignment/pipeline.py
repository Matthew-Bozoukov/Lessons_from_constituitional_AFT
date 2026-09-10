# ABOUTME: Deliberative SFT generation: sample Qwen with the constitution, then export without it.
# ABOUTME: Owns configuration, generation, resumable checkpoints, accounting and HF publication.

from __future__ import annotations

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


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_config(cfg: dict) -> None:
    """Reject unsupported options before any generation or publication."""
    allowed = {"method", "pipeline", "source", "constitution", "model", "sampling",
               "generation_prompt", "workers", "budget_usd", "limit", "output_dir",
               "hf_push", "hf_private", "smoke"}
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
    pin = provider_pin(cfg["model"])
    if pin.get("order") != ["alibaba"] or pin.get("allow_fallbacks") is not False:
        raise ValueError("Qwen must be pinned to Alibaba with allow_fallbacks: false in providers.yaml")
    price = provider_price(cfg["model"])
    if not price or any(not math.isfinite(float(price[k])) or float(price[k]) <= 0
                        for k in ("in", "out")):
        raise ValueError("The Alibaba pin must have positive input/output prices")
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


def _effective(cfg: dict, smoke: bool) -> dict:
    result = OmegaConf.to_container(OmegaConf.merge(cfg, cfg.get("smoke", {}) if smoke else {}), resolve=True)
    validate_config(result)
    return result


def _completion(result, record: dict) -> dict:
    """Require one complete native reasoning response; never quality-filter rows."""
    if result.finish_reason not in {"stop", "tool_calls"}:
        raise ValueError(f"Incomplete completion: finish_reason={result.finish_reason!r}")
    if result.provider.casefold() != "alibaba":
        raise ValueError(f"Unexpected serving provider: {result.provider!r}")
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
    assistant = {"role": "assistant", "content": answer, "reasoning_content": trace}
    if result.tool_calls:
        assistant["tool_calls"] = result.tool_calls
    return assistant


def _sample(client, record: dict, cfg: dict, augmentation: str) -> dict:
    sampling = copy.deepcopy(cfg["sampling"])
    reasoning = sampling.pop("reasoning")
    kwargs = {"tools": record["tools"]} if record.get("tools") else {}
    result = client.chat(model=cfg["model"], messages=generation_messages(record, augmentation),
                         extra_body={"reasoning": reasoning}, **sampling, **kwargs)
    attempt = {"id": record["id"], "response": asdict(result)}
    try:
        assistant = _completion(result, record)
        # Validate export/tool schemas while the response is still checkpointed as an attempt.
        export_row(record, assistant, {})
        attempt["assistant"] = assistant
    except (ValueError, TypeError) as exc:
        attempt["error"] = str(exc)
    return attempt


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


def run(cfg: dict, smoke: bool = False, resume: str | None = None) -> dict:
    """Generate every selected final-row prompt, preserving membership and order.

    A malformed response fails the run instead of silently selecting a smaller corpus.
    Resume retries failed/missing rows and reuses successful responses. The budget is
    checked before each worker batch; in-flight calls and transport retries can exceed it.
    """
    started = time.monotonic()
    cfg = _effective(cfg, smoke)
    constitution = Path(cfg["constitution"]).read_text(encoding="utf-8")
    if not constitution.strip():
        raise ValueError("Constitution is empty")
    augmentation = cfg["generation_prompt"].format(constitution=constitution)
    pin, price = provider_pin(cfg["model"]), provider_price(cfg["model"])
    # Operational controls may change on resume; all data-affecting settings stay fixed.
    identity = {k: v for k, v in cfg.items() if k not in {"workers", "budget_usd", "output_dir", "smoke"}}
    signature = _digest({"config": identity, "augmentation": augmentation, "provider": pin, "price": price})
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
                    "provider_pin": pin, "price_per_million": price, "hf_repo": repo,
                    "smoke": smoke, "git_sha": git_sha(), "commands": [], "run_dir": str(run_dir)}
        previous_seconds = 0
    card = {"experiment": f"Deliberative SFT: {cfg['pipeline']}; native Qwen reasoning, no judge filtering",
            "date_generated": ts, "constitution": cfg["constitution"],
            "source_repo": f"{origin_url()} @ {manifest['git_sha']}",
            "models": f"{cfg['model']} through Alibaba/OpenRouter (API revision not exposed)",
            "generation_config": "manifest.json contains resolved config, provider pin, pricing and usage",
            "schema": "dataset.jsonl: messages + metadata + optional tools; stages/ snapshots",
            "provenance": f"{shlex.join(sys.argv)}; prompts from {source['repo']} @ {source['revision']} / dataset.jsonl"}
    cache = StageCache(run_dir, repo, private=cfg["hf_private"], card_fields=card,
                       tags=training_data_tags("synth", cfg["pipeline"], cfg["constitution"], smoke=smoke))
    attempts_path = run_dir / "generations.partial.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    known_ids = {r["id"] for r in records}
    if any(a["id"] not in known_ids for a in attempts):
        raise ValueError("Checkpoint contains an unknown prompt id")
    done = {a["id"]: a for a in attempts if "assistant" in a}
    manifest["commands"].append({"command": shlex.join(sys.argv), "git_sha": git_sha(), "resume": resume})
    manifest.update(config=cfg, selected_rows=len(records), status="running", dataset=None, aborted=None)

    def save_state():
        manifest.update(completed_rows=len(done), usage=_usage(attempts, price),
                        wall_clock_s=round(previous_seconds + time.monotonic() - started, 2))
        cache.save_json("manifest.json", manifest)
        cache.save_json("run_meta.json", {"git_sha": manifest["git_sha"], "config": cfg,
                        "timestamp": ts, "source": source, "commands": manifest["commands"],
                        "constitution_sha256": manifest["constitution_sha256"]})
        cache.mirror(attempts_path)

    try:
        # Replay publication on resume too: a local snapshot may have survived an upload failure.
        cache.save(1, "prompts", records)
        prompt_file = run_dir / "generation_prompt.txt"
        prompt_file.write_text(augmentation, encoding="utf-8")
        cache.mirror(prompt_file)
        save_state()
        todo = [r for r in records if r["id"] not in done]
        client = OpenRouterClient() if todo else None
        workers = int(cfg["workers"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(todo), workers):
                if _usage(attempts, price)["total_usd"] >= float(cfg["budget_usd"]):
                    raise RuntimeError("Generation budget reached; raise budget_usd and resume")
                batch = todo[offset:offset + workers]
                futures = {executor.submit(_sample, client, r, cfg, augmentation): r for r in batch}
                errors = []
                for future in as_completed(futures):
                    record = futures[future]
                    try:
                        attempt = future.result()
                    except Exception as exc:
                        # HTTP exception text may contain credentials; record only the class.
                        attempt = {"id": record["id"], "error": type(exc).__name__}
                    attempts.append(attempt)
                    with attempts_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                        handle.flush()
                    if "assistant" in attempt:
                        done[record["id"]] = attempt
                    else:
                        errors.append(f"row {record['id']}: {attempt['error']}")
                save_state()
                print(f">>> generated {len(done)}/{len(records)}; ${manifest['usage']['total_usd']:.4f}")
                if errors:
                    raise RuntimeError("Generation failed; successful rows checkpointed. " + "; ".join(errors))
        if len(done) != len(records):
            raise RuntimeError("Incomplete generation; refusing to publish a partial dataset")
        generated = [done[r["id"]] for r in records]
        cache.save(2, "responses", generated)
        provenance = {"source": source, "model": cfg["model"], "provider": "Alibaba",
                      "constitution_sha256": manifest["constitution_sha256"]}
        rows = [export_row(r, done[r["id"]]["assistant"], provenance) for r in records]
        cache.save(3, "export_sft", rows)
        cache.publish_final(rows)
        manifest.update(status="complete", dataset="dataset.jsonl")
        save_state()
    except BaseException as exc:
        manifest.update(status="aborted", aborted={"type": type(exc).__name__}, dataset=None)
        # Preserve local diagnosis even when the remote mirror itself is failing.
        try:
            save_state()
        except Exception:
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    return manifest
