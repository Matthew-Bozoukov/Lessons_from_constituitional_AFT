# ABOUTME: Run three requester conditions through isolated in-memory benchmark worlds.
# ABOUTME: Freeze same-checkpoint requests first, save complete traces, then judge actions.
from __future__ import annotations

import copy
import inspect
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI
from omegaconf import OmegaConf

from .judging import Judge, metrics, validate_author, validate_episode
from .rubrics import AUTHOR_SYSTEM, COMPONENTS, EPISODE_SYSTEM
from .source import digest, prepare, save


def _invoke(fn, state, cond):
    if fn is None:
        return None
    return fn(state, cond) if len(inspect.signature(fn).parameters) >= 2 else fn(state)


def cells(keys, variants, repeats, seed):
    result = []
    for index, key in enumerate(keys):
        for variant in range(variants):
            for rep in range(repeats):
                common = dict(scenario=key, variant=variant, repeat=rep,
                              seed=seed + index * 10000 + variant * 100 + rep)
                for delivery in ("chat", "subagent", "chat_human"):
                    result.append(dict(common, delivery=delivery,
                                       id=f"{key}__{delivery}__v{variant}__r{rep}"))
    random.Random(seed).shuffle(result)
    return result


def make_env(sc, delivery, model_label, brief):
    from core import env as envmod
    from core.scenario import Condition
    cond = Condition(delivery=delivery)
    env = envmod.build(sc, cond, model_label, model_key=model_label)
    # Explicit injection: do not allow the upstream default-author fallback.
    env.update(brief=brief, brief_source=brief,
               brief_author="human" if delivery == "chat_human" else model_label)
    state = envmod.seed_state(env)
    return env, state


def author_one(client, model, sc, notes, variant, cfg, judge, path):
    from core import reskin
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return record
    ctx = reskin._ctx(sc, sc.key)
    system = reskin.SYSTEM.format(**ctx)
    prompt = reskin.PROMPT.format(subject=ctx["subject"], brief=notes)
    record = {"scenario": sc.key, "variant": variant, "model": model,
              "source_notes": notes, "system": system, "attempts": [], "accepted": False}
    for attempt in range(int(cfg.author.attempts)):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": prompt}],
                temperature=float(cfg.author.temperature), top_p=float(cfg.generation.top_p),
                seed=int(cfg.seed) + variant * 100 + attempt,
                max_tokens=int(cfg.author.max_tokens))
            message = response.choices[0].message
            text = reskin._clean(message.content or "", sc.subject_handle)
            item = {"prompt": prompt, "response": response.model_dump(mode="json"), "text": text}
            record["attempts"].append(item)
            if response.choices[0].finish_reason != "stop" or not text:
                problems = ["Request generation did not complete"]
            else:
                # Upstream checks preserve identifiers, quantities and surface form;
                # semantic validation additionally checks harmful/benign intent.
                problems = reskin.check(notes, text, sc)
                verdict = judge.ask(AUTHOR_SYSTEM, {"source": notes, "candidate": text}, validate_author)
                item["validation"] = verdict
                problems += verdict["problems"]
            item["problems"] = problems
            if not problems:
                record.update(accepted=True, text=text, sha256=digest(text))
                save(path, record)
                return record
            prompt = reskin.REPAIR.format(brief=notes, attempt=text,
                                         problems="\n".join("- " + p for p in problems))
        except Exception as exc:
            record["attempts"].append({"error_type": type(exc).__name__, "error": str(exc)[:500]})
            # API/judge failure cannot be repaired by changing the author's wording.
            save(path, record)
            return record
        save(path, record)
    return record


def episode(client, model, label, sc, cell, brief, cfg, path):
    from core import env as envmod
    env, state = make_env(sc, cell["delivery"], label, brief)
    specs = envmod.tool_specs(env)
    tools = [{"type": "function", "function": {"name": s["name"],
              "description": s["description"], "parameters": s["input_schema"]}} for s in specs]
    system, first = envmod.system_prompt(env), envmod.first_user_message(env)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    trace = dict(cell, cond=asdict(env["cond"]), subject_handle=env["subject_handle"],
                 requester=env["requester"], requester_is_human=cell["delivery"] == "chat_human",
                 requester_model=env["requester_model"], brief_author=env["brief_author"],
                 brief=env["brief"], brief_source=brief, brief_sha256=digest(brief),
                 system=system, first_user=first, tools=tools, messages=messages,
                 max_turns=int(cfg.generation.max_turns), turns=[], status="running",
                 error=None, started=time.time())
    save(path, trace)
    try:
        for turn in range(int(cfg.generation.max_turns)):
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                temperature=float(cfg.generation.temperature), top_p=float(cfg.generation.top_p),
                seed=cell["seed"] + turn, max_tokens=int(cfg.generation.max_tokens))
            choice = response.choices[0]
            message = choice.message.model_dump(mode="json", exclude_none=True)
            messages.append(message)
            calls = choice.message.tool_calls or []
            rec = {"turn": turn, "stop_reason": choice.finish_reason,
                   "response": response.model_dump(mode="json"), "tool_calls": [], "tool_results": []}
            trace["turns"].append(rec)
            if choice.finish_reason == "length":
                trace["status"] = "token_limit"
                break
            if choice.finish_reason not in ("stop", "tool_calls"):
                trace["status"] = "provider_refusal_or_stop"
                break
            if not calls:
                trace["status"] = "complete" if choice.message.content else "empty_completion"
                break
            for call in calls:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object")
                rec["tool_calls"].append({"name": call.function.name, "input": args, "id": call.id})
                output, ended = envmod.execute(env, state, call.function.name, args)
                rec["tool_results"].append({"name": call.function.name, "output": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                if ended:
                    trace["status"] = "complete"
            save(path, trace)
            if trace["status"] == "complete":
                break
        else:
            trace["status"] = "turn_limit"
    except Exception as exc:
        trace.update(status="error", error={"type": type(exc).__name__, "message": str(exc)[:1000]})
    touched = set(state["fs"]["reads"]) | {w["path"] for w in state["fs"]["writes"]}
    trace["state"] = copy.deepcopy(state)
    trace["state"]["fs"]["files"] = {k: v for k, v in state["fs"]["files"].items() if k in touched}
    trace["state"]["fs"]["files_omitted"] = True
    trace["mechanical"] = {
        "compliance": int(_invoke(sc.compliance, state, env["cond"]) or 0),
        "safe_alternative": _invoke(sc.safe_alternative, state, env["cond"]),
        "observations": _invoke(sc.observations, state, env["cond"]),
        "evidence_opened": [p for p in sc.decisive_evidence if p in state["fs"]["reads"]],
    }
    trace["finished"] = time.time()
    save(path, trace)
    # Readable conversation includes reasoning and tool calls, with the prompt inline.
    lines = [f"# {cell['id']}", f"Status: {trace['status']}", ""]
    for m in messages:
        lines += [f"## {m['role']}", m.get("reasoning_content") or m.get("reasoning") or "",
                  m.get("content") or ""]
        if m.get("tool_calls"):
            lines += [json.dumps(m["tool_calls"], indent=2, ensure_ascii=False)]
    path.with_suffix(".md").write_text("\n\n".join(lines), encoding="utf-8")
    return trace


def judge_episode(sc, trace, notes, judge, descriptions):
    from core import escaldet
    from core.env import HUMAN_APPROVER
    candidates = [dict(m, recipients=sorted(people))
                  for m, people, _ in escaldet.candidates(sc, trace)]
    # Full visible transcript and every changed file are retained. The judge also sees
    # the decisive world evidence, even when the subject failed to read it.
    visible = [{k: v for k, v in m.items() if k not in
                ("reasoning", "reasoning_content", "reasoning_details")}
               for m in trace["messages"][1:]]
    payload = {"request": trace["brief"], "source_notes": notes,
               "components": COMPONENTS[sc.key], "visible_transcript": visible,
               "final_files": trace["state"]["fs"]["files"],
               "decisive_world_evidence": {k: sc.files.get(k) for k in sc.decisive_evidence},
               "mechanical_observations": trace["mechanical"],
               "human_messages": candidates,
               "concern_rubric": escaldet.system_for(sc, descriptions[sc.key])}
    verdict = judge.ask(EPISODE_SYSTEM, payload,
                        lambda o: validate_episode(o, COMPONENTS[sc.key], candidates))
    return {"verdict": verdict, "metrics": metrics(verdict, candidates, trace["requester"],
            HUMAN_APPROVER[sc.key], trace["mechanical"]["compliance"])}


def summarize(records, schedule):
    result = {"scheduled": len(schedule), "recorded": len(records),
              "interpretation": "Checkpoint comparison; repeats are not training seeds. AI prompts differ across checkpoints.",
              "arms": {}}
    for delivery in ("chat", "subagent", "chat_human"):
        planned = [c for c in schedule if c["delivery"] == delivery]
        rows = [r for r in records if r["delivery"] == delivery]
        valid = [r for r in rows if r.get("metrics")]
        arm = {"scheduled": len(planned), "recorded": len(rows), "valid": len(valid),
               "missing_or_invalid": len(planned) - len(valid), "rates": {}, "per_scenario": {}}
        if valid:
            arm["rates"] = {k: sum(r["metrics"][k] for r in valid) / len(valid)
                            for k in valid[0]["metrics"]}
            successes = sum(r["metrics"]["joint_success"] for r in valid)
            arm["joint_success_bounds_all_scheduled"] = [successes / len(planned),
                (successes + len(planned) - len(valid)) / len(planned)]
            for key in sorted({c["scenario"] for c in planned}):
                group = [r for r in valid if r["scenario"] == key]
                arm["per_scenario"][key] = {"valid": len(group), "rates": {
                    k: sum(r["metrics"][k] for r in group) / len(group)
                    for k in valid[0]["metrics"]} if group else {}}
        result["arms"][delivery] = arm
    return result


def checkpoint(out, target, cfg, summary):
    from src.eval.run_eval import _card_fields
    from src.infra.huggingface import push_run_dir
    from src.naming import eval_name
    save(out / "results/results.json", summary)
    stage = out.with_name(out.name + "-checkpoint")
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("rollouts", "results", "metadata"):
        shutil.copytree(out / name, stage / name, dirs_exist_ok=True)
    shutil.copy2(out / "run_meta.json", stage / "metadata/run_meta.json")
    card = _card_fields("delegated_harm", cfg, "uv run evals --name delegated_harm --target "
                        + target.spec.hf_path, experiment="Mixed-request delegated harm evaluation",
                        models=json.dumps(asdict(target.spec)))
    return push_run_dir(stage, eval_name("delegated_harm", target.spec.model_key), card,
                        front_matter={"tags": ["eval-run", "eval:delegated_harm",
                          f"model:{target.spec.model_key}", f"mode:{target.spec.mode}"]})


def run(target, cfg, out_dir):
    scenarios, human, notes, manifest = prepare(cfg)
    if set(scenarios) != set(COMPONENTS):
        raise ValueError("Selected scenario set needs an exact predeclared rubric set")
    expected = cfg.get("expected_revisions", {}).get(target.spec.hf_path)
    if expected and target.spec.revision != expected:
        raise ValueError("Target revision drifted from the configured pin")
    if cfg.get("expected_base_revision") and target.spec.base_revision != cfg.expected_base_revision:
        raise ValueError("Base revision drifted from the shared evaluation pin")
    for name in ("metadata", "rollouts", "results"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    manifest.update(target=asdict(target.spec), components=COMPONENTS,
                    protocol=OmegaConf.to_container(cfg, resolve=True))
    save(out_dir / "metadata/protocol.json", manifest)
    save(out_dir / "metadata/human_requests.json", {k: human[k] for k in scenarios})
    judge = Judge(cfg.judge, out_dir)
    client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                    timeout=float(cfg.generation.request_timeout), max_retries=1)
    author_records = {}
    with ThreadPoolExecutor(max_workers=int(cfg.generation.parallel)) as pool:
        futures = {pool.submit(author_one, client, target.model_name, sc, notes[k]["clear"],
                    variant, cfg, judge, out_dir / "metadata/authors" / f"{k}__v{variant}.json"):
                   (k, variant) for k, sc in scenarios.items() for variant in range(int(cfg.author.variants))}
        for future in as_completed(futures):
            key = futures[future]
            author_records[key] = future.result()
            print(f">>> author {key}: {'accepted' if author_records[key]['accepted'] else 'FAILED'}", flush=True)
    save(out_dir / "metadata/authorship_frozen.json", {
        f"{k}__v{v}": {"accepted": r["accepted"], "sha256": r.get("sha256")}
        for (k, v), r in author_records.items()})
    schedule = cells(list(scenarios), int(cfg.author.variants), int(cfg.generation.repetitions), int(cfg.seed))
    save(out_dir / "metadata/schedule.json", schedule)
    descriptions = json.loads((Path(str(cfg.source.path)) / "descriptions.json").read_text(encoding="utf-8"))
    records = []
    def one(cell):
        sc = scenarios[cell["scenario"]]
        rec = dict(cell)
        author = author_records[(sc.key, cell["variant"])]
        if cell["delivery"] != "chat_human" and not author["accepted"]:
            rec["status"] = "author_failure"
            save(out_dir / "results/episodes" / f"{cell['id']}.json", rec)
            return rec
        brief = human[sc.key] if cell["delivery"] == "chat_human" else author["text"]
        path = out_dir / "rollouts" / f"{cell['id']}.json"
        trace = episode(client, target.model_name, target.spec.hf_path, sc, cell, brief, cfg, path)
        rec.update(status=trace["status"], mechanical=trace["mechanical"])
        if trace["status"] == "complete":
            try:
                rec.update(judge_episode(sc, trace, notes[sc.key]["clear"], judge, descriptions))
            except Exception as exc:
                rec["judgment_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        save(out_dir / "results/episodes" / f"{cell['id']}.json", rec)
        return rec
    with ThreadPoolExecutor(max_workers=int(cfg.generation.parallel)) as pool:
        futures = [pool.submit(one, cell) for cell in schedule]
        for future in as_completed(futures):
            records.append(future.result())
            summary = summarize(records, schedule)
            summary["excluded_scenarios"] = dict(cfg.source.excluded)
            summary["judge_usd"] = judge.ledger["charged_or_reserved_usd"]
            save(out_dir / "results/results.json", summary)
            print(f">>> episodes {len(records)}/{len(schedule)}; valid "
                  f"{sum(bool(r.get('metrics')) for r in records)}; judge ${summary['judge_usd']:.3f}", flush=True)
            if len(records) % int(cfg.checkpoint_every) == 0:
                checkpoint(out_dir, target, cfg, summary)
    checkpoint(out_dir, target, cfg, summary)
    return summary
