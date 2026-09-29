# ABOUTME: Compare two matched eight-prompt Arena runs that differ only in output budget.
# ABOUTME: Preserve raw evidence and optional GPT-4.1 preferences as an instrument study.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.capabilities.arena_hard import arena_hard_judge as arena_judge
from src.eval.capabilities.arena_hard.runner import _generation_settings, isolate_harness
from src.eval.layout import assert_layout, publish_layout, run_tags
from src.eval.run_eval import _card_fields
from src.infra.huggingface import push_run_dir
from src.naming import artifact_name
from src.utils import read_jsonl, write_run_meta


BUDGETS = {"budget6000": 6000, "budget12000": 12000}
SUBJECT = "arena-hard-output-budget-comparison"
LIMITATION = (
    "Instrument study on the same fixed eight prompts (four hard, four creative). "
    "The output-budget difference is deliberate. This is not a training capability "
    "comparison or representative judge calibration. Character counts describe saved "
    "text, not token counts, reasoning quality, answer correctness or model health. "
    "Any historical run is a descriptive replication reference only, outside the matched pair."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_arm(metadata, rollouts, *, budget=None, fresh=False):
    """Validate a standard arm's recorded identity, prompts and actual allowances."""
    metadata, rollouts = Path(metadata), Path(rollouts)
    meta = read(metadata / "run_meta.json")
    protocol = read(metadata / "generation_protocol.json")
    sources = read(metadata / "sources.json")
    cfg = OmegaConf.load(metadata / "arena_hard_config.yaml")
    rows = read_jsonl(rollouts / "answers.jsonl")
    by_uid = {row["uid"]: row for row in rows}
    if len(rows) != 8 or len(by_uid) != 8:
        raise ValueError("Each input must contain exactly eight unique answered UIDs")
    identity = {key: meta.get(key) for key in (
        "target", "target_revision", "base_model", "base_model_revision", "mode")}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("Input has missing target/base identity or thinking mode")
    for key in ("target_revision", "base_model_revision"):
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", identity[key]):
            raise ValueError(f"Input {key} must be an exact commit, not a moving revision")
    expected_target = {"target": identity["target"], "revision": identity["target_revision"],
                       "base_revision": identity["base_model_revision"], "mode": identity["mode"]}
    if OmegaConf.to_container(cfg.target_identity, resolve=True) != expected_target:
        raise ValueError("Arm config identity differs from run metadata")
    if fresh and sources.get("answers", {}).get("generated") != identity["target"]:
        raise ValueError("The matched pair requires newly generated standard arms, not reused-answer runs")
    if (protocol.get("schema_version") != 1 or protocol.get("status") != "recorded"
            or protocol.get("bench_name") != str(cfg.bench_name)
            or protocol.get("settings") != _generation_settings(cfg, identity["mode"])):
        raise ValueError("Generation protocol is missing or differs from the recorded config")
    if (not isinstance(protocol.get("effective_context_window"), int)
            or protocol["effective_context_window"] <= 0 or not protocol.get("output_budget_policy")):
        raise ValueError("Effective context/output-budget policy was not recorded")
    if set(protocol.get("questions", {})) != set(by_uid):
        raise ValueError("Answer UIDs differ from the generation protocol")
    if budget is not None and int(protocol["settings"]["max_tokens"]) != budget:
        raise ValueError(f"Expected configured max_tokens={budget}")
    identities = {}
    for uid, row in by_uid.items():
        generation = row.get("generation_record") or {}
        messages = row.get("messages") or []
        prompt = generation.get("prompt")
        if (generation.get("uid") != uid or not isinstance(prompt, str)
                or not messages or messages[0].get("role") != "user"
                or messages[0].get("content") != prompt):
            raise ValueError(f"Missing or mismatched raw prompt for {uid}")
        question = {"category": generation.get("category"),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        item = protocol["questions"][uid]
        if item.get("identity") != question:
            raise ValueError(f"Prompt/category identity differs for {uid}")
        actual_budget = generation.get("max_tokens")
        if (not isinstance(actual_budget, int) or actual_budget <= 0
                or item.get("max_tokens") != actual_budget
                or budget is not None and actual_budget != budget):
            raise ValueError(f"Actual output budget for {uid} must equal {budget or item.get('max_tokens')}")
        if (not row.get("request_hash") or generation.get("request_hash") != row["request_hash"]
                or any(not isinstance(generation.get(key), str) for key in ("raw", "think", "answer", "finish_reason"))):
            raise ValueError(f"Incomplete raw generation evidence for {uid}")
        visible = generation["answer"] if cfg.generation.strip_think_for_judging else generation["raw"]
        if messages[-1].get("content", {}).get("answer") != visible:
            raise ValueError(f"Recorded extraction differs from the judged answer for {uid}")
        identities[uid] = question
    if Counter(item["category"] for item in identities.values()) != {"hard_prompt": 4, "creative_writing": 4}:
        raise ValueError("The study requires exactly four hard and four creative prompts")
    generation_settings = OmegaConf.to_container(cfg.generation, resolve=True)
    for operational in ("endpoint", "api_key", "max_tokens"):
        generation_settings.pop(operational, None)
    settings = {key: value for key, value in protocol["settings"].items() if key != "max_tokens"}
    return {"identity": identity, "protocol": protocol, "rows": by_uid, "questions": identities,
            "shared_settings": settings, "generation_settings": generation_settings,
            "serving_settings": OmegaConf.to_container(cfg.serving, resolve=True),
            "meta": meta, "sources": sources}


def matched_pair(low, high):
    for key in ("identity", "questions", "shared_settings", "generation_settings", "serving_settings"):
        if low[key] != high[key]:
            raise ValueError(f"The matched inputs differ in {key}; only max_tokens may vary")
    for key in ("bench_name", "effective_context_window", "output_budget_policy"):
        if low["protocol"].get(key) != high["protocol"].get(key):
            raise ValueError(f"The matched generation protocols differ in {key}")
    return {"status": "matched_except_deliberate_output_budget", "identity": low["identity"],
            "questions": low["questions"], "shared_settings": low["shared_settings"],
            "generation_settings_except_budget": low["generation_settings"],
            "serving_settings": low["serving_settings"],
            "effective_context_window": low["protocol"]["effective_context_window"],
            "output_budget_policy": low["protocol"]["output_budget_policy"],
            "budgets": BUDGETS, "n_prompts": 8}


def snapshot_arm(out, alias, *, budget=None):
    return read_arm(out / "metadata/inputs" / alias, out / "rollouts/inputs" / alias, budget=budget)


def verify_snapshot(out):
    for relative, expected in read(out / "metadata/snapshot_manifest.json").items():
        if digest(out / relative) != expected:
            raise ValueError(f"Prepared study input changed: {relative}")
    low = snapshot_arm(out, "budget6000", budget=6000)
    high = snapshot_arm(out, "budget12000", budget=12000)
    matched = matched_pair(low, high)
    if matched != read(out / "metadata/sources.json")["matched_pair"]:
        raise ValueError("Prepared study protocol no longer matches its saved provenance")
    return low, high


def prepare(out, low, high, historical=None):
    out = out.resolve()
    inputs = {"budget6000": low.resolve(), "budget12000": high.resolve()}
    if historical:
        inputs["historical"] = historical.resolve()
    if len(set(inputs.values())) != len(inputs):
        raise ValueError("Supply distinct original run directories")
    if any(out == source or out in source.parents or source in out.parents for source in inputs.values()):
        raise ValueError("Study output and input run directories must be separate")
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare requires an empty output directory; use analyze/judge to resume")
    arms = {alias: read_arm(source / "metadata", source / "rollouts",
                           budget=BUDGETS.get(alias, 6000), fresh=alias in BUDGETS)
            for alias, source in inputs.items()}
    matched = matched_pair(arms["budget6000"], arms["budget12000"])
    if historical and any(arms["historical"][key] != arms["budget6000"][key] for key in ("identity", "questions")):
        raise ValueError("Historical descriptive replication must have the same target/base pins and eight prompts")
    if any(list(source.rglob("generated_api_config.yaml")) for source in inputs.values()):
        raise ValueError("An input contains a generated judge credential file")
    rollouts, results, metadata = publish_layout(out)
    manifest = {}
    for alias, source in inputs.items():
        for section in ("metadata", "rollouts", "results"):
            original = source / section
            if not original.exists():
                continue
            before = {p.relative_to(original): digest(p) for p in original.rglob("*") if p.is_file()}
            destination = out / section / "inputs" / alias
            shutil.copytree(original, destination)
            for relative, expected in before.items():
                copied = destination / relative
                if digest(copied) != expected or digest(original / relative) != expected:
                    raise ValueError(f"Source changed while snapshotting: {original / relative}")
                manifest[copied.relative_to(out).as_posix()] = expected

    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.output_dir = str(metadata / "judge_work")
    cfg.smoke = True
    cfg.base_model = matched["identity"]["base_model"]
    cfg.generation = {"varied_parameter": "max_tokens", "values": [6000, 12000], "shared": matched["shared_settings"]}
    cfg.judge.model = "openai/gpt-4.1"
    cfg.judge.extra_body = {}
    cfg.judge.max_attempts = 3
    cfg.judge_validation.enabled = False
    cfg.baseline_arm = "budget6000"
    cfg.arms = [dict(name=alias, adapter=matched["identity"]["target"], role=role,
                    n_hard_prompt=4, n_creative_writing=4)
                for alias, role in (("budget6000", "baseline"), ("budget12000", "target"))]
    isolate_harness(cfg, metadata)
    bench = Path(cfg.vendor_dir) / "data" / cfg.bench_name
    questions_path = bench / "question.jsonl"
    benchmark_hash = digest(questions_path)
    full_questions = read_jsonl(questions_path)
    selected = [q for q in full_questions if q["uid"] in matched["questions"]]
    if len(selected) != 8 or set(q["uid"] for q in selected) != set(matched["questions"]) or any(arms["budget6000"]["rows"][q["uid"]]["generation_record"]["prompt"] != q["prompt"]
                                 or matched["questions"][q["uid"]]["category"] != q["category"] for q in selected):
        raise ValueError("Source prompts are not the same eight vendored benchmark questions")
    questions_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in selected), encoding="utf-8")
    manifest[questions_path.relative_to(out).as_posix()] = digest(questions_path)
    (bench / "model_answer").mkdir(parents=True, exist_ok=True)
    for alias in BUDGETS:
        rows = copy.deepcopy(list(arms[alias]["rows"].values()))
        for row in rows:
            row["model"] = alias
        destination = bench / "model_answer" / f"{alias}.jsonl"
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        shutil.copy2(destination, rollouts / f"answers_{alias}.jsonl")
        manifest[destination.relative_to(out).as_posix()] = digest(destination)
        manifest[f"rollouts/answers_{alias}.jsonl"] = digest(destination)
    protocol = {"kind": "output_budget_instrument_study", "capability_comparison": False,
                "calibration_status": "not_assessed", "limitations": LIMITATION,
                "matched_pair": matched, "benchmark_source_sha256": benchmark_hash,
                "sources": {alias: {"directory": str(source),
                    "role": "descriptive_replication_only" if alias == "historical" else "matched_budget_arm",
                    "original_timestamp_utc": arms[alias]["meta"].get("timestamp_utc"),
                    "original_git_sha": arms[alias]["meta"].get("git_sha")} for alias, source in inputs.items()},
                "compute_hardware": "Not recorded by standard arm metadata; consult the operator's infrastructure evidence.",
                "judge": {"model": "openai/gpt-4.1", "required_complete_games": 16,
                          "max_attempts_per_unresolved_game_per_invocation": 3,
                          "maximum_attempts_per_fresh_invocation": 48},
                "repo_name": artifact_name(SUBJECT)}
    dump(metadata / "sources.json", protocol)
    OmegaConf.save(cfg, metadata / "comparison_config.yaml")
    generated = write_run_meta(out, OmegaConf.to_container(cfg, resolve=True),
                               extra={"command": " ".join(sys.argv), "instrument_study": True,
                                      "capability_comparison": False, "sources": protocol})
    shutil.move(str(generated), metadata / "run_meta.json")
    for name in ("sources.json", "comparison_config.yaml", "run_meta.json"):
        manifest[f"metadata/{name}"] = digest(metadata / name)
    dump(metadata / "snapshot_manifest.json", manifest)
    analyze(out)
    print(f"Prepared {out}: eight matched prompts, 6000 vs 12000; judging has not run.")


def observation(row):
    generation = row["generation_record"]
    return {"finish_reason": generation["finish_reason"], "max_tokens": generation["max_tokens"],
            "empty_answer": not generation["answer"].strip(),
            "reasoning_chars": len(generation["think"]), "answer_chars": len(generation["answer"]),
            "answer_sha256": hashlib.sha256(generation["answer"].encode("utf-8")).hexdigest()}


def replication_observation(current, historical):
    result = {}
    for key in ("raw", "think", "answer"):
        left, right = current["generation_record"][key], historical["generation_record"][key]
        prefix = 0
        for a, b in zip(left, right):
            if a != b:
                break
            prefix += 1
        result[key] = {"identical": left == right, "common_prefix_chars": prefix,
                       "fresh_is_prefix_of_historical": right.startswith(left),
                       "historical_is_prefix_of_fresh": left.startswith(right)}
    return result


def analyze(out):
    low, high = verify_snapshot(out)
    arms = {"budget6000": low, "budget12000": high}
    provenance = read(out / "metadata/sources.json")
    if "historical" in provenance["sources"]:
        arms["historical"] = snapshot_arm(out, "historical", budget=6000)
    per_uid = [{"uid": uid, "category": low["questions"][uid]["category"],
                **{alias: observation(arm["rows"][uid]) for alias, arm in arms.items()}}
               for uid in sorted(low["rows"])]
    aggregate = {}
    for alias in arms:
        aggregate[alias] = {}
        for category in ("hard_prompt", "creative_writing"):
            values = [row[alias] for row in per_uid if row["category"] == category]
            block = {"n": len(values), "finish_reasons": dict(Counter(v["finish_reason"] for v in values)),
                     "empty_answers": sum(v["empty_answer"] for v in values)}
            for field in ("reasoning_chars", "answer_chars"):
                block[field] = {"sum": sum(v[field] for v in values),
                                "mean": sum(v[field] for v in values) / len(values),
                                "min": min(v[field] for v in values), "max": max(v[field] for v in values)}
            aggregate[alias][category] = block
    historical_replication = None
    if "historical" in arms:
        historical_replication = {
            "role": "descriptive_replication_only",
            "by_uid": {uid: replication_observation(low["rows"][uid], arms["historical"]["rows"][uid])
                       for uid in sorted(low["rows"])},
            "recorded_setting_differences": {key: {"fresh": low[key], "historical": arms["historical"][key]}
                for key in ("generation_settings", "serving_settings", "shared_settings")
                if low[key] != arms["historical"][key]},
            "interpretation": "Text equality/prefix diagnostics only; differences do not identify their cause."}
    manual_review = (out / "results/manual_review.md").exists()
    summary = {"instrument_study": True, "capability_comparison": False, "calibration_status": "not_assessed",
               "limitations": LIMITATION, "matched_pair": provenance["matched_pair"],
               "by_uid": per_uid, "by_slice": aggregate,
               "historical_role": "descriptive_replication_only" if "historical" in arms else None,
               "historical_replication": historical_replication,
               "manual_review": "results/manual_review.md" if manual_review else None,
               "judge_result": read(out / "results/judge_summary.json") if (out / "results/judge_summary.json").exists() else None,
               "judge_failures": [read(p) for p in sorted((out / "results/failures").glob("*.json"))]}
    dump(out / "results/results.json", summary)
    lines = ["# Arena-Hard output-budget instrument study", "", LIMITATION, "",
             "| UID | Slice | 6k finish | 6k empty | 6k reasoning chars | 6k answer chars | 12k finish | 12k empty | 12k reasoning chars | 12k answer chars |",
             "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in per_uid:
        fields = [row["uid"], row["category"]]
        for alias in BUDGETS:
            fields += [str(row[alias][key]) for key in ("finish_reason", "empty_answer", "reasoning_chars", "answer_chars")]
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "Full slice aggregates and any historical replication are in `results.json`.",
              "Original metadata and raw transcripts are retained under `metadata/inputs/` and `rollouts/inputs/`.",
              "No quality or health conclusion is inferred from these counts; qualitative review is separate."]
    if manual_review:
        lines += ["", "[Manual qualitative review](manual_review.md)"]
    (out / "results/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def judge(out):
    verify_snapshot(out)
    cfg = OmegaConf.load(out / "metadata/comparison_config.yaml")
    if (str(cfg.judge.model) != "openai/gpt-4.1" or cfg.judge.get("extra_body")
            or int(cfg.judge.get("max_attempts", 3)) != 3
            or str(cfg.baseline_arm) != "budget6000"):
        raise ValueError("This study permits only the frozen GPT-4.1 judging protocol")
    for arm in cfg.arms:
        if int(arm.n_hard_prompt) != 4 or int(arm.n_creative_writing) != 4:
            raise ValueError("Judging is bounded to four hard and four creative prompts")
    try:
        result = arena_judge.judge_arm(cfg, "budget12000", None, "openai/gpt-4.1")
        result.update(instrument_study=True, capability_comparison=False,
                      calibration_status="not_assessed", limitations=LIMITATION)
        dump(out / "results/judge_summary.json", result)
    except BaseException as exc:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dump(out / "results/failures" / f"{name}_judge.json",
             {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        raw = Path(cfg.vendor_dir) / "data" / cfg.bench_name / "model_judgment"
        if raw.exists():
            shutil.copytree(raw, out / "rollouts/judgments", dirs_exist_ok=True)
        analyze(out)


def publish(out):
    summary = analyze(out)
    if list(out.rglob("generated_api_config.yaml")):
        raise ValueError("Refusing publication with generated endpoint credentials")
    cfg = OmegaConf.load(out / "metadata/comparison_config.yaml")
    sources = read(out / "metadata/sources.json")
    fields = _card_fields("arena_hard", cfg, " ".join(sys.argv),
        experiment="Arena-Hard 6000 vs 12000 output-budget instrument study; eight fixed prompts; no training capability claim",
        models=json.dumps({"target": sources["matched_pair"]["identity"], "judge": sources["judge"],
                           "sources": sources["sources"], "limitations": LIMITATION}),
        source_revision=read(out / "metadata/run_meta.json")["git_sha"])
    assert_layout(out)
    url = push_run_dir(out, sources["repo_name"], fields, front_matter={"tags":
        run_tags("arena_hard", "output_budget_instrument_study", sources["matched_pair"]["identity"]["mode"])
        + ["instrument-study", "output-budget-comparison"]})
    print(url)
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description=LIMITATION)
    parser.add_argument("action", choices=("prepare", "analyze", "judge", "publish"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low", type=Path, help="Fresh standard 6000-token Arena arm output")
    parser.add_argument("--high", type=Path, help="Fresh standard 12000-token Arena arm output")
    parser.add_argument("--historical", type=Path, help="Optional frozen standard eight-answer run, descriptive only")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.low is None or args.high is None:
            parser.error("prepare requires --low and --high standard arm directories")
        prepare(args.out, args.low, args.high, args.historical)
    else:
        if args.low or args.high or args.historical:
            parser.error("Input directories are supplied only to prepare; later actions use the frozen snapshot")
        if args.action in {"judge", "publish"}:
            load_dotenv(".env", override=True)
        if args.action == "analyze":
            result = analyze(args.out)
            print(json.dumps(result["by_slice"], indent=2))
        elif args.action == "judge":
            judge(args.out)
        else:
            publish(args.out)


if __name__ == "__main__":
    main()
