# ABOUTME: Pairwise Arena judgments with exact-request caches and bounded recovery.
# ABOUTME: Every attempt is retained; only complete, parseable orderings are scored.
import copy
import hashlib
import json
import yaml
import argparse
import os
import concurrent.futures
import threading
import uuid
from pathlib import Path

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_model_answers,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS

JUDGMENT_PROTOCOL = "arena-paired-completion-v1"
VALID_SCORES = {"A>>B", "A>B", "A=B", "B>A", "B>>A"}


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def public_settings(value):
    """Remove credentials and operational concurrency from request identity."""
    if isinstance(value, dict):
        return {key: public_settings(item) for key, item in value.items()
                if key.lower() not in {"api_key", "authorization", "parallel"}}
    if isinstance(value, list):
        return [public_settings(item) for item in value]
    return value


def judgment_request(question, baseline, answer, reference, configs, settings, ordering=0):
    """Build the actual rubric and exact content identity before consulting cache."""
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    if isinstance(reference, list):
        if len(reference) != 1:
            raise ValueError("Pairwise judging supports exactly one rubric reference")
        reference = reference[0]
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    identity = {
        "protocol": JUDGMENT_PROTOCOL,
        "question": question,
        "ordering": ordering,
        "messages": messages,
        "answers": [{key: row.get(key) for key in ("model", "request_hash")}
                    for row in (baseline, answer)],
        "reference": reference,
        "judge_model": configs["judge_model"],
        "temperature": configs["temperature"],
        "max_tokens": configs["max_tokens"],
        "regex_patterns": configs["regex_patterns"],
        "settings": public_settings(settings),
    }
    request_hash = hashlib.sha256(json.dumps(
        identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return messages, request_hash


def complete_game(game, request_hash=None):
    """Old score-only rows are analysis inputs, never certified cache entries."""
    if not isinstance(game, dict):
        return False
    output = game.get("judgment") or {}
    return (game.get("status") == "complete"
            and game.get("score") in VALID_SCORES
            and bool(game.get("request_hash"))
            and (request_hash is None or game["request_hash"] == request_hash)
            and output.get("finish_reason") == "stop"
            and not output.get("error")
            and isinstance(output.get("answer"), str)
            and bool(output["answer"].strip()))


def pairwise_judgment(question, baseline, answer, reference, configs, settings, ordering=0):
    messages, request_hash = judgment_request(
        question, baseline, answer, reference, configs, settings, ordering
    )
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
        "judge_attempt": True,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    try:
        output = api_completion_func(**kwargs)
    except Exception as exc:
        message = str(exc)
        for endpoint in settings.get("endpoints") or []:
            if endpoint.get("api_key"):
                message = message.replace(endpoint["api_key"], "[REDACTED]")
        output = {"answer": None, "finish_reason": None,
                  "error": {"type": type(exc).__name__, "message": message}}
    if not isinstance(output, dict):
        output = {"answer": None, "finish_reason": None, "raw_response": output,
                  "error": {"type": "InvalidCompletion", "message": "No completion object"}}
    text = output.get("answer")
    parsed = get_score(text, configs["regex_patterns"]) if isinstance(text, str) else None
    reasons = []
    if output.get("error"):
        reasons.append("transport_error")
    if output.get("finish_reason") != "stop":
        reasons.append("incomplete_completion")
    if parsed not in VALID_SCORES:
        reasons.append("invalid_verdict")
    return {
        "attempt_id": uuid.uuid4().hex,
        "score": None if reasons else parsed,
        "parsed_score": parsed,
        "judgment": output,
        "prompt": messages,
        "request_hash": request_hash,
        "status": "invalid" if reasons else "complete",
        "invalid_reasons": reasons,
    }


class JudgmentStore:
    """Atomic snapshots retain completed orderings even if another worker fails."""

    def __init__(self, output_file):
        self.path = Path(output_file)
        self.lock = threading.Lock()
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row["uid"] in self.records:
                        raise ValueError(f"Duplicate judgment uid in {self.path}: {row['uid']}")
                    self.records[row["uid"]] = row

    def save(self, row):
        with self.lock:
            self.records[row["uid"]] = copy.deepcopy(row)
            temporary = self.path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for uid in sorted(self.records):
                    stream.write(json.dumps(self.records[uid], ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "judgment_protocol": JUDGMENT_PROTOCOL,
        "games": [None, None],
    }
    store = args.get("store") or JudgmentStore(args["output_file"])
    existing = store.records.get(output["uid"], {})
    old_games = existing.get("games") or []
    expected = [judgment_request(args['question'], a, b, args['reference'],
                                args['configs'], args['settings'], order)[1]
                for order, (a, b) in enumerate(((baseline, answer), (answer, baseline)))]
    identity_matches = (existing.get("judgment_protocol") == JUDGMENT_PROTOCOL
                        and len(old_games) == 2
                        and all(isinstance(game, dict) and game.get("request_hash") == key
                                for game, key in zip(old_games, expected)))
    # Preserve old identities too; otherwise changing a setting erases the evidence
    # for the invalidated cache entry. Do not recursively duplicate earlier history.
    history = copy.deepcopy(existing.get("prior_judgments") or [])
    if existing and not identity_matches:
        history.append({key: value for key, value in existing.items()
                        if key != "prior_judgments"})
    if history:
        output["prior_judgments"] = history
    # Copy both matching orderings before the first checkpoint; recovering ordering
    # zero must not overwrite an already-complete ordering one if interrupted again.
    for ordering, previous in enumerate(old_games[:2]):
        if (existing.get("judgment_protocol") == JUDGMENT_PROTOCOL
                and isinstance(previous, dict)
                and previous.get("request_hash") == expected[ordering]):
            output["games"][ordering] = copy.deepcopy(previous)
    max_attempts = int(args['configs'].get("max_attempts", 3))
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for ordering, (first, second) in enumerate(((baseline, answer), (answer, baseline))):
        previous = old_games[ordering] if ordering < len(old_games) else None
        matches = (existing.get("judgment_protocol") == JUDGMENT_PROTOCOL
                   and isinstance(previous, dict)
                   and previous.get("request_hash") == expected[ordering])
        if matches:
            output["games"][ordering] = copy.deepcopy(previous)
        if matches and complete_game(previous, expected[ordering]):
            continue
        attempts = copy.deepcopy(previous.get("attempts") or []) if matches else []
        for _ in range(max_attempts):
            result = pairwise_judgment(
                question=args['question'], baseline=first, answer=second,
                reference=args['reference'], configs=args['configs'],
                settings=args['settings'], ordering=ordering,
            )
            attempts.append(copy.deepcopy(result))
            output["games"][ordering] = result | {"attempts": attempts}
            store.save(output)
            if complete_game(result, expected[ordering]):
                break
    store.save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    # PATCH (capability eval spec §4): baseline override. Upstream's packaged baselines
    # are right for a public leaderboard and wrong for us — our estimand is a treated
    # checkpoint against its own untreated sibling, and the comparison is only
    # interpretable when the baseline IS that sibling. The arm name arrives in the
    # setting file written by the driver (from run_eval.py's --target/--reference flow);
    # deliberately not an env var, so the invocation is the complete record.
    _baseline_override = configs.get("baseline_override")
    if _baseline_override:
        for _category in JUDGE_SETTINGS:
            JUDGE_SETTINGS[_category]["baseline"] = _baseline_override
        print(f"INFO: baseline_override -> {_baseline_override}")

    print(f'judge model: {configs["judge_model"]}, reference: {configs["reference"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    questions = load_questions(question_file)

    # PATCH (capability eval spec §9): staged sampling. `question_limit` is a
    # {category: n} mapping. Exact-request caches reuse completed earlier orderings.
    _limits = configs.get("question_limit") or {}
    if _limits:
        _seen = {}
        _kept = []
        for _q in questions:
            _cat = _q["category"]
            _n = _seen.get(_cat, 0)
            if _cat in _limits and _n >= _limits[_cat]:
                continue
            _seen[_cat] = _n + 1
            _kept.append(_q)
        print(f"INFO: question_limit {_limits} -> {len(_kept)}/{len(questions)} questions")
        questions = _kept

    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    if configs["reference"]:
        assert not configs["reference"] in models, "ERROR: one of the models being evaluated is used as reference."
        ref_answers = [model_answers[model] for model in configs["reference"]]
    else:
        ref_answers = None
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    stores = {model: JudgmentStore(path) for model, path in output_files.items()}

    endpoint_settings = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_settings["parallel"]) as executor:
        futures = []
        for model in models:
            for question in questions:
                uid = question["uid"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not uid in model_answers[model]:
                    print(f"Warning: {model} answer to {question['uid']} cannot be found.")
                    continue

                kwargs["answer"] = model_answers[model][uid]
                kwargs["baseline"] = model_answers[
                    JUDGE_SETTINGS[question["category"]]["baseline"]
                ][uid]
                
                if ref_answers:
                    kwargs["reference"] = [ref_answer[uid] for ref_answer in ref_answers]
                else:
                    kwargs["reference"] = None
                    
                kwargs["configs"] = configs
                kwargs["settings"] = endpoint_settings
                kwargs["output_file"] = output_files[model]
                kwargs["store"] = stores[model]
                                
                future = executor.submit(judgment, kwargs)
                futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
