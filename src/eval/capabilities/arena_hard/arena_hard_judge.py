# ABOUTME: Drive Arena-Hard pairwise judging of one arena-hard-eval arm against arm A,
# ABOUTME: with staged sampling. Run: uv run python src/eval/capabilities/arena_hard_judge.py --arm arm_d_synth40

"""Pairwise judging for the capability regression eval.

Thin driver over the vendored, patched `gen_judgment.py`. It owns the things the
vendored harness has no opinion about: which baseline to compare against, how many
questions this stage judges, which judge is pinned, and what the run cost.

Two modes:

**`judge`** (default) — run one arm against the baseline arm for one stage of the
staged-sampling ladder. Judgment caching verifies the exact request for each ordering,
so re-running a later stage reuses completed earlier requests. That is what
makes 150 → 300 → 500 cost the same as going straight to 500 while giving a read within
the first hour.

**`validate`** — dual-judge the configured panel and report agreement, win-rate gap
and swap consistency. The primary requires complete paired judgments. Legacy `gate`
policy also requires complete auxiliary coverage and enforces agreement thresholds;
`diagnostic` policy reports auxiliary failures and disagreement without blocking the
primary comparison.

    uv run python src/eval/capabilities/arena_hard_judge.py --arm arm_b_synth10 --stage 150
    uv run python src/eval/capabilities/arena_hard_judge.py --mode validate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.eval.capabilities.arena_hard.arena_hard_stats import (  # noqa: E402
    battles_from_judgments,
    per_prompt_scores,
    win_tie_loss,
)
from src.eval.capabilities.arena_hard.arena_hard_gen import _select_questions
from src.utils import read_jsonl, timestamp, write_run_meta  # noqa: E402

load_dotenv()


def _arm(cfg: DictConfig, name: str) -> DictConfig:
    """Look up an arm by name, failing loudly if it is not in the config."""
    for arm in cfg.arms:
        if arm.name == name:
            return arm
    raise SystemExit(f"Unknown arm {name!r}. Known: {', '.join(a.name for a in cfg.arms)}")


def _judge_settings(cfg: DictConfig, judge_model: str) -> DictConfig:
    """Resolve each judge's settings without inheriting another model's controls."""
    settings = OmegaConf.create(OmegaConf.to_container(cfg.judge, resolve=True))
    validation = cfg.get("judge_validation")
    if (judge_model != str(cfg.judge.model) and validation
            and judge_model == str(validation.get("reference_judge", ""))):
        for key in ("api_base", "api_key_env", "temperature", "max_tokens", "parallel", "max_attempts"):
            if key in validation:
                settings[key] = validation[key]
        settings.extra_body = validation.get("extra_body") or {}
    settings.model = judge_model
    return settings


def _write_endpoint_config(cfg: DictConfig, vendor: Path, judge_model: str) -> Path:
    """Emit the vendored harness's api_config entry for our judge.

    Generated rather than hand-edited so the pinned judge ID, reasoning effort and
    concurrency all trace back to `configs/eval/arena_hard.yaml` — one source of truth,
    and a `third_party/` wipe cannot take the settings with it.

    Args:
        cfg: Loaded arena-hard-eval config.
        vendor: Path to the vendored harness.
        judge_model: OpenRouter model id to pin.

    Returns:
        Path to the written endpoint config.
    """
    judge = _judge_settings(cfg, judge_model)
    key = os.environ.get(str(judge.api_key_env))
    if not key:
        raise SystemExit(
            f"{judge.api_key_env} is not set. All model calls in this repo route through "
            f"OpenRouter; put the key in .env."
        )
    entry: dict[str, Any] = {
        "model": judge_model,
        "endpoints": [{"api_base": str(judge.api_base), "api_key": key}],
        "api_type": "openai",
        "parallel": int(judge.parallel),
        "max_tokens": int(judge.max_tokens),
        "temperature": float(judge.temperature),
    }
    if judge.get("extra_body"):
        entry["extra_body"] = OmegaConf.to_container(judge.extra_body, resolve=True)

    path = vendor / "config" / "generated_api_config.yaml"
    path.write_text(yaml.safe_dump({judge_model: entry}, sort_keys=False), encoding="utf-8")
    return path


def _write_setting_config(
    cfg: DictConfig,
    vendor: Path,
    judge_model: str,
    models: list[str],
    limits: dict[str, int],
) -> Path:
    """Emit the vendored harness's judging setting file.

    Args:
        cfg: Loaded arena-hard-eval config.
        vendor: Path to the vendored harness.
        judge_model: OpenRouter model id to pin.
        models: Arms to judge (everything except the baseline).
        limits: `{category: n}` staged-sampling limits.

    Returns:
        Path to the written setting file.
    """
    upstream = yaml.safe_load((vendor / "config" / f"{cfg.bench_name}.yaml").read_text(encoding="utf-8"))
    judge = _judge_settings(cfg, judge_model)
    setting = {
        "judge_model": judge_model,
        "temperature": float(judge.temperature),
        "max_tokens": int(judge.max_tokens),
        "max_attempts": int(judge.get("max_attempts", 3)),
        "bench_name": str(cfg.bench_name),
        "reference": None,
        # Reuse upstream's verdict regexes and prompt template verbatim: the rubric is
        # the part that was validated, and rewriting it would silently change the estimand.
        "regex_patterns": upstream["regex_patterns"],
        "prompt_template": upstream["prompt_template"],
        "model_list": models,
        "question_limit": limits,
        # Read by the vendored gen_judgment.py's baseline patch: every category judges
        # against this arm instead of upstream's packaged leaderboard baselines. Flows
        # from run_eval.py's --target/--reference, never from an env var.
        "baseline_override": str(cfg.baseline_arm),
    }
    path = vendor / "config" / "generated_judge_config.yaml"
    path.write_text(yaml.safe_dump(setting, sort_keys=False), encoding="utf-8")
    return path


def _load_judgments(vendor: Path, cfg: DictConfig, judge_model: str, arm: str) -> list[dict]:
    """Read one arm's judgment records for a given judge."""
    path = (
        vendor / "data" / cfg.bench_name / "model_judgment" / judge_model / f"{arm}.jsonl"
    )
    if not path.exists():
        return []
    return read_jsonl(path)


def _cost(records: list[dict], judge_model: str) -> dict[str, Any]:
    """Total judge token usage and dollar cost over a set of judgment records.

    Footgun §10.3: Gemini 3.x Flash bills reasoning tokens as output. The projection in
    spec §11 assumes ~1,600 output tokens per question; verify against reality after the
    first stage rather than trusting it.

    Args:
        records: Judgment records carrying per-call `usage` from the patched client.
        judge_model: The pinned model id, used to pick a price.

    Returns:
        Token totals, observed per-question averages, and estimated USD.
    """
    # OpenRouter per-token prices for the models this eval pins. `:batch` is exactly half.
    prices = {
        "google/gemini-3-flash-preview": (0.5e-6, 3.0e-6),
        "google/gemini-3-flash-preview:batch": (0.25e-6, 1.5e-6),
        "openai/gpt-4.1": (2.0e-6, 8.0e-6),
        "openai/gpt-4.1:batch": (1.0e-6, 4.0e-6),
        "anthropic/claude-sonnet-4.5": (3.0e-6, 15.0e-6),
        "anthropic/claude-sonnet-4.5:batch": (1.5e-6, 7.5e-6),
    }
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    calls = 0
    reported_cost = 0.0
    reported_cost_calls = 0
    seen_attempts = set()
    pending = list(records)
    while pending:
        rec = pending.pop()
        pending.extend(rec.get("prior_judgments") or [])
        for game in rec.get("games") or []:
            if not game:
                continue
            # Each saved attempt is a paid call, including truncation or parse
            # failure; the final judgment also appears in attempts and is not extra.
            for attempt in game.get("attempts") or [game]:
                identity = attempt.get("attempt_id")
                if identity and identity in seen_attempts:
                    continue
                if identity:
                    seen_attempts.add(identity)
                usage = (attempt.get("judgment") or {}).get("usage") or {}
                prompt_tokens += usage.get("prompt_tokens") or 0
                completion_tokens += usage.get("completion_tokens") or 0
                reasoning_tokens += usage.get("reasoning_tokens") or 0
                cost = (attempt.get("judgment") or {}).get("reported_cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    reported_cost += cost
                    reported_cost_calls += 1
                calls += 1

    in_price, out_price = prices.get(judge_model, (0.0, 0.0))
    n_questions = len(records) or 1
    return {
        "judge_model": judge_model,
        "n_questions": len(records),
        "n_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        # Compare these against spec §11's ~3,700 in / ~1,600 out per question.
        "input_tokens_per_question": prompt_tokens / n_questions,
        "output_tokens_per_question": completion_tokens / n_questions,
        "usd": prompt_tokens * in_price + completion_tokens * out_price,
        "priced": judge_model in prices,
        "reported_usd": reported_cost if reported_cost_calls else None,
        "reported_cost_calls": reported_cost_calls,
    }


def _run_vendor(vendor: Path, setting: Path, endpoint: Path) -> None:
    """Invoke the vendored gen_judgment.py.

    The baseline override travels inside the setting file (`baseline_override`, spec
    §4's required deviation 1 — what makes the self-comparison land near 50%), so the
    written config is the complete record of the invocation.
    """
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
    try:
        subprocess.run(
            [sys.executable, "gen_judgment.py", "--setting-file",
             str(setting.relative_to(vendor)), "--endpoint-file",
             str(endpoint.relative_to(vendor))],
            cwd=vendor,
            env=env,
            check=True,
        )
    finally:
        # This generated file contains a credential; retain judgments, never the key.
        endpoint.unlink(missing_ok=True)


def _expected_questions(cfg, limits: dict[str, int]) -> list[dict]:
    questions = read_jsonl(Path(cfg.vendor_dir) / "data" / cfg.bench_name / "question.jsonl")
    arm = OmegaConf.create({"n_hard_prompt": limits["hard_prompt"],
                           "n_creative_writing": limits["creative_writing"]})
    selected = _select_questions(questions, arm)
    if not selected or len(selected) != sum(limits.values()):
        raise ValueError("Arena-Hard question set does not cover requested limits")
    if len({q['uid'] for q in selected}) != len(selected):
        raise ValueError("Arena-Hard duplicate question IDs")
    return selected


def _validate_answers(cfg, arm: str, questions: list[dict]) -> dict:
    path = Path(cfg.vendor_dir) / "data" / cfg.bench_name / "model_answer" / f"{arm}.jsonl"
    rows = read_jsonl(path)
    by_uid = {r['uid']: r for r in rows}
    if len(by_uid) != len(rows):
        raise ValueError(f"Arena-Hard duplicate answers for {arm}")
    for question in questions:
        row = by_uid.get(question['uid'])
        if row is None or row['messages'][0]['content'] != question['prompt']:
            raise ValueError(f"Arena-Hard missing or different prompt for {arm}: {question['uid']}")
        if not isinstance(row['messages'][-1]['content']['answer'], str):
            raise ValueError(f"Arena-Hard invalid answer schema for {arm}")
    return by_uid


def _is_complete_judgment(row: dict) -> bool:
    games = row.get("games") or []
    if (row.get("judgment_protocol") != "arena-paired-completion-v1"
            or not isinstance(games, list) or len(games) != 2):
        return False
    for game in games:
        if not isinstance(game, dict):
            return False
        output = game.get("judgment") or {}
        if (not isinstance(output, dict) or game.get("status") != "complete"
                or not game.get("request_hash") or output.get("finish_reason") != "stop"
                or output.get("error") or not isinstance(output.get("answer"), str)
                or not output["answer"].strip()):
            return False
    return len(battles_from_judgments([row])) == 2


def _complete_judgments(records: list[dict], questions: list[dict]) -> list[dict]:
    """Require certified complete orderings for a new comparison.

    Historical score-only rows can still be analyzed directly with the stats module;
    their missing completion evidence must never certify a new run.
    """
    by_uid = {r['uid']: r for r in records}
    if len(by_uid) != len(records):
        raise ValueError("Arena-Hard duplicate judgment IDs")
    selected = []
    for question in questions:
        row = by_uid.get(question['uid'])
        if (row is None or row['category'] != question['category']
                or not _is_complete_judgment(row)):
            raise ValueError(f"Arena-Hard incomplete paired judgments: {question['uid']}")
        selected.append(row)
    return selected


def _summarise(records: list[dict], baseline: str) -> dict[str, Any]:
    """Per-slice win/tie/loss and mean score for a set of judgment records."""
    battles = battles_from_judgments(records)
    out: dict[str, Any] = {"baseline": baseline, "by_slice": {}}
    for category in sorted({b["category"] for b in battles}):
        subset = [b for b in battles if b["category"] == category]
        _, scores = per_prompt_scores(subset)
        out["by_slice"][category] = win_tie_loss(subset) | {"mean_score": float(scores.mean())}
    return out


def judge_arm(cfg: DictConfig, arm: str, stage: int | None, judge_model: str) -> dict[str, Any]:
    """Judge one arm against the baseline for one stage of the ladder."""
    vendor = Path(cfg.vendor_dir)
    arm_cfg = _arm(cfg, arm)
    baseline = str(cfg.baseline_arm)
    if arm == baseline:
        # Spec §7 lists A-vs-A as an instrument sanity check. It is a real comparison —
        # the harness judges the baseline's answers against themselves — and should land
        # at 50% with a very high tie rate. Anything else means the judge is not stable.
        print(">>> A-vs-A instrument sanity check (expect ~50% and a high tie rate)")

    limits = {
        "hard_prompt": int(stage) if stage else int(arm_cfg.n_hard_prompt),
        "creative_writing": int(arm_cfg.n_creative_writing),
    }
    # An arm cannot be judged past the number of answers it actually has.
    limits["hard_prompt"] = min(limits["hard_prompt"], int(arm_cfg.n_hard_prompt))

    questions = _expected_questions(cfg, limits)
    for name in (arm, baseline):
        _validate_answers(cfg, name, questions)
    endpoint = _write_endpoint_config(cfg, vendor, judge_model)
    setting = _write_setting_config(cfg, vendor, judge_model, [arm], limits)

    print(f">>> arm:      {arm}  vs baseline {baseline}")
    print(f">>> judge:    {judge_model}")
    print(f">>> stage:    {limits}")
    _run_vendor(vendor, setting, endpoint)

    records = _complete_judgments(_load_judgments(vendor, cfg, judge_model, arm), questions)
    return {"arm": arm, "limits": limits} | _summarise(records, baseline) | {
        "cost": _cost(records, judge_model)
    }


class JudgeValidationError(ValueError):
    """A strict coverage failure carrying the report callers must retain."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _judgment_coverage(records, questions, *, judge_model, arm, baseline,
                       expected_answers, allowed_questions):
    allowed = {q["uid"]: q for q in allowed_questions}
    requested = {q["uid"]: q for q in questions}
    if len(requested) != len(questions) or len(allowed) != len(allowed_questions):
        raise ValueError("Arena-Hard duplicate question IDs")
    by_uid = {}
    for row in records:
        uid = row["uid"]
        if uid in by_uid:
            raise ValueError(f"Arena-Hard duplicate judgment IDs: {uid}")
        if (uid not in allowed or row.get("category") != allowed[uid]["category"]
                or row.get("judge") != judge_model):
            raise ValueError(f"Arena-Hard unexpected judgment identity: {uid}")
        # Cached larger stages are retained, but only requested pairs enter this panel.
        if uid in requested:
            if (row.get("model") != expected_answers[arm][uid]["model"]
                    or row.get("baseline") != expected_answers[baseline][uid]["model"]):
                raise ValueError(f"Arena-Hard unexpected answer identity: {uid}")
        by_uid[uid] = row
    missing = [uid for uid in requested if uid not in by_uid]
    incomplete = [uid for uid in requested if uid in by_uid and not _is_complete_judgment(by_uid[uid])]
    complete = [by_uid[uid] for uid in requested if uid in by_uid and _is_complete_judgment(by_uid[uid])]
    return complete, {
        "status": "complete" if not missing and not incomplete else "incomplete",
        "n_expected": len(questions), "n_complete": len(complete),
        "n_missing": len(missing), "n_incomplete": len(incomplete),
        "missing_uids": missing, "incomplete_uids": incomplete,
        "n_unrequested_cached": len(set(by_uid) - set(requested)),
    }


def summarise_judge_validation(cfg, questions, records_by_judge, *, expected_answers,
                              allowed_questions=None, execution_errors=None) -> dict[str, Any]:
    """Analyze saved judge records without inference or silently filling missing pairs.

    ``expected_answers`` maps arm names to their by-UID answer records. The primary
    must cover every requested question under either policy. Diagnostic auxiliary
    coverage is reported explicitly; agreement uses only complete shared pairs.
    """
    val = cfg.judge_validation
    arm = str(val.comparison_arm)
    baseline = str(cfg.baseline_arm)
    primary = str(cfg.judge.model)
    reference = str(val.reference_judge)
    policy = str(val.get("policy", "gate"))
    if policy not in {"gate", "diagnostic"}:
        raise ValueError(f"Unknown Arena-Hard judge validation policy: {policy}")
    if primary == reference:
        raise ValueError("Arena-Hard primary and auxiliary judges must be distinct")
    if len(questions) != int(val.n_questions) or any(q["category"] != val.slice for q in questions):
        raise ValueError("Arena-Hard validation questions do not match requested panel")
    allowed_questions = questions if allowed_questions is None else allowed_questions
    execution_errors = execution_errors or {}
    coverage, scored = {}, {}
    for judge_model in (primary, reference):
        complete, coverage[judge_model] = _judgment_coverage(
            records_by_judge.get(judge_model, []), questions, judge_model=judge_model,
            arm=arm, baseline=baseline, expected_answers=expected_answers,
            allowed_questions=allowed_questions)
        if judge_model in execution_errors:
            coverage[judge_model]["execution_error"] = execution_errors[judge_model]
            coverage[judge_model]["status"] = "execution_failed"
        battles = battles_from_judgments(complete)
        uids, scores = per_prompt_scores(battles)
        scored[judge_model] = {
            "by_uid": dict(zip(uids, scores)),
            "records": complete,
        }

    shared = sorted(set(scored[primary]["by_uid"]) & set(scored[reference]["by_uid"]))
    # Agreement on the per-prompt verdict, collapsed to win/tie/loss so a "slightly" vs
    # "significantly" difference is not counted as disagreement.
    def bucket(x: float) -> str:
        return "win" if x > 0.5 else ("loss" if x < 0.5 else "tie")

    agreement = wr_primary = wr_reference = gap_pp = thresholds_met = None
    thresholds = val.thresholds
    if shared:
        agreement = sum(bucket(scored[primary]["by_uid"][uid]) == bucket(scored[reference]["by_uid"][uid])
                        for uid in shared) / len(shared)
        wr_primary = float(sum(scored[primary]["by_uid"][u] for u in shared) / len(shared))
        wr_reference = float(sum(scored[reference]["by_uid"][u] for u in shared) / len(shared))
        gap_pp = abs(wr_primary - wr_reference) * 100
        thresholds_met = (agreement >= float(thresholds.verdict_agreement_min)
                          and gap_pp <= float(thresholds.win_rate_gap_max_pp))
    swap = {}
    for judge_model in (primary, reference):
        matched = [row for row in scored[judge_model]["records"] if row["uid"] in shared]
        swap[judge_model] = win_tie_loss(battles_from_judgments(matched))["swap_consistency"] if matched else None
    primary_complete = coverage[primary]["status"] == "complete"
    auxiliary_complete = coverage[reference]["status"] == "complete"
    report = {
        "policy": policy,
        "primary_judge": primary,
        "reference_judge": reference,
        "comparison_arm": arm,
        "slice": str(val.slice),
        "n_requested": len(questions),
        "primary_complete": primary_complete,
        "n_primary_complete": coverage[primary]["n_complete"],
        "n_auxiliary_complete": coverage[reference]["n_complete"],
        "coverage": coverage,
        "n_compared": len(shared),
        "compared_uids": shared,
        "metrics_scope": "complete_shared_pairs_only",
        "diagnostic_status": ("complete" if auxiliary_complete else "partial") if shared else "unavailable",
        "verdict_agreement": agreement,
        "agreement_threshold": float(thresholds.verdict_agreement_min),
        "win_rate_primary": wr_primary,
        "win_rate_reference": wr_reference,
        "win_rate_gap_pp": gap_pp,
        "gap_threshold_pp": float(thresholds.win_rate_gap_max_pp),
        "swap_consistency": swap,
        "thresholds_met_on_sample": thresholds_met,
        "passes": bool(primary_complete and auxiliary_complete and thresholds_met) if policy == "gate" else None,
        "fallback_judge": val.get("fallback_judge"),
        "cost": {
            judge: _cost([row for row in records_by_judge.get(judge, [])
                          if row["uid"] in {q["uid"] for q in questions}], judge)
            for judge in (primary, reference)
        },
    }
    if not primary_complete:
        raise JudgeValidationError("Arena-Hard primary judge has incomplete paired coverage", report)
    if policy == "gate" and not auxiliary_complete:
        raise JudgeValidationError("Arena-Hard auxiliary judge has incomplete paired coverage", report)
    return report


def validate_judge(cfg: DictConfig) -> dict[str, Any]:
    """Collect the configured panel, then enforce gate or diagnostic policy."""
    val = cfg.judge_validation
    policy = str(val.get("policy", "gate"))
    if policy not in {"gate", "diagnostic"}:
        raise ValueError(f"Unknown Arena-Hard judge validation policy: {policy}")
    if not bool(cfg.get("smoke", False)) and int(val.n_questions) != 100:
        raise ValueError("Full Arena-Hard judge validation requires 100 questions; use smoke for wiring checks")
    arm, baseline = str(val.comparison_arm), str(cfg.baseline_arm)
    primary, reference = str(cfg.judge.model), str(val.reference_judge)
    if primary == reference:
        raise ValueError("Arena-Hard primary and auxiliary judges must be distinct")
    limits = {"hard_prompt": 0, "creative_writing": 0}
    if val.slice not in limits:
        raise ValueError(f"Unsupported Arena-Hard validation slice: {val.slice}")
    limits[str(val.slice)] = int(val.n_questions)
    vendor = Path(cfg.vendor_dir)
    questions = _expected_questions(cfg, limits)
    allowed = read_jsonl(vendor / "data" / cfg.bench_name / "question.jsonl")
    answers = {name: _validate_answers(cfg, name, questions) for name in (arm, baseline)}
    records, errors = {}, {}
    for judge_model in (primary, reference):
        endpoint = _write_endpoint_config(cfg, vendor, judge_model)
        setting = _write_setting_config(cfg, vendor, judge_model, [arm], limits)
        print(f"\n>>> judging {val.n_questions} {val.slice} panel with {judge_model} ({policy})")
        try:
            _run_vendor(vendor, setting, endpoint)
        except subprocess.CalledProcessError as exc:
            errors[judge_model] = {"type": type(exc).__name__, "returncode": exc.returncode}
        records[judge_model] = _load_judgments(vendor, cfg, judge_model, arm)
        if judge_model == primary:
            _, coverage = _judgment_coverage(
                records[primary], questions, judge_model=primary, arm=arm, baseline=baseline,
                expected_answers=answers, allowed_questions=allowed)
            if coverage["status"] != "complete" or primary in errors:
                # Attach a report before any auxiliary spend when primary integrity fails.
                return summarise_judge_validation(cfg, questions, records, expected_answers=answers,
                                                   allowed_questions=allowed, execution_errors=errors)
    return summarise_judge_validation(cfg, questions, records, expected_answers=answers,
                                       allowed_questions=allowed, execution_errors=errors)


def main(
    config: str = "configs/eval/arena_hard.yaml",
    mode: str = "judge",
    arm: str = "",
    stage: int = 0,
    judge_model: str = "",
) -> None:
    """Judge an arm against the baseline, or validate the judge.

    Args:
        config: Path to the arena-hard-eval config.
        mode: `judge` or `validate`.
        arm: Arm to judge (required in `judge` mode).
        stage: hard_prompt questions to judge this stage; 0 uses the arm's full count.
        judge_model: Override the pinned judge (e.g. to run the fallback).
    """
    cfg = OmegaConf.load(config)
    vendor = Path(cfg.vendor_dir)
    if not (vendor / "gen_judgment.py").exists():
        raise SystemExit(
            f"No vendored harness at {vendor} — the tree is TRACKED in git (patched, "
            "pruned; see its VENDORED_FROM.txt). Restore it: git checkout -- "
            f"{vendor}")
    # A missing patch means the judge would silently compare against upstream's packaged
    # baseline instead of arm A, producing a number that looks fine and means nothing.
    if "baseline_override" not in (vendor / "gen_judgment.py").read_text():
        raise SystemExit(
            f"Vendored gen_judgment.py has lost the baseline_override patch (upstream "
            f"re-clone?). The tracked tree carries the patches as ordinary diffs — "
            f"restore it: git checkout -- {vendor}")

    out_dir = Path(cfg.output_dir) / "judging" / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "validate":
        name = "judge_validation"
        try:
            result = validate_judge(cfg)
        except JudgeValidationError as exc:
            (out_dir / f"{name}.json").write_text(json.dumps(exc.report, indent=2), encoding="utf-8")
            write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={"mode": mode})
            raise
        def percentage(value):
            return f"{value:.1%}" if value is not None else "unavailable"
        print(f"\n=== Judge panel ({result['policy']}) ===")
        print(f"  primary complete {result['n_primary_complete']}/{result['n_requested']}")
        print(f"  auxiliary pairs  {result['n_auxiliary_complete']}/{result['n_requested']}")
        print(f"  compared pairs   {result['n_compared']} (complete shared pairs only)")
        threshold_label = "reference" if result["policy"] == "diagnostic" else "need"
        print(f"  agreement       {percentage(result['verdict_agreement'])} "
              f"({threshold_label} >= {result['agreement_threshold']:.0%})")
        print(f"  win rate        {result['primary_judge']}: {percentage(result['win_rate_primary'])}")
        print(f"                  {result['reference_judge']}: {percentage(result['win_rate_reference'])}")
        gap = f"{result['win_rate_gap_pp']:.1f}pp" if result['win_rate_gap_pp'] is not None else "unavailable"
        print(f"  gap             {gap} "
              f"({threshold_label} <= {result['gap_threshold_pp']:.0f}pp)")
        for judge, consistency in result["swap_consistency"].items():
            print(f"  swap consist.   {judge}: {percentage(consistency)}")
        verdict = "DIAGNOSTIC ONLY" if result["policy"] == "diagnostic" else ("PASS" if result["passes"] else "FAIL")
        print(f"  VERDICT         {verdict}")
    elif mode == "judge":
        if not arm:
            raise SystemExit("--arm is required in judge mode")
        result = judge_arm(cfg, arm, stage or None, judge_model or str(cfg.judge.model))
        name = f"judgment_{arm}"
        print(f"\n=== {arm} vs {result['baseline']} ===")
        for category, block in result["by_slice"].items():
            print(
                f"  {category:18} n={block['n_prompts']:<4} "
                f"win={block['win_rate']:.1%} tie={block['tie_rate']:.1%} "
                f"loss={block['loss_rate']:.1%} swap={block['swap_consistency']:.1%}"
            )
        cost = result["cost"]
        print(
            f"  cost            ${cost['usd']:.2f} over {cost['n_calls']} calls "
            f"({cost['output_tokens_per_question']:.0f} out-tok/question"
            f"{'' if cost['priced'] else ', UNPRICED MODEL'})"
        )
        # Footgun §10.3. Spec §11 budgets ~1,600 output tokens per question, but that is
        # measurably too low for this judge: Gemini 3 Flash spends 300-500 reasoning
        # tokens per call even at `effort: low` (verified by A/B — low genuinely reduces
        # them, it is not being ignored), so ~3,100 per question across both orderings is
        # the normal operating point, not an alarm. The threshold below is set above that
        # so it fires on a real blowout — a dropped `extra_body`, or a judge silently
        # swapped for one that reasons harder — rather than on every run.
        if (cost["judge_model"] == "google/gemini-3-flash-preview"
                and cost["output_tokens_per_question"] > 5000):
            print(
                f"  WARNING: output tokens/question is {cost['output_tokens_per_question']:.0f}, "
                f"far above the ~3,100 observed for this judge at `effort: low`. Check that "
                f"`judge.extra_body` is reaching the provider and that the judge ID is pinned."
            )
    else:
        raise SystemExit(f"Unknown mode {mode!r}; expected 'judge' or 'validate'")

    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={"mode": mode})
    print(f"\n>>> {out_dir / f'{name}.json'}")


if __name__ == "__main__":
    fire.Fire(main)
