# ABOUTME: Drive Arena-Hard pairwise judging of one capability-eval arm against arm A,
# ABOUTME: with staged sampling. Run: uv run python src/experiments/capability_judge.py --arm arm_d_synth40

"""Pairwise judging for the capability regression eval.

Thin driver over the vendored, patched `gen_judgment.py`. It owns the things the
vendored harness has no opinion about: which baseline to compare against, how many
questions this stage judges, which judge is pinned, and what the run cost.

Two modes:

**`judge`** (default) — run one arm against the baseline arm for one stage of the
staged-sampling ladder. Judgment caching keys on `uid`, so re-running a later stage
re-reads the earlier stage from disk and pays only for the new questions. That is what
makes 150 → 300 → 500 cost the same as going straight to 500 while giving a read within
the first hour.

**`validate`** — spec §4's judge validation, which is not optional. Gemini 3 Flash is a
tier below arena-hard-auto's validated judges, and the rubric requires the judge to draft
its own answer to a hard software-engineering prompt before comparing: if it cannot do
that well, the reference answer is weak and every comparison degrades. Our candidates are
~27B models, so the judge has to be clearly stronger than what it is judging. This
dual-judges 100 questions and reports agreement, win-rate gap and swap consistency.

    uv run python src/experiments/capability_judge.py --arm arm_b_synth10 --stage 150
    uv run python src/experiments/capability_judge.py --mode validate
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.capability_stats import (  # noqa: E402
    battles_from_judgments,
    per_prompt_scores,
    win_tie_loss,
)
from src.utils import read_jsonl, timestamp, write_run_meta  # noqa: E402

load_dotenv()


def _arm(cfg: DictConfig, name: str) -> DictConfig:
    """Look up an arm by name, failing loudly if it is not in the config."""
    for arm in cfg.arms:
        if arm.name == name:
            return arm
    raise SystemExit(f"Unknown arm {name!r}. Known: {', '.join(a.name for a in cfg.arms)}")


def _write_endpoint_config(cfg: DictConfig, vendor: Path, judge_model: str) -> Path:
    """Emit the vendored harness's api_config entry for our judge.

    Generated rather than hand-edited so the pinned judge ID, reasoning effort and
    concurrency all trace back to `configs/capability_eval.yaml` — one source of truth,
    and a `third_party/` wipe cannot take the settings with it.

    Args:
        cfg: Loaded capability-eval config.
        vendor: Path to the vendored harness.
        judge_model: OpenRouter model id to pin.

    Returns:
        Path to the written endpoint config.
    """
    judge = cfg.judge
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
    path.write_text(yaml.safe_dump({judge_model: entry}, sort_keys=False))
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
        cfg: Loaded capability-eval config.
        vendor: Path to the vendored harness.
        judge_model: OpenRouter model id to pin.
        models: Arms to judge (everything except the baseline).
        limits: `{category: n}` staged-sampling limits.

    Returns:
        Path to the written setting file.
    """
    upstream = yaml.safe_load((vendor / "config" / f"{cfg.bench_name}.yaml").read_text())
    setting = {
        "judge_model": judge_model,
        "temperature": float(cfg.judge.temperature),
        "max_tokens": int(cfg.judge.max_tokens),
        "bench_name": str(cfg.bench_name),
        "reference": None,
        # Reuse upstream's verdict regexes and prompt template verbatim: the rubric is
        # the part that was validated, and rewriting it would silently change the estimand.
        "regex_patterns": upstream["regex_patterns"],
        "prompt_template": upstream["prompt_template"],
        "model_list": models,
        "question_limit": limits,
    }
    path = vendor / "config" / "generated_judge_config.yaml"
    path.write_text(yaml.safe_dump(setting, sort_keys=False))
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
    for rec in records:
        for game in rec.get("games") or []:
            usage = ((game or {}).get("judgment") or {}).get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
            reasoning_tokens += usage.get("reasoning_tokens") or 0
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
    }


def _run_vendor(vendor: Path, setting: Path, endpoint: Path, baseline: str) -> None:
    """Invoke the vendored gen_judgment.py with our baseline override.

    `ARENA_HARD_BASELINE` is read by the patch in `utils/judge_utils.py` — spec §4's
    required deviation 1, which is what makes the self-comparison land near 50%.
    """
    env = os.environ | {"ARENA_HARD_BASELINE": baseline, "PYTHONUNBUFFERED": "1"}
    subprocess.run(
        [
            sys.executable,
            "gen_judgment.py",
            "--setting-file",
            str(setting.relative_to(vendor)),
            "--endpoint-file",
            str(endpoint.relative_to(vendor)),
        ],
        cwd=vendor,
        env=env,
        check=True,
    )


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

    endpoint = _write_endpoint_config(cfg, vendor, judge_model)
    setting = _write_setting_config(cfg, vendor, judge_model, [arm], limits)

    print(f">>> arm:      {arm}  vs baseline {baseline}")
    print(f">>> judge:    {judge_model}")
    print(f">>> stage:    {limits}")
    _run_vendor(vendor, setting, endpoint, baseline)

    records = _load_judgments(vendor, cfg, judge_model, arm)
    return {"arm": arm, "limits": limits} | _summarise(records, baseline) | {
        "cost": _cost(records, judge_model)
    }


def validate_judge(cfg: DictConfig) -> dict[str, Any]:
    """Spec §4 judge validation: dual-judge one arm and compare the two judges.

    Deliberately validates against GPT-4.1 rather than a Sonnet-class judge. Claude
    generated our synthetic corpus, so a Claude validator would import the very
    generator-family self-preference confound we avoided by choosing Gemini. GPT-4.1 is
    a third family *and* arena-hard-auto's own primary validated judge, which makes it
    the stronger reference on both counts.

    Returns:
        Agreement rate, per-judge win rates and their gap, swap consistency per judge,
        the pass/fail verdict, and the cost of the exercise.
    """
    val = cfg.judge_validation
    arm = str(val.comparison_arm)
    primary = str(cfg.judge.model)
    reference = str(val.reference_judge)
    limits = {"hard_prompt": int(val.n_questions), "creative_writing": 0}

    vendor = Path(cfg.vendor_dir)
    baseline = str(cfg.baseline_arm)

    results = {}
    for judge_model in (primary, reference):
        endpoint = _write_endpoint_config(cfg, vendor, judge_model)
        setting = _write_setting_config(cfg, vendor, judge_model, [arm], limits)
        print(f"\n>>> validating with {judge_model} on {val.n_questions} {val.slice} questions")
        _run_vendor(vendor, setting, endpoint, baseline)
        results[judge_model] = _load_judgments(vendor, cfg, judge_model, arm)

    # Compare only questions both judges actually returned a parseable verdict on.
    scored = {}
    for judge_model, records in results.items():
        battles = [b for b in battles_from_judgments(records) if b["category"] == val.slice]
        uids, scores = per_prompt_scores(battles)
        scored[judge_model] = {
            "by_uid": dict(zip(uids, scores)),
            "split": win_tie_loss(battles) if battles else {},
        }

    shared = sorted(set(scored[primary]["by_uid"]) & set(scored[reference]["by_uid"]))
    if not shared:
        raise SystemExit("No overlapping judged questions; cannot validate.")

    # Agreement on the per-prompt verdict, collapsed to win/tie/loss so a "slightly" vs
    # "significantly" difference is not counted as disagreement.
    def bucket(x: float) -> str:
        return "win" if x > 0.5 else ("loss" if x < 0.5 else "tie")

    agree = sum(
        1
        for uid in shared
        if bucket(scored[primary]["by_uid"][uid]) == bucket(scored[reference]["by_uid"][uid])
    )
    agreement = agree / len(shared)
    wr_primary = float(sum(scored[primary]["by_uid"][u] for u in shared) / len(shared))
    wr_reference = float(sum(scored[reference]["by_uid"][u] for u in shared) / len(shared))
    gap_pp = abs(wr_primary - wr_reference) * 100

    thresholds = val.thresholds
    passes = agreement >= float(thresholds.verdict_agreement_min) and gap_pp <= float(
        thresholds.win_rate_gap_max_pp
    )

    return {
        "primary_judge": primary,
        "reference_judge": reference,
        "comparison_arm": arm,
        "slice": str(val.slice),
        "n_compared": len(shared),
        "verdict_agreement": agreement,
        "agreement_threshold": float(thresholds.verdict_agreement_min),
        "win_rate_primary": wr_primary,
        "win_rate_reference": wr_reference,
        "win_rate_gap_pp": gap_pp,
        "gap_threshold_pp": float(thresholds.win_rate_gap_max_pp),
        "swap_consistency": {
            judge: scored[judge]["split"].get("swap_consistency") for judge in scored
        },
        "passes": passes,
        # If this fails, the fallback is not a saving — a cheap judge that disagrees with
        # a good one costs a full rerun once discovered.
        "fallback_judge": str(val.fallback_judge),
        "cost": {
            judge: _cost(records, judge) for judge, records in results.items()
        },
    }


def main(
    config: str = "configs/capability_eval.yaml",
    mode: str = "judge",
    arm: str = "",
    stage: int = 0,
    judge_model: str = "",
) -> None:
    """Judge an arm against the baseline, or validate the judge.

    Args:
        config: Path to the capability-eval config.
        mode: `judge` or `validate`.
        arm: Arm to judge (required in `judge` mode).
        stage: hard_prompt questions to judge this stage; 0 uses the arm's full count.
        judge_model: Override the pinned judge (e.g. to run the fallback).
    """
    cfg = OmegaConf.load(config)
    vendor = Path(cfg.vendor_dir)
    if not (vendor / "gen_judgment.py").exists():
        raise SystemExit(
            f"No vendored harness at {vendor}. Clone it, then run "
            f"`uv run python scripts/patch_arena_hard.py`."
        )
    # A missing patch means the judge would silently compare against upstream's packaged
    # baseline instead of arm A, producing a number that looks fine and means nothing.
    if "_ARENA_HARD_BASELINE" not in (vendor / "utils" / "judge_utils.py").read_text():
        raise SystemExit(
            "Vendored harness is unpatched. Run: uv run python scripts/patch_arena_hard.py"
        )

    out_dir = Path(cfg.output_dir) / "judging" / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "validate":
        result = validate_judge(cfg)
        name = "judge_validation"
        print("\n=== Judge validation (spec §4) ===")
        print(f"  agreement       {result['verdict_agreement']:.1%} "
              f"(need >= {result['agreement_threshold']:.0%})")
        print(f"  win rate        {result['primary_judge']}: {result['win_rate_primary']:.1%}")
        print(f"                  {result['reference_judge']}: {result['win_rate_reference']:.1%}")
        print(f"  gap             {result['win_rate_gap_pp']:.1f}pp "
              f"(need <= {result['gap_threshold_pp']:.0f}pp)")
        for judge, consistency in result["swap_consistency"].items():
            print(f"  swap consist.   {judge}: {consistency:.1%}")
        print(f"  VERDICT         {'PASS' if result['passes'] else 'FAIL'}")
        if not result["passes"]:
            print(f"  -> fall back to {result['fallback_judge']} for the full sweep and "
                  f"re-run every comparison; a cheap judge that disagrees is not a saving.")
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
        if cost["output_tokens_per_question"] > 5000:
            print(
                f"  WARNING: output tokens/question is {cost['output_tokens_per_question']:.0f}, "
                f"far above the ~3,100 observed for this judge at `effort: low`. Check that "
                f"`judge.extra_body` is reaching the provider and that the judge ID is pinned."
            )
    else:
        raise SystemExit(f"Unknown mode {mode!r}; expected 'judge' or 'validate'")

    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={"mode": mode})
    print(f"\n>>> {out_dir / f'{name}.json'}")


if __name__ == "__main__":
    fire.Fire(main)
