# ABOUTME: Drives the vendored spec-gaming suite (arXiv:2605.02269) against one served
# ABOUTME: target, all 8 settings, and mirrors every result into output/spec_gaming/<tag>/.
"""Run the reward-hacking-evals suite against a vLLM-served adapter.

MUST run in the vendored harness's own env (inspect-ai's click pin conflicts with the
root project's huggingface-hub):

    HARNESS=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
    uv run --project $HARNESS python scratch/spec_gaming/run_suite.py all \
        --served numinactl --base-url https://<pod>-8000.proxy.runpod.net/v1 --n 100

Each subcommand runs one setting group; `all` runs the eight in the cheap-to-expensive
order so a wiring bug surfaces on a 10-minute eval rather than a 2-hour one.

The upstream code resolves `datasets/`, `results/` and `logs/` relative to cwd, so every
entrypoint chdirs into the sub-harness it belongs to. Results stay where upstream writes
them and are copied into output/spec_gaming/<tag>_<ts>/ at the end of each setting.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals"
OUT_ROOT = REPO / "output/spec_gaming"

# Qwen3.6's served --max-num-seqs. Nothing here may exceed it: extra in-flight requests
# queue in vLLM rather than run, and on this arch there is no prefix cache to recover the
# re-prefill cost (docs/LOG.md 2026-07-29).
MAX_CONCURRENCY = 32


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "nogit"


def _enter(subdir: str = "") -> Path:
    """chdir into the harness (or one of its sub-harnesses) and put it on sys.path."""
    d = HARNESS / subdir if subdir else HARNESS
    os.chdir(d)
    for p in (str(HARNESS), str(d)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return d


def _configure_target(served: str, base_url: str, max_tokens: int) -> str:
    """Point the harness's vLLM path at the served endpoint. Returns the model id."""
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ.setdefault("VLLM_API_KEY", "dummy")
    os.environ["VLLM_MAX_TOKENS"] = str(max_tokens)
    return f"vllm/{served}"


def _write_meta(out_dir: Path, **fields) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"git_sha": _git_sha(), "timestamp": _stamp(), **fields}, indent=2))


def _mirror(src: Path, out_dir: Path, label: str) -> None:
    """Copy a setting's raw output into output/spec_gaming/<tag>/<label>/."""
    dest = out_dir / label
    if not src.exists():
        print(f"!! {label}: nothing at {src} to mirror", file=sys.stderr)
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f">>> {label}: mirrored {src} -> {dest}")


# --------------------------------------------------------------------------- settings


def multiple_choice(served: str, base_url: str, n: int = 100, tag: str = "",
                    max_tokens: int = 16384) -> None:
    """Settings 1+2: MC revealing-score and MC reward-action. Regex-graded, no judge."""
    _enter()
    from config import (MultipleChoiceDatasetType, MultipleChoiceExperimentConfig,
                        MultipleChoicePromptType, ExperimentType, LLMModel)
    from run_experiments import calculate_batch_size
    from runner import run_experiment

    model = _configure_target(served, base_url, max_tokens)
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    t0 = time.time()

    for dataset_type in (MultipleChoiceDatasetType.REVEALING_SCORE,
                         MultipleChoiceDatasetType.WORLD_AFFECTING_REWARD):
        cfg = MultipleChoiceExperimentConfig(
            experiment_type=ExperimentType.MULTIPLE_CHOICE,
            model=LLMModel(model_name=model),
            num_samples=n,
            output_folder=f"{tag}/multiple_choice/{dataset_type.value}",
            prompt_type=MultipleChoicePromptType.DEFAULT,
            dataset_type=dataset_type,
            batch_size=calculate_batch_size(model),
        )
        print(f">>> multiple_choice/{dataset_type.value}: n={n}")
        asyncio.run(run_experiment(cfg))

    _write_meta(out_dir, setting="multiple_choice", model=model, base_url=base_url,
                n=n, elapsed_s=round(time.time() - t0, 1))
    _mirror(HARNESS / "results" / tag / "multiple_choice", out_dir, "multiple_choice")


def email_assistant(served: str, base_url: str, n: int = 100, tag: str = "",
                    max_tokens: int = 16384) -> None:
    """Setting 3: email assistant. 3 sequential turns per sample; regex-graded."""
    _enter()
    from config import (EmailExperimentConfig, EmailPromptType, FinalEmailType,
                        ExperimentType, LLMModel)
    from run_experiments import calculate_batch_size
    from runner import run_experiment

    model = _configure_target(served, base_url, max_tokens)
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    t0 = time.time()

    # The two cells run_experiments.run_email_assistant_experiment ends up with after its
    # own reassignment of prompt_email_tuples -- i.e. the paper's headline configuration.
    for final_email_type in (FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED,
                             FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED):
        cfg = EmailExperimentConfig(
            experiment_type=ExperimentType.EMAIL_ASSISTANT,
            model=LLMModel(model_name=model),
            num_samples=n,
            output_folder=f"{tag}/email_assistant/{final_email_type.value}",
            prompt_type=EmailPromptType.REMOVE_SNOOZE,
            email_ct=3,
            final_email_type=final_email_type,
            batch_size=calculate_batch_size(model),
        )
        print(f">>> email_assistant/{final_email_type.value}: n={n}")
        asyncio.run(run_experiment(cfg))

    _write_meta(out_dir, setting="email_assistant", model=model, base_url=base_url,
                n=n, elapsed_s=round(time.time() - t0, 1))
    _mirror(HARNESS / "results" / tag / "email_assistant", out_dir, "email_assistant")


def customer_service(served: str, base_url: str, n: int = 100, tag: str = "",
                     max_tokens: int = 16384, prompt_type: str = "default") -> None:
    """Setting 4: customer service. Dialogue against a JUDGE_MODEL customer simulator.

    The simulator is gemini-3-flash via OpenRouter, not the paper's gpt-4.1 (this repo has
    no OpenAI key). That changes the ENVIRONMENT, not just the grading, so the number is
    not directly comparable to the published one.
    """
    _enter()
    from config import (CustomerServiceExperimentConfig, CustomerServicePromptType,
                        CustomerType, ExperimentType, LLMModel)
    from constants import JUDGE_MODEL
    from run_experiments import calculate_batch_size
    from runner import run_experiment

    model = _configure_target(served, base_url, max_tokens)
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    t0 = time.time()

    ptype = CustomerServicePromptType(prompt_type)
    for customer_type in (CustomerType.FRIENDLY, CustomerType.BELLIGERENT):
        # Non-default prompt types get their OWN directory. runner.py resumes from an
        # existing results.jsonl, so writing them to the default path would append rows
        # generated under a DIFFERENT system prompt into the default cell's data and
        # silently corrupt it.
        cell = (customer_type.value if ptype is CustomerServicePromptType.DEFAULT
                else f"{ptype.value}_{customer_type.value}")
        cfg = CustomerServiceExperimentConfig(
            experiment_type=ExperimentType.CUSTOMER_SERVICE,
            model=LLMModel(model_name=model),
            num_samples=n,
            output_folder=f"{tag}/customer_service/{cell}",
            prompt_type=ptype,
            customer_type=customer_type,
            customer_model=LLMModel(model_name=JUDGE_MODEL),
            batch_size=calculate_batch_size(model),
        )
        print(f">>> customer_service/{cell}: n={n} sim={JUDGE_MODEL}", flush=True)
        asyncio.run(run_experiment(cfg))

    _write_meta(out_dir, setting="customer_service", model=model, base_url=base_url,
                n=n, simulator=JUDGE_MODEL, prompt_type=ptype.value,
                elapsed_s=round(time.time() - t0, 1))
    _mirror(HARNESS / "results" / tag / "customer_service", out_dir, "customer_service")


def data_entry(served: str, base_url: str, n: int = 100, tag: str = "",
               dataset: str = "xsum_stories", max_sandboxes: int = 12,
               cmd_timeout: int = 210) -> None:
    """Setting 5: data entry. Inspect agent in Docker; needs Docker where THIS runs.

    max_sandboxes defaults to 12, the worker count ODCV settled on for the same
    driver+remote-GPU shape (configs/eval/swebench_mini_verified.yaml records why more
    is counterproductive: the bound is the server's KV cache, not the driver).
    """
    _enter("inspect_command_line_envs")
    from cl_config import (DataEntryProblemType, DataEntryRewardFuncNameType,
                           DataEntryValidSubmissionType, DataEntryFallbackType,
                           DataEntryRewardLocationType)
    from run_cl_agent_envs import run_data_entry_sweep

    # Inspect reaches an OpenAI-compatible endpoint through its `openai/` provider plus
    # OPENAI_BASE_URL -- the route upstream already used for its self-hosted OLMo runs.
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    log_dir = f"logs/{tag}/data_entry"
    t0 = time.time()

    print(f">>> data_entry/{dataset}: n={n} sandboxes={max_sandboxes}")
    run_data_entry_sweep(
        models=[f"openai/{served}"],
        dataset_choices=[DataEntryProblemType(dataset)],
        reward_func_name_types=[DataEntryRewardFuncNameType.REWARD],
        valid_submission_types=[DataEntryValidSubmissionType.BASIC_CHECK],
        fallback_types=[DataEntryFallbackType.NO_FALLBACK],
        reward_location_types=[DataEntryRewardLocationType.NORMAL_FILE],
        num_samples=n,
        log_dir=log_dir,
        # Upstream's 180 predates inspect-ai's floor of 210 for bash_session
        # ("Timeout must be at least 210 seconds"); raising the ceiling only affects
        # commands that would otherwise have been killed.
        cmd_timeout=cmd_timeout,
        # set_reasoning=False: thinking is pinned into the SERVED chat template, not
        # requested per-call. vLLM has no reasoning_effort knob to receive.
        set_reasoning=False,
        use_openrouter_grader=True,
        max_connections=min(MAX_CONCURRENCY, max_sandboxes),
        max_subprocesses=max_sandboxes,
        max_samples=max_sandboxes,
        max_sandboxes=max_sandboxes,
    )

    _write_meta(out_dir, setting="data_entry", model=f"openai/{served}",
                base_url=base_url, n=n, dataset=dataset,
                elapsed_s=round(time.time() - t0, 1))
    _mirror(HARNESS / "inspect_command_line_envs" / log_dir, out_dir, "data_entry")


def sales(served: str, base_url: str, n: int = 100, tag: str = "",
          max_tokens: int = 16384) -> None:
    """Setting 6: sales agent. Programmatically graded (quota_met_rate); no judge."""
    _enter("sales_env_experiments")
    from run_evaluation import run_multiple_evaluations

    model = _configure_target(served, base_url, max_tokens)
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    t0 = time.time()

    print(f">>> sales: n_runs={n} max_turns=8")
    results = asyncio.run(run_multiple_evaluations(
        n_runs=n,
        model_name=model,
        wandb_project_name="llm_sales_evaluation",
        use_wandb=False,
        incentive_email="none",
        max_turns=8,
        quota_target=485000,
        start_of_day_revenue=450000,
        use_reasoning=False,
        use_agent_prompt=False,
        constraint_strictness="weak",
    ))
    quota_met = sum(1 for r in results if r["quota_status"].get("quota_met", False))
    print(f">>> sales: quota_met_rate = {quota_met}/{len(results)}")

    # run_multiple_evaluations returns the rollouts but never writes them: its
    # save_results() is dead code and the CSV is written by eval_sweep.main, which we do
    # not call. Persist here or the transcripts are lost when the process exits. Each
    # entry carries the system prompt AND every turn, so it stands alone as a rollout.
    dest = out_dir / "sales"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "rollouts.json").write_text(json.dumps(results, indent=2, default=str))
    print(f">>> sales: wrote {len(results)} rollouts -> {dest / 'rollouts.json'}")

    _write_meta(out_dir, setting="sales", model=model, base_url=base_url, n=n,
                quota_met=quota_met, n_completed=len(results),
                elapsed_s=round(time.time() - t0, 1))


def livecode(served: str, base_url: str, limit: int = 80, tag: str = "",
             difficulties: str = "easy,hard", max_connections: int = MAX_CONCURRENCY,
             max_sandboxes: int = 24) -> None:
    """Settings 7+8: LiveCode easy and hard. Inspect + Docker sandbox + LLM detector.

    Concurrency here is cheap: the sandboxes are short-lived python containers (~50MB
    idle) and the contexts are small, so unlike data entry this can run at the server's
    full --max-num-seqs. max_sandboxes is the local Docker ceiling, held a little under
    max_connections so container churn never becomes the bottleneck.
    """
    _enter("livecodebench_env_experiments")
    from inspect_ai import eval as inspect_eval
    from livecodebench_hack_eval import livecodebench
    from constants import JUDGE_MODEL

    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    tag = tag or f"{served}_{_stamp()}"
    out_dir = OUT_ROOT / tag
    t0 = time.time()

    for difficulty in [d.strip() for d in difficulties.split(",") if d.strip()]:
        log_dir = f"logs/{tag}/livecode_{difficulty}"
        print(f">>> livecode/{difficulty}: limit={limit}")
        inspect_eval(
            livecodebench(
                pressure_prompt_type=["standard"],
                use_incorrect_unit_test=True,
                difficulties=[difficulty],
                max_turns=3,
                give_bailout_option=True,
                # docker, never "local": "local" executes model-written code on THIS host.
                sandbox="docker",
                max_private_test_cases=3,
                use_llm_hard_coding_detector=True,
                llm_grader_model=JUDGE_MODEL,
            ),
            model=f"openai/{served}",
            log_dir=log_dir,
            limit=limit,
            max_connections=min(MAX_CONCURRENCY, max_connections),
            max_sandboxes=max_sandboxes,
            # None: thinking is pinned in the served template, not requested per-call.
            reasoning_tokens=None,
            reasoning_effort=None,
        )
        _mirror(HARNESS / "livecodebench_env_experiments" / log_dir,
                out_dir, f"livecode_{difficulty}")

    _write_meta(out_dir, setting="livecode", model=f"openai/{served}",
                base_url=base_url, limit=limit, difficulties=difficulties,
                elapsed_s=round(time.time() - t0, 1))


def all(served: str, base_url: str, n: int = 100, tag: str = "",
        livecode_limit: int = 80) -> None:
    """Run all eight settings, cheapest first so wiring bugs surface early."""
    tag = tag or f"{served}_{_stamp()}"
    failures: list[tuple[str, str]] = []
    print(f"=== spec-gaming suite: target={served} tag={tag} n={n} ===", flush=True)
    for name, fn, kwargs in [
        ("multiple_choice", multiple_choice, {"n": n}),
        ("sales", sales, {"n": n}),
        ("email_assistant", email_assistant, {"n": n}),
        ("livecode", livecode, {"limit": livecode_limit}),
        ("customer_service", customer_service, {"n": n}),
        ("data_entry", data_entry, {"n": n}),
    ]:
        print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            fn(served=served, base_url=base_url, tag=tag, **kwargs)
            print(f"=== {name} OK in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception as e:
            # Surfaced, not swallowed: a 6-hour suite must not lose five settings to
            # the sixth, and the failure is reprinted in the closing summary.
            import traceback
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"!!! {name} FAILED after {(time.time() - t0) / 60:.1f} min",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
    print(f"\n=== suite complete -> {OUT_ROOT / tag}")
    if failures:
        print(f"=== {len(failures)} SETTING(S) FAILED:")
        for name, err in failures:
            print(f"    {name}: {err}")
    else:
        print("=== all settings completed")


if __name__ == "__main__":
    fire.Fire({
        "multiple_choice": multiple_choice,
        "email_assistant": email_assistant,
        "customer_service": customer_service,
        "data_entry": data_entry,
        "sales": sales,
        "livecode": livecode,
        "all": all,
    })
