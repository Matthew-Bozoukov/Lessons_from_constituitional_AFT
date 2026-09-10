import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent))  # PATCH: repo-root constants.py
from constants import JUDGE_MODEL
from cl_config import DataEntryProblemType, DataEntryValidSubmissionType, DataEntryRewardFuncNameType, DataEntryFallbackType, DataEntryRewardLocationType, DataEntryConfig, CommandLineEnvConfig
from typing import Optional
from datetime import datetime
from inspect_ai import eval_set
from cl_agent_envs import command_line_agent_env

def run_data_entry_sweep(
    models: list[str],
    dataset_choices: list[DataEntryProblemType],
    reward_func_name_types: list[DataEntryRewardFuncNameType],
    valid_submission_types: list[DataEntryValidSubmissionType],
    fallback_types: list[DataEntryFallbackType],
    reward_location_types: list[DataEntryRewardLocationType],
    n: int = 3,
    num_samples: int = 300,
    log_dir: str | None = None,
    attempts: int = 1,
    cmd_timeout: int = 180,
    message_limit: int = 60,
    time_limit: int = 1800,
    compose_file: Optional[str] = 'compose.yaml',
    grader_model: str = JUDGE_MODEL,
    use_openrouter_grader: bool = False,  # Use OpenRouter for grader (when OPENAI_BASE_URL is overridden)
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
    set_reasoning: bool = True,
    reasoning_effort: str = "low",
    reasoning_tokens: int = 2480,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Use OpenRouter for grader when OPENAI_BASE_URL is overridden (e.g., for vLLM)
    if use_openrouter_grader:
        grader_model = JUDGE_MODEL

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []
    for ds in dataset_choices:
        for rfn in reward_func_name_types:
            for vs in valid_submission_types:
                for ft in fallback_types:
                    for rlt in reward_location_types:
                        de_config = DataEntryConfig(
                            dataset_choice=ds,
                            reward_func_name_type=rfn,
                            valid_submission_type=vs,
                            fallback_type=ft,
                            reward_location_type=rlt,
                            n=n,
                        )
                        command_line_env_config = CommandLineEnvConfig(
                            env_specific_config=de_config,
                            num_samples=num_samples,
                            attempts=attempts,
                            cmd_timeout=cmd_timeout,
                            message_limit=message_limit,
                            time_limit=time_limit,
                            compose_file=compose_file,
                        )
                        env_str = f"ds={ds.value},rfn={rfn.value},vs={vs.value},ft={ft.value},rlt={rlt.value}"
                        t = command_line_agent_env(command_line_env_config, env_str=env_str)
                        tasks.append(t)

    # Run the eval set across all models
    if set_reasoning:
        success, logs = eval_set(
            tasks=tasks,
            model=models,
            model_roles={"grader": grader_model},
            log_dir=log_dir,
            max_tasks=max_tasks,
            max_connections=max_connections,
            max_subprocesses=max_subprocesses,
            max_samples=max_samples,
            max_sandboxes=max_sandboxes,
            reasoning_summary="auto",
            reasoning_effort=reasoning_effort,
            reasoning_tokens=reasoning_tokens,
            reasoning_history="auto",
        )
    else:
        success, logs = eval_set(
            tasks=tasks,
            model=models,                     # solver models to sweep
            model_roles={"grader": grader_model},  # fixed LLM-judge for your scorer
            log_dir=log_dir,                  # REQUIRED for eval sets
            # Optional throughput tuning:
            max_tasks=max_tasks,              # defaults to max(4, len(models)) if None
            max_connections=max_connections,  # per-provider API concurrency
            max_subprocesses=max_subprocesses, # sandbox subprocess concurrency
            max_samples=max_samples, # max number of samples to run
            max_sandboxes=max_sandboxes, # max number of sandboxes to run
        )
    return success, logs

def de_sweep_20251025(num_samples: int = 30, log_dir: str = "logs/data-entry-redux-sweep-20251025"):
    models = [
    "openai/o3",
    # "openai/o3-mini",
    "openai/gpt-5",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-5-20250929",
    "openrouter/google/gemini-2.5-pro",
    "grok/grok-4",
    ]

    datasets = [
        DataEntryProblemType.MATH,
        DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.XSUM_STORIES,
        DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
    )

def de_sweep_20251104_nongrok_only(num_samples: int = 300, log_dir: str = "logs/data-entry-redux-sweep-20251104-different-locations-nongrok"):
    models = [
    "openai/o3",
    # "openai/o3-mini",
    "openai/gpt-5",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-5-20250929",
    "openrouter/google/gemini-2.5-pro",
    # "grok/grok-4",
    ]

    datasets = [
        # DataEntryProblemType.MATH,
        # DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.XSUM_STORIES,
        DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    non_grok_models = [model for model in models if model != "grok/grok-4"]
    run_data_entry_sweep(
        models=non_grok_models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
        set_reasoning=True,
    )


def de_sweep_20251104_grok_only(num_samples: int = 300, log_dir: str = "logs/data-entry-redux-sweep-20251104-different-locations-grok-only"):

    datasets = [
        # DataEntryProblemType.MATH,
        # DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.XSUM_STORIES,
        DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    grok_models = ["grok/grok-4"]
    run_data_entry_sweep(
        models=grok_models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
        set_reasoning=False,
    )

def de_sweep_20251104_full(num_samples: int = 300, log_dir: str = "logs/TEST"):
    models = [
    # "openai/o3",
    # "openai/o3-mini",
    # "openai/gpt-5",
    # "anthropic/claude-3-7-sonnet-20250219",
    # "anthropic/claude-opus-4-20250514",
    # "anthropic/claude-sonnet-4-5-20250929",
    # "openrouter/google/gemini-2.5-pro",
    "grok/grok-4",
    ]

    models = MODEL_RL_LIST

    datasets = [
        # DataEntryProblemType.MATH,
        # DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.XSUM_STORIES,
        # DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
        set_reasoning=True,
    )

def de_sweep_20251120_rl_models(num_samples: int = 300, log_dir: str = "logs/TEST"):

    MODEL_RL_LIST = [
    'openrouter/deepseek/deepseek-chat',
    'openrouter/deepseek/deepseek-r1',
    'openrouter/mistralai/magistral-medium-2506:thinking',
    'openrouter/mistralai/mistral-medium-3',
]
    models = MODEL_RL_LIST

    datasets = [
        # DataEntryProblemType.MATH,
        # DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.XSUM_STORIES,
        # DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
    )

def de_sweep_20260116_gemini_3_pro_preview(num_samples: int = 10, log_dir: str = "logs/data-entry-sweep-20260116-gemini-3-pro-preview"):
    models = [
        'openrouter/google/gemini-3-pro-preview',
    ]

    datasets_full = [
        DataEntryProblemType.XSUM_STORIES,
        DataEntryProblemType.MATH,
        DataEntryProblemType.APPS_COMPETITION,
        DataEntryProblemType.MMLU,
    ]

    datasets_reduced = [
        DataEntryProblemType.XSUM_STORIES,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets_reduced,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir + '_high',
        set_reasoning=True,
        reasoning_effort="high",
        reasoning_tokens=16000,
        max_subprocesses=10,
        max_connections=10,
        max_sandboxes=10,
    )

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets_full,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir + '_low',
        set_reasoning=True,
        max_subprocesses=10,
        max_connections=10,
        max_sandboxes=10,
    )

def de_sweep_20260116_gemini_3_pro_preview_code_only(num_samples: int = 10, log_dir: str = "logs/data-entry-sweep-20260116-gemini-3-pro-preview-code-only"):
    models = [
        'openrouter/google/gemini-3-pro-preview',
    ]

    datasets = [
        DataEntryProblemType.APPS_COMPETITION,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir + '_low',
        set_reasoning=True,
        max_subprocesses=10,
        max_connections=10,
        max_sandboxes=10,
    )

def de_sweep_20251218_rl_models(num_samples: int = 20, log_dir: str = "logs/data-entry-sweep-20251218-rl-models-test", model_list: list[str] = None):
    if model_list is None:
        models = [
            # 'openrouter/deepseek/deepseek-chat',
            # 'openrouter/deepseek/deepseek-r1',
            # 'openrouter/mistralai/mistral-medium-3',
            # 'openrouter/qwen/qwq-32b',
            # 'openrouter/allenai/olmo-3-32b-think:free',
            # 'openrouter/allenai/olmo-3.1-32b-think:free',
            # 'mistral/magistral-medium-2509',
            # 'together/kei.nishimuragasparian@gmail.com/Qwen/Qwen2.5-32B-2a5ff838',
            # 'together/kei.nishimuragasparian@gmail.com/Qwen/QwQ-32B-91390a4b'
            'openai/Olmo-3-32B-Think-SFT',
        ]
    else:
        models = model_list

    datasets = [
        DataEntryProblemType.XSUM_STORIES,
        # DataEntryProblemType.MATH,
        # DataEntryProblemType.APPS_COMPETITION,
        # DataEntryProblemType.MMLU,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        DataEntryFallbackType.NO_FALLBACK,
        # DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir,
        set_reasoning=False,
        max_subprocesses=5,
        max_connections=5,
        max_sandboxes=5,
        use_openrouter_grader=True,
    )


def de_sweep_20260126_gemini_3_pro_fallback(num_samples: int = 300, log_dir: str = "logs/data-entry-sweep-20260126-gemini-3-pro-fallback"):
    models = [
            "openai/o3",
    "openai/gpt-5",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/google/gemini-3-pro-preview",
    "openrouter/x-ai/grok-4",
    ]

    datasets = [
        DataEntryProblemType.XSUM_STORIES,
    ]

    reward_func_name_types = [
        DataEntryRewardFuncNameType.REWARD,
        # DataEntryRewardFuncNameType.IS_TASK_COMPLETED,
        # DataEntryRewardFuncNameType.SCORE,
    ]

    valid_submission_types = [
        DataEntryValidSubmissionType.BASIC_CHECK,
        # DataEntryValidSubmissionType.CHECK_NONEMPTY,
    ]

    fallback_types = [
        # DataEntryFallbackType.NO_FALLBACK,
        DataEntryFallbackType.FALLBACK,
    ]

    reward_location_types = [
        DataEntryRewardLocationType.NORMAL_FILE,
        # DataEntryRewardLocationType.PROMPT_AND_FILE,
        # DataEntryRewardLocationType.PROMPT_ONLY,
        # DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY,
    ]

    run_data_entry_sweep(
        models=models,
        dataset_choices=datasets,
        reward_func_name_types=reward_func_name_types,
        valid_submission_types=valid_submission_types,
        fallback_types=fallback_types,
        reward_location_types=reward_location_types,
        num_samples=num_samples,
        log_dir=log_dir + '_low',
        set_reasoning=True,
        max_subprocesses=10,
        max_connections=10,
        max_sandboxes=10,
    )

if __name__ == "__main__":
    # de_sweep_20251025(num_samples=3, log_dir="logs/data-entry-redux-sweep-20250115-different-datasets")
    # de_sweep_20251104_nongrok_only(num_samples=300)
    # de_sweep_20251104_grok_only(num_samples=300)
    # de_sweep_20260116_gemini_3_pro_preview(num_samples=300, log_dir="logs/data-entry-sweep-20260116-gemini-3-pro-preview_reduced")
    # de_sweep_20260116_gemini_3_pro_preview_code_only(num_samples=300, log_dir="logs/data-entry-sweep-20260116-gemini-3-pro-preview-code-only2")
    # de_sweep_20251218_rl_models(num_samples=300, log_dir="logs/data-entry-sweep-20251223-rl-models-deepseek-mistral")
    # de_sweep_20251218_rl_models(num_samples=1, log_dir="logs/data-entry-sweep-20251223-rl-models-deepseek-mistral")
    # de_sweep_20251218_rl_models(num_samples=300, log_dir="logs/data-entry-sweep-20251223-rl-models-qwen")
    # de_sweep_20251218_rl_models(num_samples=20, log_dir="logs/data-entry-sweep-20251223-rl-models-olmo-test")
    # de_sweep_20251218_rl_models(num_samples=300, log_dir="logs/data-entry-sweep-20251227-rl-models-olmo-SFTv2", model_list=['openai/Olmo-3-32B-Think-SFT'])
    # de_sweep_20251218_rl_models(num_samples=300, log_dir="logs/data-entry-sweep-20251227-rl-models-olmo-3v2", model_list=['openai/Olmo-3-32B-Think'])
    de_sweep_20260126_gemini_3_pro_fallback(num_samples=300, log_dir="logs/data-entry-sweep-20260126-gemini-3-pro-fallback")