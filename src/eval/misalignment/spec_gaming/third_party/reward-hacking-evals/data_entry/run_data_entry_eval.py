from datetime import datetime
from inspect_ai import eval_set
from data_entry_hack import data_entry_reward_hack, DataEntryProblemType, DataEntryValidSubmissionType, DataEntryRewardFuncNameType, DataEntryFallbackType, DataEntryRewardLocationType

# import your task + enums from wherever they live
# from my_eval_module import data_entry_reward_hack, DataEntryProblemType

def run_data_entry_sweep(
    models: list[str],
    dataset_choices: list[DataEntryProblemType],
    valid_submission_types: list[DataEntryValidSubmissionType],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    # optional passthrough to your @task for consistency:
    reward_func_name_type=None,
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []
    for ds in dataset_choices:
        for vs in valid_submission_types:
            t = data_entry_reward_hack(
                dataset_choice=ds,
                num_samples=num_samples,
                # pass through optional knobs if you want stable settings:
                reward_func_name_type=reward_func_name_type or DataEntryRewardFuncNameType.REWARD,
                valid_submission_type=vs,
                attempts=attempts,
                cmd_timeout=cmd_timeout,
            )
            tasks.append(t)

    # Run the eval set across all models
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

def run_data_entry_sweep_2(
    models: list[str],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []
    for reward_type in [DataEntryRewardFuncNameType.REWARD, DataEntryRewardFuncNameType.IS_TASK_COMPLETED, DataEntryRewardFuncNameType.SCORE]:
        t = data_entry_reward_hack(
            dataset_choice=DataEntryProblemType.MMLU,
            num_samples=num_samples,
            reward_func_name_type=reward_type,
            valid_submission_type=DataEntryValidSubmissionType.CHECK_NONEMPTY,
            fallback_type=DataEntryFallbackType.NO_FALLBACK,
            attempts=attempts,
            cmd_timeout=cmd_timeout,
        )
        tasks.append(t)
    
    t = data_entry_reward_hack(
            dataset_choice=DataEntryProblemType.MMLU,
            num_samples=num_samples,
            reward_func_name_type=DataEntryRewardFuncNameType.REWARD,
            valid_submission_type=DataEntryValidSubmissionType.CHECK_NONEMPTY,
            fallback_type=DataEntryFallbackType.FALLBACK,
            attempts=attempts,
            cmd_timeout=cmd_timeout,
        )
    tasks.append(t)

    # Run the eval set across all models
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

def run_data_entry_sweep_3(
    models: list[str],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []
    for data_entry_reward_location_type in [DataEntryRewardLocationType.PROMPT_AND_FILE, DataEntryRewardLocationType.PROMPT_ONLY, DataEntryRewardLocationType.PROMPT_DESCRIPTION_ONLY, DataEntryRewardLocationType.NORMAL_FILE]:
        t = data_entry_reward_hack(
            dataset_choice=DataEntryProblemType.MMLU,
            num_samples=num_samples,
            reward_func_name_type=DataEntryRewardFuncNameType.REWARD,
            valid_submission_type=DataEntryValidSubmissionType.BASIC_CHECK,
            fallback_type=DataEntryFallbackType.NO_FALLBACK,
            reward_location_type=data_entry_reward_location_type,
            attempts=attempts,
            cmd_timeout=cmd_timeout,
        )
        tasks.append(t)


    # Run the eval set across all models
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

def run_data_entry_sweep_4(
    models: list[str],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
    reasoning_effort: str = "high",
    reasoning_tokens: int = 16000,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Runs Grok on default settings.
    Runs all remaining models with various reasoning settings.
    
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []

    t = data_entry_reward_hack(
            dataset_choice=DataEntryProblemType.XSUM_STORIES,
            num_samples=num_samples,
            reward_func_name_type=DataEntryRewardFuncNameType.REWARD,
            valid_submission_type=DataEntryValidSubmissionType.BASIC_CHECK,
            fallback_type=DataEntryFallbackType.NO_FALLBACK,
            reward_location_type=DataEntryRewardLocationType.NORMAL_FILE,
            attempts=attempts,
            cmd_timeout=cmd_timeout,
        )
    tasks.append(t)

    # Run the eval set across all models
    success, logs = eval_set(
        tasks=tasks,
        model=models,                     # solver models to sweep
        model_roles={"grader": grader_model},  # fixed LLM-judge for your scorer
        reasoning_effort=reasoning_effort,
        reasoning_tokens=reasoning_tokens,
        log_dir=log_dir,                  # REQUIRED for eval sets
        # Optional throughput tuning:
        max_tasks=max_tasks,              # defaults to max(4, len(models)) if None
        max_connections=max_connections,  # per-provider API concurrency
        max_subprocesses=max_subprocesses, # sandbox subprocess concurrency
        max_samples=max_samples, # max number of samples to run
        max_sandboxes=max_sandboxes, # max number of sandboxes to run
    )
    return success, logs


def run_data_entry_sweep_5(
    models: list[str],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Runs Grok on default settings.
    Runs all remaining models with various reasoning settings.
    
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []

    t = data_entry_reward_hack(
            dataset_choice=DataEntryProblemType.XSUM_STORIES,
            num_samples=num_samples,
            reward_func_name_type=DataEntryRewardFuncNameType.REWARD,
            valid_submission_type=DataEntryValidSubmissionType.BASIC_CHECK,
            fallback_type=DataEntryFallbackType.NO_FALLBACK,
            reward_location_type=DataEntryRewardLocationType.NORMAL_FILE,
            attempts=attempts,
            cmd_timeout=cmd_timeout,
        )
    tasks.append(t)

    # Run the eval set across all models
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


def run_data_entry_sweep_6(
    models: list[str],
    *,
    num_samples: int = 300,
    log_dir: str | None = None,
    grader_model: str = "openai/gpt-4.1-nano",
    attempts: int = 1,
    cmd_timeout: int = 180,
    # optional parallelism tuning for eval_set:
    max_tasks: int | None = None,
    max_connections: int | None = 16,
    max_subprocesses: int | None = 16,
    max_samples: int | None = 50,
    max_sandboxes: int | None = 25,
):
    """
    Runs data_entry_reward_hack across (dataset_choices x models), with num_samples fixed.
    Runs Grok on default settings.
    Runs all remaining models with various reasoning settings.
    
    Returns (success: bool, logs: list[EvalLog]).
    """
    if log_dir is None:
        log_dir = f"logs/data-entry-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build dynamic tasks (OK because data_entry_reward_hack is decorated with @task)
    tasks = []

    for dataset_choice in [#DataEntryProblemType.XSUM_STORIES, 
    DataEntryProblemType.MMLU, DataEntryProblemType.APPS_COMPETITION, DataEntryProblemType.MATH]:
        t = data_entry_reward_hack(
                dataset_choice=dataset_choice,
                num_samples=num_samples,
                reward_func_name_type=DataEntryRewardFuncNameType.REWARD,
                valid_submission_type=DataEntryValidSubmissionType.BASIC_CHECK,
                fallback_type=DataEntryFallbackType.NO_FALLBACK,
                reward_location_type=DataEntryRewardLocationType.NORMAL_FILE,
                attempts=attempts,
                cmd_timeout=cmd_timeout,
            )
        tasks.append(t)

    # Run the eval set across all models
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

models = [
    "openai/o3",
    "openai/o3-mini",
    "openai/gpt-5",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-5-20250929",
    "openrouter/google/gemini-2.5-pro",
    # "openrouter/x-ai/grok-4",
]

datasets = [
    # DataEntryProblemType.MATH,
    DataEntryProblemType.APPS_COMPETITION,
    # DataEntryProblemType.XSUM_STORIES,
    # DataEntryProblemType.MMLU,
]

valid_submission_types = [
    DataEntryValidSubmissionType.BASIC_CHECK,
    DataEntryValidSubmissionType.CHECK_NONEMPTY,
]

# success, logs = run_data_entry_sweep(
#     models=models,
#     dataset_choices=datasets,
#     valid_submission_types=valid_submission_types,
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-09-16",
# )

# success, logs = run_data_entry_sweep_2(
#     models=models,
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-09-23",
# )

# success, logs = run_data_entry_sweep_3(
#     models=models,
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-09-30",
# )

# success, logs = run_data_entry_sweep_4(
#     models=models,
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-10-13-high-16k",
#     reasoning_effort="high",
#     reasoning_tokens=16000,
# )

# success, logs = run_data_entry_sweep_4(
#     models=models,
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-10-13-low-2480",
#     reasoning_effort="low",
#     reasoning_tokens=2480,
# )

# success, logs = run_data_entry_sweep_5(
#     models=['grok/grok-4'],
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-10-13-grok-4",
# )

# success, logs = run_data_entry_sweep_4(
#     models=[model for model in models if 'o3' not in model and 'gemini' not in model],
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-10-13-least-reasoning",
#     reasoning_effort="minimal",
#     reasoning_tokens=None,
# )

# success, logs = run_data_entry_sweep_6(
#     models=['grok/grok-4'],
#     num_samples=300,
#     log_dir="logs/data-entry-sweep-2025-10-13-grok-remaining",
# )

success, logs = run_data_entry_sweep_6(
    models=['grok/grok-4'],
    num_samples=10,
    log_dir="logs/data-entry-sweep-2025-10-25-test",
)