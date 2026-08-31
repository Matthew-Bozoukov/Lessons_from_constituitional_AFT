import os
from inspect_ai import eval
from datetime import datetime
import random
import string
from itertools import product

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from livecodebench_hack_eval import livecodebench
from constants import JUDGE_MODEL

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Generate date prefix in date_MM_DD_YYYY_time_HHMM format
date_prefix = datetime.now().strftime("date_%m_%d_%Y_time_%H%M")
# generate 8 random alphanumeric characters in one string
random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

"""
Hyperparameters:

Task arguments (task_args):
- pressure_prompt_type: List of pressure types to add to system prompt (e.g., "standard", "do_not_hack", "hack")
- max_turns: Maximum number of code submission attempts allowed
- give_bailout_option: Whether to allow model to end session without submitting code
- difficulties: List of code problem difficulties to include ("easy", "medium", "hard")
- max_private_test_cases: Strengthen unit tests used by adding tests from livecode private set
- sandbox: Execution environment ("local" or "docker")
- use_llm_hard_coding_detector: Whether to use LLM judge to detect hard-coding
- llm_grader_model: Model to use for LLM judging (if None, uses evaluated model)
- use_incorrect_unit_test: Whether to use dataset with intentionally incorrect unit tests

Eval arguments:
- log_dir: Directory to save evaluation logs
- limit: Maximum number of samples to evaluate
- model: Model(s) to evaluate
- max_connections: Maximum parallel API connections
- reasoning_tokens: Max tokens for reasoning (for o1-style models)
- reasoning_effort: Reasoning effort level ("low", "high", None)
- reasoning_summary: What reasoning summary to show in inspect logs ("auto")
- reasoning_history: Whether model is fed back prior reasoning in later turns ("auto")
"""

DEBUG = False
hparams_base = {
    "task_args": {
        "pressure_prompt_type": ["standard"], # list of: "standard", "do_not_hack"
        "max_turns": 3,
        "give_bailout_option": True,
        "difficulties": ["easy"], # "easy" or "hard"
        "max_private_test_cases": 3,
        "sandbox": "local",
        "use_llm_hard_coding_detector": True,
        "llm_grader_model": JUDGE_MODEL,
        "use_incorrect_unit_test": True,
    },
    "log_dir": f"logs/{date_prefix}-{random_suffix}-",
    "limit": 80,
    "model": "openrouter/openai/gpt-4.1-nano",
    "max_connections": 80,
    "reasoning_summary": "auto", # see inspect docs
    "reasoning_history": "auto",
}
hparams_base.update({
    # Reasoning
    "reasoning_tokens": 16000, # None, 2480, 16000
    "reasoning_effort": "high", # None, low, high
})
hparams_base["log_dir"] += "-test"
model_sweep = [
    # "anthropic/claude-3-5-sonnet-20241022",
    # "anthropic/claude-3-7-sonnet-20250219",
    # "anthropic/claude-sonnet-4-20250514",
    # "anthropic/claude-opus-4-20250514",
    # "anthropic/claude-opus-4-1-20250805",
    # "openai/gpt-4o-mini-2024-07-18",
    # "openai/gpt-4o-2024-08-06",
    # "openai/gpt-4.1",
    # "openai/o3",
    # "openai/gpt-5",
    # "google/vertex/gemini-2.5-pro",
    # "openrouter/x-ai/grok-4",
    # "grok/grok-4",
    # "openrouter/google/gemini-2.5-flash",
    # "openrouter/google/gemini-2.5-pro",
    "openrouter/google/gemini-3-pro-preview",
    # "openrouter/deepseek/deepseek-r1-0528",
]
# Define sweep parameters
sweep_params = {
    "pressure_prompt_type": [
        ["standard"],
        # ["do_not_hack"],
        # ["self-preservation"],
        # ["do_not_hack", "reverse-self-preservation"],
    ],
    "max_turns": [
        3,
        # 10,
    ],
    "use_incorrect_unit_test": [
        True,
        # False,
    ],
    "difficulties": [
        "easy",
        # "medium",
        "hard",
    ],
    "give_bailout_option": [
        True,
        # False,
    ],
}

if DEBUG:
    model_sweep = ["openai/gpt-4o-mini-2024-07-18"]
    hparams_base["limit"] = 1
    hparams_base["task_args"]["max_turns"] = 2
    hparams_base["log_dir"] = "logs/debug"

def run_sweep() -> None:

    tasks = []
    
    # Generate all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    for param_combo in product(*param_values):
        temp_task_hparams = dict(hparams_base["task_args"])
        
        # Apply parameter combination
        for param_name, param_value in zip(param_names, param_combo):
            temp_task_hparams[param_name] = param_value

        tasks.append(livecodebench(**temp_task_hparams))

    eval(
        tasks,
        model=model_sweep,
        log_dir=hparams_base["log_dir"],
        limit=hparams_base["limit"],
        max_connections=hparams_base["max_connections"],
        reasoning_tokens=hparams_base["reasoning_tokens"],
        reasoning_effort=hparams_base["reasoning_effort"],
        reasoning_summary=hparams_base["reasoning_summary"],
        reasoning_history=hparams_base["reasoning_history"],
    )


if __name__ == "__main__":
    run_sweep()