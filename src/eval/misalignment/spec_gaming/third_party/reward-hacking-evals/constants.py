# PATCH (teaching_claude_why_replication): this repo has no OPENAI_API_KEY -- every
# Claude/OpenAI call goes through OpenRouter. The gpt-4.1 customer simulator, the
# gpt-4.1-nano data-entry grader and the gpt-4.1 LiveCode hard-coding detector are all
# routed here instead. Changing the customer simulator changes the ENVIRONMENT, not just
# the grading, so numbers are not directly comparable to the paper's.
JUDGE_MODEL = "openrouter/google/gemini-3-flash-preview"

MODEL_FULL_NAME_LIST = [
    "openai/gpt-4o-2024-08-06",
    "openai/o3-mini-2025-01-31",
    "openai/o3-2025-04-16",
    "openai/gpt-5",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "openrouter/x-ai/grok-4",
    "openrouter/google/gemini-2.5-pro",
]

ANTHROPIC_REASONING_MODEL_LIST = [
    'anthropic/claude-3-7-sonnet-20250219',
    'anthropic/claude-sonnet-4-20250514',
    'anthropic/claude-opus-4-20250514',
]

MODEL_MAPPING_DICT = {
    'openai/o3-2025-04-16': 'o3',
    'openai/o3': 'o3',
    'openai/o3-mini-2025-01-31': 'o3-mini',
    'openai/o3-mini': 'o3-mini',
    'openai/o1-2024-12-17': 'o1',
    'openai/gpt-4o-2024-08-06': 'gpt-4o',
    'gpt-4o-2024-08-06': 'gpt-4o',
    'o1-2024-12-17': 'o1',
    'anthropic/claude-3-5-sonnet-20241022': 'claude sonnet-3.5',
    'anthropic/claude-3-7-sonnet-20250219': 'claude sonnet-3.7',
    'anthropic/claude-sonnet-4-20250514': 'claude sonnet-4',
    'anthropic/claude-opus-4-20250514': 'claude opus-4',
    'openrouter/x-ai/grok-4': 'grok-4',
    'grok/grok-4': 'grok-4',
    'xai/grok-4': 'grok-4',
    'openrouter/google/gemini-2.5-pro': 'gemini-2.5-pro',
    'openai/gpt-5': 'gpt-5',
    'anthropic/claude-sonnet-4-5-20250929': 'claude sonnet-4.5',
    'openrouter/mistralai/magistral-medium-2506:thinking': 'magistral-medium',
    'openrouter/mistralai/mistral-medium-3': 'mistral-medium-3',
    'openrouter/deepseek/deepseek-chat': 'deepseek-chat',
    'openrouter/deepseek/deepseek-r1': 'deepseek-r1',
    'openrouter/google/gemini-3-pro-preview': 'gemini-3-pro',
    'google/gemini-3-pro-preview': 'gemini-3-pro',
    'together_ai/kei.nishimuragasparian@gmail.com/Qwen/Qwen2.5-32B-2a5ff838': 'qwen2.5-32b',
    'together_ai/kei.nishimuragasparian@gmail.com/Qwen/QwQ-32B-91390a4b': 'qwq-32b',
    'together/kei.nishimuragasparian@gmail.com/Qwen/Qwen2.5-32B-2a5ff838': 'qwen2.5-32b',
    'together/kei.nishimuragasparian@gmail.com/Qwen/QwQ-32B-91390a4b': 'qwq-32b',
    'mistral/magistral-medium-2509': 'magistral-medium-1.2',
    'openrouter/qwen/qwq-32b': 'qwq-32b',
    'openrouter/allenai/olmo-3-32b-think:free': 'olmo-3-32b-think',
    'openrouter/allenai/olmo-3.1-32b-think:free': 'olmo-3.1-32b-think',
    # vLLM-served OLMo models
    'vllm/Olmo-3-32B-Think-SFT': 'olmo-3-32b-sft',
    'vllm/Olmo-3-32B-Think-DPO': 'olmo-3-32b-dpo',
    'vllm/Olmo-3.1-32B-Think': 'olmo-3.1-32b-think',
}

# OLMo RLVR checkpoints (for vLLM serving)
OLMO_RLVR_CHECKPOINTS = [f'step_{i:04d}' for i in range(50, 2350, 50)]

MODEL_RL_LIST = [
    'openrouter/deepseek/deepseek-chat',
    'openrouter/deepseek/deepseek-r1',
    'openrouter/mistralai/magistral-medium-2506:thinking',
    'openrouter/mistralai/mistral-medium-3',
]