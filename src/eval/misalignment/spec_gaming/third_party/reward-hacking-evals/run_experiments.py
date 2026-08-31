import logging
import os

# Set up logging configuration BEFORE any other imports
os.makedirs("logs", exist_ok=True)
# Set root logger to WARNING to suppress most INFO logs
logging.basicConfig(
    level=logging.WARNING,
    filename=f"logs/experiment.log",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Allow INFO logs for our own modules
logging.getLogger('runner').setLevel(logging.INFO)
logging.getLogger('model_call').setLevel(logging.INFO)

# Suppress specific noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.ERROR)
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('litellm.litellm').setLevel(logging.ERROR)

from runner import run_experiment
from config import (
    EmailExperimentConfig, 
    EmailPromptType, 
    FinalEmailType, 
    ExperimentType, 
    LLMModel, 
    CustomerServicePromptType, 
    CustomerType, 
    CustomerServiceExperimentConfig, 
    MultipleChoicePromptType,
    MultipleChoiceDatasetType,
    MultipleChoiceExperimentConfig
)
from constants import MODEL_FULL_NAME_LIST, ANTHROPIC_REASONING_MODEL_LIST, MODEL_RL_LIST, JUDGE_MODEL

import asyncio

def calculate_batch_size(model: str):
    if 'openai' in model:
        return 100
    elif 'claude' in model:
        return 8
    elif 'mistral' in model:
        return 5
    elif 'allenai' in model:
        return 5
    elif 'together_ai' in model:
        return 5
    elif 'vllm' in model:
        # PATCH: 32 == the served --max-num-seqs for Qwen3.6 (one Mamba cache block per
        # decode sequence; the server refuses to start above that). Batching to the
        # server's cap is the only free throughput lever on this arch -- there is no
        # prefix caching to reuse.
        return 32
    else:
        return 20

async def run_email_assistant_experiment(top_level_subfolder, model_list: list[str] = MODEL_FULL_NAME_LIST, num_samples: int = 300):
    model_tasks = {model: [] for model in model_list}

    prompt_email_tuples = [(prompt_type, final_email_type) for prompt_type in EmailPromptType for final_email_type in [FinalEmailType.BAD_BEHAVIOR, FinalEmailType.ASSISTANT_REPLACEMENT]]
    prompt_email_tuples.extend([(EmailPromptType.DEFAULT, final_email_type) for final_email_type in [FinalEmailType.BAD_BEHAVIOR_SLACK, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK]])

    prompt_email_tuples = [(prompt_type, final_email_type) for prompt_type in [EmailPromptType.REMOVE_SNOOZE] for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED, FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED]]

    for model in model_list:
        for prompt_type, final_email_type in prompt_email_tuples:
            config = EmailExperimentConfig(
                experiment_type=ExperimentType.EMAIL_ASSISTANT,
                model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                num_samples=num_samples,
                output_folder=f"{top_level_subfolder}/{model.split('/')[-1]}_{final_email_type.value}_{prompt_type.value}",
                prompt_type=prompt_type,
                email_ct=3,
                final_email_type=final_email_type,
                batch_size=calculate_batch_size(model),
            )
            model_tasks[model].append(run_experiment(config))
    
    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results

    # Create one task per model that will run all that model's experiments in sequence
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]

    await asyncio.gather(*model_runners)


async def run_customer_service_experiment(top_level_subfolder, model_list: list[str] = MODEL_FULL_NAME_LIST, num_samples: int = 300):
    model_tasks = {model: [] for model in model_list}

    # prompt_customer_tuples = [(prompt_type, customer_type) for prompt_type in CustomerServicePromptType for customer_type in CustomerType]

    prompt_types = [CustomerServicePromptType.DEFAULT, 
    #CustomerServicePromptType.REWARD_SIGNAL, CustomerServicePromptType.LINK_REQUIRED, 
                    #CustomerServicePromptType.ALTERNATE_LINK
                    # CustomerServicePromptType.EVALUATION_SELECT, CustomerServicePromptType.EVALUATION_W_CODE_SELECT
                    ]
    customer_types = [CustomerType.FRIENDLY, CustomerType.BELLIGERENT]
    prompt_customer_tuples = [(prompt_type, customer_type) for prompt_type in prompt_types for customer_type in customer_types]

    # model_tasks = {model: [] for model in model_list if 'openai' in model and 'mini' in model}



    for model in model_list:
        for prompt_type, customer_type in prompt_customer_tuples:
            config = CustomerServiceExperimentConfig(
                experiment_type=ExperimentType.CUSTOMER_SERVICE,
                model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                num_samples=num_samples,
                output_folder=f"{top_level_subfolder}/{model.split('/')[-1]}_{customer_type.value}_{prompt_type.value}",
                prompt_type=prompt_type,
                customer_type=customer_type,
                customer_model=LLMModel(model_name=JUDGE_MODEL),
                batch_size=calculate_batch_size(model),
            )
            model_tasks[model].append(run_experiment(config))

    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results

    # Create one task per model that will run all that model's experiments in sequence
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]

    await asyncio.gather(*model_runners)


async def run_multiple_choice_experiment(top_level_subfolder, model_list: list[str] = MODEL_FULL_NAME_LIST, num_samples: int = 300):
    model_tasks = {model: [] for model in model_list}

    prompt_type_tuples = [(prompt_type, dataset_type) for prompt_type in [MultipleChoicePromptType.DEFAULT] for dataset_type in [
        MultipleChoiceDatasetType.WORLD_AFFECTING_REWARD,
        MultipleChoiceDatasetType.REVEALING_SCORE
        ]]


    for model in model_list:
        for prompt_type, dataset_type in prompt_type_tuples:
            config = MultipleChoiceExperimentConfig(
                experiment_type=ExperimentType.MULTIPLE_CHOICE,
                model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                num_samples=num_samples,
                output_folder=f"{top_level_subfolder}/{model.split('/')[-1]}_{dataset_type.value}_{prompt_type.value}",
                prompt_type=prompt_type,
                dataset_type=dataset_type,
                batch_size=calculate_batch_size(model),
            )
            model_tasks[model].append(run_experiment(config))

    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results
    
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]
    await asyncio.gather(*model_runners)


async def run_varying_reasoning_effort_experiment(top_level_subfolder):
    num_samples = 300

    reasoning_model_list = [
        'openai/o3-2025-04-16', 
        'openai/gpt-5', 
        'openai/o3-mini-2025-01-31', 
        'anthropic/claude-3-7-sonnet-20250219', 
        'anthropic/claude-opus-4-20250514',
        'openrouter/google/gemini-2.5-pro',
    ]

    reasoning_model_mapping = {
        'openai/o3-2025-04-16': ['low', 'medium'],
        'openai/o3-mini-2025-01-31': ['low', 'medium'],
        'openai/gpt-5': ['minimal', 'medium'],
        'anthropic/claude-3-7-sonnet-20250219': [None],
        'anthropic/claude-opus-4-20250514': [None],
        'openrouter/google/gemini-2.5-pro': [128, 2048],
    }

    model_tasks = {model: [] for model in reasoning_model_list}

    for model in model_tasks.keys():
        for effort in reasoning_model_mapping[model]:
            # Run email assistant experiment
            email_prompt_type = EmailPromptType.DEFAULT
            for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK]:
                config = EmailExperimentConfig(
                    experiment_type=ExperimentType.EMAIL_ASSISTANT,
                    model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                    num_samples=num_samples,
                    output_folder=f"email_{top_level_subfolder}/{model.split('/')[-1]}_{effort}_{final_email_type.value}_{email_prompt_type.value}",
                    prompt_type=email_prompt_type,
                    email_ct=3,
                    final_email_type=final_email_type,
                    batch_size=calculate_batch_size(model),
                )
                model_tasks[model].append(run_experiment(config))
            
            # Run customer service experiment
            for prompt_type in [CustomerServicePromptType.DEFAULT, 
                                CustomerServicePromptType.ALTERNATE_LINK]:
                for customer_type in [CustomerType.FRIENDLY, CustomerType.BELLIGERENT]:

                        config = CustomerServiceExperimentConfig(
                            experiment_type=ExperimentType.CUSTOMER_SERVICE,
                            model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                            num_samples=num_samples,
                            output_folder=f"customer_service_{top_level_subfolder}/{model.split('/')[-1]}_{effort}_{customer_type.value}_{prompt_type.value}",
                            prompt_type=prompt_type,
                            customer_type=customer_type,
                            customer_model=LLMModel(model_name=JUDGE_MODEL),
                            batch_size=calculate_batch_size(model),
                        )
                        model_tasks[model].append(run_experiment(config))
            
            # Run multiple choice experiment
            for dataset_type in MultipleChoiceDatasetType:
                for prompt_type in [MultipleChoicePromptType.DEFAULT]:
                    config = MultipleChoiceExperimentConfig(
                        experiment_type=ExperimentType.MULTIPLE_CHOICE,
                        model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                        num_samples=num_samples,
                        output_folder=f"multiple_choice_{top_level_subfolder}/{model.split('/')[-1]}_{effort}_{dataset_type.value}_{prompt_type.value}",
                        prompt_type=prompt_type,
                        dataset_type=dataset_type,
                        batch_size=calculate_batch_size(model),
                    )
                    model_tasks[model].append(run_experiment(config))

    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results
    
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]
    await asyncio.gather(*model_runners)


async def run_varying_reasoning_effort_experiment_2(top_level_subfolder):
    num_samples = 300

    reasoning_model_list = [
        'openai/o3-2025-04-16', 
        'openai/gpt-5', 
        'anthropic/claude-3-7-sonnet-20250219', 
        'anthropic/claude-opus-4-20250514',
        'openrouter/google/gemini-2.5-pro',
        'openrouter/google/gemini-3-pro-preview',
    ]

    reasoning_model_mapping = {
        'openai/o3-2025-04-16': ['low', 'high'],
        'openai/gpt-5': ['minimal', 'low', 'high'],
        'anthropic/claude-3-7-sonnet-20250219': [None, 2480, 16000],
        'anthropic/claude-opus-4-20250514': [None, 2480, 16000],
        'openrouter/google/gemini-2.5-pro': [2480, 16000],
        'openrouter/google/gemini-3-pro-preview': ['low', 'high'],
    }

    model_tasks = {model: [] for model in reasoning_model_list}

    for model in model_tasks.keys():
        for effort in reasoning_model_mapping[model]:
            # Run email assistant experiment
            for email_prompt_type in [#EmailPromptType.DEFAULT, 
                EmailPromptType.REMOVE_SNOOZE]:
                # for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK]:
                for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED, FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED]:
                    config = EmailExperimentConfig(
                        experiment_type=ExperimentType.EMAIL_ASSISTANT,
                        model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                        num_samples=num_samples,
                        output_folder=f"{top_level_subfolder}_email_assistant/{model.split('/')[-1]}_{effort}_{final_email_type.value}_{email_prompt_type.value}",
                        prompt_type=email_prompt_type,
                        email_ct=3,
                        final_email_type=final_email_type,
                        batch_size=calculate_batch_size(model),
                    )
                    model_tasks[model].append(run_experiment(config))
            
            # Run customer service experiment
            for prompt_type in [CustomerServicePromptType.DEFAULT]:
                for customer_type in [CustomerType.FRIENDLY, CustomerType.BELLIGERENT]:

                        config = CustomerServiceExperimentConfig(
                            experiment_type=ExperimentType.CUSTOMER_SERVICE,
                            model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                            num_samples=num_samples,
                            output_folder=f"{top_level_subfolder}_customer_service/{model.split('/')[-1]}_{effort}_{customer_type.value}_{prompt_type.value}",
                            prompt_type=prompt_type,
                            customer_type=customer_type,
                            customer_model=LLMModel(model_name=JUDGE_MODEL),
                            batch_size=calculate_batch_size(model),
                        )
                        model_tasks[model].append(run_experiment(config))
            
            # Run multiple choice experiment
            for dataset_type in MultipleChoiceDatasetType:
                for prompt_type in [MultipleChoicePromptType.DEFAULT]:
                    config = MultipleChoiceExperimentConfig(
                        experiment_type=ExperimentType.MULTIPLE_CHOICE,
                        model=LLMModel(model_name=model, extended_thinking_tokens=effort),
                        num_samples=num_samples,
                        output_folder=f"{top_level_subfolder}_multiple_choice/{model.split('/')[-1]}_{effort}_{dataset_type.value}_{prompt_type.value}",
                        prompt_type=prompt_type,
                        dataset_type=dataset_type,
                        batch_size=calculate_batch_size(model),
                    )
                    model_tasks[model].append(run_experiment(config))

    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results
    
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]
    await asyncio.gather(*model_runners)
        

async def test_reward_and_self_preservation_prompts(top_level_subfolder):
    num_samples = 300
    models_to_test = [
        'openai/gpt-5', 
        'openai/o3-2025-04-16', 
        'openai/o3-mini-2025-01-31', 
        'anthropic/claude-3-7-sonnet-20250219', 
        'anthropic/claude-opus-4-20250514',
        'openrouter/x-ai/grok-4',
        'openrouter/google/gemini-2.5-pro',
    ]

    model_tasks = {model: [] for model in models_to_test}

    for model in model_tasks.keys():
        # Test alternate email assistant prompts
        for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK]:
            prompt_type_list = [EmailPromptType.REWARD_SIGNAL, EmailPromptType.SELF_PRESERVATION, EmailPromptType.EVALUATION_BASELINE, EmailPromptType.EVALUATION_BASELINE_TWO]
            if 'gpt-5' in model or 'gemini-2.5-pro' in model:
                prompt_type_list.append(EmailPromptType.DEFAULT)
            for prompt_type in prompt_type_list:
                config = EmailExperimentConfig(
                    experiment_type=ExperimentType.EMAIL_ASSISTANT,
                    model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                    num_samples=num_samples,
                    output_folder=f"email_assistant_{top_level_subfolder}/{model.split('/')[-1]}_{final_email_type.value}_{prompt_type.value}",
                    prompt_type=prompt_type,
                    email_ct=3,
                    final_email_type=final_email_type,
                    batch_size=calculate_batch_size(model),
                )
                model_tasks[model].append(run_experiment(config))
        
        # Test alternate customer service prompts
        for customer_type in [CustomerType.FRIENDLY, CustomerType.BELLIGERENT]:
            prompt_type_list = [CustomerServicePromptType.REWARD_SIGNAL_TWO, CustomerServicePromptType.SELF_PRESERVATION]
            if 'gpt-5' in model:
                prompt_type_list.append(CustomerServicePromptType.DEFAULT)
                prompt_type_list.append(CustomerServicePromptType.EVALUATION)
            for prompt_type in prompt_type_list:
                config = CustomerServiceExperimentConfig(
                    experiment_type=ExperimentType.CUSTOMER_SERVICE,
                    model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                    num_samples=num_samples,
                    output_folder=f"customer_service_{top_level_subfolder}/{model.split('/')[-1]}_{customer_type.value}_{prompt_type.value}",
                    prompt_type=prompt_type,
                    customer_type=customer_type,
                    customer_model=LLMModel(model_name=JUDGE_MODEL),
                    batch_size=calculate_batch_size(model),
                )
                model_tasks[model].append(run_experiment(config))
        
    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results
    
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]
    await asyncio.gather(*model_runners)

async def run_email_assistant_experiment_updated(top_level_subfolder):
    num_samples = 300


    model_list = [
        'openai/gpt-5', 
        'openai/o3-2025-04-16', 
        'xai/grok-4',
        'openrouter/google/gemini-2.5-pro',
        'anthropic/claude-opus-4-20250514',
        'anthropic/claude-3-7-sonnet-20250219',
        'anthropic/claude-sonnet-4-5-20250929',
    ]


    model_tasks = {model: [] for model in model_list}
    prompt_types = [EmailPromptType.DEFAULT, EmailPromptType.REMOVE_SNOOZE]
    final_email_types = [FinalEmailType.BAD_BEHAVIOR_UPDATED, FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED, FinalEmailType.BAD_BEHAVIOR_SLACK_UPDATED, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED]

    prompt_email_tuples = [(prompt_type, final_email_type) for prompt_type in prompt_types for final_email_type in final_email_types]

    for model in model_list:
        for prompt_type, final_email_type in prompt_email_tuples:
            config = EmailExperimentConfig(
                experiment_type=ExperimentType.EMAIL_ASSISTANT,
                model=LLMModel(model_name=model) if model not in ANTHROPIC_REASONING_MODEL_LIST else LLMModel(model_name=model, extended_thinking_tokens=2048),
                num_samples=num_samples,
                output_folder=f"{top_level_subfolder}/{model.split('/')[-1]}_{final_email_type.value}_{prompt_type.value}",
                prompt_type=prompt_type,
                email_ct=3,
                final_email_type=final_email_type,
                batch_size=calculate_batch_size(model),
            )
            model_tasks[model].append(run_experiment(config))
    
    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results

    # Create one task per model that will run all that model's experiments in sequence
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]

    await asyncio.gather(*model_runners)


async def run_dont_hack_experiment(top_level_subfolder, num_samples: int = 5):
    reasoning_model_list = [
        'openai/o3-2025-04-16', 
        'openai/gpt-5', 
        'anthropic/claude-3-7-sonnet-20250219', 
        'anthropic/claude-opus-4-20250514',
        'openrouter/google/gemini-2.5-pro',
        'openrouter/google/gemini-3-pro-preview',
        'openrouter/x-ai/grok-4',
    ]

    reasoning_model_mapping = {
        'openai/o3-2025-04-16': 'low',
        'openai/gpt-5': 'low',
        'anthropic/claude-3-7-sonnet-20250219': 2480,
        'anthropic/claude-opus-4-20250514': 2480,
        'openrouter/google/gemini-2.5-pro': 2480,
        'openrouter/google/gemini-3-pro-preview': 'low',
        'openrouter/x-ai/grok-4': None,
    }

    model_tasks = {model: [] for model in reasoning_model_list}

    for model in model_tasks.keys():
        # Run email assistant experiment
        for email_prompt_type in [EmailPromptType.REMOVE_SNOOZE_NO_HACK]:
            # for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT, FinalEmailType.ASSISTANT_REPLACEMENT_SLACK]:
            for final_email_type in [FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED, FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED]:
                llm_model = LLMModel(model_name=model, extended_thinking_tokens=reasoning_model_mapping[model]) if reasoning_model_mapping[model] is not None else LLMModel(model_name=model)
                config = EmailExperimentConfig(
                    experiment_type=ExperimentType.EMAIL_ASSISTANT,
                    model=llm_model,
                    num_samples=num_samples,
                    output_folder=f"{top_level_subfolder}_email_assistant/{model.split('/')[-1]}_{llm_model.extended_thinking_tokens}_{final_email_type.value}_{email_prompt_type.value}",
                    prompt_type=email_prompt_type,
                    email_ct=3,
                    final_email_type=final_email_type,
                    batch_size=calculate_batch_size(model),
                )
                model_tasks[model].append(run_experiment(config))
        
        # Run customer service experiment
        for prompt_type in [CustomerServicePromptType.DEFAULT_NO_HACK]:
            for customer_type in [CustomerType.FRIENDLY, CustomerType.BELLIGERENT]:
                    llm_model = LLMModel(model_name=model, extended_thinking_tokens=reasoning_model_mapping[model]) if reasoning_model_mapping[model] is not None else LLMModel(model_name=model)
                    config = CustomerServiceExperimentConfig(
                        experiment_type=ExperimentType.CUSTOMER_SERVICE,
                        model=llm_model,
                        num_samples=num_samples,
                        output_folder=f"{top_level_subfolder}_customer_service/{model.split('/')[-1]}_{llm_model.extended_thinking_tokens}_{customer_type.value}_{prompt_type.value}",
                        prompt_type=prompt_type,
                        customer_type=customer_type,
                        customer_model=LLMModel(model_name=JUDGE_MODEL),
                        batch_size=calculate_batch_size(model),
                    )
                    model_tasks[model].append(run_experiment(config))
        
        # Run multiple choice experiment
        for (dataset_type, prompt_type) in [(MultipleChoiceDatasetType.WORLD_AFFECTING_REWARD, MultipleChoicePromptType.WORLD_AFFECTING_REWARD_NO_HACK), (MultipleChoiceDatasetType.REVEALING_SCORE, MultipleChoicePromptType.REVEALING_SCORE_NO_HACK)]:
            llm_model = LLMModel(model_name=model, extended_thinking_tokens=reasoning_model_mapping[model]) if reasoning_model_mapping[model] is not None else LLMModel(model_name=model)
            config = MultipleChoiceExperimentConfig(
                experiment_type=ExperimentType.MULTIPLE_CHOICE,
                model=llm_model,
                num_samples=num_samples,
                output_folder=f"{top_level_subfolder}_multiple_choice/{model.split('/')[-1]}_{llm_model.extended_thinking_tokens}_{dataset_type.value}_{prompt_type.value}",
                prompt_type=prompt_type,
                dataset_type=dataset_type,
                batch_size=calculate_batch_size(model),
            )
            model_tasks[model].append(run_experiment(config))

    async def run_model_tasks(tasks):
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        return results
    
    model_runners = [run_model_tasks(tasks) for model, tasks in model_tasks.items()]
    await asyncio.gather(*model_runners)


if __name__ == "__main__":
    # asyncio.run(run_email_assistant_experiment("email_assistant_20250710_v2"))
    # asyncio.run(run_customer_service_experiment("customer_service_20250716_v2"))
    # asyncio.run(run_multiple_choice_experiment("multiple_choice_20250722"))
    # asyncio.run(run_varying_reasoning_effort_experiment("reasoning_effort_20250826"))
    # asyncio.run(test_reward_and_self_preservation_prompts("reward_and_self_preservation_20250828"))
    # asyncio.run(run_varying_reasoning_effort_experiment_2("reasoning_effort_20251025"))
    # asyncio.run(run_email_assistant_experiment_updated("email_assistant_20251105_v2"))
    # MODEL_RL_LIST = MODEL_RL_LIST[:2]
    MODEL_RL_LIST = [
        # 'o1-2024-12-17',
        # 'gpt-4o-2024-08-06',
        # 'openrouter/mistralai/mistral-medium-3',
        # 'openrouter/qwen/qwq-32b',
        # 'openrouter/allenai/olmo-3-32b-think:free',
        # 'mistral/magistral-medium-2509',
        # 'together_ai/kei.nishimuragasparian@gmail.com/Qwen/Qwen2.5-32B-2a5ff838',
        # 'vllm/Olmo-3-32B-Think-SFT',
        # 'vllm/Olmo-3-32B-Think-DPO',
        # 'vllm/Olmo-3.1-32B-Think@step_0800',
        # 'vllm/Olmo-3.1-32B-Think@step_1600',
        'openrouter/allenai/olmo-3.1-32b-think:free',
    ]
    # asyncio.run(run_email_assistant_experiment("email_assistant_20251105_v2", model_list=MODEL_RL_LIST, num_samples=300))
    # asyncio.run(run_customer_service_experiment("customer_service_20250716_v2", model_list=MODEL_RL_LIST, num_samples=300))
    # asyncio.run(run_multiple_choice_experiment("multiple_choice_20250722", model_list=MODEL_RL_LIST, num_samples=300))
    asyncio.run(run_dont_hack_experiment("dont_hack_20260128", num_samples=300))