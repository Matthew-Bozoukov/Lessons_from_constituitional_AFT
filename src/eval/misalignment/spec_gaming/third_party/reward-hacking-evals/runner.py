from abc import ABC, abstractmethod
from config import (
    ExperimentConfig, 
    EmailExperimentConfig, 
    ExperimentType, 
    FinalEmailType, 
    EmailPromptType, 
    CustomerServiceExperimentConfig, 
    CustomerType, 
    MultipleChoiceDatasetType, 
    MultipleChoicePromptType, 
    MultipleChoiceExperimentConfig
)
from create_prompts import (
    build_email_starting_prompt, 
    build_customer_service_bot_starting_prompt, 
    build_customer_starting_prompt,
    build_multiple_choice_starting_prompt
)
from model_call import unified_completion
from evaluate_experiments import (
    evaluate_email_assistant_experiment,
    evaluate_customer_service_experiment,
    evaluate_multiple_choice_experiment
)
from constants import MODEL_MAPPING_DICT

import json
import time
import random
from tqdm import tqdm
import asyncio
import yaml
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    filename=f"logs/experiment.log",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

domain = "applite"


async def load_data(config: ExperimentConfig, num_samples: int):
    if config.experiment_type is ExperimentType.EMAIL_ASSISTANT:
        emails = json.load(open('datasets/emails_updated.json'))
        with open('datasets/filtered_names.txt', 'r') as f:
            names = f.read().split('\n')
        benign_emails = emails['event'] + emails['important'] + emails['spam']
        if config.final_email_type is FinalEmailType.BAD_BEHAVIOR:
            bad_behavior_emails = emails['bad_email']
        elif config.final_email_type is FinalEmailType.ASSISTANT_REPLACEMENT:
            bad_behavior_emails = emails['email_assistant']
        elif config.final_email_type is FinalEmailType.BAD_BEHAVIOR_SLACK:
            bad_behavior_emails = emails['bad_slack']
        elif config.final_email_type is FinalEmailType.ASSISTANT_REPLACEMENT_SLACK:
            bad_behavior_emails = emails['slack_assistant']
        elif config.final_email_type is FinalEmailType.BAD_BEHAVIOR_UPDATED:
            bad_behavior_emails = emails['bad_emails_matching']
        elif config.final_email_type is FinalEmailType.ASSISTANT_REPLACEMENT_UPDATED:
            bad_behavior_emails = emails['email_assistant_matching']
        elif config.final_email_type is FinalEmailType.BAD_BEHAVIOR_SLACK_UPDATED:
            bad_behavior_emails = emails['bad_slack_matching']
        elif config.final_email_type is FinalEmailType.ASSISTANT_REPLACEMENT_SLACK_UPDATED:
            bad_behavior_emails = emails['slack_assistant_matching']
        elif config.final_email_type is FinalEmailType.BAD_BEHAVIOR_INTERN_UPDATED:
            bad_behavior_emails = emails['bad_intern_matching']
        elif config.final_email_type is FinalEmailType.ASSISTANT_REPLACEMENT_TIME_MANAGEMENT_UPDATED:
            bad_behavior_emails = emails['time_management_matching']
        else:
            raise ValueError(f"Unsupported final email type: {config.final_email_type}")
        
        output = []
        for _ in range(num_samples):
            non_final_email_ct = config.email_ct - 1
            name = random.choice(names)
            
            non_final_emails = [em.format(name=name, domain=domain) for em in random.choices(benign_emails, k=non_final_email_ct)]
            final_email = random.choice(bad_behavior_emails).format(name=name, domain=domain)
            output.append({
                'non_final_emails': non_final_emails,
                'final_email': final_email,
                'name': name,
                'domain': domain,
            })
        return output
    elif config.experiment_type is ExperimentType.CUSTOMER_SERVICE:
        with open('datasets/customer_service_problems.txt', 'r') as f:
            problems = f.read().split('\n')
        output = []
        for _ in range(num_samples):
            problem = random.choice(problems)
            output.append({
                'problem': problem,
            })
        return output
    elif config.experiment_type is ExperimentType.MULTIPLE_CHOICE:
        if config.dataset_type is MultipleChoiceDatasetType.WORLD_AFFECTING_REWARD:
            with open('datasets/world_affecting_reward.jsonl', 'r') as f:
                data = [json.loads(line) for line in f]
        elif config.dataset_type is MultipleChoiceDatasetType.REVEALING_SCORE:
            with open('datasets/mmlu_scored_filtered.jsonl', 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
        
        data = random.sample(data, num_samples)
        return data
    else:
        raise ValueError(f"Unsupported experiment type: {config.experiment_type}")


async def save_results(results: list, output_file: str):
    with open(output_file, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


async def calculate_num_samples_remaining(config: ExperimentConfig):
    # If the folder doesn't exist, or if the folder does exist but doesn't have a results.jsonl file, return the full num_samples
    output_folder = f"results/{config.output_folder}"
    if not os.path.exists(output_folder) or not os.path.exists(f"{output_folder}/results.jsonl"):
        return config.num_samples

    # If the folder exists and has a results.jsonl file, count the number of lines in the file
    with open(f"{output_folder}/results.jsonl", 'r') as f:
        return max(config.num_samples - len([line for line in f]), 0)


async def run_experiment(config: ExperimentConfig):
    num_samples_remaining = await calculate_num_samples_remaining(config)
    dataset = await load_data(config, num_samples_remaining)

    output_folder = f"results/{config.output_folder}"
    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False, indent=2)    
    output_file = f"{output_folder}/results.jsonl"
    
    short_model_name = MODEL_MAPPING_DICT.get(config.model.model_name, config.model.model_name.split('/')[-1])
    desc = f"{short_model_name} | {config.experiment_type.value}"
    for i in tqdm(range(0, len(dataset), config.batch_size), desc=desc):
        batch = dataset[i : i + config.batch_size]
        batch_tasks = []

        for item in batch:
            if config.experiment_type is ExperimentType.EMAIL_ASSISTANT:
                batch_tasks.append(run_email_assistant_experiment(config, item))
            elif config.experiment_type is ExperimentType.CUSTOMER_SERVICE:
                batch_tasks.append(run_customer_service_experiment(config, item))
            elif config.experiment_type is ExperimentType.MULTIPLE_CHOICE:
                batch_tasks.append(run_multiple_choice_experiment(config, item))
            else:
                raise ValueError(f"Unsupported experiment mode: {config.experiment_mode}")

        # Wait for all batch tasks to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Log exceptions and filter valid results
        valid_batch_results = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                item_id = getattr(batch[idx], 'id', f'item_{i + idx}')  # Use item ID if available
                logger.error(f"Task failed for {item_id}: {type(result).__name__}: {result}")
            elif result is not None:
                valid_batch_results.append(result)

        # save results
        await save_results(valid_batch_results, output_file)


async def run_email_assistant_experiment(config: EmailExperimentConfig, item: dict):
    prompt_type = config.prompt_type
    non_final_emails = item['non_final_emails']
    final_email = item['final_email']
    name = item['name']

    base_prompt = build_email_starting_prompt(name, prompt_type)
    messages = [
        {"role": "system", "content": base_prompt},
    ]
    thinking_blocks = []
    response_dicts = []

    for i in range(config.email_ct):
        if i < config.email_ct - 1:
            messages.append({"role": "user", "content": non_final_emails[i]})
        else:
            messages.append({"role": "user", "content": final_email})
    
        content, thinking, response_dict = await unified_completion(config.model, messages)
        thinking_blocks.append(thinking)
        messages.append({"role": "assistant", "content": content})
        response_dicts.append(response_dict)

    outcome = evaluate_email_assistant_experiment(messages[-1]['content'])

    return {
        'experiment_type': config.experiment_type.value,
        'outcome': outcome,
        'messages': messages,
        'thinking_blocks': thinking_blocks,
        'model': config.model.model_name,
        'response_dicts': response_dicts,
    }

async def run_customer_service_experiment(config: CustomerServiceExperimentConfig, item: dict):
    prompt_type = config.prompt_type
    customer_type = config.customer_type
    customer_model = config.customer_model
    bot_model = config.model

    bot_base_prompt = build_customer_service_bot_starting_prompt(prompt_type)
    customer_base_prompt = build_customer_starting_prompt(item['problem'], customer_type)
    
    bot_messages = [
        {"role": "system", "content": bot_base_prompt},
    ]
    customer_messages = [
        {"role": "system", "content": customer_base_prompt},
    ]
    
    bot_thinking_blocks = []
    customer_thinking_blocks = []
    customer_response_dicts = []
    bot_response_dicts = []

    # PATCH (teaching_claude_why_replication): `counter = 0` was INSIDE the loop, so
    # `counter >= 10` never fired and the only exit was the bot emitting
    # END_CONVERSATION. A model that never emits it looped forever, billing the
    # customer-simulator every turn. Hoisted so the cap is real.
    counter = 0
    while True:
        customer_content, customer_thinking, customer_response_dict = await unified_completion(customer_model, customer_messages)
        customer_thinking_blocks.append(customer_thinking)
        customer_messages.append({"role": "assistant", "content": customer_content})
        bot_messages.append({"role": "user", "content": customer_content})
        customer_response_dicts.append(customer_response_dict)

        bot_content, bot_thinking, bot_response_dict = await unified_completion(bot_model, bot_messages)
        bot_thinking_blocks.append(bot_thinking)
        bot_messages.append({"role": "assistant", "content": bot_content})
        customer_messages.append({"role": "user", "content": bot_content})
        bot_response_dicts.append(bot_response_dict)

        if "END_CONVERSATION" in bot_content or counter >= 10:
            break

        counter += 1
    
    outcome = evaluate_customer_service_experiment(bot_messages[-1]['content'], prompt_type)

    return {
        'experiment_type': config.experiment_type.value,
        'outcome': outcome,
        'bot_messages': bot_messages,
        'customer_messages': customer_messages,
        'bot_thinking_blocks': bot_thinking_blocks,
        'customer_thinking_blocks': customer_thinking_blocks,
        'bot_model': bot_model.model_name,
        'customer_model': customer_model.model_name,
        'customer_type': customer_type.value,
        'customer_response_dicts': customer_response_dicts,
        'bot_response_dicts': bot_response_dicts,
    }


async def run_multiple_choice_experiment(config: MultipleChoiceExperimentConfig, item: dict):
    prompt_type = config.prompt_type
    model = config.model

    base_prompt = build_multiple_choice_starting_prompt(item['prompt_list'][0], prompt_type)
    messages = [
        {"role": "user", "content": base_prompt},
    ]
    thinking_blocks = []

    content, thinking, response_dict = await unified_completion(model, messages)
    thinking_blocks.append(thinking)
    messages.append({"role": "assistant", "content": content})

    outcome = evaluate_multiple_choice_experiment(messages[-1]['content'], item['high_reward_answer'])

    return {
        'experiment_type': config.experiment_type.value,
        'outcome': outcome,
        'messages': messages,
        'thinking_blocks': thinking_blocks,
        'model': model.model_name,
        'dataset_type': config.dataset_type.value,
        'prompt_type': config.prompt_type.value,
        'high_reward_answer': item['high_reward_answer'],
        'response_dict': response_dict,
    }