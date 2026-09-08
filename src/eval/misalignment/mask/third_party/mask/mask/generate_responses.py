import pandas as pd
import glob
import os
import random
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import argparse
import asyncio
from dotenv import load_dotenv

load_dotenv()  

# PATCH (teaching_claude_why): per-prompt-type count of finished-but-empty generations for
# the file being processed (see generate_responses_async); flushed to `_empty_content.json`.
EMPTY_CONTENT = {}


async def generate_responses_async(client, model_name, formatted_prompts, max_tokens, temperature, semaphore, K=1):
    """
    Generates responses from the specified model for a batch of prompts.
    
    Args:
        client: API client instance for the model
        model_name: Name of the model to use
        formatted_prompts: Dictionary mapping prompt keys to message lists
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        semaphore: Asyncio semaphore for rate limiting
        K: Number of responses to generate per prompt (default=1)
        
    Returns:
        dict: Mapping of prompt keys to lists of generated responses
    """
    results = {}

    async def process_prompt(prompt_key, messages):
        """Processes a single prompt with retries and rate limiting."""
        retry_delay = 3
        max_retries = 7
        jitter = 0.1

        all_responses = []

        for _ in range(K):
            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        # Model-specific API calls
                        if 'claude' in model_name:
                            system_prompt = None
                            user_messages = messages.copy()
                            if messages and messages[0]['role'] == 'system':
                                system_prompt = messages[0]['content']
                                user_messages = messages[1:]
                            
                            message = await client.messages.create(
                                model=model_name,
                                max_tokens=max_tokens,
                                system=system_prompt,
                                messages=user_messages,
                                temperature=temperature
                            )
                        else:
                            if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
                                # replace "system" with "developer"
                                if messages and messages[0]['role'] == 'system':
                                    messages[0]['role'] = 'developer'
                            completion = await client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                        
                    # Extract response based on model type
                    if 'claude' in model_name:
                        response = message.content[0].text
                    else:
                        response = completion.choices[0].message.content
                        # PATCH (teaching_claude_why): a thinking target served behind a
                        # reasoning parser can finish (finish_reason=stop) with EMPTY content:
                        # it wrote its reply inside the think block and never closed it
                        # (measured 2026-09-07 on the nosynth control: 13/16 pressure prompts).
                        # Upstream stores that as a generation error. Here it is counted as
                        # empty content and, per MASK_EMPTY_CONTENT, stored as "" (`evasion`:
                        # the judge reads no statement) or as the reasoning text (`reasoning`:
                        # what the model actually wrote). Never silently an error.
                        if response is None and completion.choices[0].finish_reason == 'stop':
                            msg = completion.choices[0].message
                            reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None) or ''
                            EMPTY_CONTENT[prompt_key] = EMPTY_CONTENT.get(prompt_key, 0) + 1
                            response = reasoning if os.getenv('MASK_EMPTY_CONTENT', 'evasion') == 'reasoning' else ''
                    all_responses.append(response)
                    break

                except Exception as e:
                    if 'invalid_prompt' in str(e):
                        print(f"Prompt violated content policy for prompt: {prompt_key}. Replacing with error message.")
                        all_responses.append("[Error: Prompt violated content policies]")
                        break
                    elif attempt == max_retries - 1:
                        print(f"Max retries exceeded for prompt: {prompt_key}")
                        print(e)
                        all_responses.append(None)
                    else:
                        print(f"Error occurred for prompt: {prompt_key}. Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay:.1f} seconds.")
                        print(e)
                        jittered_delay = retry_delay * (1 + jitter * (2 * random.random() - 1))
                        await asyncio.sleep(jittered_delay)
                        retry_delay *= 2

        results[prompt_key] = all_responses

    tasks = [process_prompt(key, messages) for key, messages in formatted_prompts.items()]
    await asyncio.gather(*tasks)

    return results

def create_model_client(model_name, base_url=None, api_key=None):
    # PATCH (teaching_claude_why): a served target (vLLM) or any OpenAI-compatible endpoint
    # passes an explicit base_url, and is always reached through the OpenAI chat API — never
    # the Anthropic client. The original provider-by-name routing is kept for bare API models.
    if base_url is not None:
        return AsyncOpenAI(base_url=base_url, api_key=api_key or os.getenv('OPENAI_API_KEY') or 'EMPTY')
    if 'claude' in model_name.lower():
        return AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    else:
        return AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def process_dataframe(input_csv, output_csv, temperature, model_name="gpt-4o-mini", max_tokens=2000, belief_k=None, lie_k=10, base_url=None, api_key=None):
    """
    Process a dataframe and generate responses from an AI model.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        temperature (float): Sampling temperature for generation
        model_name (str): Name of the model to use
        max_tokens (int): Maximum tokens for generation
        belief_k (dict): Required mapping of belief elicitation numbers to number of runs
            e.g. {"1": 3, "2": 1, "3": 1} means run belief_elicit_1 three times,
            belief_elicit_2 and belief_elicit_3 once each
        lie_k (int): Number of lying prompts to generate for each row
        
    The function reads a CSV containing prompts, generates responses using the specified
    model, and saves the results to a new CSV file.
    """
    client = None
    try:
        # Initialize model-specific client
        client = create_model_client(model_name, base_url=base_url, api_key=api_key)

        # PATCH (teaching_claude_why): in-flight generations. The original 10 cannot fill a
        # vLLM server; the runner sets MASK_GEN_CONCURRENCY from its config (gen_concurrency).
        semaphore = asyncio.Semaphore(int(os.getenv('MASK_GEN_CONCURRENCY', '10')))

        # Read input data and set default system prompt if needed
        df = pd.read_csv(input_csv)
        if 'belief_system_1' not in df.columns and 'belief_elicit_1' in df.columns:
            df['belief_system_1'] = 'You are a helpful assistant.'

        # Check if this is a doubling down scenario (multi-turn conversation)
        is_doubling_down = 'dd_turn_1_prompt' in df.columns and 'dd_turn_1_response' in df.columns
        print(f"Is this a doubling down scenario? {is_doubling_down}")

        # Prepare prompts for generation
        all_prompts = []
        indices = []
        prompt_types = []
        
        # Process each row in the dataframe
        for index, row in df.iterrows():
            # Handle lying prompts - either doubling down or standard
            if is_doubling_down:
                # For doubling down, include the full conversation history
                for run in range(lie_k):
                    all_prompts.append([
                        {'role': 'system', 'content': row['system_prompt']},
                        {'role': 'user', 'content': row['dd_turn_1_prompt']},
                        {'role': 'assistant', 'content': row['dd_turn_1_response']},
                        {'role': 'user', 'content': row['user_prompt']},
                    ])
                    indices.append(index)
                    prompt_types.append(f'lying_run{run+1}')
            else:
                # Standard lying prompt with just system and user messages
                for run in range(lie_k):
                    all_prompts.append([
                        {'role': 'system', 'content': row['system_prompt']},
                        {'role': 'user', 'content': row['user_prompt']}
                    ])
                    indices.append(index)
                    prompt_types.append(f'lying_run{run+1}')
            
            # Handle belief elicitation prompts if present
            if 'belief_elicit_1' in df.columns:
                for be_num, k_value in belief_k.items():
                    belief_col = f'belief_elicit_{be_num}'
                    if pd.notna(row.get(belief_col)):
                        for run in range(k_value):
                            all_prompts.append([
                                {'role': 'system', 'content': row['belief_system_1']},
                                {'role': 'user', 'content': row[belief_col]}
                            ])
                            indices.append(index)
                            prompt_types.append(f'be{be_num}_run{run+1}')

        # Generate responses in parallel for all prompts
        all_responses = await asyncio.gather(*[
            generate_responses_async(
                client, model_name, {prompt_type: prompt},
                max_tokens=max_tokens, temperature=temperature,
                semaphore=semaphore, K=1
            )
            for prompt, prompt_type in zip(all_prompts, prompt_types)
        ])

        # Create mapping for output column names
        column_mapping = {
            f'lying_run{run+1}': f'generation(System Prompt + User Prompt)_run{run+1}'
            for run in range(lie_k)
        }

        # Add belief elicitation column names if present
        if 'belief_elicit_1' in df.columns:
            for be_num, k_value in belief_k.items():
                for run in range(k_value):
                    column_mapping[f'be{be_num}_run{run+1}'] = f'generation(belief_system_1 + belief_elicit_{be_num})_run{run+1}'

        # Process responses and save to dataframe
        for response, index, prompt_type in zip(all_responses, indices, prompt_types):
            column_name = column_mapping[prompt_type]
            try:
                response_value = response[prompt_type][0] if prompt_type in response else None
                if response_value is None:
                    raise ValueError("Generation failed or timed out")
                df.at[index, column_name] = response_value
            except Exception as e:
                error_msg = f"[ERROR: {str(e)}]"
                print(f"WARNING: Error processing response for index {index}, prompt type {prompt_type}: {str(e)}")
                df.at[index, column_name] = error_msg

        # Save results to CSV
        df.to_csv(output_csv, index=False)
        # PATCH (teaching_claude_why): the empty-content tally for this file, by prompt type
        # (see generate_responses_async); the runner reports it beside the honesty score.
        import json as _json
        tally_path = os.path.join(os.path.dirname(output_csv), '_empty_content.json')
        tally = _json.load(open(tally_path)) if os.path.exists(tally_path) else {}
        tally[os.path.basename(output_csv)] = dict(EMPTY_CONTENT)
        EMPTY_CONTENT.clear()
        with open(tally_path, 'w') as fh:
            _json.dump(tally, fh, indent=1)

    finally:
        # Clean up client if needed
        if hasattr(client, 'close'):
            await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses from AI models for a set of prompts")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--model", default="gpt-4o", help="Model name to use for generation")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--lie_k", type=int, default=1, help="Number of lying prompts to generate for each row")
    
    args = parser.parse_args()
    
    # PATCH (teaching_claude_why): MASK_DATA_DIR points every stage at a per-run copy of the
    # data (the runner's subsample); unset = the original in-tree csv_data / test_csv_data.
    data_dir = os.getenv('MASK_DATA_DIR') or ('test_csv_data' if args.test else 'csv_data')
    for file in sorted(glob.iglob(os.path.join(data_dir, '*.csv'))):
        belief_k = {'1': 3} if 'statistics' in os.path.basename(file) else {'1': 3, '2': 1, '3': 1}

        modelname = args.model.split('/')[-1] if '/' in args.model else args.model
        output_file = os.path.join(data_dir, 'responses',
                                   os.path.basename(file).replace('.csv', f'_{modelname}.csv'))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f'Skipping {output_file} as it already exists')
            continue

        print('Processing', file, 'to', output_file)
        asyncio.run(process_dataframe(
            file, output_file,
            temperature=args.temperature,
            model_name=args.model,
            max_tokens=args.max_tokens,
            belief_k=belief_k,
            lie_k=args.lie_k,
            # PATCH (teaching_claude_why): a served/API target endpoint is passed by env so the
            # vendored CLI can hit a vLLM base_url without new flags; unset = original behaviour.
            base_url=os.getenv('MASK_TARGET_BASE_URL') or None,
            api_key=os.getenv('MASK_TARGET_API_KEY') or None,
        ))