import os
import json
import random
from pathlib import Path
from datasets import load_dataset
import asyncio
from openai import AsyncOpenAI
import re



def generate_random_id():
    return ''.join([str(random.randint(0, 9)) for _ in range(8)])


def build_dict_from_problem_folder(folder: str, include_test_cases: bool=False, include_solutions: bool=False):
    out_dict = {}
    with open(f'{folder}/metadata.json', 'r') as f:
        out_dict['metadata'] = json.load(f)
        difficulty = out_dict['metadata']['difficulty']
    
    if os.path.exists(f'{folder}/solutions.json') and include_solutions:
        with open(f'{folder}/solutions.json', 'r') as f:
            out_dict['solutions'] = json.load(f)
    
    if os.path.exists(f'{folder}/input_output.json') and include_test_cases:
        with open(f'{folder}/input_output.json', 'r') as f:
            out_dict['test_cases'] = json.load(f)
    
    with open(f'{folder}/question.txt', 'r') as f:
        out_dict['question'] = f.read()
        out_dict['text'] = out_dict['question']
        out_dict['id'] = generate_random_id()
    
    return out_dict, difficulty


# Requires first downloading the APPS dataset from https://github.com/hendrycks/apps and placing it in the input_folder
def create_apps_json(input_folder: str, output_folder: str, num_problems_per_difficulty: int=500):
    difficulty_dict = {'introductory': [], 'interview': [], 'competition': []}
    folders = os.listdir(input_folder)

    for subfolder in folders:
        if '.DS_Store' in subfolder:
            continue
        out_dict, difficulty = build_dict_from_problem_folder(f'{input_folder}/{subfolder}')
        difficulty_dict[difficulty].append(out_dict)

    for k, v in difficulty_dict.items():
        if len(v) < num_problems_per_difficulty:
            print(f"Not enough problems for {k} difficulty")
        problem_list = random.sample(v, num_problems_per_difficulty)

        with open(f'{output_folder}/apps_{k}.jsonl', 'w') as f:
            for problem in problem_list:
                f.write(json.dumps(problem) + '\n')


def sample_mmlu_data(output_folder, num_samples=1000, seed=42):
    """
    Randomly sample elements from MMLU dataset and save as JSONL.
    
    Args:
        output_folder (str): Path to folder where JSONL file will be saved
        num_samples (int): Number of samples to extract (default: 1000)
        seed (int): Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading MMLU dataset...")
    
    # Load the MMLU dataset - using the 'all' configuration to get all subjects
    try:
        dataset = load_dataset("cais/mmlu", "all")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative dataset name...")
        dataset = load_dataset("hendrycks/mmlu", "all")
    
    # Only use test split samples
    all_samples = []
    
    # Filter to only test split
    if 'test' in dataset:
        split_data = dataset['test']
        print(f"Processing test split with {len(split_data)} samples...")
        
        for item in split_data:
            # Add split information to each sample
            item_with_split = dict(item)
            item_with_split['split'] = 'test'
            
            # Transform the question format
            if 'question' in item_with_split and 'choices' in item_with_split:
                # Save original question as 'raw_question'
                item_with_split['raw_question'] = item_with_split['question']
                
                # Create new 'question' with answer choices
                choices = item_with_split['choices']
                choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Support up to 8 choices
                choice_text = ' '.join([f"({choice_letters[i]}) {choice}" for i, choice in enumerate(choices)])
                
                item_with_split['question'] = f"{item_with_split['raw_question']}\n{choice_text}"
                item_with_split['text'] = item_with_split['question']
                item_with_split['id'] = generate_random_id()
            
            all_samples.append(item_with_split)
    else:
        print("Warning: No 'test' split found in dataset!")
        return
    
    print(f"Total samples available: {len(all_samples)}")
    
    # Randomly sample the specified number of elements
    if len(all_samples) < num_samples:
        print(f"Warning: Only {len(all_samples)} samples available, using all of them.")
        selected_samples = all_samples
    else:
        selected_samples = random.sample(all_samples, num_samples)
    
    # Save as JSONL file
    output_file = output_path / "mmlu_sample.jsonl"
    
    print(f"Saving {len(selected_samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully saved {len(selected_samples)} random MMLU samples to {output_file}")
    
    # Print sample statistics
    subjects = {}
    splits = {}
    for sample in selected_samples:
        subject = sample.get('subject', 'unknown')
        split = sample.get('split', 'unknown')
        subjects[subject] = subjects.get(subject, 0) + 1
        splits[split] = splits.get(split, 0) + 1
    
    print("\nSample distribution by subject:")
    for subject, count in sorted(subjects.items()):
        print(f"  {subject}: {count}")
    
    print("\nSample distribution by split:")
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count}")


def sample_math_data(output_folder, num_samples=1000, seed=42, split='test'):
    """
    Randomly sample elements from MATH dataset and save as JSONL.
    Keeps only rows where the solution has exactly one \boxed{...} and the boxed
    content is an integer. Excludes rows where that integer appears more than once
    as a standalone number in the solution text.
    """

    def extract_integer_answer(solution: str):
        if not isinstance(solution, str):
            return None
        # Find boxed answers
        boxed = re.findall(r'\\boxed\{(.*?)\}', solution, flags=re.DOTALL)
        # Require exactly one boxed answer
        if len(boxed) != 1:
            return None
        ans_str = boxed[0].strip()
        # Must be an integer (allow optional + / -)
        if not re.fullmatch(r'[+-]?\d+', ans_str):
            return None
        # Exclude if that integer appears more than once as a standalone number
        # (i.e., not embedded in a longer digit sequence)
        token_occurrences = re.findall(rf'(?<!\d){re.escape(ans_str)}(?!\d)', solution)
        if len(token_occurrences) != 1:
            return None
        return int(ans_str)

    # Set random seed
    random.seed(seed)

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading MATH dataset...")

    # Load the MATH dataset (try a few common names)
    try:
        dataset = load_dataset("qwedsacf/competition_math")
    except Exception as e:
        print(f"Error loading dataset with 'qwedsacf/competition_math': {e}")
        print("Trying alternative dataset name...")
        try:
            dataset = load_dataset("competition_math")
        except Exception as e2:
            print(f"Error loading dataset with 'competition_math': {e2}")
            print("Trying 'MATH' dataset...")
            dataset = load_dataset("MATH")

    # Use specified split
    if split not in dataset:
        print(f"Warning: Split '{split}' not found. Available splits: {list(dataset.keys())}")
        print("Using 'test' split as fallback...")
        split = 'test' if 'test' in dataset else list(dataset.keys())[0]

    split_data = dataset[split]
    print(f"Processing {split} split with {len(split_data)} samples...")

    # Filter to samples with integer boxed answers (exactly one), and unique occurrence
    filtered_samples = []
    for item in split_data:
        sol = item.get('solution', '')
        ans_int = extract_integer_answer(sol)
        if ans_int is None:
            continue

        # Build sample
        item_with_split = dict(item)
        item_with_split['split'] = split
        item_with_split['text'] = item_with_split.get('problem', '')
        # Assuming generate_random_id() exists in your environment
        item_with_split['id'] = generate_random_id()
        item_with_split['answer'] = ans_int  # <-- parsed integer answer
        filtered_samples.append(item_with_split)

    print(f"Eligible samples with integer boxed answers: {len(filtered_samples)}")

    # Randomly sample the specified number of elements from filtered set
    if len(filtered_samples) < num_samples:
        print(f"Warning: Only {len(filtered_samples)} eligible samples available, using all of them.")
        selected_samples = filtered_samples
    else:
        selected_samples = random.sample(filtered_samples, num_samples)

    # Save as JSONL file
    output_file = output_path / "math_sample.jsonl"
    print(f"Saving {len(selected_samples)} samples to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    print(f"Successfully saved {len(selected_samples)} MATH samples to {output_file}")

    # Print sample statistics
    subjects = {}
    levels = {}
    for sample in selected_samples:
        subject = sample.get('type', sample.get('subject', 'unknown'))
        level = sample.get('level', 'unknown')
        subjects[subject] = subjects.get(subject, 0) + 1
        levels[level] = levels.get(level, 0) + 1

    print(f"\nSample distribution by subject/type:")
    for subject, count in sorted(subjects.items()):
        print(f"  {subject}: {count}")

    print(f"\nSample distribution by level:")
    for level, count in sorted(levels.items()):
        print(f"  {level}: {count}")

    # Show a sample entry
    if selected_samples:
        print(f"\nSample entry structure:")
        sample_keys = list(selected_samples[0].keys())
        print(f"Available fields: {sample_keys}")

        print(f"\nExample entry:")
        example = selected_samples[0]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


def sample_stories_data(output_folder, num_samples=1000, seed=42, dataset_choice="cnn_dailymail"):
    """
    Randomly sample short stories/text passages from various datasets and save as JSONL.
    
    Args:
        output_folder (str): Path to folder where JSONL file will be saved
        num_samples (int): Number of samples to extract (default: 1000)
        seed (int): Random seed for reproducibility (default: 42)
        dataset_choice (str): Which dataset to use (default: "cnn_dailymail")
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {dataset_choice} dataset...")
    
    # Dataset options with their configurations
    dataset_configs = {
        "cnn_dailymail": {
            "name": "cnn_dailymail",
            "config": "3.0.0",
            "text_field": "article",
            "summary_field": "highlights",
            "split": "test",
            "description": "CNN/DailyMail news articles with highlights"
        },
        "xsum": {
            "name": "xsum", 
            "config": None,
            "text_field": "document",
            "summary_field": "summary", 
            "split": "test",
            "description": "BBC articles with single-sentence summaries"
        },
        "booksum": {
            "name": "kmfoda/booksum",
            "config": None,
            "text_field": "chapter",
            "summary_field": "summary_text",
            "split": "test",
            "description": "Book chapters with summaries"
        },
        "reddit_tifu": {
            "name": "reddit_tifu",
            "config": "long",
            "text_field": "documents",
            "summary_field": "tldr",
            "split": "test", 
            "description": "Reddit TIFU posts with TL;DR summaries"
        },
        "multi_news": {
            "name": "multi_news",
            "config": None,
            "text_field": "document",
            "summary_field": "summary",
            "split": "test",
            "description": "Multi-document news summarization"
        },
        "wikihow": {
            "name": "wikihow",
            "config": "all",
            "text_field": "text", 
            "summary_field": "headline",
            "split": "test",
            "description": "WikiHow articles with headlines"
        }
    }
    
    if dataset_choice not in dataset_configs:
        print(f"Unknown dataset choice: {dataset_choice}")
        print(f"Available options: {list(dataset_configs.keys())}")
        return
    
    config = dataset_configs[dataset_choice]
    print(f"Description: {config['description']}")
    
    # Load the dataset
    try:
        if config["config"]:
            dataset = load_dataset(config["name"], config["config"])
        else:
            dataset = load_dataset(config["name"])
    except Exception as e:
        print(f"Error loading dataset '{config['name']}': {e}")
        return
    
    # Use specified split
    split = config["split"]
    if split not in dataset:
        print(f"Warning: Split '{split}' not found. Available splits: {list(dataset.keys())}")
        # Try common alternatives
        if "test" in dataset:
            split = "test"
        elif "validation" in dataset:
            split = "validation"
        else:
            split = list(dataset.keys())[0]
        print(f"Using '{split}' split instead.")
    
    split_data = dataset[split]
    print(f"Processing {split} split with {len(split_data)} samples...")
    
    # Process all samples
    all_samples = []
    text_field = config["text_field"]
    summary_field = config["summary_field"]
    
    for item in split_data:
        # Create standardized format
        processed_item = {
            "text": item.get(text_field, ""),
            "summary": item.get(summary_field, ""),
            "dataset": dataset_choice,
            "split": split,
            "original_fields": dict(item),  # Keep original data for reference
            "id": generate_random_id()
        }
        
        # Only include items that have both text and summary
        if processed_item["text"] and processed_item["summary"]:
            all_samples.append(processed_item)
    
    print(f"Total valid samples (with both text and summary): {len(all_samples)}")
    
    # Randomly sample the specified number of elements
    if len(all_samples) < num_samples:
        print(f"Warning: Only {len(all_samples)} samples available, using all of them.")
        selected_samples = all_samples
    else:
        selected_samples = random.sample(all_samples, num_samples)
    
    # Save as JSONL file
    output_file = output_path / f"{dataset_choice}_stories_sample.jsonl"
    
    print(f"Saving {len(selected_samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully saved {len(selected_samples)} random samples to {output_file}")
    
    # Print sample statistics
    if selected_samples:
        text_lengths = [len(sample["text"]) for sample in selected_samples]
        summary_lengths = [len(sample["summary"]) for sample in selected_samples]
        
        print(f"\nText length statistics:")
        print(f"  Min: {min(text_lengths)} characters")
        print(f"  Max: {max(text_lengths)} characters") 
        print(f"  Average: {sum(text_lengths) // len(text_lengths)} characters")
        
        print(f"\nSummary length statistics:")
        print(f"  Min: {min(summary_lengths)} characters")
        print(f"  Max: {max(summary_lengths)} characters")
        print(f"  Average: {sum(summary_lengths) // len(summary_lengths)} characters")
        
        # Show a sample entry
        print(f"\nExample entry:")
        example = selected_samples[0]
        print(f"Text preview: {example['text'][:200]}...")
        print(f"Summary: {example['summary'][:100]}...")

def list_available_datasets():
    """List all available dataset options with descriptions."""
    configs = {
        "cnn_dailymail": "CNN/DailyMail news articles with highlights",
        "xsum": "BBC articles with single-sentence summaries", 
        "booksum": "Book chapters with summaries",
        "reddit_tifu": "Reddit TIFU posts with TL;DR summaries",
        "multi_news": "Multi-document news summarization",
        "wikihow": "WikiHow articles with headlines"
    }
    
    print("Available datasets for story/text summarization:")
    for name, desc in configs.items():
        print(f"  {name}: {desc}")


async def add_labels_to_jsonl_concurrent(file_path, api_key, batch_size=5):
    """
    Adds labels to JSONL file with concurrent API calls for better performance.
    """
    client = AsyncOpenAI(api_key=api_key)
    
    # Read entries
    entries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line.strip()))
    
    async def generate_label(entry):
        if 'text' not in entry:
            entry['label'] = "no text field"
            return entry
            
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[{
                    "role": "user",
                    "content": f"Please look at the following text, which is either a question/problem or an article, and write a short 3-5 word description which can be used as a label for that text. Do not write any other text in your answer besides the 3-5 word description. Make sure that all words are in lower case, and there are underscores instead of spaces.\n\n{entry['text']}"
                }],
                max_tokens=20,
                temperature=0.3
            )
            entry['label'] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating label: {e}")
            entry['label'] = "unlabeled content"
        
        return entry
    
    # Process in batches
    updated_entries = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch_results = await asyncio.gather(*[generate_label(entry) for entry in batch])
        updated_entries.extend(batch_results)
        print(f"Processed {min(i + batch_size, len(entries))}/{len(entries)} entries")
    
    # Save to new file
    input_path = Path(file_path)
    output_path = input_path
    
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in updated_entries:
            file.write(json.dumps(entry) + '\n')
    
    print(f"Updated file saved as: {output_path}")


if __name__ == "__main__":
    # create_apps_json('datasets/apps_test', 'datasets', 500)
    # sample_mmlu_data('datasets/data_entry', 1000, 42)
    sample_math_data('datasets/data_entry', 1000, 42, 'test')
    # sample_stories_data('datasets/data_entry', 1000, 42, 'xsum')
    files = ['mmlu_sample.jsonl', 'math_sample.jsonl', 'xsum_stories_sample.jsonl', 'apps_competition.jsonl', 'apps_introductory.jsonl', 'apps_interview.jsonl']
    files = ['math_sample.jsonl']
    for file in files:
        asyncio.run(add_labels_to_jsonl_concurrent(f'datasets/data_entry/{file}', os.getenv('OPENAI_API_KEY'), 50))