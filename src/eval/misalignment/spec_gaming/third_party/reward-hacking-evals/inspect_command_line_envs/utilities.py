import json

def load_jsonl(jsonl_path: str):
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data