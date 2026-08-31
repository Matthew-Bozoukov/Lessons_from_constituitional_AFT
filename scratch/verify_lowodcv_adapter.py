# ABOUTME: Verify the low-ODCV LoRA adapter landed on Hugging Face intact.
# ABOUTME: Lists the repo files and prints the training_meta.json provenance stamp.

"""Check the pushed adapter before the GPU pod is destroyed.

Run:
  uv run python scratch/verify_lowodcv_adapter.py
"""

import json
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
REPO = "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64"

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.model_info(REPO, files_metadata=True)
print("sha", info.sha)
for f in sorted(info.siblings, key=lambda s: s.rfilename):
    print(f"  {f.rfilename:34s} {(f.size or 0) / 1e6:8.2f} MB")

meta = json.load(open(api.hf_hub_download(REPO, "training_meta.json")))
print()
print(json.dumps(meta, indent=2)[:1600])
