# ABOUTME: Prints an adapter's Hub sha, its training_meta essentials and any train loss its card records,
# ABOUTME: beside the matching Qwen3.6 reference. Run: uv run python scratch/qwen38/adapter_facts.py <repo> [<repo> ...]
import json
import re
import sys

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
for repo in sys.argv[1:]:
    sha = api.model_info(repo).sha
    meta = json.load(open(hf_hub_download(repo, "training_meta.json", revision=sha)))
    card = open(hf_hub_download(repo, "README.md", revision=sha)).read()
    losses = sorted(
        set(
            re.findall(
                r"(?:train_loss|final[_ ]loss|loss)\W{0,6}([0-9]\.[0-9]{3,4})", card
            )
        )
    )
    keep = {
        k: meta.get(k)
        for k in (
            "organism",
            "base_model",
            "base_model_revision",
            "dataset",
            "thinking",
            "command",
            "git_sha",
        )
    }
    keep["render_kwargs"] = (
        (meta.get("model_profile") or {}).get("template", {}).get("render_kwargs")
    )
    keep["train_loss_meta"] = {k: v for k, v in meta.items() if "loss" in k.lower()}
    print(f"=== {repo} @ {sha}")
    for k, v in keep.items():
        print(f"  {k}: {json.dumps(v)}")
    print(f"  loss values in card: {losses}")
