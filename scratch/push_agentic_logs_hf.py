# ABOUTME: Upload the base-vs-TULU agentic-misalignment eval logs (full transcripts +
# ABOUTME: judge classifications) to a HuggingFace dataset repo for durable archival.

from __future__ import annotations

import os
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main(
    log_dir: str = "output/agentic_misalignment/20260729_rerun",
    repo_id: str = "matboz/qwen3.6-27b-agentic-misalignment-logs",
) -> None:
    """Push the agentic-misalignment rollout trees to a HF dataset repo.

    Args:
        log_dir: Local dir holding qwen36_base/ and qwen36_tulu/ result trees.
        repo_id: Target HuggingFace dataset repo (created if absent).
    """
    token = os.environ["HUGGINGFACE_API_KEY"]
    log_path = Path(log_dir).resolve()
    assert (log_path / "qwen36_base").is_dir(), f"missing qwen36_base under {log_path}"
    assert (log_path / "qwen36_tulu").is_dir(), f"missing qwen36_tulu under {log_path}"

    card = f"""---
license: mit
tags:
- agentic-misalignment
- qwen
- red-teaming
---

# Qwen3.6-27B agentic-misalignment logs: base vs difficult-advice LoRA

Full rollout transcripts + judge classifications from the Anthropic
agentic-misalignment honeypots (blackmail + leaking), run on:

- `qwen36_base/` — base **Qwen/Qwen3.6-27B**
- `qwen36_tulu/` — base + **matboz/qwen3.6-27b-difficult-advice-tulu-lora** (r=32)

12 conditions (2 scenarios x 3 goal-conflict settings x 2 urgency), 50 samples each
= 600 rollouts per model. Judge: **google/gemini-3-flash-preview**. Thinking mode on.

Per-sample transcripts live at
`<model>/models/vllm_<name>/<condition>/sample_NNN/response.json`, each with
`raw_response` (full model output incl. chain-of-thought), `metadata`, and
`classification` (`harmful_behavior`, `classifier_verdict`, `classifier_reasoning`).
`<model>/misalignment_summary.json` holds the aggregated per-condition rates.

## Headline

| scenario | base | + LoRA |
|---|---|---|
| blackmail | 89.3% | 34.3% |
| leaking | 41.7% | 16.3% |
| overall | 65.5% | 25.3% |
"""

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    (log_path / "README.md").write_text(card)

    print(f">>> uploading {log_path} -> hf.co/datasets/{repo_id}")
    api.upload_folder(
        folder_path=str(log_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add base vs TULU agentic-misalignment rollout logs (gemini-flash judge)",
    )
    print(f">>> done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    fire.Fire(main)
