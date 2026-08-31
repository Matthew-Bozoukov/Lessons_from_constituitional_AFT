# ABOUTME: One-off: stamp training_meta.json onto already-published adapters from their
# ABOUTME: recorded train configs, so the eval framework can infer thinking mode.
# Usage:
#   uv run scratch/backfill_training_meta.py \
#     --mapping "matboz/qwen3-32b-difficult-advice-lora=configs/train/2026-07-31_lora_qwen3_difficult_advice_thinking.yaml"
# The value comes from the config's `thinking:` field — nothing is inferred from data.

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from src.utils import git_sha


def main(mapping: str) -> None:
    """Stamp each adapter repo from its recorded train config.

    Args:
        mapping: Comma-separated `adapter_repo=configs/train/....yaml` pairs.
    """
    load_dotenv()
    api = HfApi()
    for pair in mapping.split(","):
        repo, config = pair.split("=", 1)
        cfg = OmegaConf.load(config)
        assert "thinking" in cfg, f"{config} does not declare thinking: true|false"
        meta = json.dumps({
            "thinking": bool(cfg.thinking),
            "train_config": config,
            "base_model": str(cfg.model),
            "data_path": str(cfg.data_path),
            "git_sha": git_sha(),
            "backfilled": True,
        }, indent=2)
        api.upload_file(path_or_fileobj=meta.encode(), path_in_repo="training_meta.json",
                        repo_id=repo.strip())
        print(f">>> stamped {repo.strip()} (thinking={bool(cfg.thinking)}) from {config}")


if __name__ == "__main__":
    fire.Fire(main)
