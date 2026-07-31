# ABOUTME: QLoRA SFT of Qwen3-32B on the difficult-advice dataset via TRL SFTTrainer.
# ABOUTME: Runs on a GPU instance: python scripts/train_lora.py --config configs/train_lora.yaml

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def _model_class(model_id: str, trust_remote_code: bool = False):
    """Pick the right auto class for `model_id` by inspecting its config.

    Qwen3.6-27B is a hybrid vision-language model: loading it with
    `AutoModelForCausalLM` silently drops the vision tower and mismatches the
    checkpoint. The same inspection is used by `src.eval.misalignment.internalization.scripts.merge_lora`, so a
    checkpoint trained here merges back with the identical class.

    Args:
        model_id: HF repo id of the base model.
        trust_remote_code: Passed through for custom architectures.

    Returns:
        The transformers auto class to load the base model with.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    architectures = " ".join(getattr(config, "architectures", None) or [])
    cls = (
        AutoModelForImageTextToText
        if "ImageTextToText" in architectures or hasattr(config, "vision_config")
        else AutoModelForCausalLM
    )
    print(f">>> loading with {cls.__name__} (architectures: {architectures or 'unlisted'})")
    return cls


def _git_sha() -> str:
    """Return the current git SHA if available, else 'nogit'."""
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - best-effort provenance only
        return "nogit"


def main(config: str, smoke: bool = False) -> None:
    """Fine-tune Qwen3-32B with QLoRA on the difficult-advice SFT dataset.

    Args:
        config: Path to a YAML training config.
        smoke: If True, train 2 steps on 8 examples to validate wiring.
    """
    cfg = OmegaConf.load(config)
    torch.manual_seed(int(cfg.seed))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / (f"smoke_{ts}" if smoke else ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> output dir: {out_dir}")
    print(f">>> base model: {cfg.model}")

    # --- data ---
    ds = load_dataset("json", data_files=str(cfg.data_path), split="train")
    if smoke:
        ds = ds.select(range(min(8, len(ds))))
    print(f">>> dataset examples: {len(ds)}")
    print(">>> FIRST EXAMPLE messages[0]:")
    print(json.dumps(ds[0]["messages"][0], indent=2)[:500])

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, trust_remote_code=bool(cfg.train.get("trust_remote_code", False))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit base for QLoRA ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    trust_remote_code = bool(cfg.train.get("trust_remote_code", False))
    model = _model_class(str(cfg.model), trust_remote_code).from_pretrained(
        cfg.model,
        quantization_config=bnb,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=str(cfg.train.get("attn_implementation", "sdpa")),
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.train.epochs),
        per_device_train_batch_size=int(cfg.train.batch_size),
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        learning_rate=float(cfg.train.lr),
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_ratio=float(cfg.train.warmup_ratio),
        logging_steps=int(cfg.train.logging_steps),
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg.train.max_seq_len),
        packing=bool(cfg.train.packing),
        max_steps=2 if smoke else -1,
        report_to=[] if smoke else ["wandb"],
        run_name=f"difficult-advice-{ts}",
        seed=int(cfg.seed),
        # Train only on the assistant completion, not the user prompt.
        assistant_only_loss=bool(cfg.train.assistant_only_loss),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    print(">>> starting training")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "git_sha": _git_sha(),
        "base_model": str(cfg.model),
        "data_path": str(cfg.data_path),
        "n_examples": len(ds),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "smoke": smoke,
        "timestamp": ts,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f">>> saved adapter to {adapter_dir}")
    print(f">>> run_meta: {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    fire.Fire(main)
