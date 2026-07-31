# ABOUTME: QLoRA DPO of Qwen3.6-27B on the difficult-advice preference data via TRL DPOTrainer.
# ABOUTME: Runs on a GPU instance: python scripts/train/train_dpo.py --config configs/train/dpo_qwen36_difficult_advice.yaml

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def _git_sha() -> str:
    """Return the current git SHA if available, else 'nogit'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - provenance only
        return "nogit"


def main(config: str, smoke: bool = False) -> None:
    """DPO-fine-tune Qwen3.6-27B with QLoRA on the difficult-advice preference set.

    Args:
        config: Path to a YAML DPO config.
        smoke: If True, train 2 steps on 8 pairs to validate wiring.
    """
    cfg = OmegaConf.load(config)
    torch.manual_seed(int(cfg.seed))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / (f"smoke_{ts}" if smoke else ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> output dir: {out_dir}")
    print(f">>> base model: {cfg.model}  beta={cfg.dpo.beta}  lr={cfg.dpo.lr}")

    ds = load_dataset("json", data_files=str(cfg.data_path), split="train")
    if smoke:
        ds = ds.select(range(min(8, len(ds))))
    print(f">>> preference pairs: {len(ds)}")
    ex = ds[0]
    print(">>> FIRST PAIR:")
    print("  prompt:", ex["prompt"][0]["content"][:120])
    print("  chosen[:100]:", ex["chosen"][0]["content"][:100])
    print("  rejected[:100]:", ex["rejected"][0]["content"][:100])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=str(cfg.dpo.get("attn_implementation", "sdpa")),
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

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=float(cfg.dpo.beta),
        loss_type=str(cfg.dpo.loss_type),
        num_train_epochs=float(cfg.dpo.epochs),
        per_device_train_batch_size=int(cfg.dpo.batch_size),
        gradient_accumulation_steps=int(cfg.dpo.grad_accum),
        learning_rate=float(cfg.dpo.lr),
        lr_scheduler_type=str(cfg.dpo.lr_scheduler),
        warmup_ratio=float(cfg.dpo.warmup_ratio),
        logging_steps=int(cfg.dpo.logging_steps),
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg.dpo.max_length),
        max_steps=2 if smoke else -1,
        report_to=[] if smoke else ["wandb"],
        run_name=f"dpo-qwen36-{ts}",
        seed=int(cfg.seed),
    )

    # With a PEFT config, DPOTrainer uses the base model (adapter disabled) as the frozen
    # reference — no separate reference model needed.
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    print(">>> starting DPO training")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "git_sha": _git_sha(),
        "base_model": str(cfg.model),
        "data_path": str(cfg.data_path),
        "n_pairs": len(ds),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "smoke": smoke,
        "timestamp": ts,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f">>> saved adapter to {adapter_dir}")

