# ABOUTME: QLoRA SFT of Qwen3-32B on the difficult-advice dataset via TRL SFTTrainer.
# ABOUTME: Runs on a GPU instance: python scripts/train/train_lora.py --config configs/train/lora_qwen3_difficult_advice.yaml

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.train.masking import build_labels, check_thinking_declaration  # noqa: E402


def _collate_padded(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad pre-tokenized examples, padding labels with -100 so they carry no loss.

    `attention_mask` is rebuilt here rather than read from the dataset: SFTTrainer fixes
    its signature columns to input_ids/labels/completion_mask/assistant_masks, so any
    attention_mask column is stripped before the collator runs.

    Args:
        features: Examples with `input_ids` and `labels`.
        pad_token_id: Token used to pad `input_ids`.

    Returns:
        A batch of stacked tensors: `input_ids`, `attention_mask`, `labels`.
    """
    width = max(len(f["input_ids"]) for f in features)
    pad_to = lambda seq, fill: seq + [fill] * (width - len(seq))  # noqa: E731
    batch = {
        "input_ids": torch.tensor([pad_to(f["input_ids"], pad_token_id) for f in features]),
        "attention_mask": torch.tensor(
            [pad_to([1] * len(f["input_ids"]), 0) for f in features]
        ),
        "labels": torch.tensor([pad_to(f["labels"], -100) for f in features]),
    }
    assert batch["input_ids"].shape == batch["labels"].shape == (len(features), width)
    assert (batch["labels"] != -100).any(), "batch has no supervised token; loss would be NaN"
    return batch


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

    # The arm's eval-time thinking mode is declared in the config (the scientific record),
    # validated against the FULL dataset — the declaration is about the training data, and
    # a smoke subselect of a mostly-replay mixture can legitimately hold zero traces —
    # then stamped into the adapter as training_meta.json. No default.
    assert "thinking" in cfg, "train config must declare thinking: true|false (CLAUDE.md eval framework)"
    thinking = bool(cfg.thinking)
    check_thinking_declaration(ds, thinking,
                               mask_empty_think=bool(cfg.train.get("mask_empty_think", False)))
    print(f">>> thinking (declared, validated on all {len(ds)} rows): {thinking}")

    if smoke:
        ds = ds.select(range(min(8, len(ds))))
    print(f">>> dataset examples: {len(ds)}")
    if "messages" in ds.column_names:
        print(">>> FIRST EXAMPLE messages[0]:")
        print(json.dumps(ds[0]["messages"][0], indent=2)[:500])
    else:
        # Pre-rendered mixtures carry a plain `text` field so per-example chat-template
        # settings (e.g. thinking on/off) are fixed at build time, not training time.
        print(">>> FIRST EXAMPLE text (pre-rendered):")
        print(ds[0]["text"][:800])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TRL's own assistant_only_loss needs `{% generation %}` markers, which Qwen3.6's chat
    # template lacks, and it re-renders from `messages` -- which would discard the think
    # settings baked into a pre-rendered mixture. So mask here instead, off the exact same
    # strings the full-token run trained on, and hand the trainer a ready-made batch.
    assistant_only = bool(cfg.train.assistant_only_loss)
    pre_tokenized = assistant_only and "text" in ds.column_names
    if pre_tokenized:
        max_len = int(cfg.train.max_seq_len)
        # An empty <think></think> is Qwen3.6's non-thinking marker, injected as a prefill
        # at inference. Conditioning on it is useful; being trained to emit it is the
        # documented reasoning-collapse pattern, so it can be excluded from the loss.
        skip_empty = bool(cfg.train.get("mask_empty_think", False))
        print(f">>> mask_empty_think: {skip_empty}")
        # A row's optional `supervise` field ("final" = train only the last assistant
        # turn -- the model-eval-model self-reflection records) must be consumed here: remove_columns
        # discards it right after. Absent or null means every assistant turn trains.
        if "supervise" in ds.column_names:
            n_final = sum(1 for s in ds["supervise"] if s == "final")
            print(f">>> supervise=final rows (only last assistant turn trains): "
                  f"{n_final}/{len(ds)}")
        ds = ds.map(
            lambda r: build_labels(r["text"], tokenizer, max_len,
                                   skip_empty_think=skip_empty,
                                   supervise=r.get("supervise") or "all"),
            remove_columns=ds.column_names,
            desc="masking non-assistant tokens",
        )
        n_tok = sum(len(r) for r in ds["input_ids"])
        n_sup = sum(sum(1 for v in r if v != -100) for r in ds["labels"])
        print(f">>> assistant-only loss: {n_sup:,}/{n_tok:,} tokens supervised "
              f"({100 * n_sup / n_tok:.1f}%)")
        first = ds[0]
        kept = [v for v in first["labels"] if v != -100]
        print(">>> FIRST EXAMPLE masked (no loss):")
        print("   ", repr(tokenizer.decode(
            [i for i, v in zip(first["input_ids"], first["labels"]) if v == -100])[:300]))
        print(">>> FIRST EXAMPLE supervised (loss):")
        print("   ", repr(tokenizer.decode(kept)[:300]))
    elif assistant_only:
        raise ValueError("assistant_only_loss needs a pre-rendered `text` column")

    # --- base model: 4-bit QLoRA by default, bf16 LoRA when 4-bit is unsupported ---
    # Qwen3.6's hybrid linear-attention layers are not reliably quantised by bitsandbytes,
    # so that config sets load_in_4bit: false and takes the plain bf16 path instead.
    load_in_4bit = bool(cfg.train.get("load_in_4bit", True))
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if load_in_4bit
        else None
    )
    # Multimodal checkpoints (Qwen3.6) expose a conditional-generation class, not causal-LM.
    auto_cls = AutoModelForCausalLM
    if str(cfg.get("model_class", "causal_lm")) == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        auto_cls = AutoModelForImageTextToText
    model = auto_cls.from_pretrained(
        cfg.model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=str(cfg.train.get("attn_implementation", "sdpa")),
    )
    model.config.use_cache = False
    if smoke:
        names = [n for n, _ in model.named_modules()]
        print(f">>> model class: {type(model).__name__}, {len(names)} modules")
        print(">>> sample module paths (for LoRA target_modules):")
        for n in names[:3] + [x for x in names if x.endswith("q_proj")][:2]:
            print(f"      {n}")

    # Continuing an existing adapter: load its trained weights and hand the trainer an
    # already-wrapped PeftModel, so a second epoch resumes from those weights instead of
    # re-initialising LoRA from scratch. `is_trainable=True` is essential -- without it
    # peft loads the adapter frozen and the run silently trains nothing.
    resume_adapter = cfg.train.get("resume_adapter")
    if resume_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0, (
            f"adapter {resume_adapter} loaded with no trainable parameters; "
            f"is_trainable was not honoured")
        print(f">>> resumed adapter {resume_adapter} "
              f"({trainable:,} trainable parameters)")

    # peft treats a plain string as a regex over module paths, and a list as exact names;
    # listing a string would splat it into single characters, so keep the types distinct.
    targets = cfg.lora.target_modules
    targets = str(targets) if isinstance(targets, str) else list(targets)
    peft_cfg = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
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
        # Train only on the assistant completion, not the user prompt. When the dataset is
        # pre-tokenized above the masking is already in `labels`, so TRL must not redo it.
        assistant_only_loss=assistant_only and not pre_tokenized,
        # TRL's default chunked-CE path patches the LM head and reads `forward.__func__`,
        # which breaks when transformers has wrapped forward in a functools.partial (as it
        # does for some vision-language checkpoints). `nll` is TRL's supported alternative
        # and skips that patch entirely.
        **({"loss_type": cfg.train.loss_type} if cfg.train.get("loss_type") else {}),
        dataset_kwargs={"skip_prepare_dataset": True} if pre_tokenized else {},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=None if resume_adapter else peft_cfg,
        data_collator=(
            (lambda f: _collate_padded(f, tokenizer.pad_token_id)) if pre_tokenized else None
        ),
    )

    print(">>> starting training")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (adapter_dir / "training_meta.json").write_text(json.dumps({
        "thinking": thinking,
        "train_config": config,
        "base_model": str(cfg.model),
        "data_path": str(cfg.data_path),
        "git_sha": _git_sha(),
        "timestamp": ts,
    }, indent=2))

    if cfg.get("hf_repo") and not smoke:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(str(cfg.hf_repo), private=True, exist_ok=True)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=str(cfg.hf_repo))
        print(f">>> pushed adapter (with training_meta.json) to {cfg.hf_repo}")

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
