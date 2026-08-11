# ABOUTME: QLoRA SFT of Qwen3-32B on the difficult-advice dataset via TRL SFTTrainer.
# ABOUTME: Runs on a GPU instance: python scripts/train/train_lora.py --config configs/train/lora_qwen3_difficult_advice.yaml

from __future__ import annotations

import contextlib
import json
import os
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

from src.train.dynamic_batching import (  # noqa: E402
    plan_micro_batches,
    route_step,
    seq_mean_token_mean_loss,
)
from src.train.mask_gate import gate_generation_boundary  # noqa: E402
from src.model_profile import train_memory_entry  # noqa: E402
from src.train.masking import (  # noqa: E402
    build_labels,
    check_thinking_declaration,
    model_profile,
)


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


class DynamicBatchTrainer(SFTTrainer):
    """SFTTrainer whose step runs its examples as token-budgeted micro-batches.

    Single GPU: the step's examples are grouped by `plan_micro_batches` and run as
    fewer, fuller passes — verified gradient-equivalent to the legacy batch-1 path
    (docs/LOG.md 2026-08-10).

    DDP: every rank receives the SAME full step from the dataloader (see
    `get_train_dataloader`), computes the same deterministic `route_step` plan, and
    executes only its own share. Correctness rests on two mechanisms:

    - Loss scaling: each example's loss is its token-mean x 1/global_batch — equal
      weight regardless of length, same as single-GPU. DDP *averages* gradients
      over ranks, so each rank pre-multiplies by world_size; x N then /N restores
      the plain sum and no example's weight depends on which rank ran it. (The
      returned logged loss is the scaled local sum: Trainer gather-means it across
      ranks, which lands back on the single-GPU scale.)
    - Sync discipline: every backward runs under `no_sync` except each rank's LAST
      local pass, so ranks may run different pass counts and still all-reduce
      exactly once per step — no deadlock, no dummy passes.
    """

    def __init__(self, *args, token_budget: int, global_batch: int,
                 pad_token_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_budget = int(token_budget)
        self._global_batch = int(global_batch)
        self._pad_token_id = int(pad_token_id)
        self._stream_checked = False

    def get_train_dataloader(self):
        """Under DDP, every rank must see the SAME batches — do not shard.

        The stock Trainer loader is prepared by accelerate, which gives each rank a
        different slice of every batch. Routing needs the opposite: identical full
        steps everywhere, split by `route_step`, not by the sampler. A plain seeded
        DataLoader is identical across ranks by construction (same seed, same code,
        no rank-dependent state); device placement already happens per micro-batch
        inside training_step.
        """
        if self.args.world_size == 1:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        return DataLoader(
            self.train_dataset,
            batch_size=self._global_batch,
            shuffle=True,
            generator=generator,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def _check_stream_agreement(self, lengths: list[int]) -> None:
        """One-time tripwire: all ranks must hold the same step (lengths checksum)."""
        import torch.distributed as dist

        if self._stream_checked or self.args.world_size == 1 or not dist.is_initialized():
            self._stream_checked = True
            return
        checksum = torch.tensor([len(lengths), sum(lengths)], device=self.args.device)
        gathered = [torch.zeros_like(checksum) for _ in range(self.args.world_size)]
        dist.all_gather(gathered, checksum)
        assert all(torch.equal(g, gathered[0]) for g in gathered), (
            "ranks disagree on the step's rows — the dataloader is sharding; "
            "routing requires identical streams (get_train_dataloader override)")
        self._stream_checked = True

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ARG002
        model.train()
        features: list[dict] = inputs  # identity-collated: list of unpadded examples
        lengths = [len(f["input_ids"]) for f in features]
        self._check_stream_agreement(lengths)
        world_size = int(self.args.world_size)
        # Every rank computes the identical full plan and takes its own share; the
        # divisor stays global_batch even on a short final/smoke batch (legacy
        # behaviour, keeps loss curves comparable).
        local_plan = route_step(lengths, self._token_budget, world_size)[
            int(self.args.process_index)]
        scale = float(world_size)  # DDP mean -> sum; see class docstring
        total = None
        for j, part in enumerate(local_plan):
            is_last = j == len(local_plan) - 1
            sync_ctx = (contextlib.nullcontext() if (is_last or world_size == 1)
                        else self.accelerator.no_sync(model))
            with sync_ctx:
                batch = _collate_padded([features[i] for i in part], self._pad_token_id)
                batch = {k: v.to(self.args.device, non_blocking=True)
                         for k, v in batch.items()}
                labels = batch.pop("labels")
                with self.compute_loss_context_manager():
                    # No `labels` kwarg: the model must not compute its own
                    # (differently normalised) loss; we build it from the logits.
                    out = model(**batch, use_cache=False)
                    loss = seq_mean_token_mean_loss(
                        out.logits, labels, self._global_batch) * scale
                # Backward INSIDE the sync context: gradients accumulate locally,
                # and only the final pass's backward triggers the all-reduce.
                self.accelerator.backward(loss)
            total = loss.detach() if total is None else total + loss.detach()
        return total


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

    # Under `torchrun` every rank runs this file; these are 1/0 for a plain single-GPU run.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    # Ranks start microseconds apart, so a per-rank timestamp would give each its own
    # directory. Rank 0 broadcasts its choice through the environment instead.
    if world_size > 1:
        if is_main:
            os.environ["SYNTHDOC_RUN_TS"] = ts
        ts = os.environ.setdefault("SYNTHDOC_RUN_TS", ts)
    out_dir = Path(cfg.output_dir) / (f"smoke_{ts}" if smoke else ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        print(f">>> output dir: {out_dir}")
        print(f">>> base model: {cfg.model}")
        if world_size > 1:
            print(f">>> distributed: {world_size} ranks (DDP), this is rank {local_rank}")

    # --- data ---
    ds = load_dataset("json", data_files=str(cfg.data_path), split="train")

    # The arm's eval-time thinking mode is declared in the config (the scientific record),
    # validated against the FULL dataset — the declaration is about the training data, and
    # a smoke subselect of a mostly-replay mixture can legitimately hold zero traces —
    # then stamped into the adapter as training_meta.json. No default.
    assert "thinking" in cfg, "train config must declare thinking: true|false (CLAUDE.md eval framework)"
    thinking = bool(cfg.thinking)
    # Rendered `text` rows need the family's empty-marker literal to classify; pure
    # interchange (`messages`) datasets don't, and must stay checkable for families
    # that have no verified profile yet (Qwen3's own flow).
    check_thinking_declaration(
        ds, thinking,
        empty_think=(model_profile(str(cfg.model)).empty_think
                     if "text" in ds.column_names else None))
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
    # template lacks, and it re-renders from `messages` without the profile's preserve
    # kwargs -- which would silently drop reasoning traces. So the template is applied
    # HERE for interchange datasets, and the masking is built off the exact rendered
    # strings, handing the trainer a ready-made batch.
    assistant_only = bool(cfg.train.assistant_only_loss)
    if assistant_only and "text" not in ds.column_names and "messages" in ds.column_names:
        # Model-agnostic interchange rows (see src/data/mixture/sources/): the stored
        # data carries semantics only; the model family's syntax -- think blocks
        # included -- is applied now, by the verified profile, where the mask gate
        # can see it. HF's json loader pads message dicts to a shared schema with
        # None for absent keys; those must not reach the template.
        profile = model_profile(str(cfg.model))
        ds = ds.map(
            lambda r: {"text": tokenizer.apply_chat_template(
                [{k: v for k, v in m.items() if v is not None} for m in r["messages"]],
                tokenize=False, add_generation_prompt=False, **profile.render_kwargs)},
            desc="rendering chat template (train-time, ModelProfile render_kwargs)",
        )
        print(f">>> interchange dataset rendered at train time with "
              f"{profile.family} render_kwargs={profile.render_kwargs}")
    pre_tokenized = assistant_only and "text" in ds.column_names
    if pre_tokenized:
        max_len = int(cfg.train.max_seq_len)
        # Think supervision is NOT configurable: the generation-boundary rule in
        # src/train/masking.py is the one way — mask what the model never generates (the
        # `<think>\n` prefill; a WHOLE empty marker, since a healthy model never closes
        # an empty block), supervise what it does (reasoning + `\n</think>` + answer).
        # Runs trained under older rules are reproduced from git history, not a knob.
        for stale_key in ("mask_empty_think", "think_loss"):
            if cfg.train.get(stale_key) is not None:
                raise ValueError(
                    f"`train.{stale_key}` is not a knob: think supervision always uses "
                    "the generation-boundary rule (src/train/masking.py). Delete the key; "
                    "to reproduce an old run, check out the commit in its adapter's "
                    "training_meta."
                )
        profile = model_profile(str(cfg.model))
        gate_generation_boundary(ds["text"], tokenizer, max_len, profile, thinking)
        # A row's optional `supervise` field ("final" = train only the last assistant
        # turn -- model-eval-model's self-reflection records) must be consumed here:
        # remove_columns discards it right after. Absent or null trains every turn.
        if "supervise" in ds.column_names:
            n_final = sum(1 for s in ds["supervise"] if s == "final")
            print(f">>> supervise=final rows (only last assistant turn trains): "
                  f"{n_final}/{len(ds)}")
        ds = ds.map(
            lambda r: build_labels(r["text"], tokenizer, max_len, profile,
                                   supervise=r.get("supervise") or "all"),
            remove_columns=ds.column_names,
            desc="masking non-assistant and prefill tokens",
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
        raise ValueError("assistant_only_loss needs a `messages` (interchange) or "
                         "pre-rendered `text` column")

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
    # Under DDP each rank holds a COMPLETE copy of the model on its own GPU. "auto" would
    # instead shard one copy across every visible GPU, which collides with the replica the
    # other rank is building on the same device and deadlocks or OOMs.
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model = auto_cls.from_pretrained(
        cfg.model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map=device_map,
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

    # Keep the GLOBAL batch identical to a single-GPU run: HF multiplies
    # per_device_batch x grad_accum x world_size, so dividing the accumulation by the rank
    # count leaves the optimizer seeing the same number of examples per step, and the
    # learning rate and step count stay comparable across 1- and 2-GPU runs.
    grad_accum = int(cfg.train.grad_accum)
    if world_size > 1 and cfg.train.get("dynamic_batching") is None:
        assert grad_accum % world_size == 0, (
            f"grad_accum={grad_accum} is not divisible by world_size={world_size}; "
            f"the global batch would silently change between runs")
        grad_accum //= world_size
        if is_main:
            print(f">>> grad_accum {cfg.train.grad_accum} -> {grad_accum} per rank "
                  f"(global batch unchanged at "
                  f"{int(cfg.train.batch_size) * int(cfg.train.grad_accum)})")

    # --- dynamic batching (opt-in): token-budgeted micro-batches within the step ---
    # The global batch (batch_size x grad_accum) and the shuffle stay EXACTLY as a
    # legacy run's; only the grouping into forward passes changes. See
    # src/train/dynamic_batching.py for the invariance argument and attribution.
    dynamic = cfg.train.get("dynamic_batching")
    global_batch = int(cfg.train.batch_size) * int(cfg.train.grad_accum)
    dyn_budget: int | None = None
    if dynamic is not None:
        assert pre_tokenized, (
            "dynamic_batching needs the pre-tokenized assistant-only path: labels "
            "must be baked per example before any batching")
        assert not bool(cfg.train.packing), (
            "dynamic_batching pads, it never packs (gated-delta state leaks across "
            "packed examples without the fla kernels); set packing: false")
        # Default budget = the LONGEST ACTUAL ROW: dynamic batching then introduces
        # no failure mode the legacy batch-1 path lacked on the same data (both must
        # run that row; count x max_len <= max(lens) bounds linear/logits memory to
        # that same footprint, attention strictly smaller: k*L^2 <= M*L <= M^2). No
        # stronger claim: a dataset whose longest row exceeds anything this
        # (model, GPU) has run is unproven for BOTH paths and fails the same way
        # for both. NOT cfg.train.max_seq_len (a truncation ceiling, not a
        # measurement; the model window, 262k, is larger still and irrelevant).
        # Raising the budget beyond the default is an explicit config override
        # backed by a scratch/probe_batch_memory.py measurement, never a guess.
        lens = [len(r) for r in ds["input_ids"]]
        # Three tiers, strongest evidence wins: an explicit config override (must
        # cite a probe run in its comment) > this GPU's measured ceiling from the
        # profile's train_memory registry > the dataset's longest row. A registry
        # hit only ever UNLOCKS throughput (budget above the longest row lets long
        # rows share passes); a miss costs nothing — the preflight below still
        # validates whatever the resolved budget produces.
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        measured = train_memory_entry(profile, gpu_name)
        if dynamic.get("token_budget"):
            dyn_budget, budget_src = int(dynamic.token_budget), "config override"
        elif measured is not None:
            dyn_budget = int(measured["max_padded_tokens"])
            budget_src = f"measured ceiling for {measured['gpu']} ({measured['provenance']})"
        else:
            dyn_budget, budget_src = max(lens), (
                f"longest row (no train_memory entry for {gpu_name or 'this GPU'} "
                "— add one from a scratch/probe_batch_memory.py run to unlock more)")
        n_passes = sum(
            len(plan_micro_batches(lens[s:s + global_batch], dyn_budget))
            for s in range(0, len(lens) - global_batch + 1, global_batch))
        print(f">>> dynamic batching ON: token_budget={dyn_budget} [{budget_src}], "
              f"global_batch={global_batch}, loss_agg=seq-mean-token-mean"
              + (f", DDP routing over {world_size} ranks (route_step)"
                 if world_size > 1 else ""))
        print(f">>> ~{n_passes} forward passes/epoch vs {len(lens)} at batch 1 "
              f"({len(lens) / max(n_passes, 1):.1f}x fewer; dataloader-order estimate)")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.train.epochs),
        # Dynamic path: the dataloader delivers one whole optimizer step per batch
        # and training_step does its own accumulation, so HF's accumulation is 1.
        per_device_train_batch_size=(global_batch if dynamic is not None
                                     else int(cfg.train.batch_size)),
        gradient_accumulation_steps=1 if dynamic is not None else grad_accum,
        # Only the LoRA adapters carry gradients; the frozen base never does. Leaving this
        # True makes DDP scan the whole graph for unused parameters every step for nothing.
        ddp_find_unused_parameters=False,
        learning_rate=float(cfg.train.lr),
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_ratio=float(cfg.train.warmup_ratio),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
        logging_steps=int(cfg.train.logging_steps),
        # Periodic checkpoints so a dead pod costs minutes, not the whole run. Writing them
        # to a persistent volume is what makes `resume_from_checkpoint` worth having.
        save_strategy=str(cfg.train.get("save_strategy", "epoch")),
        save_steps=int(cfg.train.get("save_steps", 500)),
        save_total_limit=int(cfg.train.get("save_total_limit", 2)),
        # Durable checkpoints without a network volume: hub_strategy "checkpoint" pushes the
        # full trainer state to a `last-checkpoint` folder on every save, so a destroyed pod
        # loses only the steps since the last one and the final adapter is already on the Hub
        # when training ends -- the pod can be torn down immediately.
        push_to_hub=bool(cfg.train.get("push_to_hub", False)),
        hub_model_id=cfg.train.get("hub_model_id"),
        hub_strategy=str(cfg.train.get("hub_strategy", "every_save")),
        hub_private_repo=bool(cfg.train.get("hub_private_repo", False)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg.train.max_seq_len),
        packing=bool(cfg.train.packing),
        max_steps=2 if smoke else -1,
        # Default to no reporter: a throwaway pod carries no W&B credentials, and an
        # unavailable reporter is a hard error in transformers, not a warning.
        report_to=[] if smoke else list(cfg.train.get("report_to", []) or []),
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

    if dynamic is not None:
        trainer = DynamicBatchTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=None if resume_adapter else peft_cfg,
            # Identity collator: the step's examples reach training_step unpadded;
            # padding happens per micro-batch inside the step.
            data_collator=lambda f: f,
            token_budget=dyn_budget,
            global_batch=global_batch,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
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

    # Resume from the newest checkpoint on the volume when one is there, so a restart
    # continues rather than silently retraining from scratch at full cost.
    resume_ckpt = None
    if bool(cfg.train.get("auto_resume", False)):
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            if is_main:
                print(f">>> resuming from {resume_ckpt}")

    print(">>> starting training")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    # Every rank holds identical adapter weights after the all-reduce, so only rank 0
    # writes the side artifacts; two ranks writing the same paths can interleave and
    # leave a truncated run_meta.json.
    if not is_main:
        return
    tokenizer.save_pretrained(str(adapter_dir))

    # The eval framework infers serve-time thinking mode from this stamp; an adapter
    # without it is a hard error at eval time (CLAUDE.md, "The eval framework").
    (adapter_dir / "training_meta.json").write_text(json.dumps({
        "thinking": thinking,
        "train_config": config,
        "base_model": str(cfg.model),
        "data_path": str(cfg.data_path),
        "git_sha": _git_sha(),
        "timestamp": ts,
    }, indent=2))

    if cfg.get("hf_repo") and not smoke:
        from src.huggingface import push_run_dir
        from src.utils import origin_url

        # Same card contract as every other artifact (CLAUDE.md: every upload carries a
        # card), derived from the run's real metadata — the human-readable half beside
        # the machine-readable training_meta.json the eval framework consumes.
        url = push_run_dir(adapter_dir, str(cfg.hf_repo), {
            "experiment": f"LoRA SFT adapter — {Path(config).stem}",
            "date_generated": ts[:8],
            "constitution": str(cfg.get("constitution") or
                                f"inherited from the training data ({cfg.data_path}); "
                                "not declared in this train config"),
            "source_repo": f"{origin_url()} @ {_git_sha()}",
            "models": f"base: {cfg.model}",
            "generation_config": json.dumps({
                "seed": int(cfg.seed), "thinking": thinking,
                "epochs": float(cfg.train.epochs), "lr": float(cfg.train.lr),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum": int(cfg.train.grad_accum),
                "max_seq_len": int(cfg.train.max_seq_len),
                "lora": {"r": int(cfg.lora.r), "alpha": int(cfg.lora.alpha),
                         "dropout": float(cfg.lora.dropout)},
                # Present only when the run used token-budgeted micro-batching;
                # gradient-equivalent to the legacy grouping (dynamic_batching.py).
                **({"dynamic_batching": {
                    "token_budget": dyn_budget,
                    "loss_agg": "seq-mean-token-mean",
                }} if dynamic is not None else {}),
            }),
            "schema": "PEFT LoRA adapter (safetensors) + tokenizer + training_meta.json "
                      "{thinking, train_config, base_model, data_path, git_sha, timestamp}",
            "provenance": f"uv run train --config {config}",
        }, private=True, repo_type="model")
        print(f">>> pushed adapter (with training_meta.json + card) to {url}")

    meta = {
        "git_sha": _git_sha(),
        "base_model": str(cfg.model),
        "data_path": str(cfg.data_path),
        "n_examples": len(ds),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "smoke": smoke,
        "timestamp": ts,
        "world_size": world_size,
        # transformers 5.x does not print the loss to stdout, so without this the loss
        # curve exists only in a reporter we may not be running.
        "log_history": trainer.state.log_history,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    for row in trainer.state.log_history:
        if "loss" in row:
            print(f">>> step {row.get('step')}  loss {row['loss']:.4f}")
    print(f">>> saved adapter to {adapter_dir}")
    print(f">>> run_meta: {out_dir / 'run_meta.json'}")


def cli() -> None:
    """Console entry (`uv run train_lora --config ...`, [project.scripts])."""
    import fire

    fire.Fire(main)
