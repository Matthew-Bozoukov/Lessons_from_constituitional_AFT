# ABOUTME: A/B loss curves for the 2026-09-20 training optimisations: main's full-logits loss on
# ABOUTME: torch kernels (A) vs supervised-only logits + fla kernels (B), identical steps, one W&B group.

"""Run ONE arm of the A/B; launch it twice, once per GPU, once per checkout.

The arms differ in what is INSTALLED and what is CHECKED OUT, not in a flag this script
flips, so each arm is the real thing:

    A  baseline   origin/main worktree + main's lock   full-sequence fp32 logits, torch gated-delta
    B  optim      jamie/train-optim  + its lock        logits_to_keep + masked upcast, fla kernels

This file is copied into both checkouts and only touches APIs main already has; the one
branch-only import (`supervised_positions`) is made inside the optim arm.

Controls, as in scratch/ab_loss_curves.py (2026-08-10): one seeded shuffle of the pinned
mixture, the first `steps` x 16 rows as the step stream (a per-step length checksum is
logged, so identical data is visible in W&B); the recipe's LoRA shape from
configs/train/sft.yaml, seeded immediately before it is initialised and checksummed;
dropout 0 and a constant LR, so the curves differ only by numerics; the SAME token budget
in both arms, so both run the identical micro-batch plan.

    CUDA_VISIBLE_DEVICES=0 uv run python scratch/ab_train_optim.py --arm baseline \
        --data_repo <org>/<mix> --data_revision <sha>
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText, AutoTokenizer

from src.infra.huggingface import resolve_dataset
from src.model_profile import model_profile, render_chat, train_memory_entry
from src.train.dynamic_batching import plan_micro_batches, seq_mean_token_mean_loss
from src.train.masking import build_labels
from src.train.train_lora import _collate_padded
from src.utils import timestamp

GLOBAL_BATCH = 16
ARMS = {"baseline": "A-baseline-main", "optim": "B-optim-train-optim"}


def _stream(rows_path: str, tokenizer, profile, max_len: int, steps: int) -> list[dict]:
    """The first steps x 16 rows of a seeded shuffle, rendered and masked as training does."""
    raw = [json.loads(line) for line in Path(rows_path).open(encoding="utf8")]
    order = np.random.default_rng(0).permutation(len(raw))[: steps * GLOBAL_BATCH]
    feats = []
    for i in order:
        r = raw[int(i)]
        text = r.get("text") or render_chat(
            tokenizer, r["messages"], r.get("tools"), render_kwargs=profile.render_kwargs)
        feats.append(build_labels(text, tokenizer, max_len, profile,
                                  supervise=r.get("supervise") or "all",
                                  mask_spans=r.get("mask_spans")))
    return feats


def _build_model(model_id: str, profile, recipe):
    from peft import LoraConfig, get_peft_model

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=profile.attn_implementation)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(0)  # the LoRA init is the only RNG draw that must match across arms
    model = get_peft_model(model, LoraConfig(
        r=int(recipe.lora.r), lora_alpha=int(recipe.lora.alpha), lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=profile.lora_target_modules))
    model.train()
    return model


def main(arm: str, data_repo: str, data_revision: str | None = None, model: str = "qwen36",
         steps: int = 30, budget: int | None = None, group: str = "train-optim-ab",
         recipe: str = "configs/train/sft.yaml") -> None:
    import wandb
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    assert arm in ARMS, f"--arm is one of {sorted(ARMS)}"
    fast = {"fla_gated_delta": qwen.chunk_gated_delta_rule is not None,
            "causal_conv1d": qwen.causal_conv1d_fn is not None}
    # Each arm must be running on the stack it claims — a baseline with fla installed, or an
    # optim arm silently on the torch fallback, would make the comparison a lie.
    assert fast["fla_gated_delta"] == (arm == "optim"), (
        f"arm {arm!r} found fla_gated_delta={fast['fla_gated_delta']}; wrong venv for this arm")

    cfg = OmegaConf.load(recipe)
    profile = model_profile(model)
    model_id = profile.model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows_path, dataset_ref = resolve_dataset(data_repo, None, data_revision)
    feats = _stream(rows_path, tokenizer, profile, int(cfg.train.max_seq_len), steps)

    gpu = torch.cuda.get_device_name(0)
    if budget is None:
        measured = train_memory_entry(profile, gpu)
        budget = int(measured["max_padded_tokens"]) if measured else max(
            len(f["input_ids"]) for f in feats)

    net = _build_model(model_id, profile, cfg)
    trainable = [p for p in net.parameters() if p.requires_grad]
    init_checksum = float(sum(p.detach().double().abs().sum() for p in trainable))
    opt = torch.optim.AdamW(trainable, lr=float(cfg.train.lr))

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    run_cfg = {"arm": arm, "git_sha": git_sha, "dataset": dataset_ref, "model": model_id,
               "steps": steps, "budget": int(budget), "lr": float(cfg.train.lr),
               "global_batch": GLOBAL_BATCH, "gpu": gpu, "lora_r": int(cfg.lora.r),
               "lora_dropout": 0.0, "lora_init_checksum": init_checksum,
               "torch": torch.__version__, **fast}
    run = wandb.init(group=group, name=ARMS[arm], config=run_cfg)  # project/entity: env
    print(f">>> [{arm}] {git_sha[:8]} budget={budget} kernels={fast} "
          f"lora_init_checksum={init_checksum:.6f}", flush=True)

    if arm == "optim":
        from src.train.dynamic_batching import supervised_positions

    history, total_wall = [], 0.0
    for s in range(steps):
        step_feats = feats[s * GLOBAL_BATCH:(s + 1) * GLOBAL_BATCH]
        lens = [len(f["input_ids"]) for f in step_feats]
        plan = plan_micro_batches(lens, budget)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        net.zero_grad(set_to_none=True)
        loss_total, padded, logit_rows = 0.0, 0, 0
        for part in plan:
            mb = _collate_padded([step_feats[i] for i in part], tokenizer.pad_token_id)
            mb = {k: v.to("cuda") for k, v in mb.items()}
            labels = mb.pop("labels")
            padded += int(mb["input_ids"].numel())
            if arm == "optim":
                keep = supervised_positions(labels)
                logits = net(**mb, use_cache=False, logits_to_keep=keep).logits
                loss = seq_mean_token_mean_loss(logits, labels, GLOBAL_BATCH, keep)
            else:
                logits = net(**mb, use_cache=False).logits
                loss = seq_mean_token_mean_loss(logits, labels, GLOBAL_BATCH)
            logit_rows += int(logits.shape[0] * logits.shape[1])
            loss.backward()
            loss_total += float(loss.detach())
            del logits, loss
        grad_norm = float(torch.norm(torch.stack([p.grad.norm() for p in trainable])))
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_wall += dt
        row = {"loss": loss_total, "grad_norm": grad_norm, "passes": len(plan),
               "padded_tokens": padded, "logit_rows": logit_rows, "step_time_s": dt,
               "cum_wall_s": total_wall,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
               "data_checksum": sum(lens)}
        run.log(row, step=s)
        history.append(row)
        print(f"[{arm}] step {s:3d} loss {loss_total:.4f} grad {grad_norm:.3f} "
              f"passes {len(plan):2d} {dt:.1f}s peak {row['peak_mem_gib']:.1f}GiB", flush=True)

    run.summary["total_wall_s"] = total_wall
    run.summary["max_peak_mem_gib"] = max(r["peak_mem_gib"] for r in history)
    out = Path("output/train_optim_ab") / f"{timestamp()}_{arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": run_cfg, "wandb_url": run.url, "steps": history},
                              indent=2))
    print(f">>> [{arm}] {total_wall:.0f}s total; wrote {out}; {run.url}", flush=True)
    run.finish()


if __name__ == "__main__":
    fire.Fire(main)
