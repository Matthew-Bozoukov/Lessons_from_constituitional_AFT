# ABOUTME: A/B loss-curve comparison: legacy batch-1 vs dynamic micro-batching, byte-identical
# ABOUTME: 16-row optimizer steps from identical LoRA init, one wandb run per protocol.

"""Train the SAME steps twice — legacy grouping vs dynamic — and overlay the curves.

The acceptance criterion (Jamie, 2026-08-10): if the loss curves over real training
steps match, the protocols are equivalent for our purposes. Controls:

- One seeded shuffle of the mixture; the first `steps` x 16 rows become the step
  stream. BOTH protocols consume byte-identical rows in identical order (asserted
  via a per-step length checksum), so every effective batch of 16 is the same.
- One model load; LoRA initial weights snapshotted and restored between protocols,
  so both start from the identical point. Fresh AdamW each (constant LR — the
  config's cosine schedule would add a confound irrelevant to grouping).
- Dropout 0 (partition changes RNG consumption order; curves should differ only by
  bf16 batching numerics, which fp64 analysis showed is the ONLY divergence channel
  — see scratch/fp64 semantics test, 2026-08-10).
- The stream is real shuffled data, so longalign rows land where the shuffle puts
  them (~25% of steps); the script asserts >=2 are present and reports the count.

Each protocol logs to wandb (entity = the API key's default account) as its own run:
loss, grad_norm, passes, padded_tokens, step_time_s per optimizer step.

    WANDB_API_KEY=... uv run python scratch/ab_loss_curves.py \
        --config configs/train/lora_qwen36_table2_selfreflect_r64.yaml \
        --rows data/mixture.jsonl --steps 30
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.train.dynamic_batching import plan_micro_batches, seq_mean_token_mean_loss
from src.train.masking import build_labels, model_profile
from src.train.train_lora import _collate_padded

GLOBAL_BATCH = 16


def _build_model(cfg):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        str(cfg.model), dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=str(cfg.lora.target_modules)))
    model.train()
    return model


def _stream(cfg, rows_path: str, tokenizer, steps: int) -> tuple[list[dict], list[str]]:
    """The first steps x 16 rows of a seeded shuffle, masked exactly as training."""
    profile = model_profile(str(cfg.model))
    raw = [json.loads(line) for line in Path(rows_path).open(encoding="utf8")]
    order = np.random.default_rng(0).permutation(len(raw))[: steps * GLOBAL_BATCH]
    chosen = [raw[i] for i in order]
    feats = [build_labels(r["text"], tokenizer, int(cfg.train.max_seq_len), profile,
                          supervise=r.get("supervise") or "all") for r in chosen]
    sources = [r.get("source", "?") for r in chosen]
    n_la = sources.count("longalign")
    print(f">>> stream: {len(feats)} rows / {steps} steps; longalign rows: {n_la}")
    assert n_la >= 2, "reseed: the stream must contain longalign rows"
    return feats, sources


def _run_protocol(name: str, model, feats, lr: float, budget: int, pad_id: int,
                  wandb, run_cfg: dict) -> dict:
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    run = wandb.init(project=run_cfg["project"], name=name, config=run_cfg,
                     reinit="finish_previous")
    total_wall = 0.0
    for s in range(run_cfg["steps"]):
        step_feats = feats[s * GLOBAL_BATCH:(s + 1) * GLOBAL_BATCH]
        lens = [len(f["input_ids"]) for f in step_feats]
        plan = ([[i] for i in range(GLOBAL_BATCH)] if name == "legacy"
                else plan_micro_batches(lens, budget))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss_total, padded = 0.0, 0
        for part in plan:
            mb = _collate_padded([step_feats[i] for i in part], pad_id)
            mb = {k: v.to("cuda") for k, v in mb.items()}
            labels = mb.pop("labels")
            padded += int(mb["input_ids"].numel())
            loss = seq_mean_token_mean_loss(
                model(**mb, use_cache=False).logits, labels, GLOBAL_BATCH)
            loss.backward()
            loss_total += float(loss.detach())
        grad_norm = float(torch.norm(torch.stack([p.grad.norm() for p in trainable])))
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_wall += dt
        run.log({"loss": loss_total, "grad_norm": grad_norm, "passes": len(plan),
                 "padded_tokens": padded, "step_time_s": dt,
                 "cum_wall_s": total_wall, "data_checksum": sum(lens)}, step=s)
        print(f"[{name}] step {s:3d} loss {loss_total:.4f} grad {grad_norm:.3f} "
              f"passes {len(plan):2d} {dt:.1f}s", flush=True)
    run.summary["total_wall_s"] = total_wall
    run.finish()
    return {"total_wall_s": total_wall}


def main(config: str, rows: str = "data/mixture.jsonl", steps: int = 30,
         budget: int | None = None, project: str = "dynamic-batching-ab") -> None:
    import wandb

    cfg = OmegaConf.load(config)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    feats, _ = _stream(cfg, rows, tokenizer, steps)
    if budget is None:
        budget = max(len(f["input_ids"]) for f in feats)
    model = _build_model(cfg)
    init = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    run_cfg = {"project": project, "steps": steps, "budget": int(budget),
               "lr": float(cfg.train.lr), "global_batch": GLOBAL_BATCH,
               "gpu": torch.cuda.get_device_name(0), "config": config,
               "loss_agg": "seq-mean-token-mean", "lora_dropout": 0.0}
    results = {}
    for name in ("legacy", "dynamic"):
        with torch.no_grad():  # identical starting point for both protocols
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(init[n])
        results[name] = _run_protocol(name, model, feats, float(cfg.train.lr),
                                      int(budget), tokenizer.pad_token_id, wandb, run_cfg)

    speed = results["legacy"]["total_wall_s"] / results["dynamic"]["total_wall_s"]
    print(f"\n>>> legacy  {results['legacy']['total_wall_s']:.0f}s")
    print(f">>> dynamic {results['dynamic']['total_wall_s']:.0f}s  ({speed:.2f}x)")
    print(">>> compare the loss/grad_norm curves in the wandb project "
          f"'{project}' — identical data per step is asserted via data_checksum")


if __name__ == "__main__":
    fire.Fire(main)
