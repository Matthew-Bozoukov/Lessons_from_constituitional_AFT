# ABOUTME: LESS step 1 — trains the warmup LoRA on a random slice of D, checkpointing
# ABOUTME: the adapter AND name-keyed Adam moments at every epoch boundary.

"""Train the warmup LoRA and emit one gradient-feature checkpoint per epoch.

    uv run python scratch/less/warmup.py --config configs/train/lora_qwen36_less_warmup_r64.yaml

LESS needs three things from each epoch boundary that ordinary SFT does not keep:

1. The adapter weights (ordinary).
2. The Adam moments `exp_avg` (m) and `exp_avg_sq` (v) for every LoRA parameter. The
   training-side gradient feature is the Adam UPDATE direction, not the raw gradient --
   that is what the "Adam" in InfAdam means -- and it cannot be recovered after the fact.
3. The learning rate that epoch actually ran at, which weights that checkpoint's
   similarities in `I = sum_i eta_i * S_i`.

Why this is a hand-written loop rather than `src/train/train_lora.py`: the HF Trainer
saves optimizer state in `optimizer.pt` keyed by the parameter's INTEGER POSITION in
`optimizer.param_groups`, not by name. Recovering the name mapping means reconstructing
the exact order the Trainer enumerated parameters in and splitting decay/no-decay groups
the same way. If that reconstruction is off by one, every Adam moment is paired with the
wrong parameter, the preconditioning is garbage, and NOTHING errors -- the run completes
and produces a confident, meaningless ranking. Keying the state by name at the point it is
saved makes that failure impossible rather than merely unlikely.

The masking is NOT hand-written: rendering and label construction call the same
`apply_chat_template(**profile.render_kwargs)` -> `build_labels` path training uses, so
CoT supervision is identical to a real run (prefill masked, whole empty marker masked,
real traces and their close supervised).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.model_profile import model_profile
from src.train.masking import build_labels, check_thinking_declaration
from src.utils import git_sha, timestamp


def load_rows(path: Path, tokenizer, profile, max_len: int) -> list[dict]:
    """Render and mask every row once, up front, so the loop is pure compute."""
    raw = [json.loads(line) for line in
           path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check_thinking_declaration(raw, thinking=True, empty_think=profile.empty_think)
    out = []
    for r in raw:
        text = tokenizer.apply_chat_template(
            [{k: v for k, v in m.items() if v is not None} for m in r["messages"]],
            tokenize=False, add_generation_prompt=False, **profile.render_kwargs)
        enc = build_labels(text, tokenizer, max_len, profile,
                           supervise=r.get("supervise") or "all")
        enc["less_id"] = r["metadata"]["less_id"]
        out.append(enc)
    return out


def save_checkpoint(model, optimizer, named_lora: list[tuple[str, torch.nn.Parameter]],
                    out: Path, epoch: int, lrs: list[float], step: int) -> None:
    """Write adapter weights, name-keyed Adam moments, and this epoch's learning rate.

    The moments are stored on CPU in fp32: preconditioning divides by sqrt(v), so bf16
    rounding here propagates straight into every similarity downstream.
    """
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    state = {}
    for name, param in named_lora:
        st = optimizer.state.get(param)
        assert st and "exp_avg" in st and "exp_avg_sq" in st, (
            f"no Adam moments for {name} at epoch {epoch}; the parameter never received a "
            f"gradient, so its feature would be undefined")
        state[name] = {"exp_avg": st["exp_avg"].detach().to("cpu", torch.float32),
                       "exp_avg_sq": st["exp_avg_sq"].detach().to("cpu", torch.float32)}
    torch.save(state, out / "adam_state.pt")

    # eta_i weights this checkpoint's similarities. The MEAN lr over the epoch describes
    # what the epoch did; the final lr is recorded too so the choice stays auditable.
    (out / "checkpoint_meta.json").write_text(json.dumps({
        "epoch": epoch, "optimizer_step": step,
        "lr_mean": sum(lrs) / len(lrs), "lr_final": lrs[-1],
        "n_lora_params": sum(p.numel() for _, p in named_lora),
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="8 rows, 2 epochs — proves the checkpoint contract, not the science")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    profile = model_profile(str(cfg.model))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(Path(cfg.data_path), tokenizer, profile, int(cfg.train.max_seq_len))
    epochs = int(cfg.train.epochs)
    if args.smoke:
        rows, epochs = rows[:8], 2
    print(f">>> {len(rows)} warmup rows, {epochs} epochs, "
          f"{sum(len(r['input_ids']) for r in rows):,} tokens")

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=str(cfg.train.get("attn_implementation", "sdpa")))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = cfg.lora.target_modules
    model = get_peft_model(model, LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout), bias="none", task_type="CAUSAL_LM",
        target_modules=str(targets) if isinstance(targets, str) else list(targets)))

    named_lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert named_lora, "LoRA produced no trainable parameters — check target_modules"
    p_total = sum(p.numel() for _, p in named_lora)
    print(f">>> {len(named_lora)} LoRA tensors, P = {p_total:,}")

    accum = int(cfg.train.grad_accum)
    steps_per_epoch = math.ceil(len(rows) / accum)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW([p for _, p in named_lora], lr=float(cfg.train.lr),
                                  weight_decay=float(cfg.train.weight_decay),
                                  betas=(0.9, 0.999), eps=1e-8)
    warm = max(1, int(float(cfg.train.warmup_ratio) * total_steps))
    # (s + 1) / warm, NOT s / warm: LambdaLR evaluates at s=0 for the first step, so the
    # bare ratio makes step 0 run at lr exactly 0 -- it trains nothing and, worse, drags
    # eta_1 down, which is the weight checkpoint 1 carries in `I = sum_i eta_i * S_i`.
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))))

    out_root = Path(str(cfg.output_dir)) / timestamp()
    out_root.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    rng = torch.Generator().manual_seed(int(cfg.seed))
    step, t0 = 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(rows), generator=rng).tolist()
        epoch_lrs, losses = [], []
        for i, idx in enumerate(order):
            r = rows[idx]
            batch = {k: torch.tensor([r[k]], device=dev)
                     for k in ("input_ids", "attention_mask", "labels")}
            loss = model(**batch).loss / accum
            loss.backward()
            losses.append(loss.item() * accum)
            if (i + 1) % accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_([p for _, p in named_lora], 1.0)
                epoch_lrs.append(sched.get_last_lr()[0])
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    print(f"    epoch {epoch} step {step}/{total_steps} "
                          f"loss {sum(losses[-accum:]) / len(losses[-accum:]):.4f} "
                          f"lr {epoch_lrs[-1]:.2e} ({time.time() - t0:.0f}s)")
        ckpt = out_root / f"ckpt_epoch{epoch}"
        save_checkpoint(model, optimizer, named_lora, ckpt, epoch, epoch_lrs, step)
        print(f">>> epoch {epoch}: loss {sum(losses) / len(losses):.4f}, "
              f"lr_mean {sum(epoch_lrs) / len(epoch_lrs):.3e} -> {ckpt}")

    (out_root / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha(), "config": OmegaConf.to_container(cfg, resolve=True),
        "smoke": args.smoke, "epochs": epochs, "n_rows": len(rows),
        "p_total": p_total, "seconds": round(time.time() - t0, 1),
        "checkpoints": [f"ckpt_epoch{e}" for e in range(1, epochs + 1)],
    }, indent=2), encoding="utf-8")
    print(f">>> done in {time.time() - t0:.0f}s -> {out_root}")


if __name__ == "__main__":
    main()
