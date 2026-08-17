# ABOUTME: LESS step 2 — per-example LoRA gradient features for every warmup checkpoint,
# ABOUTME: Adam-preconditioned for training rows and raw for validation rows.

"""Build the projected gradient datastore.

    uv run python scratch/less/gradients.py --warmup output/less_warmup/<ts> \\
        --rows data/less/d_full.jsonl --split train --out output/less_grads/<ts> \\
        [--shard 0 --num-shards 4] [--limit 32]

The train/validation asymmetry is the whole method and is easy to get silently wrong, so
it is spelled out here. A first-order Adam step on training point z changes the loss at
validation point z' by

    l(z'; θ_{t+1}) - l(z'; θ_t)  ≈  -η_t * ∇l(z'; θ_t)ᵀ Γ(z, θ_t)

so the TRAINING side contributes the Adam update direction Γ, and the VALIDATION side
contributes a plain gradient. Using plain gradients on both sides computes a different
quantity (SGD influence) and quietly discards the "Adam" in InfAdam.

Γ follows the reference implementation exactly, including its omission of bias correction
and its placement of eps INSIDE the square root:

    m' = β1·m + (1-β1)·g      v' = β2·v + (1-β2)·g²      Γ = m' / sqrt(v' + eps)

where m and v are the moments saved at that checkpoint by scratch/less/warmup.py, keyed by
parameter name. They are read, never updated -- so every example is independent of every
other, which is what makes this stage shardable across GPUs with no communication.

Sharding is BY EXAMPLE, not by checkpoint, on purpose: the 54GB base model is identical
across all four checkpoints and only the adapter differs, so one worker loads the base
once and swaps adapter weights in place. Sharding by checkpoint would pay the base-model
load four times over.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.model_profile import model_profile
from src.train.masking import build_labels
from src.utils import git_sha

from projection import DEFAULT_DIM, CountSketchProjector  # noqa: E402  (same-dir module)

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def adam_precondition(grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The Adam update direction Γ for one example, per the LESS reference implementation.

    Written to allocate exactly two P-sized temporaries rather than five: at P=319M the
    naive arithmetic expression costs ~6.4 GiB of transient memory per example. `m` and `v`
    are read-only here -- they are the checkpoint's frozen moments, reused for every row --
    so the leading `.mul()` deliberately copies while the rest operate in place on that copy.
    """
    m_new = m.mul(BETA1).add_(grad, alpha=1 - BETA1)
    v_new = v.mul(BETA2).addcmul_(grad, grad, value=1 - BETA2)
    return m_new.div_(v_new.add_(EPS).sqrt_())


def ordered_lora_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """LoRA parameters in a fixed, name-sorted order.

    The order defines the flat vector's layout, so it must be identical between the
    gradient, the Adam moments and every shard. Sorting by name makes that independent of
    module traversal order and of how many adapters happen to be loaded.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert named, "no trainable LoRA parameters; the adapter loaded frozen"
    return sorted(named, key=lambda kv: kv[0])


def flatten_moments(state: dict, named: list[tuple[str, torch.nn.Parameter]],
                    device) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate saved Adam moments into the same layout as the flat gradient."""
    missing = [n for n, _ in named if n not in state]
    assert not missing, (
        f"{len(missing)} LoRA parameters have no saved Adam moment (e.g. {missing[:2]}); "
        f"the checkpoint does not match this adapter geometry")
    m = torch.cat([state[n]["exp_avg"].reshape(-1) for n, _ in named]).to(device, torch.float32)
    v = torch.cat([state[n]["exp_avg_sq"].reshape(-1) for n, _ in named]).to(device, torch.float32)
    for name, param in named:
        assert state[name]["exp_avg"].numel() == param.numel(), (
            f"shape mismatch for {name}: moment has {state[name]['exp_avg'].numel()} "
            f"elements, parameter has {param.numel()}")
    return m, v


def encode(rows: list[dict], tokenizer, profile, max_len: int) -> list[dict]:
    """Render and mask rows through the exact path training uses."""
    out = []
    for r in rows:
        text = tokenizer.apply_chat_template(
            [{k: v for k, v in msg.items() if v is not None} for msg in r["messages"]],
            tokenize=False, add_generation_prompt=False, **profile.render_kwargs)
        enc = build_labels(text, tokenizer, max_len, profile,
                           supervise=r.get("supervise") or "all")
        meta = r["metadata"]
        assert "less_id" in meta, (
            f"row has no `less_id` (keys: {sorted(meta)}); it is the join key tying a "
            f"gradient row back to its source — stamp it in whichever builder wrote this "
            f"file (scratch/less/convert_dval.py or prepare_data.py)")
        enc["less_id"] = meta["less_id"]
        enc["subtask"] = meta.get("subtask")
        out.append(enc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", required=True, type=Path, help="warmup run dir")
    ap.add_argument("--rows", required=True, type=Path, help="jsonl of rows to featurise")
    ap.add_argument("--split", required=True, choices=("train", "val"),
                    help="train = Adam-preconditioned Γ; val = raw gradient")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--proj-seed", type=int, default=0,
                    help="MUST match across train, val and every shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: N rows, taken STRIDED across the file — D is ordered by "
                         "trait, so a head slice samples one trait and makes any "
                         "trait-composition diagnostic meaningless")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    meta = json.loads((args.warmup / "run_meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    ckpts = sorted((args.warmup / c for c in meta["checkpoints"]), key=lambda p: p.name)
    profile = model_profile(str(cfg["model"]))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    rows = [json.loads(l) for l in
            args.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit < len(rows):
        step = len(rows) / args.limit
        rows = [rows[int(i * step)] for i in range(args.limit)]
    # Strided rather than contiguous: rows vary in length, so a strided slice gives every
    # shard a comparable token load and they finish together.
    mine = rows[args.shard::args.num_shards]
    print(f">>> shard {args.shard}/{args.num_shards}: {len(mine)}/{len(rows)} rows, "
          f"{len(ckpts)} checkpoints, split={args.split}")

    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation=cfg["train"].get("attn_implementation", "sdpa"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = cfg["lora"]["target_modules"]
    model = get_peft_model(model, LoraConfig(
        r=int(cfg["lora"]["r"]), lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=0.0,  # deterministic features: dropout would randomise the gradient
        bias="none", task_type="CAUSAL_LM",
        target_modules=str(targets) if isinstance(targets, str) else list(targets)))

    # Determinism WITHOUT model.eval(). The obvious way to make a gradient a function of
    # the row alone is eval mode, but transformers guards activation recomputation with
    # `if self.gradient_checkpointing and self.training:` -- so eval() silently turns
    # gradient checkpointing OFF and retains every layer's activations. On a 27B model at
    # 2k tokens that is ~70 GB of extra activation memory and an OOM on a 143 GB H200.
    # Train mode keeps checkpointing alive; dropout is what actually had to go, so it is
    # zeroed directly (LoRA dropout is already 0.0 in the config above).
    model.train()
    n_dropout = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p, n_dropout = 0.0, n_dropout + 1
    print(f">>> train mode with gradient checkpointing; {n_dropout} dropout modules zeroed")

    named = ordered_lora_params(model)
    p_total = sum(p.numel() for _, p in named)
    dev = next(model.parameters()).device
    print(f">>> P = {p_total:,} over {len(named)} tensors on {dev}")

    proj = CountSketchProjector(p_total, dim=args.dim, seed=args.proj_seed, device=dev)
    print(f">>> projector d={args.dim} seed={args.proj_seed} "
          f"({proj.memory_bytes() / 2**30:.1f} GiB resident)")

    encoded = encode(mine, tokenizer, profile, args.max_seq_len)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def row_gradient(r: dict) -> torch.Tensor:
        """One example's flat LoRA gradient, mean over its supervised tokens."""
        batch = {k: torch.tensor([r[k]], device=dev)
                 for k in ("input_ids", "attention_mask", "labels")}
        model.zero_grad(set_to_none=True)
        # Batch size 1, so HF's CE over -100 labels is already the mean over THIS row's
        # supervised tokens -- long rows cannot dominate the datastore.
        model(**batch).loss.backward()
        return torch.cat([p.grad.reshape(-1).to(torch.float32) for _, p in named])

    # Reproducibility gate. Features are compared by cosine across shards and splits, so a
    # gradient that depends on more than the row corrupts every similarity downstream
    # without erroring. Train mode is required for gradient checkpointing, so this measures
    # the claim that dropout was the only stochastic thing rather than assuming it.
    #
    # The bar is a cosine tolerance, not bit-equality, because bit-equality is unachievable
    # here and demanding it would be the wrong test. Measured on this model (2026-08-14):
    # the backward pass alone differs by 3.9e-03 RELATIVE between two runs of the same row
    # -- CUDA atomics plus gradient-checkpoint recomputation -- while the projector is
    # clean to 1.9e-06. End to end that is 1-cos = 5.2e-05, which is ~100x smaller than the
    # count-sketch's own approximation error (rmse 4.8e-03 at d=32768) and ~3000x below the
    # spread of true cosines. It cannot move a ranking. A live dropout mask, by contrast,
    # perturbs 1-cos by order 1e-2, so REPRODUCIBILITY_TOL separates the two by ~20x.
    REPRODUCIBILITY_TOL = 1e-3
    a = proj.project(row_gradient(encoded[0]))
    b = proj.project(row_gradient(encoded[0]))
    drift = 1 - float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    assert drift < REPRODUCIBILITY_TOL, (
        f"the same row produced materially different features twice (1-cos = {drift:.3e} "
        f">= {REPRODUCIBILITY_TOL}); something stochastic is live beyond float "
        f"nondeterminism — check for dropout modules the zeroing pass missed")
    print(f">>> reproducibility gate passed (1-cos = {drift:.2e} on a repeated row, "
          f"tolerance {REPRODUCIBILITY_TOL})")
    del a, b

    for ckpt in ckpts:
        set_peft_model_state_dict(model, load_peft_weights(str(ckpt)))
        state = torch.load(ckpt / "adam_state.pt", map_location="cpu", weights_only=True)
        m_flat, v_flat = (flatten_moments(state, named, dev)
                          if args.split == "train" else (None, None))
        del state

        feats = torch.zeros(len(encoded), args.dim, dtype=torch.float32)
        for i, r in enumerate(encoded):
            flat = row_gradient(r)
            if args.split == "train":
                flat = adam_precondition(flat, m_flat, v_flat)
            feats[i] = proj.project(flat).cpu()
            del flat
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"    {ckpt.name}: {i + 1}/{len(encoded)} ({rate:.2f} rows/s)")

        stem = f"{args.split}_{ckpt.name}_shard{args.shard}of{args.num_shards}"
        torch.save({"features": feats,
                    "less_ids": [r["less_id"] for r in encoded],
                    "subtasks": [r["subtask"] for r in encoded],
                    "checkpoint": ckpt.name, "split": args.split,
                    "dim": args.dim, "proj_seed": args.proj_seed,
                    "lr_mean": json.loads((ckpt / "checkpoint_meta.json")
                                          .read_text(encoding="utf-8"))["lr_mean"]},
                   args.out / f"{stem}.pt")
        del m_flat, v_flat
        torch.cuda.empty_cache()
        print(f">>> {stem}: {tuple(feats.shape)} in {time.time() - t0:.0f}s")

    (args.out / f"meta_{args.split}_shard{args.shard}.json").write_text(json.dumps({
        "git_sha": git_sha(), "warmup": str(args.warmup), "rows": str(args.rows),
        "split": args.split, "shard": args.shard, "num_shards": args.num_shards,
        "n_rows": len(encoded), "dim": args.dim, "proj_seed": args.proj_seed,
        "p_total": p_total, "seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f">>> shard done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
