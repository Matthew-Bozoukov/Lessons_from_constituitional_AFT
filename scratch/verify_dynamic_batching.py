# ABOUTME: On-pod gates for dynamic batching: gradient/loss equivalence vs the legacy
# ABOUTME: batch-1 path, a token-mean negative control, and a throughput A/B with host-lane timing.

"""Prove dynamic batching changes throughput and nothing else. GPU host only.

Three gates, in order; the script exits non-zero on the first failure:

1. EQUIVALENCE — the crux. Sixteen real mixture rows (stratified by length,
   forced to include the longest row so a singleton over-budget micro-batch is
   exercised). LoRA dropout is forced to 0 (a different partition consumes
   dropout RNG in a different order, so exact comparison requires it off).
   Legacy reference: 16 batch-1 forward+backwards, each loss = row token-mean/16.
   Dynamic: plan_micro_batches + padded micro-batches + seq_mean_token_mean_loss.
   Asserts: |loss_a - loss_b| relative < 1e-3, LoRA-grad cosine > 0.9999, and max
   relative grad-norm error < 1e-2 (bf16 accumulation-order tolerance). This one
   gate simultaneously tests the weighting, the batch>1 padding path through the
   gated-delta layers (apply_mask_to_padding_states is a no-op at batch 1 and
   active here), and the accumulate-then-step plumbing.

2. NEGATIVE CONTROL — the same comparison with token-mean aggregation must FAIL
   the loss match. If it doesn't, the equivalence gate has no teeth.

3. THROUGHPUT A/B — N optimizer steps legacy vs dynamic, same rows: wall-clock,
   pass count, host-lane time (plan+collate, the "is the CPU a bottleneck"
   number), and peak memory. Expectation from the length distribution: >=2x
   fewer passes; peak memory <= legacy (budget == max_seq_len bounds it).

Run (pod, repo root, data/mixture.jsonl staged as for training):

    uv run python scratch/verify_dynamic_batching.py \
        --config configs/train/lora_qwen36_table2_selfreflect_r64.yaml \
        [--budget 8192] [--steps 6] [--rows data/mixture.jsonl]

Reuses train_lora's own loading/masking helpers so what is verified is what
trains. Results print as a markdown table; paste into docs/LOG.md with the run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.train.dynamic_batching import plan_micro_batches, seq_mean_token_mean_loss
from src.train.masking import build_labels, model_profile
from src.train.train_lora import _collate_padded


def _load_rows(cfg, rows_path: str, tokenizer, n: int = 16) -> list[dict]:
    """Tokenize+mask `n` rows exactly as train_lora would.

    The step is DELIBERATELY adversarial for dynamic batching: the TWO longest rows
    of the corpus (longalign, ~8k — the rows that force batch 1 today) plus an even
    sweep of the remaining length distribution. A random step usually holds 0-1 long
    rows, so this composition makes the timing comparison conservative, not
    flattering.
    """
    profile = model_profile(str(cfg.model))
    max_len = int(cfg.train.max_seq_len)
    raw = [json.loads(line) for line in Path(rows_path).open(encoding="utf8")]
    if "text" not in raw[0]:
        raise SystemExit("verify needs a pre-rendered mixture (text column), like training")
    order = sorted(range(len(raw)), key=lambda i: len(raw[i]["text"]))
    sweep = [order[int(i * (len(order) - 3) / (n - 3))] for i in range(n - 2)]
    chosen = [raw[i] for i in order[-2:] + sweep]  # 2 longest + (n-2) stratified
    feats = [build_labels(r["text"], tokenizer, max_len, profile,
                          supervise=r.get("supervise") or "all") for r in chosen]
    lens = sorted(len(f["input_ids"]) for f in feats)
    srcs = [r.get("source", "?") for r in chosen]
    print(f">>> {len(feats)} rows, lengths {lens[0]}..{lens[-1]} (median {lens[len(lens) // 2]})")
    print(f">>> sources: { {s: srcs.count(s) for s in sorted(set(srcs))} }")
    assert len(feats) == n
    return feats


def _build_model(cfg):
    """The trainer's model path: bf16, image_text_to_text, LoRA attached, dropout 0."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    assert not bool(cfg.train.get("load_in_4bit", True)), \
        "verify assumes the qwen36 bf16 path (load_in_4bit: false)"
    model = AutoModelForImageTextToText.from_pretrained(
        str(cfg.model), dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
        lora_dropout=0.0,  # determinism: dropout RNG order differs per partition
        bias="none", task_type="CAUSAL_LM",
        target_modules=str(cfg.lora.target_modules)))
    model.train()
    return model


def _forward_loss(model, feats: list[dict], pad_id: int, global_batch: int,
                  agg: str = "seq-mean-token-mean"):
    batch = _collate_padded(feats, pad_id)
    batch = {k: v.to("cuda") for k, v in batch.items()}
    labels = batch.pop("labels")
    out = model(**batch, use_cache=False)
    if agg == "seq-mean-token-mean":
        return seq_mean_token_mean_loss(out.logits, labels, global_batch)
    # token-mean, for the negative control only
    shift_logits = out.logits[:, :-1, :].float().flatten(0, 1)
    shift_labels = labels[:, 1:].flatten()
    per = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none")
    return per.sum() / shift_labels.ne(-100).sum()


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.grad is not None}


def _run_step(model, feats, pad_id, gb, plan: list[list[int]], agg: str):
    """One optimizer step's backward under a given partition. Returns (loss, grads)."""
    model.zero_grad(set_to_none=True)
    total = 0.0
    scale = {"seq-mean-token-mean": 1.0, "token-mean": 1.0 / len(plan)}[agg]
    for part in plan:
        loss = _forward_loss(model, [feats[i] for i in part], pad_id, gb, agg) * scale
        loss.backward()
        total += float(loss.detach())
    return total, _grads(model)


def main(config: str, rows: str = "data/mixture.jsonl",
         budget: int | None = None, steps: int = 6) -> None:
    cfg = OmegaConf.load(config)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    gb = 16

    feats = _load_rows(cfg, rows, tokenizer, n=gb)
    lens = [len(f["input_ids"]) for f in feats]
    if budget is None:  # the trainer's own default: the longest actual row (the
        budget = max(lens)  # picked set includes the dataset's longest by construction)
    model = _build_model(cfg)

    legacy_plan = [[i] for i in range(gb)]
    dyn_plan = plan_micro_batches(lens, budget)
    print(f">>> dynamic plan: {len(dyn_plan)} passes "
          f"{sorted(len(p) for p in dyn_plan)} vs {gb} at batch 1")

    # --- gate 1: equivalence -----------------------------------------------------
    loss_a, grads_a = _run_step(model, feats, pad_id, gb, legacy_plan, "seq-mean-token-mean")
    loss_b, grads_b = _run_step(model, feats, pad_id, gb, dyn_plan, "seq-mean-token-mean")
    rel_loss = abs(loss_a - loss_b) / max(abs(loss_a), 1e-9)
    flat_a = torch.cat([grads_a[k].flatten() for k in sorted(grads_a)])
    flat_b = torch.cat([grads_b[k].flatten() for k in sorted(grads_b)])
    cosine = float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0))
    rel_norm = float((flat_a - flat_b).norm() / flat_a.norm())
    print(f"\n| gate | metric | value | threshold | pass |\n|---|---|---|---|---|")
    ok1 = rel_loss < 1e-3 and cosine > 0.9999 and rel_norm < 1e-2
    print(f"| equivalence | loss rel diff | {rel_loss:.2e} | <1e-3 | {rel_loss < 1e-3} |")
    print(f"| equivalence | grad cosine | {cosine:.6f} | >0.9999 | {cosine > 0.9999} |")
    print(f"| equivalence | grad rel norm err | {rel_norm:.2e} | <1e-2 | {rel_norm < 1e-2} |")

    # --- gate 2: negative control ------------------------------------------------
    loss_c, _ = _run_step(model, feats, pad_id, gb, dyn_plan, "token-mean")
    rel_c = abs(loss_a - loss_c) / max(abs(loss_a), 1e-9)
    ok2 = rel_c > 1e-3  # token-mean MUST disagree, or gate 1 proves nothing
    print(f"| negative control | token-mean rel diff | {rel_c:.2e} | >1e-3 | {ok2} |")

    # --- gate 3: throughput A/B ---------------------------------------------------
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    results = {}
    for name, plan in (("legacy", legacy_plan), ("dynamic", dyn_plan)):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        host_lane = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            h0 = time.perf_counter()
            p = plan_micro_batches(lens, budget) if name == "dynamic" else plan
            mbs = [_collate_padded([feats[i] for i in part], pad_id) for part in p]
            host_lane += time.perf_counter() - h0
            for mb in mbs:
                mb = {k: v.to("cuda") for k, v in mb.items()}
                labels = mb.pop("labels")
                loss = seq_mean_token_mean_loss(
                    model(**mb, use_cache=False).logits, labels, gb)
                loss.backward()
            opt.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        results[name] = dict(wall=wall / steps, host=host_lane / steps,
                             passes=len(plan), peak=torch.cuda.max_memory_allocated() / 2**30)
        print(f"| throughput | {name}: s/step, host ms, passes, peak GiB "
              f"| {wall / steps:.2f}, {1000 * host_lane / steps:.1f}, {len(plan)}, "
              f"{results[name]['peak']:.1f} | — | — |")
    speedup = results["legacy"]["wall"] / results["dynamic"]["wall"]
    mem_ok = results["dynamic"]["peak"] <= results["legacy"]["peak"] * 1.05
    print(f"| throughput | dynamic speedup | {speedup:.2f}x | >=2x expected | {speedup >= 2} |")
    print(f"| throughput | peak mem <= legacy+5% | "
          f"{results['dynamic']['peak']:.1f} vs {results['legacy']['peak']:.1f} GiB | — | {mem_ok} |")

    if not (ok1 and ok2 and mem_ok):
        sys.exit(">>> GATES FAILED — do not train with dynamic_batching until resolved")
    print(f"\n>>> ALL GATES PASS (speedup {speedup:.2f}x on this 16-row step; "
          "epoch-level number comes from a real run)")


if __name__ == "__main__":
    fire.Fire(main)
