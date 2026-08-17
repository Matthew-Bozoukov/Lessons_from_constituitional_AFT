# ABOUTME: Times each stage of per-example feature extraction, to show whether the cost is
# ABOUTME: the backward pass or the count-sketch scatter before committing to a long run.

"""Break the ~3.8s per row-checkpoint into its parts.

    uv run python scratch/less/profile_stages.py --warmup output/less_warmup/<ts> --rows 6

The question this answers is narrow and worth money: the sketch scatters P=319M values into
d buckets with CUDA atomics, so ~9,700 adds collide per bucket at d=32768. High-contention
atomicAdd is slow, and if it is a large share of the per-row cost then RAISING d is a free
speedup -- 4x fewer collisions at d=131072 -- which also measured BETTER on rank
preservation (Spearman 0.9999 vs 0.9995). Both widths are timed here so the trade is
decided on numbers rather than on the reasoning above, which is only a hypothesis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.model_profile import model_profile

from gradients import (adam_precondition, encode, flatten_moments,  # noqa: E402
                       ordered_lora_params)
from projection import CountSketchProjector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", required=True, type=Path)
    ap.add_argument("--pool", type=Path, default=Path("data/less/d_full.jsonl"))
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--dims", type=int, nargs="+", default=[32768, 131072])
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    meta = json.loads((args.warmup / "run_meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    prof = model_profile(cfg["model"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    raw = [json.loads(l) for l in
           args.pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    step = len(raw) / args.rows
    rows = [raw[int(i * step)] for i in range(args.rows)]

    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = cfg["lora"]["target_modules"]
    model = get_peft_model(model, LoraConfig(
        r=int(cfg["lora"]["r"]), lora_alpha=int(cfg["lora"]["alpha"]), lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=str(targets) if isinstance(targets, str) else list(targets)))
    model.train()

    ckpt = args.warmup / meta["checkpoints"][0]
    set_peft_model_state_dict(model, load_peft_weights(str(ckpt)))
    named = ordered_lora_params(model)
    dev = next(model.parameters()).device
    state = torch.load(ckpt / "adam_state.pt", map_location="cpu", weights_only=True)
    m_flat, v_flat = flatten_moments(state, named, dev)
    del state

    enc = encode(rows, tok, prof, 8192)
    projs = {d: CountSketchProjector(sum(p.numel() for _, p in named), dim=d,
                                     seed=0, device=dev) for d in args.dims}

    def clk() -> float:
        torch.cuda.synchronize()
        return time.time()

    acc: dict[str, float] = {"backward": 0.0, "concat": 0.0, "adam": 0.0}
    acc.update({f"sketch_d{d}": 0.0 for d in args.dims})

    for r in enc:
        batch = {k: torch.tensor([r[k]], device=dev)
                 for k in ("input_ids", "attention_mask", "labels")}
        t = clk(); model.zero_grad(set_to_none=True); model(**batch).loss.backward()
        acc["backward"] += clk() - t
        t = clk(); flat = torch.cat([p.grad.reshape(-1).to(torch.float32) for _, p in named])
        acc["concat"] += clk() - t
        t = clk(); gamma = adam_precondition(flat, m_flat, v_flat); acc["adam"] += clk() - t
        for d in args.dims:
            t = clk(); projs[d].project(gamma); acc[f"sketch_d{d}"] += clk() - t
        del flat, gamma

    n = len(enc)
    med = sorted(len(r["input_ids"]) for r in enc)[n // 2]
    primary = args.dims[0]
    pipeline = acc["backward"] + acc["concat"] + acc["adam"] + acc[f"sketch_d{primary}"]
    print(f"\nper row over {n} rows (median {med} tokens), pipeline uses d={primary}:")
    for key, total in acc.items():
        share = "" if key.startswith("sketch_d") and not key.endswith(str(primary)) \
            else f"{100 * total / pipeline:5.1f}%"
        print(f"  {key:<14} {total / n:7.3f}s  {share}")
    print(f"  {'TOTAL':<14} {pipeline / n:7.3f}s")
    print(f"\nfull run projection: 2203 rows x 4 ckpts x {pipeline / n:.3f}s = "
          f"{2203 * 4 * pipeline / n / 3600:.2f} GPU-hours")


if __name__ == "__main__":
    main()
