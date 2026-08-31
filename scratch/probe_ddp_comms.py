# ABOUTME: Measures what DDP's gradient all-reduce actually costs per step — the EXPOSED
# ABOUTME: cost after backward-pass overlap, not the raw collective time.

"""How much does inter-GPU communication really cost this fine-tune?

Three numbers, because they answer different questions and only the last one is the answer:

  1. **Raw all-reduce** — time a collective on a tensor the size of the LoRA gradients, with
     nothing else running. This is the number napkin math produces. It OVERSTATES the cost.
  2. **Exposed cost** — real training steps with DDP syncing, versus the same steps inside
     `no_sync()` (which accumulates gradients locally and skips the collective entirely).
     The difference is what communication actually adds to wall-clock. DDP buckets gradients
     and launches each bucket's all-reduce *during* backward, so most of (1) hides behind
     compute that was going to happen anyway; the exposed cost can be near zero.
  3. **Straggler cost** — the same steps with every rank given IDENTICAL work, versus ranks
     given REAL sampled sequence lengths. That difference is rank imbalance, not comms.

Distinguishing 2 from 3 is the point. Both look like "distributed overhead" in a wall-clock
comparison against one GPU, but only one is fixed by faster interconnect — the other is fixed
by how you batch.

    PYTHONPATH=. torchrun --nproc_per_node=4 scratch/probe_ddp_comms.py --steps 12
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import fire


def main(model: str = "Qwen/Qwen3.6-27B",
         seq_len: int = 1024,
         steps: int = 12,
         warmup: int = 3,
         lora_r: int = 64,
         lengths_file: str | None = None) -> None:
    """Measure raw, exposed and straggler costs under torchrun.

    Args:
        model: Base model.
        seq_len: Fixed sequence length for the balanced (comms-isolating) measurement.
        steps: Timed steps per condition.
        warmup: Untimed steps first — CUDA graphs, allocator growth and NCCL channel setup
            all land on the first few iterations and would otherwise dominate.
        lora_r: LoRA rank, so the gradient payload matches training.
        lengths_file: Optional jsonl of training rows; if given, a third condition draws
            real per-rank lengths to separate straggler cost from comms cost.
    """
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForImageTextToText

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    log = print if rank == 0 else (lambda *a, **k: None)

    m = AutoModelForImageTextToText.from_pretrained(
        model, dtype=torch.bfloat16, device_map={"": local})
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(m, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    ))
    m.train()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_train > 1e6, f"LoRA matched almost nothing ({n_train:,} params)"
    ddp = DDP(m, device_ids=[local], find_unused_parameters=False)
    log(f">>> world={world} trainable={n_train/1e6:.0f}M "
        f"payload={n_train*2/1e6:.0f} MB (bf16)")

    # --- 1. raw collective, nothing else running -------------------------------------
    buf = torch.empty(n_train, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    raw_ms = (time.perf_counter() - t0) / 20 * 1000
    log(f">>> raw all-reduce of the gradient payload: {raw_ms:.1f} ms "
        f"({n_train*2/1e6/raw_ms:.0f} MB/s effective)")

    def run(sync: bool, lengths: list[int] | None) -> float:
        """Median step time (ms). `sync=False` skips the collective via no_sync()."""
        import contextlib
        times = []
        for i in range(warmup + steps):
            L = lengths[i % len(lengths)] if lengths else seq_len
            ids = torch.randint(1000, 100000, (1, L), device="cuda")
            ctx = contextlib.nullcontext() if sync else ddp.no_sync()
            dist.barrier(); torch.cuda.synchronize()
            t = time.perf_counter()
            with ctx:
                out = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
                out.loss.backward()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) * 1000
            ddp.zero_grad(set_to_none=True)
            if i >= warmup:
                times.append(dt)
        # Every rank waits for the slowest, so the honest step time is the max across ranks.
        t = torch.tensor([statistics.median(times)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    # --- 2. exposed comms cost: identical work on every rank, sync vs no_sync ---------
    synced = run(True, None)
    unsynced = run(False, None)
    log(f"\n>>> BALANCED work ({seq_len} tokens on every rank):")
    log(f"    with all-reduce   : {synced:8.1f} ms/step")
    log(f"    with no_sync()    : {unsynced:8.1f} ms/step")
    log(f"    EXPOSED comms cost: {synced-unsynced:8.1f} ms "
        f"({(synced-unsynced)/synced*100:.2f}% of step)")
    log(f"    hidden by overlap : {raw_ms-(synced-unsynced):8.1f} ms of the {raw_ms:.1f} ms raw")

    # --- 3. straggler cost: real length distribution, same collective ------------------
    if lengths_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        rows = [json.loads(l) for l in Path(lengths_file).open()]
        allv = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
        # Each rank draws its own lengths -- that imbalance is exactly the effect.
        g = torch.Generator().manual_seed(1234 + rank)
        picks = [allv[int(torch.randint(len(allv), (1,), generator=g))]
                 for _ in range(warmup + steps)]
        real = run(True, picks)
        log(f"\n>>> REAL length distribution (ranks draw independently):")
        log(f"    step time         : {real:8.1f} ms/step")
        log(f"    vs balanced       : {real-synced:8.1f} ms "
            f"({(real-synced)/real*100:.1f}% of step lost to rank imbalance)")
        log("    ^ this is straggler cost, NOT communication. Faster interconnect will not "
            "touch it; length bucketing will.")

    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
