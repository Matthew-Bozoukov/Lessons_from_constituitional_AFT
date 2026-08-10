# ABOUTME: Empirically finds the largest per-device batch that survives a real forward +
# ABOUTME: backward at the corpus's LONGEST sequence, by allocating until CUDA OOMs.

"""Measure the batch-size wall, don't estimate it.

Napkin math over hidden size, layer count and vocab gets the order of magnitude right and
the boundary wrong: it misses allocator fragmentation, cuDNN workspaces, the hybrid Mamba
layers' real footprint, and whatever the loss path materialises. This runs the actual
thing — same dtype, same gradient checkpointing, same LoRA config, same `loss_type` path
as training — on a batch of `batch` copies of the LONGEST row in the corpus, and reports
peak allocated memory or the OOM.

Every case is run in a fresh subprocess. A CUDA OOM can leave the allocator and the
autograd graph in a state where the *next* measurement is wrong or spuriously fails, so
in-process sweeps quietly under-report the wall.

    uv run python scratch/probe_batch_memory.py --data data/mixture.jsonl
    uv run python scratch/probe_batch_memory.py --data data/mixture.jsonl --only 16
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import fire


def _one_case(model: str, seq_len: int, batch: int, grad_ckpt: bool, lora_r: int) -> None:
    """Run a single forward+backward and print the peak memory, or the OOM. Child process."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # The SAME auto class the trainer uses for this checkpoint (`model_class:
    # image_text_to_text`, train_lora.py:215). AutoModelForCausalLM loads a different
    # module tree, so the LoRA target regex matches nothing and every case dies before
    # allocating anything -- a "failure" that says nothing about memory.
    m = AutoModelForImageTextToText.from_pretrained(
        model, dtype=torch.bfloat16, device_map={"": 0})
    if grad_ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(m, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    ))
    m.train()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_train > 1e6, f"LoRA matched almost nothing ({n_train:,} trainable params)"
    after_load = torch.cuda.max_memory_allocated() / 1024**3

    # Real token ids, not zeros: a degenerate batch can take different kernel paths.
    ids = torch.randint(1000, 100000, (batch, seq_len), device="cuda")
    labels = ids.clone()
    out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    out.loss.backward()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"RESULT ok batch={batch} seq={seq_len} lora={n_train/1e6:.0f}M weights={after_load:.1f} "
          f"peak={peak:.1f} reserved={reserved:.1f} card={total:.1f}")


def main(data: str = "data/mixture.jsonl",
         model: str = "Qwen/Qwen3.6-27B",
         batches="1,2,4,6,8,16",
         seq_len: int | None = None,
         grad_ckpt: bool = True,
         lora_r: int = 64,
         only: int | None = None) -> None:
    """Sweep per-device batch sizes at the corpus's longest sequence length.

    Args:
        data: Training jsonl (`text` per row); its longest row sets the sequence length.
        model: Base model to load.
        batches: Comma-separated batch sizes to try.
        seq_len: Override the measured longest length.
        grad_ckpt: Match training's gradient checkpointing setting.
        lora_r: LoRA rank, to match the adapter's memory footprint.
        only: Run just this one batch size.
    """
    if os.environ.get("_PROBE_CASE"):          # child process: run one case and exit
        c = json.loads(os.environ["_PROBE_CASE"])
        _one_case(c["model"], c["seq_len"], c["batch"], c["grad_ckpt"], c["lora_r"])
        return

    if seq_len is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        rows = [json.loads(l) for l in Path(data).open()]
        lens = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
        seq_len = max(lens)
        print(f">>> corpus: {len(rows):,} rows, longest {seq_len:,} tokens "
              f"(median {sorted(lens)[len(lens)//2]:,})")

    # fire turns a bare `--batches 1,4,8` into a TUPLE, not a string, so str() of it is
    # "(1, 4, 8)" and a naive split on commas yields "(1". Accept either shape.
    if isinstance(batches, (list, tuple)):
        todo = [int(b) for b in batches]
    else:
        todo = [int(b) for b in str(batches).strip("()[] ").split(",") if b.strip()]
    if only:
        todo = [int(only)]
    print(f">>> probing batches {todo} at seq_len {seq_len:,}, "
          f"gradient_checkpointing={grad_ckpt}\n")
    results = []
    for b in todo:
        env = {**os.environ, "_PROBE_CASE": json.dumps(
            {"model": model, "seq_len": seq_len, "batch": b,
             "grad_ckpt": grad_ckpt, "lora_r": lora_r})}
        r = subprocess.run([sys.executable, __file__], env=env,
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
        if line:
            print(f"  batch {b:>3}: {line[len('RESULT ok '):]}")
            results.append((b, True))
        else:
            oom = "OutOfMemoryError" in r.stderr or "out of memory" in r.stderr.lower()
            why = "CUDA OOM" if oom else (r.stderr.strip().splitlines() or ["failed"])[-1]
            print(f"  batch {b:>3}: FAILED — {why[:110]}")
            results.append((b, False))

    ok = [b for b, good in results if good]
    print(f"\n>>> largest batch that completes fwd+bwd at {seq_len:,} tokens: "
          f"{max(ok) if ok else 'NONE'}")


if __name__ == "__main__":
    fire.Fire(main)
