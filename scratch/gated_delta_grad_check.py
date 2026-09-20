# ABOUTME: One gated-delta op, two implementations: how far fla's bf16 kernel's output and gradients
# ABOUTME: sit from transformers' torch reference (which runs the recurrence in fp32). Needs a GPU + fla.

"""Which backward is the accurate one? (2026-09-20, the train-optim A/B.)

At step 0 of scratch/ab_train_optim.py — identical weights, identical data — the fla arms
report loss 0.8968 / grad-norm 2.789 against the torch-fallback arms' 0.8961 / 2.555. The
loss curves cannot adjudicate that: an arm that is mathematically IDENTICAL to the baseline
drifts from it by as much as the fla arms do. So compare the op in isolation, at the model's
own head shapes, against the torch implementation — which upcasts to fp32 internally and is
the definition the fused kernel approximates.

    uv run python scratch/gated_delta_grad_check.py            # fla's default backward (TileLang on Hopper)
    FLA_TILELANG=0 uv run python scratch/gated_delta_grad_check.py   # fla's Triton backward
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def _inputs(cfg, seq_len: int, seed: int):
    """q/k/v/beta in bf16 and g in fp32, shaped and scaled as Qwen3_5GatedDeltaNet feeds them."""
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    rand = lambda *s: torch.randn(*s, device="cuda", generator=g_)  # noqa: E731
    q = rand(1, seq_len, hv, dk).bfloat16()
    k = rand(1, seq_len, hv, dk).bfloat16()
    v = rand(1, seq_len, hv, dv).bfloat16()
    beta = rand(1, seq_len, hv).sigmoid().bfloat16()
    g = -torch.rand(hv, device="cuda", generator=g_).mul(16).log1p() * F.softplus(rand(1, seq_len, hv))
    dout = rand(1, seq_len, hv, dv).bfloat16()
    return [q, k, v, beta, g.float()], dout


def _run(fn, tensors, dout):
    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    q, k, v, beta, g = leaves
    out, _ = fn(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True)
    out.backward(dout.to(out.dtype))
    return out.detach().float(), [t.grad.detach().float() for t in leaves]


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def main() -> None:
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").text_config
    print(f"fla backward backend: FLA_TILELANG={os.environ.get('FLA_TILELANG', '<default>')}  "
          f"gpu={torch.cuda.get_device_name(0)}")
    print(f"{'T':>6} {'seed':>4} | {'out':>9} | " + " ".join(f"{n:>9}" for n in
          ("dq", "dk", "dv", "dbeta", "dg")) + " | grad-norm ratio fla/torch")
    for seq_len in (512, 2048, 8000):
        for seed in (0, 1):
            tensors, dout = _inputs(cfg, seq_len, seed)
            ref_out, ref_grads = _run(torch_chunk_gated_delta_rule, tensors, dout)
            fla_out, fla_grads = _run(chunk_gated_delta_rule, tensors, dout)
            total = lambda gs: torch.sqrt(sum(x.norm() ** 2 for x in gs))  # noqa: E731
            print(f"{seq_len:>6} {seed:>4} | {_rel(fla_out, ref_out):9.2e} | "
                  + " ".join(f"{_rel(a, b):9.2e}" for a, b in zip(fla_grads, ref_grads))
                  + f" | {float(total(fla_grads) / total(ref_grads)):.4f}")


if __name__ == "__main__":
    main()
