# ABOUTME: The packing gate: every example's logits and its LoRA gradient must be the same packed as alone.
# ABOUTME: Runs on the live Qwen3.6 with the trainer's own collators. Run on a pod: uv run python scratch/pack_equality_check.py

"""Is a packed forward the same computation as the unpacked one? Measure it, per example.

A pack is several examples in one row. If every boundary is respected — varlen attention
from the flash kwargs, `cu_seqlens` in the gated-delta kernel, `seq_idx` in the causal conv —
each example's logits equal the logits it gets alone (to bf16 noise), and the step's LoRA
gradient equals the padded path's. If a boundary leaks, the first tokens of an example see
the tail of the previous one, and the difference concentrates there: the conv has kernel 4,
so a torch-fallback conv leaks exactly 3 positions; unmasked attention leaks everywhere.
This prints the difference per example AND per position band so the two are told apart.

    uv run python scratch/pack_equality_check.py --data_repo <org>/<mix> --data_revision <sha> [--n 6]
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from src.infra.huggingface import resolve_dataset
from src.model_profile import model_profile, render_chat
from src.train.dynamic_batching import seq_mean_token_mean_loss, supervised_positions
from src.train.masking import build_labels
from src.train.train_lora import _collate_packed, _collate_padded


def _rows(data_repo, data_revision, tokenizer, profile, max_len, n, seed=0):
    path, _ = resolve_dataset(data_repo, None, data_revision)
    raw = [json.loads(line) for line in Path(path).open(encoding="utf8")]
    order = np.random.default_rng(seed).permutation(len(raw))
    feats = []
    for i in order:
        r = raw[int(i)]
        text = r.get("text") or render_chat(tokenizer, r["messages"], r.get("tools"),
                                            render_kwargs=profile.render_kwargs)
        f = build_labels(text, tokenizer, max_len, profile, supervise=r.get("supervise") or "all",
                         mask_spans=r.get("mask_spans"))
        if len(f["input_ids"]) <= 1500:  # several must fit one 8000-token pack
            feats.append(f)
        if len(feats) == n:
            return feats
    raise RuntimeError("not enough short rows")


def main(data_repo: str, data_revision: str | None = None, model: str = "qwen36", n: int = 6,
         attn: str = "flash_attention_2", recipe: str = "configs/train/sft.yaml") -> None:
    from peft import LoraConfig, get_peft_model

    cfg = OmegaConf.load(recipe)
    profile = model_profile(model)
    tokenizer = AutoTokenizer.from_pretrained(profile.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    feats = _rows(data_repo, data_revision, tokenizer, profile, int(cfg.train.max_seq_len), n)
    lens = [len(f["input_ids"]) for f in feats]
    print(f">>> {n} rows, lengths {lens}, pack of {sum(lens)} tokens | attn={attn} | "
          f"fla={qwen.chunk_gated_delta_rule is not None} causal_conv1d={qwen.causal_conv1d_fn is not None}")

    net = AutoModelForImageTextToText.from_pretrained(profile.model, dtype=torch.bfloat16,
                                                       device_map={"": 0}, attn_implementation=attn)
    net.config.use_cache = False
    torch.manual_seed(0)
    net = get_peft_model(net, LoraConfig(r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=0.0,
                                         bias="none", task_type="CAUSAL_LM",
                                         target_modules=profile.lora_target_modules))
    net.eval()  # no dropout; gradients still flow to the LoRA params
    trainable = [p for p in net.parameters() if p.requires_grad]
    gb = 16

    # --- alone: one padded pass per example (batch 1, no padding at all) --------------------
    alone_logits, alone_loss = [], torch.zeros((), device="cuda")
    net.zero_grad(set_to_none=True)
    for f in feats:
        b = {k: v.to("cuda") for k, v in _collate_padded([f], tokenizer.pad_token_id).items()}
        labels = b.pop("labels")
        out = net(**b, use_cache=False).logits
        alone_logits.append(out[0].detach().float())
        loss = seq_mean_token_mean_loss(out, labels, gb)
        loss.backward()
        alone_loss += loss.detach()
    alone_grad = torch.cat([p.grad.flatten().float() for p in trainable]).clone()

    # --- packed: the same examples end to end, through the trainer's collator -----------------
    net.zero_grad(set_to_none=True)
    b = _collate_packed(feats)
    b = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
    labels, segments = b.pop("labels"), b.pop("segments")
    out = net(**b, use_cache=False).logits[0].detach().float()
    keep = supervised_positions(labels)
    net.zero_grad(set_to_none=True)
    out_k = net(**b, use_cache=False, logits_to_keep=keep).logits
    packed_loss = seq_mean_token_mean_loss(out_k, labels, gb, keep, segments)
    packed_loss.backward()
    packed_grad = torch.cat([p.grad.flatten().float() for p in trainable])

    # --- report ------------------------------------------------------------------------------
    print(f"\nloss   alone {float(alone_loss):.6f}   packed {float(packed_loss):.6f}   "
          f"rel {abs(float(alone_loss - packed_loss)) / float(alone_loss):.2e}")
    cos = float((alone_grad @ packed_grad) / (alone_grad.norm() * packed_grad.norm()))
    print(f"LoRA grad   cosine {cos:.6f}   rel diff {float((packed_grad - alone_grad).norm() / alone_grad.norm()):.2e}")
    print(f"\n{'example':>7} {'len':>5} | max|dlogit| all positions | positions 0-2 | positions 3+  (bf16 noise ~1e-2..1e-1 on logits of scale ~10)")
    off = 0
    worst = 0.0
    for k, (f, a) in enumerate(zip(feats, alone_logits)):
        L = lens[k]
        p = out[off:off + L]
        d = (p - a).abs()
        worst = max(worst, float(d.max()))
        print(f"{k:7d} {L:5d} | {float(d.max()):12.4f}          | {float(d[:3].max()):11.4f} | {float(d[3:].max()):11.4f}")
        off += L
    print(f"\n>>> verdict: {'PACKED == ALONE (differences at bf16 noise)' if worst < 0.5 and cos > 0.999 else 'DIFFERENT — a boundary leaks; see the position bands'}")


if __name__ == "__main__":
    fire.Fire(main)
