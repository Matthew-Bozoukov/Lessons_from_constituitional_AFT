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

    def run(feats_in, packed: bool, pad_extra: int = 0):
        """Per-example float logits at every position (list, in feats_in order) and the LoRA grad."""
        net.zero_grad(set_to_none=True)
        outs = []
        if packed:
            b = _collate_packed(feats_in)
            b = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            labels, segments = b.pop("labels"), b.pop("segments")
            full = net(**b, use_cache=False).logits[0].detach().float()
            off = 0
            for f in feats_in:
                outs.append(full[off:off + len(f["input_ids"])]); off += len(f["input_ids"])
            keep = supervised_positions(labels)
            net.zero_grad(set_to_none=True)
            loss = seq_mean_token_mean_loss(net(**b, use_cache=False, logits_to_keep=keep).logits,
                                            labels, gb, keep, segments)
            loss.backward()
        else:
            loss = torch.zeros((), device="cuda")
            for f in feats_in:
                f2 = f if not pad_extra else {"input_ids": f["input_ids"] + [tokenizer.pad_token_id] * pad_extra,
                                              "labels": f["labels"] + [-100] * pad_extra}
                b = {k: v.to("cuda") for k, v in _collate_padded([f2], tokenizer.pad_token_id).items()}
                if pad_extra:
                    b["attention_mask"][:, -pad_extra:] = 0
                labels = b.pop("labels")
                out = net(**b, use_cache=False).logits
                outs.append(out[0, : len(f["input_ids"])].detach().float())
                l = seq_mean_token_mean_loss(out, labels, gb); l.backward(); loss = loss + l.detach()
        grad = torch.cat([p.grad.flatten().float() for p in trainable]).clone()
        return outs, float(loss), grad

    def compare(name, A, B, feats_a, feats_b):
        """Mean |dlogit| and top-1 disagreement over each example's SUPERVISED positions."""
        rows = []
        for k, f in enumerate(feats_a):
            kb = feats_b.index(f)
            sup = torch.tensor([i for i, v in enumerate(f["labels"][1:]) if v != -100], device="cuda")
            a, b = A[k][sup], B[kb][sup]
            rows.append((float((a - b).abs().mean()), float((a.argmax(-1) != b.argmax(-1)).float().mean()),
                         float((a - b).abs().max())))
        print(f"{name:52s} mean|d| {np.mean([r[0] for r in rows]):.4f}  top1 disagree {100*np.mean([r[1] for r in rows]):.2f}%  max|d| {max(r[2] for r in rows):.2f}")
        return rows

    alone, alone_loss, alone_grad = run(feats, packed=False)
    alone_pad, _, _ = run(feats, packed=False, pad_extra=37)         # NOISE FLOOR: same example, other shape
    pack1, pack_loss, pack_grad = run(feats, packed=True)              # the trainer's pack
    shuffled = feats[::-1]                                             # LEAK PROBE: every example gets new neighbours
    pack2, _, _ = run(shuffled, packed=True)

    print(f"\nloss  alone {alone_loss:.6f}  packed {pack_loss:.6f}  rel {abs(alone_loss - pack_loss) / alone_loss:.2e}")
    cos = float((alone_grad @ pack_grad) / (alone_grad.norm() * pack_grad.norm()))
    print(f"LoRA grad, packed vs alone: cosine {cos:.6f}  rel diff {float((pack_grad - alone_grad).norm() / alone_grad.norm()):.2e}\n")
    floor = compare("noise floor: alone vs alone+37 pad (kernel shape only)", alone, alone_pad, feats, feats)
    leak = compare("LEAK PROBE: packed vs packed with reversed neighbours", pack1, pack2, feats, shuffled)
    compare("packed vs alone", pack1, alone, feats, feats)
    f_m, l_m = np.mean([r[0] for r in floor]), np.mean([r[0] for r in leak])
    print(f"\n>>> verdict: {'NO LEAK — neighbours change nothing beyond kernel-shape noise' if l_m <= 2 * f_m + 1e-6 else 'LEAK — an example depends on its neighbours'}"
          f"  (leak probe mean|d| {l_m:.4f} vs noise floor {f_m:.4f})")


if __name__ == "__main__":
    fire.Fire(main)
