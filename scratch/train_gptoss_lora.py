# ABOUTME: LoRA SFT of gpt-oss-120b on Harmony-rendered data: 4-bit NF4 base, MoE-aware
# ABOUTME: targeting, token-budgeted dynamic batching, and Harmony channel-aware loss masking.

"""Fine-tune gpt-oss-120b with QLoRA.

Run: uv run python scratch/train_gptoss_lora.py --config configs/train/lora_gptoss120b_t2_da716.yaml

Why this is a separate script rather than a flag on src/train/train_lora.py: that trainer
refuses unprofiled families by design (`MODEL_PROFILES = (QWEN36_PROFILE,)`), and every
assumption in its masking is a Qwen assumption — `<think>` prefill, an empty-think marker,
`<|im_end|>` turn ends. Harmony has none of those. Bolting a second family into it would
mean two masking regimes behind one interface, which is how a mask silently trains the
wrong tokens. This lives in scratch/ until a human promotes it (CLAUDE.md).

HARMONY MASKING, verified by rendering the model's own template (2026-08-15):

    prompt      : ...<|start|>user<|message|>Q<|end|><|start|>assistant
    generated   : <|channel|>analysis<|message|>R<|end|>
                  <|start|>assistant<|channel|>final<|message|>A<|return|>

The generation prompt stops at `<|start|>assistant`, so the model itself emits the channel
marker — including the CHOICE of whether to open an analysis channel at all. Supervision
therefore begins immediately after the first `<|start|>assistant` of each assistant span
and runs to `<|return|>`/`<|end|>`. That is the same generation-boundary principle
src/train/masking.py applies to Qwen (never train a token the model does not generate),
expressed in Harmony's vocabulary.

NO EMPTY-ANALYSIS MARKER. Qwen needs `<think>\\n\\n</think>\\n\\n` on non-reasoning turns
because its template always renders the tags. Harmony omits the analysis channel entirely
instead, so build_harmony_dataset.py drops the marker rather than translating it.
"""

import json
import math
import os
import sys
from pathlib import Path

import fire
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.dynamic_batching import plan_micro_batches  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ASSISTANT_HEADER = "<|start|>assistant"
TURN_ENDS = ("<|return|>", "<|end|>")


def build_labels(text: str, tokenizer, max_length: int) -> dict:
    """Tokenize a Harmony conversation, supervising only assistant-generated tokens.

    Everything up to and including each `<|start|>assistant` is context the model is given,
    not output it produces, so it is masked to -100. Tokens from just after that header
    through the turn's terminator are supervised — this includes the `<|channel|>analysis`
    marker, because choosing whether to reason IS part of what the model generates.

    Args:
        text: A fully rendered Harmony conversation.
        tokenizer: The gpt-oss tokenizer.
        max_length: Truncation ceiling.

    Returns:
        `input_ids`, `attention_mask` and `labels`.
    """
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)

    # Character spans the model generates: after each assistant header, to the turn end.
    spans, cur = [], 0
    while True:
        h = text.find(ASSISTANT_HEADER, cur)
        if h == -1:
            break
        start = h + len(ASSISTANT_HEADER)
        ends = [text.find(t, start) for t in TURN_ENDS]
        ends = [e + len(t) for e, t in zip(ends, TURN_ENDS) if e != -1]
        end = min(ends) if ends else len(text)
        spans.append((start, end))
        cur = end

    assert spans, "no assistant span found; nothing would be supervised"
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if any(s >= a and e <= b for a, b in spans):
            labels[i] = ids[i]
    assert any(x != -100 for x in labels), "mask supervises nothing"
    return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}


def moe_target_parameters(n_layers: int, every: int) -> list[str]:
    """Expert projections to adapt, per the OpenAI cookbook's MoE guidance.

    The cookbook targets `<layer>.mlp.experts.gate_up_proj` and `down_proj` at a few spread
    layers (7/15/23 on the 20b) alongside `target_modules="all-linear"`, noting it is
    "important to include the projection layers within the expert modules". A dense-only
    regex — like the Qwen config's q/k/v/o + gate/up/down — matches none of them, so the
    adapter would miss the parameters that carry MoE capacity.

    Args:
        n_layers: Transformer layers in the model.
        every: Adapt one layer in this many.

    Returns:
        Parameter-name suffixes for LoraConfig.target_parameters.
    """
    layers = list(range(every - 1, n_layers, every))
    return [f"{i}.mlp.experts.{p}" for i in layers for p in ("gate_up_proj", "down_proj")]


def main(config: str) -> None:
    """Train the adapter as described by the config."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = OmegaConf.load(config)
    out_dir = Path(cfg.output_dir) / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> output {out_dir}\n>>> base {cfg.model}")

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=str(cfg.data_path), split="train")
    print(f">>> rows {len(ds)}")

    max_len = int(cfg.train.max_seq_len)
    feats = [build_labels(r["text"], tok, max_len) for r in ds]
    lens = [len(f["input_ids"]) for f in feats]
    print(f">>> tokens: total {sum(lens):,}  max {max(lens)}  median {sorted(lens)[len(lens)//2]}")
    sup = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
    print(f">>> supervised {sup:,} / {sum(lens):,} tokens ({sup/sum(lens)*100:.1f}%)")

    # --- dynamic batching plan -------------------------------------------------
    # Budget is REQUIRED here, unlike the Qwen trainer which can fall back to the longest
    # row. That fallback packs to an unverified ceiling and its failure mode is an OOM
    # partway through a multi-hour run, which on a 120B model is expensive to discover.
    budget = int(cfg.train.get("token_budget", 0))
    assert budget > 0, (
        "train.token_budget must be set explicitly for gpt-oss: no measured "
        "train_memory entry exists. Run scratch/probe_batch_memory.py on the target GPU.")
    gb = int(cfg.train.batch_size) * int(cfg.train.grad_accum)
    steps = math.ceil(len(ds) / gb * float(cfg.train.epochs))
    passes = sum(len(plan_micro_batches(lens[i:i + gb], budget))
                 for i in range(0, len(lens) - gb + 1, gb))
    print(f">>> dynamic batching: budget {budget} padded tokens, global_batch {gb}")
    print(f">>> {steps} optimizer steps | ~{passes} passes/epoch vs {len(lens)} at batch 1 "
          f"({len(lens)/max(passes,1):.1f}x fewer)")

    if cfg.train.get("plan_only", False):
        print(">>> plan_only: stopping before model load")
        return

    # --- base model ------------------------------------------------------------
    # gpt-oss ships in MXFP4. bitsandbytes NF4 is a DIFFERENT scheme, so this path
    # dequantizes MXFP4 and requantizes to NF4 — quantizing twice. The cookbook instead
    # loads Mxfp4Config(dequantize=True) and trains in bf16 (~240GB, 4xH200). Set
    # quantization: mxfp4_bf16 to follow the cookbook exactly; nf4 trades unmeasured
    # quality for fitting on ONE GPU.
    mode = str(cfg.train.get("quantization", "nf4"))
    if mode == "nf4":
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
    elif mode == "mxfp4_bf16":
        from transformers import Mxfp4Config
        qcfg = Mxfp4Config(dequantize=True)
    else:
        raise ValueError(f"unknown quantization mode {mode!r}")
    print(f">>> quantization: {mode}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, quantization_config=qcfg, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation=str(cfg.train.get("attn", "eager")),
        use_cache=False, trust_remote_code=True)
    if mode == "nf4":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(cfg.train.get("gradient_checkpointing", True)))

    n_layers = model.config.num_hidden_layers
    tp = moe_target_parameters(n_layers, int(cfg.lora.get("expert_every", 8)))
    print(f">>> {n_layers} layers; adapting expert projections on "
          f"{len(tp)//2} of them: {tp[:4]}...")
    peft_cfg = LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout), bias="none", task_type="CAUSAL_LM",
        target_modules=str(cfg.lora.get("target_modules", "all-linear")),
        target_parameters=tp)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    (out_dir / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha(), "base_model": cfg.model, "data_path": str(cfg.data_path),
        "n_rows": len(ds), "tokens": sum(lens), "supervised_tokens": sup,
        "quantization": mode, "token_budget": budget, "steps": steps,
        "passes_per_epoch": passes, "target_parameters": tp,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, indent=2))
    print(f">>> wrote {out_dir/'run_meta.json'}")
    print(">>> TRAINER NOT YET WIRED — see the note at the bottom of this file.")


if __name__ == "__main__":
    fire.Fire(main)

# WHAT IS AND IS NOT DONE
#
# Done and testable without a GPU: Harmony rendering (build_harmony_dataset.py), the
# channel-aware label mask, MoE target-parameter selection, and the dynamic-batching plan.
# `plan_only: true` exercises all of it offline.
#
# NOT done: the training loop itself. src/train/dynamic_batching.py's DynamicBatchTrainer
# subclasses TRL's SFTTrainer and computes seq-mean-token-mean loss over a routed step; it
# assumes the Qwen collator and label shapes. Wiring it here needs either that trainer
# generalised or a small local equivalent, and either way it needs a REAL probe run for
# token_budget first — planning against a guessed budget is what turns a 4-hour run into an
# OOM at hour three.
