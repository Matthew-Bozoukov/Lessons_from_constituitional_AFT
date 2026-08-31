# ABOUTME: 100-row Tinker smoke for gpt-oss-120b: builds Harmony Datums by hand (no cookbook
# ABOUTME: renderer exists for gpt-oss) and measures real throughput before a full run.

"""Measure what a Tinker fine-tune of gpt-oss-120b actually costs in time.

Run: TINKER_API_KEY=... uv run python scratch/tinker_gptoss_smoke.py [--rows 100]

WHY THIS EXISTS. Tinker publishes price ($0.737/M training tokens for gpt-oss-120b at 32K)
but NOT throughput, queueing or per-step latency, so a full-run duration cannot be derived
from the docs. This runs a real 100-row slice and reports tokens/sec and per-step
round-trip, from which the 6.33M-token full run is arithmetic. At ~63k tokens the smoke
itself costs about $0.05.

NO GPT-OSS RENDERER. `tinker_cookbook.renderers.get_renderer` covers qwen3_5, deepseekv3,
nemotron3 and kimi_k26 — there is no gpt-oss/Harmony entry (docs checked 2026-08-15), so
`conversation_to_datum` is unavailable and Datums are built here instead: Harmony text from
the model's own chat template, then a per-token weight vector from the same
generation-boundary rule used in scratch/train_gptoss_lora.py.

HYPERPARAMETERS mirror configs/train/lora_qwen36_t2_9284_da716_dynbatch_2xh200.yaml so the
arms stay comparable: rank 64, lr 1e-4, cosine with 5% warmup, global batch 16, 1 epoch.
Tinker's API takes a learning rate per optim_step, so the schedule is computed here rather
than configured.

UNVERIFIED, AND THE FIRST THING TO CHECK AGAINST A REAL RUN: whether `weights` align with
the token at the same index or are shifted by one for next-token prediction. The docs say
only "Weight 0 = prompt, 1 = completion". If Tinker shifts internally, the mask here is
correct as written; if it does not, every weight must move one position left. Getting this
wrong trains on the token BEFORE each supervised token — a silent, plausible-looking error.
`--inspect` prints the aligned (token, weight) pairs so a human can eyeball it.
"""

import json
import math
import time
from pathlib import Path

import fire

MODEL = "openai/gpt-oss-120b"
ASSISTANT_HEADER = "<|start|>assistant"
# <|call|> ends a turn exactly as the other two do; a tool-calling turn closes with it and
# nothing else, so omitting it here let a span run past the end of the message.
TURN_ENDS = ("<|return|>", "<|end|>", "<|call|>")
# The opener a turn uses when it does NOT reason. Harmony has no empty-analysis form, so
# this exact sequence IS the "did not reason" decision -- see rule 2 in supervised_spans.
NO_REASON_OPENER = "<|channel|>final<|message|>"
DATA = "data/t2_da716_harmony.jsonl"

# From the Qwen arm, so the two runs differ in model rather than recipe.
# Rank 32, NOT the Qwen arm's 64: Tinker rejects 64 for this model --
# 'lora_config.rank 64 exceeds max LoRA rank 32 for model openai/gpt-oss-120b'
# (observed 2026-08-15). A cross-model comparison therefore differs in adapter
# capacity as well as base model, and the write-up must say so.
RANK, LR, WARMUP_FRAC, GLOBAL_BATCH, EPOCHS = 32, 1.0e-4, 0.05, 16, 1


def supervised_spans(text: str, mask_channel_choice: bool = True) -> list[tuple[int, int]]:
    """Character spans the model must generate, per the generation-boundary rule.

    Supervise exactly what the model produces at inference — no more, no less. Two rules,
    both from observed failures of the naive "everything after each assistant header"
    version (2026-08-15):

    1. CONTINUATION HEADERS ARE GENERATED, NOT PREFILLED. Only the FIRST
       ``<|start|>assistant`` of a response comes from the harness
       (``add_generation_prompt``). When a response spans two messages — analysis then
       final — the model itself must emit ``<|start|>assistant<|channel|>final<|message|>``
       after closing the trace. Masking it trained a model that ended its analysis and then
       improvised ``<|start|>final<|message|>``, a sequence present in none of the 10,000
       training rows and in no base-model sample.

    2. THE CHANNEL CHOICE IS SUPERVISED ONLY WHERE THE ROW DEMONSTRATES A CHOICE. Harmony
       encodes "did not reason" by OMITTING the analysis channel, so supervising
       ``<|channel|>final<|message|>`` on a traceless row is a gradient on the decision not
       to reason — and 8,280 of 10,000 rows carry it. Masking that opener keeps everything
       those rows are for (style, refusals, instruction following, tool calls) while
       leaving the reason/don't-reason prior where the base model had it. Rows that DO
       reason keep their opener supervised, so the only gradient on channel choice
       encourages reasoning.

    A tool-calling turn opens ``<|start|>assistant to=functions.NAME<|channel|>commentary``
    rather than with the bare opener, so rule 2 never masks it: choosing to invoke a tool
    is behaviour we are training.

    Args:
        text: A rendered Harmony conversation.
        mask_channel_choice: Apply rule 2. False reproduces the pre-fix behaviour, for
            an A/B of the mask itself.

    Returns:
        (start, end) character offsets of the regions to supervise.
    """
    spans, cur, prev_end = [], 0, None
    while True:
        h = text.find(ASSISTANT_HEADER, cur)
        if h == -1:
            return spans
        body = h + len(ASSISTANT_HEADER)
        # A header sitting exactly where the previous assistant turn ended continues the
        # same response, so the model emits it (rule 1). Anything else is the prefill.
        continuation = prev_end is not None and h == prev_end
        start = h if continuation else body
        if not continuation and mask_channel_choice \
                and text.startswith(NO_REASON_OPENER, body):
            start = body + len(NO_REASON_OPENER)          # rule 2
        ends = [text.find(t, body) for t in TURN_ENDS]
        ends = [e + len(t) for e, t in zip(ends, TURN_ENDS) if e != -1]
        end = min(ends) if ends else len(text)
        spans.append((start, end))
        prev_end = end
        cur = end


def build_weights(text: str, tokenizer,
                  mask_channel_choice: bool = True
                  ) -> tuple[list[int], list[int], list[float]]:
    """Tokenize into (inputs, targets, weights) for next-token prediction.

    Tinker's cross_entropy does NOT shift internally — it requires `target_tokens` in
    loss_fn_inputs beside `weights` (the SDK raises
    "CELossExtraArgs.__init__() missing 1 required positional argument: 'target_tokens'"
    otherwise, observed 2026-08-15). So the shift is explicit here: position i predicts
    token i+1, and its weight is set from whether that TARGET is generated text.

    Args:
        text: Rendered Harmony conversation.
        tokenizer: Tinker's tokenizer for the base model.
        mask_channel_choice: Passed to supervised_spans; see rule 2 there.

    Returns:
        (input_ids, target_tokens, weights), all length N-1.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans = supervised_spans(text, mask_channel_choice)
    assert spans, "no assistant span; nothing would be supervised"
    gen = [(s != e and any(s >= a and e <= b for a, b in spans)) for s, e in offs]
    inputs, targets = ids[:-1], ids[1:]
    weights = [1.0 if gen[i + 1] else 0.0 for i in range(len(inputs))]
    assert len(inputs) == len(targets) == len(weights)
    assert any(x > 0 for x in weights), "weight vector is all zero"
    return inputs, targets, weights


def lr_at(step: int, total: int) -> float:
    """Cosine schedule with linear warmup, matching the Qwen config's shape."""
    warm = max(1, int(total * WARMUP_FRAC))
    if step < warm:
        return LR * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return LR * 0.5 * (1 + math.cos(math.pi * p))


def main(rows: int = 100, data: str = DATA, model: str = MODEL,
         inspect: bool = False, dry_run: bool = False, name: str = "smoke",
         mask_channel_choice: bool = True) -> None:
    """Run a training pass and report throughput.

    Args:
        rows: Dataset rows to use. The default 100 is the smoke; pass the full row count
            for a real run — the code path is identical, which is the point.
        data: Harmony-rendered JSONL from scratch/build_harmony_dataset.py.
        model: Tinker base-model id.
        inspect: Print aligned (token, weight) pairs for the first example and stop.
        dry_run: Build Datums and print the plan without contacting Tinker.
        name: Checkpoint name on Tinker. Name real runs for the data they trained on;
            several checkpoints called "smoke" are indistinguishable later.
        mask_channel_choice: Leave the reason/don't-reason decision unsupervised on rows
            that carry no trace. See rule 2 in supervised_spans.
    """
    import asyncio
    asyncio.run(_run(rows, data, model, inspect, dry_run, name, mask_channel_choice))


async def _run(rows, data, model, inspect, dry_run, name, mask_channel_choice=True):
    """Async body: Tinker's client API is async throughout."""
    src = [json.loads(l) for l in Path(data).open(encoding="utf-8")][:rows]
    assert src, f"no rows read from {data}"

    if dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        client = None
    else:
        import tinker
        service = tinker.ServiceClient()
        client = await service.create_lora_training_client_async(
            base_model=model, rank=RANK)
        tok = client.get_tokenizer()
        print(f">>> Tinker client up: {model} rank={RANK}")

    datums, n_tok, n_sup = [], 0, 0
    lengths = []   # Datum.loss_fn_inputs holds TensorData, which has no len()
    for r in src:
        inp, tgt, w = build_weights(r["text"], tok, mask_channel_choice)
        n_tok += len(inp)
        n_sup += int(sum(w))
        lengths.append(len(inp))
        if dry_run:
            datums.append({"tokens": len(inp), "supervised": int(sum(w))})
        else:
            import tinker
            datums.append(tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(inp),
                loss_fn_inputs={"weights": w, "target_tokens": tgt}))

    steps = math.ceil(len(src) / GLOBAL_BATCH * EPOCHS)
    print(f">>> {len(src)} rows | {n_tok:,} tokens | {n_sup:,} supervised "
          f"({n_sup/n_tok*100:.1f}%)")
    print(f">>> {steps} steps at global batch {GLOBAL_BATCH}, "
          f"lr {LR} cosine w/ {WARMUP_FRAC:.0%} warmup, rank {RANK}")
    print(f">>> mask_channel_choice={mask_channel_choice} "
          f"(rule 2: traceless rows do not train the channel decision)")
    print(f">>> cost at $0.737/M: ${n_tok/1e6*0.737:.3f}")

    if inspect:
        inp, tgt, w = build_weights(src[0]["text"], tok, mask_channel_choice)
        print("\n=== (input -> target, weight) around the first supervised target ===")
        first = next(i for i, x in enumerate(w) if x > 0)
        for i in range(max(0, first - 3), min(len(inp), first + 8)):
            mark = "TRAIN" if w[i] > 0 else "     "
            print(f"  {i:5d} {mark} {w[i]:.0f}  {tok.decode([inp[i]])!r:>16} -> "
                  f"{tok.decode([tgt[i]])!r}")
        print("\nCheck: the first TRAIN row's TARGET is the first token the model "
              "generates (`<|channel|>`).")
        return
    if dry_run:
        print(">>> dry run: no Tinker call made")
        return

    import tinker
    t0 = time.time()
    per_step = []
    for step in range(steps):
        batch = datums[step * GLOBAL_BATCH:(step + 1) * GLOBAL_BATCH]
        if not batch:
            break
        s0 = time.time()
        fb = await client.forward_backward_async(batch, "cross_entropy")
        op = await client.optim_step_async(
            tinker.AdamParams(learning_rate=lr_at(step, steps)))
        fb_r, _ = await fb.result_async(), await op.result_async()
        dt = time.time() - s0
        per_step.append(dt)
        toks = sum(lengths[step * GLOBAL_BATCH:(step + 1) * GLOBAL_BATCH])
        print(f"  step {step+1}/{steps}  {dt:5.1f}s  {toks:6d} tok  "
              f"{toks/dt:7.0f} tok/s  lr {lr_at(step, steps):.2e}")

    total = time.time() - t0
    tps = n_tok / total
    print(f"\n=== RESULT ===")
    print(f"  wall clock        : {total:.1f}s for {n_tok:,} tokens")
    print(f"  throughput        : {tps:,.0f} tokens/sec")
    print(f"  per-step round trip: mean {sum(per_step)/len(per_step):.1f}s  "
          f"min {min(per_step):.1f}s  max {max(per_step):.1f}s")
    # Only meaningful for a partial run; after a full pass the measurement IS the answer.
    # n-1 per row: build_weights drops the last token to form the target shift.
    full = 6_128_470
    if n_tok < full:
        print(f"\n  EXTRAPOLATED full run ({full:,} tokens):")
        print(f"    by throughput   : {full/tps/60:.0f} min")
        print(f"    cost            : ${full/1e6*0.737:.2f}")

    # Save so the export path can be exercised: a run that trains but produces no
    # retrievable checkpoint answers only half the question.
    # BOTH kinds, because they are not interchangeable: save_state writes a resumable
    # training state under weights/, but only save_weights_for_sampler's sampler_weights/
    # checkpoint can be downloaded and converted to PEFT. Saving state alone strands the
    # run behind an extra load-and-resave round trip (hit on 2026-08-15).
    print("\n>>> saving training state")
    state = await (await client.save_state_async(name=name)).result_async()
    print(f"    state  : {state.path}")
    print(">>> saving sampler weights")
    sampler = await (await client.save_weights_for_sampler_async(name=name)).result_async()
    print(f"    sampler: {sampler.path}")
    print("    export with: uv run --no-sync python scratch/tinker_export_adapter.py \\")
    print(f"        --tinker_path {sampler.path} --repo_id LASR-Callum/<name>")


if __name__ == "__main__":
    fire.Fire(main)
