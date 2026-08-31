# ABOUTME: Pre-flight the t2_9284_da716 mixture before a training run: exact think-marker
# ABOUTME: form on every assistant turn, token-length ceiling, and assistant-only mask sanity.

"""Verify the mixture is safe to train on.

Run: uv run python scratch/verify_t2_da716_mixture.py [--path data/t2_9284_da716_10k.jsonl]

Three things decide whether a run is valid, and all three are silent failures if wrong:

1. THINK MARKERS. Qwen3.6 renders think tags on every assistant turn. A non-reasoning row
   must carry the EXACT empty marker `<think>\\n\\n</think>\\n\\n`; a bare or malformed one
   changes what the generation-boundary rule masks. Counted exactly, not by regex presence.
2. TOKEN CEILING. max_seq_len must exceed the longest row or rows truncate mid-turn, which
   trains the model to stop early. Reported as the true max, not an assumption.
3. ASSISTANT-ONLY MASK. build_labels must supervise assistant completions and mask prompt
   tokens. Checked on real rows by counting supervised vs masked positions.
"""

import json
from collections import Counter
from pathlib import Path

import fire

EMPTY = "<think>\n\n</think>\n\n"


def main(path: str = "data/t2_9284_da716_10k.jsonl", model: str = "Qwen/Qwen3.6-27B",
         synth_label: str = "difficult_advice_v2") -> None:
    """Report think-marker counts, token lengths and label-mask behaviour.

    Args:
        path: Built mixture JSONL.
        model: Model whose ModelProfile and tokenizer define the rules.
        synth_label: `source` value of the reasoning half, so the mask check samples one
            row of each kind (real trace vs empty marker).
    """
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    print(f"rows: {len(rows)}   sources: {dict(Counter(r['source'] for r in rows).most_common(4))}")

    # ---- 1. think markers, counted exactly --------------------------------------
    # RULE: every assistant turn carries a think block — a real trace where the response
    # reasoned, the exact empty marker where it did not. This is stricter than Qwen3.6's
    # stock template, which strips reasoning from conversation HISTORY and so leaves
    # earlier assistant turns bare; the builder re-inserts the empty marker on those
    # (362 of them here). Matching the repo's preserve-thinking policy is what makes
    # `model_profile.think_census` report absent == 0, and keeps the generation-boundary
    # rule in `src/train/masking.py` applying uniformly: a turn with no think block has no
    # forced span and would be supervised whole, under a different rule than every other.
    real = empty = bare = malformed = 0
    turns = 0
    for r in rows:
        for seg in r["text"].split("<|im_start|>assistant\n")[1:]:
            turns += 1
            body = seg.split("<|im_end|>")[0]
            if body.startswith(EMPTY):
                empty += 1
            elif body.startswith("<think>\n") and "</think>" in body:
                real += 1
            elif "<think>" not in body:
                bare += 1
            else:
                malformed += 1
    print(f"\nassistant turns: {turns}  (rows: {len(rows)})")
    print(f"  real reasoning trace   : {real}")
    print(f"  exact empty marker     : {empty}")
    print(f"  BARE (no think block)  : {bare}   <- must be 0")
    print(f"  MALFORMED              : {malformed}   <- must be 0")
    assert bare == 0 and malformed == 0, "not every assistant turn carries a think block"

    # ---- 2. token ceiling --------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    lens = [len(tok(r["text"])["input_ids"]) for r in rows]
    lens.sort()
    n = len(lens)
    print(f"\ntoken length: max {lens[-1]}  p99 {lens[int(n * .99)]}  "
          f"p95 {lens[int(n * .95)]}  median {lens[n // 2]}  mean {sum(lens) / n:.0f}")
    for cap in (4096, 8192):
        over = sum(1 for x in lens if x > cap)
        print(f"  rows exceeding {cap}: {over}")
    print(f"  total tokens in epoch: {sum(lens):,}")

    # ---- 3. assistant-only mask --------------------------------------------------
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model_profile import model_profile, think_census
    from src.train.masking import build_labels

    profile = model_profile(model)
    da = next(r for r in rows if r["source"] == synth_label)
    t2 = next(r for r in rows if r["source"] != synth_label)
    census = think_census([r["text"] for r in rows])
    print(f"\nthink_census (repo yardstick): {census}")
    assert census["absent"] == 0, "absent must be 0 under thinking: true"

    print("\nassistant-only mask (build_labels):")
    for name, r in ((f"{synth_label} (real trace)", da), ("table2 (empty marker)", t2)):
        out = build_labels(r["text"], tok, 8192, profile, "all")
        labels = out["labels"]
        sup = sum(1 for x in labels if x != -100)
        print(f"  {name:30s} {len(labels):5d} tokens, {sup:5d} supervised "
              f"({sup / len(labels) * 100:.1f}%)")
        assert 0 < sup < len(labels), "mask supervises everything or nothing"
    print("\nPASS")


if __name__ == "__main__":
    fire.Fire(main)
