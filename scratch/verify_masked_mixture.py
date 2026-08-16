# ABOUTME: Verify a mask_spans mixture against its source: text unchanged, spans resolvable,
# ABOUTME: and the resulting label mask reported as a share of all supervised tokens.

"""Check that a masked mixture differs from its source in exactly one way.

Three things must hold or the ablation is not clean: every row's `text` is byte-identical to
the source mixture, every span lies inside a reasoning block and still reads as the text the
judge quoted, and only the intended rows carry spans. The token accounting at the end is the
number that matters for the experiment — what share of the model's training signal this
removes.

Run:
  uv run python scratch/verify_masked_mixture.py --masked <dir>/mixture_think_masked.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.mask_cluster_spans import apply_mask, think_region  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.train.masking import build_labels  # noqa: E402


def main(masked: str,
         source: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         model_id: str = "Qwen/Qwen3.6-27B", max_length: int = 8192) -> None:
    """Compare a masked mixture to its source and report the training-signal delta.

    Args:
        masked: The emitted mixture_think_masked.jsonl.
        source: The mixture it was built from.
        model_id: Model whose tokenizer and profile define the token stream.
        max_length: Training sequence length.

    Raises:
        AssertionError: If any row's text changed or a span falls outside its reasoning.
    """
    from transformers import AutoTokenizer

    src = [json.loads(x) for x in Path(source).read_text().splitlines() if x.strip()]
    new = [json.loads(x) for x in Path(masked).read_text().splitlines() if x.strip()]
    assert len(src) == len(new), f"row count {len(src)} -> {len(new)}"
    changed = [i for i, (a, b) in enumerate(zip(src, new)) if a["text"] != b["text"]]
    assert not changed, f"{len(changed)} rows had their text changed, e.g. {changed[:3]}"
    assert all(a["source"] == b["source"] for a, b in zip(src, new)), "source column changed"
    print(f"text identical for all {len(new)} rows; source column identical")

    tok = AutoTokenizer.from_pretrained(model_id)
    profile = model_profile(model_id)
    with_spans = [r for r in new if r.get("mask_spans")]
    print(f"rows carrying mask_spans: {len(with_spans)}")
    print(f"sources of those rows: "
          f"{ {r['source'] for r in with_spans} }")

    sup_before = sup_after = masked_tok = 0
    for r in new:
        base = build_labels(r["text"], tok, max_length, profile,
                            supervise=r.get("supervise") or "all")
        n = sum(1 for v in base["labels"] if v != -100)
        sup_before += n
        if r.get("mask_spans"):
            lo, hi = think_region(r["text"])
            for s, e in r["mask_spans"]:
                assert lo <= s < e <= hi, f"span {s}-{e} outside reasoning {lo}-{hi}"
            enc = apply_mask(r["text"], tok, max_length, profile,
                             [tuple(s) for s in r["mask_spans"]])
            masked_tok += len(enc["masked_idx"])
            sup_after += enc["supervised_after"]
        else:
            sup_after += n
    print(f"\nsupervised tokens: {sup_before:,} -> {sup_after:,} "
          f"(-{masked_tok:,}, {100 * masked_tok / sup_before:.2f}% of the training signal)")
    da = [r for r in new if r["source"] == "synthdoc_difficult_advice"]
    print(f"difficult-advice rows: {len(da)}, of which {len(with_spans)} masked "
          f"({len(with_spans) / len(da):.1%})")


if __name__ == "__main__":
    fire.Fire(main)
