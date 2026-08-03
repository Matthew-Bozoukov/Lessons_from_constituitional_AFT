# ABOUTME: Hard gate before SFT: proves the assistant-only label mask supervises assistant
# ABOUTME: tokens and nothing else, on real mixture rows, using an independent role parse.

"""Verify the training label mask on real rows of a built mixture.

A mask defect does not show up in the loss curve -- the tool-calling arm retracted on
2026-08-03 trained on the wrong tokens while its loss fell 2.753 -> 1.057 and looked
textbook. This script is the check that would have caught it, so it must not simply
re-run the code under test: it re-derives the role regions of each rendered conversation
with its own parser and asserts the mask agrees.

    python scratch/verify_mask.py --mixture data/mixture.jsonl --max-seq-len 4096

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.masking import build_labels  # noqa: E402

_START = "<|im_start|>"
_END = "<|im_end|>"


def _role_regions(text: str) -> list[tuple[str, int, int]]:
    """Parse a rendered Qwen conversation into (role, content_start, content_end) regions.

    Written independently of src/train/masking.py: it keys off the role name in each
    `<|im_start|>{role}\\n` header rather than searching for the assistant header alone,
    so agreement between the two is evidence rather than tautology. `content_end` is
    exclusive and includes the closing `<|im_end|>`, matching what the mask supervises.
    """
    regions: list[tuple[str, int, int]] = []
    for m in re.finditer(re.escape(_START), text):
        head = m.end()
        nl = text.find("\n", head)
        assert nl != -1, f"unterminated header at char {m.start()}"
        role = text[head:nl].strip()
        end = text.find(_END, nl + 1)
        assert end != -1, f"{role} turn at char {m.start()} has no {_END}"
        regions.append((role, nl + 1, end + len(_END)))
    assert regions, "no chat turns parsed; the row is not in rendered Qwen format"
    return regions


def _covered(a: int, b: int, spans: list[tuple[int, int]]) -> bool:
    """True if character range [a, b) lies wholly inside one of `spans`."""
    return any(a >= s and b <= e for s, e in spans)


def _check_row(text: str, tok, max_seq_len: int) -> dict:
    """Mask one row and cross-check every supervised token against the role parse."""
    enc = build_labels(text, tok, max_seq_len)
    ids, labels = enc["input_ids"], enc["labels"]
    offsets = tok(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
    )["offset_mapping"]
    assert len(offsets) == len(labels), "offset/label length mismatch"

    regions = _role_regions(text)
    asst = [(s, e) for role, s, e in regions if role == "assistant"]
    other = [(s, e) for role, s, e in regions if role != "assistant"]
    # Headers (`<|im_start|>assistant\n` and every other turn's) sit outside every content
    # region, so a supervised token there is caught by the not-in-assistant check below.

    n_sup = sum(1 for v in labels if v != -100)
    leaked_other, leaked_outside = [], []
    for (a, b), v in zip(offsets, labels):
        if v == -100 or b <= a:
            continue
        if _covered(a, b, other):
            leaked_other.append(text[a:b])
        elif not _covered(a, b, asst):
            leaked_outside.append(text[a:b])

    # Every assistant turn that survives truncation must actually be supervised: a mask
    # that only ever hit the first turn would pass a whole-row percentage check.
    horizon = max((b for a, b in offsets if b > a), default=0)
    turns_in_window = sum(1 for s, e in asst if e <= horizon)
    sup_spans = {
        next(i for i, (s, e) in enumerate(asst) if a >= s and b <= e)
        for (a, b), v in zip(offsets, labels)
        if v != -100 and b > a and _covered(a, b, asst)
    }
    return {
        "n_tok": len(ids),
        "n_sup": n_sup,
        "turns": len(asst),
        "turns_in_window": turns_in_window,
        "turns_supervised": len(sup_spans),
        "leaked_other": leaked_other,
        "leaked_outside": leaked_outside,
        "truncated": horizon < len(text),
        "tool_calls": text.count("<tool_call>"),
        "tool_calls_in_window": text[:horizon].count("<tool_call>"),
        "tool_calls_closed_in_window": text[:horizon].count("</tool_call>"),
    }


def main(
    mixture: str,
    tokenizer: str = "Qwen/Qwen3.6-27B",
    max_seq_len: int = 4096,
    n: int = 24,
) -> None:
    """Run the gate over a sample of mixture rows and print the numbers.

    Args:
        mixture: Path to mixture.jsonl (rows with `text` and `source`).
        tokenizer: Tokenizer of the model being trained.
        max_seq_len: Must equal `train.max_seq_len` in the training config.
        n: Rows to sample per source, plus the most multi-turn rows overall.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = [json.loads(line) for line in Path(mixture).open(encoding="utf-8")]
    print(f">>> {len(rows):,} rows in {mixture}; max_seq_len={max_seq_len}")

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r.get("source", "?"), []).append(r)

    # Deliberately not a uniform sample: the mask fails most plausibly on long multi-turn
    # tool-calling conversations, so those are over-represented on purpose.
    picked: list[tuple[str, dict]] = []
    for src, srows in sorted(by_source.items()):
        picked += [(src, r) for r in srows[:n]]
        multi = sorted(srows, key=lambda r: -r["text"].count(f"{_START}assistant"))[:n]
        picked += [(src, r) for r in multi]
    tooly = sorted(rows, key=lambda r: -r["text"].count("<tool_call>"))[:n]
    picked += [(r.get("source", "?"), r) for r in tooly]

    agg: dict[str, list[dict]] = {}
    fails = 0
    for src, r in picked:
        res = _check_row(r["text"], tok, max_seq_len)
        agg.setdefault(src, []).append(res)
        if res["leaked_other"] or res["leaked_outside"]:
            fails += 1
            print(f"!!! LEAK in a {src} row: "
                  f"{res['leaked_other'][:5]} {res['leaked_outside'][:5]}")
        if res["turns_supervised"] != res["turns_in_window"]:
            fails += 1
            print(f"!!! {src} row: {res['turns_in_window']} assistant turns in window but "
                  f"{res['turns_supervised']} supervised")

    print(f"\n{'source':<12}{'rows':>6}{'tokens':>10}{'supervised':>12}{'share':>8}"
          f"{'multiturn':>11}{'trunc':>7}")
    tot_tok = tot_sup = 0
    for src, res in sorted(agg.items()):
        t = sum(x["n_tok"] for x in res)
        s = sum(x["n_sup"] for x in res)
        tot_tok, tot_sup = tot_tok + t, tot_sup + s
        print(f"{src:<12}{len(res):>6}{t:>10,}{s:>12,}{100 * s / t:>7.1f}%"
              f"{sum(1 for x in res if x['turns'] > 1):>11}"
              f"{sum(1 for x in res if x['truncated']):>7}")
    print(f"{'TOTAL':<12}{len(picked):>6}{tot_tok:>10,}{tot_sup:>12,}"
          f"{100 * tot_sup / tot_tok:>7.1f}%")

    tc = sum(x["tool_calls"] for x in agg.get("agentic", []))
    tc_in = sum(x["tool_calls_in_window"] for x in agg.get("agentic", []))
    tc_closed = sum(x["tool_calls_closed_in_window"] for x in agg.get("agentic", []))
    print(f"\n<tool_call> spans in sampled agentic rows: {tc} total, {tc_in} inside the "
          f"{max_seq_len}-token window, {tc_closed} of those closed "
          f"({tc_in - tc_closed} severed by truncation)")

    multiturn = [x for res in agg.values() for x in res if x["turns"] > 1]
    print(f"multi-turn rows checked: {len(multiturn)}; assistant turns in window "
          f"{sum(x['turns_in_window'] for x in multiturn)}, all supervised: "
          f"{all(x['turns_supervised'] == x['turns_in_window'] for x in multiturn)}")

    share = 100 * tot_sup / tot_tok
    print(f"\nsupervised fraction: {share:.1f}%")
    if share > 95:
        print("!!! ~100% supervised means the mask is not masking (full-sequence defect)")
        fails += 1
    if tot_sup == 0:
        print("!!! 0% supervised means an all-zero mask; nothing would train")
        fails += 1
    if fails:
        raise SystemExit(f"MASK GATE FAILED: {fails} problem(s). Do not train.")
    print("MASK GATE PASSED: every supervised token lies inside an assistant turn, "
          "no system/user/tool token is supervised, and every in-window assistant turn "
          "is supervised.")


if __name__ == "__main__":
    fire.Fire(main)
