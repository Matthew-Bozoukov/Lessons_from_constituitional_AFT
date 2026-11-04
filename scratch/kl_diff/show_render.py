# ABOUTME: Print rendered steps with their token labels, for a human check before any GPU time.
# ABOUTME: Run: uv run python scratch/kl_diff/show_render.py --steps output/kl_diff/steps_mask.jsonl --k 3
"""Prints, per step: the exact rendered text the models will read (prompt + target), then the
target again split into labelled runs: [reasoning] ... [visible] ... [tool_call] ... [format].
Also asserts the labeller's invariants on every step in the file (not just the shown ones)."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from scratch.kl_diff.common import load_tokenizer, render_step, token_labels


def runs(tok, ids, first, labs):
    out, cur, buf = [], None, []
    for i, lab in zip(ids[first:], labs):
        if lab != cur and buf:
            out.append((cur, tok.decode(buf)))
            buf = []
        cur = lab
        buf.append(i)
    if buf:
        out.append((cur, tok.decode(buf)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--writer", default="nosynth")
    a = ap.parse_args()
    tok = load_tokenizer()
    steps = [json.loads(l) for l in open(a.steps, encoding="utf-8")]
    counts, n_tok = Counter(), []
    shown = 0
    for s in steps:
        r = render_step(tok, s["messages"], s["tools"], s["target"])
        ids, first, labs = token_labels(tok, r)
        counts.update(labs)
        n_tok.append(len(ids))
        if s["writer"] == a.writer and shown < a.k:
            shown += 1
            print("=" * 100)
            print(
                f"STEP {s['eval']} {s['item']} step {s['step']} written by {s['writer']}  "
                f"({len(ids)} tokens, {len(ids) - first} scored)"
            )
            print("-" * 40, "EXACT TEXT THE MODELS READ (prompt part)", "-" * 40)
            print(r.text[: r.prompt_chars])
            print("-" * 40, "TARGET, BY LABEL", "-" * 40)
            for lab, txt in runs(tok, ids, first, labs):
                print(f"[{lab}] {txt}")
    print("=" * 100)
    print(
        json.dumps(
            {
                "steps": len(steps),
                "target_tokens_by_label": counts,
                "max_total_tokens": max(n_tok),
                "mean_total_tokens": round(sum(n_tok) / len(n_tok)),
            }
        )
    )


if __name__ == "__main__":
    main()
