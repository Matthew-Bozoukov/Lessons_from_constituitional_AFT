# ABOUTME: Print a scored smoke run for a human: the exact input, the reply by label with per-token KL, the top-5 tokens where the models disagree most.
# ABOUTME: Run: uv run python scratch/kl_diff/report_smoke.py --scores output/kl_diff/scores_mask_smoke.jsonl --k 3
"""Per step: per-label mean KL for each trained model, then the reply with each run of same-label
tokens shown with its mean KL, then the three highest-KL tokens per model with what each model
would rather have written (its top-5 next tokens there)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from scratch.kl_diff.common import LABELS, load_tokenizer


def per_label(rec, m):
    acc = defaultdict(list)
    for lab, k in zip(rec["labels"], rec["kl"][m]):
        acc[lab].append(k)
    return {lab: (sum(v) / len(v), len(v)) for lab, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--k", type=int, default=3, help="steps to print per writer")
    ap.add_argument("--full-prompt", action="store_true")
    a = ap.parse_args()
    tok = load_tokenizer()
    recs = [json.loads(l) for l in open(a.scores, encoding="utf-8")]
    models = [m for m in recs[0]["kl"]]
    shown = defaultdict(int)
    for rec in recs:
        if shown[rec["writer"]] >= a.k:
            continue
        shown[rec["writer"]] += 1
        print("=" * 100)
        print(
            f"STEP {rec['eval']} {rec['item']} step {rec['step']} written by {rec['writer']} "
            f"({rec['n_prompt']} prompt tokens, {len(rec['tokens'])} scored)"
        )
        print("-" * 30, "mean KL (nats) by label", "-" * 30)
        for m in models:
            pl = per_label(rec, m)
            print(
                f"  {m:9s} "
                + "  ".join(
                    f"{lab}={pl[lab][0]:.3f} (n={pl[lab][1]})"
                    for lab in LABELS
                    if lab in pl
                )
            )
        print("-" * 30, "reply, each same-label run with mean KL per model", "-" * 30)
        toks, labs = rec["tokens"], rec["labels"]
        i = 0
        while i < len(toks):
            j = i
            while j < len(toks) and labs[j] == labs[i]:
                j += 1
            kls = "  ".join(
                f"{m}={sum(rec['kl'][m][i:j]) / (j - i):.3f}" for m in models
            )
            print(f"[{labs[i]} | {kls}] {tok.decode(toks[i:j])}")
            i = j
        print("-" * 30, "3 highest-KL tokens per model", "-" * 30)
        for m in models:
            order = sorted(range(len(toks)), key=lambda p: -rec["kl"][m][p])[:3]
            for p in order:
                ctx = tok.decode(toks[max(0, p - 12) : p])
                actual = tok.decode([toks[p]])
                t5 = {
                    mm: [tok.decode([i]) for i in rec["top5"][mm]["ids"][p]]
                    for mm in rec["top5"]
                }
                print(
                    f"  {m} KL={rec['kl'][m][p]:.2f} at #{p} [{labs[p]}]  …{ctx!r} -> wrote {actual!r}"
                )
                for mm, cands in t5.items():
                    print(f"      {mm:9s} would write: {cands}")
    print("=" * 100)
    agg = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        for m in models:
            for lab, k in zip(rec["labels"], rec["kl"][m]):
                agg[(rec["eval"], rec["writer"], m)][lab].append(k)
    print("mean KL by eval / writer / model / label over all scored steps:")
    for (ev, w, m), d in sorted(agg.items()):
        print(
            f"  {ev} {w:9s} {m:9s} "
            + "  ".join(f"{lab}={sum(v) / len(v):.3f}" for lab, v in d.items())
        )


if __name__ == "__main__":
    main()
