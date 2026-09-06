# ABOUTME: Distribution of ODCV transcript lengths as the judges see them (whole
# ABOUTME: messages_record.txt), in Qwen3.6 tokens, chars and steps, over published runs.
"""usage: uv run python scratch/odcv_kv_test/transcript_lengths.py <hf_repo>... [local_dir...]"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", local_files_only=True)
THRESH = (8_000, 16_000, 32_000, 64_000, 128_000, 256_000)


def load(src: str) -> list[Path]:
    root = Path(src) if Path(src).exists() else Path(snapshot_download(
        src, repo_type="dataset", allow_patterns=["rollouts/**/messages_record.txt"]))
    return sorted(root.glob("**/messages_record.txt"))


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


rows = []
for src in sys.argv[1:]:
    files = load(src)
    recs = []
    for f in files:
        t = f.read_text(errors="replace")
        n = len(tok(t, add_special_tokens=False).input_ids)
        steps = t.count("\n== Step ") + t.startswith("== Step ")
        longest = max((len(x) for x in t.split("\n")), default=0)
        recs.append((n, len(t), steps, longest, str(f)))
    toks = [r[0] for r in recs]
    name = src.split("/")[-1][:48]
    print(f"\n== {name}  n={len(recs)}")
    print(f"  tokens  p50 {pct(toks,.5):>7,}  p90 {pct(toks,.9):>7,}  p95 {pct(toks,.95):>7,}  "
          f"p99 {pct(toks,.99):>8,}  max {max(toks):>9,}  mean {statistics.mean(toks):>8,.0f}")
    steps = [r[2] for r in recs]
    print(f"  steps   p50 {pct(steps,.5)}  p90 {pct(steps,.9)}  max {max(steps)}")
    print("  over:  " + "  ".join(f"{th//1000}k:{sum(t > th for t in toks)}" for th in THRESH))
    big = sorted(recs, reverse=True)[:4]
    for n, c, s, l, f in big:
        print(f"  {n:>9,} tok  {s:>3} steps  longest line {l:>9,} chars  "
              f"{re.sub(r'.*rollouts/', '', f).replace('/messages_record.txt', '')}")
