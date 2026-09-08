# ABOUTME: Exact token accounting for the normal (non-verbose) 716-row difficult-advice slice:
# ABOUTME: total/masked/trainable, split into reasoning trace vs visible answer, with quantiles.
"""Run: uv run python scratch/token_audit/da716_exact.py [repo file source_label]

Mirrors src/train/masking.build_labels segment-for-segment (same cuts at forced-span
boundaries) but keeps the offset mapping, so each supervised token can be attributed to
the reasoning trace or to the visible answer.
"""
import json, sys, statistics
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from src.infra.huggingface import hf_token
from src.model_profile import model_profile
from src.train.masking import assistant_spans, forced_spans, build_labels

MODEL = "Qwen/Qwen3.6-27B"
REPO = sys.argv[1] if len(sys.argv) > 1 else "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
FILE = sys.argv[2] if len(sys.argv) > 2 else "t2_9284_da716_10k.jsonl"
SRC = sys.argv[3] if len(sys.argv) > 3 else "difficult_advice_v2"
MAX_LEN = 8192

tok = AutoTokenizer.from_pretrained(MODEL)
prof = model_profile(MODEL)
path = hf_hub_download(REPO, FILE, repo_type="dataset", token=hf_token())

def ids_with_offsets(text):
    """Same segmented tokenization build_labels uses, but returning offsets too."""
    kw = dict(header=prof.assistant_header, turn_end=prof.turn_end)
    prefills = forced_spans(text, assistant_spans(text, **kw), prof.prefill, prof.empty_think)
    cuts = sorted({0, len(text), *(e for sp in prefills for e in sp)})
    ids, offs = [], []
    for a, b in zip(cuts, cuts[1:]):
        enc = tok(text[a:b], add_special_tokens=False, return_offsets_mapping=True)
        ids += enc["input_ids"]
        offs += [(a + s, a + e) for s, e in enc["offset_mapping"]]
    return ids[:MAX_LEN], offs[:MAX_LEN]

CLOSE = "</think>"
tot = sup = reason = answer = 0
per_row_tot, per_row_sup, per_row_reason, per_row_ans = [], [], [], []
n = n_think = 0

with open(path, encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("source") != SRC:
            continue
        text = r["text"]
        lab = build_labels(text, tok, MAX_LEN, prof,
                           supervise=r.get("supervise") or "all",
                           mask_spans=r.get("mask_spans"))["labels"]
        _, offs = ids_with_offsets(text)
        assert len(offs) == len(lab)
        close = text.find(CLOSE)
        close_end = close + len(CLOSE) if close != -1 else -1
        n += 1
        n_think += close != -1
        t = len(lab)
        s = rr = aa = 0
        for (a, b), v in zip(offs, lab):
            if v == -100:
                continue
            s += 1
            if close_end != -1 and b <= close_end:
                rr += 1
            else:
                aa += 1
        tot += t; sup += s; reason += rr; answer += aa
        per_row_tot.append(t); per_row_sup.append(s)
        per_row_reason.append(rr); per_row_ans.append(aa)

def q(xs):
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return (f"min {xs[0]:,}  p10 {p(.10):,}  p25 {p(.25):,}  median {p(.50):,}  "
            f"p75 {p(.75):,}  p90 {p(.90):,}  max {xs[-1]:,}  mean {statistics.mean(xs):,.1f}")

print(f"{REPO}/{FILE}  source={SRC}  max_seq_len={MAX_LEN}")
print(f"rows: {n:,}   (rows with a real <think> block: {n_think:,})\n")
print(f"  rendered (total)      {tot:>10,}   {tot/n:>8,.1f} /row")
print(f"  masked (no loss)      {tot-sup:>10,}   {(tot-sup)/n:>8,.1f} /row   {100*(tot-sup)/tot:.1f}%")
print(f"  TRAINABLE             {sup:>10,}   {sup/n:>8,.1f} /row   {100*sup/tot:.1f}%")
print(f"    of which reasoning  {reason:>10,}   {reason/n:>8,.1f} /row   {100*reason/sup:.1f}% of trainable")
print(f"    of which answer     {answer:>10,}   {answer/n:>8,.1f} /row   {100*answer/sup:.1f}% of trainable")
print(f"\n  per-row rendered   {q(per_row_tot)}")
print(f"  per-row trainable  {q(per_row_sup)}")
print(f"  per-row reasoning  {q(per_row_reason)}")
print(f"  per-row answer     {q(per_row_ans)}")
