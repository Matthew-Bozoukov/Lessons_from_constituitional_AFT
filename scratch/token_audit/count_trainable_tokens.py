# ABOUTME: Counts TRAINABLE (unmasked-label) tokens in published SFT mixtures on HF, using the
# ABOUTME: exact train-time path: ModelProfile render_kwargs -> build_labels -> labels != -100.
"""Run: uv run python scratch/token_audit/count_trainable_tokens.py

For each mixture repo it renders the interchange rows with the Qwen3.6 chat template and
applies src/train/masking.build_labels at the arm's real max_seq_len, so the reported
number is the same one train_lora.py prints as ">>> assistant-only loss: n/N tokens".
"""
import json, sys, collections
from transformers import AutoTokenizer
from src.huggingface import hf_token
from huggingface_hub import hf_hub_download
from src.model_profile import model_profile
from src.train.masking import build_labels

MODEL = "Qwen/Qwen3.6-27B"

# The table2 instruction pool is byte-identical across every "t2_9284 + <synth>_716" arm,
# so `--synth-only` skips it and counts just the synthetic slice under study.
TABLE2_SOURCES = {"no_robots", "numinamath_cot", "tulu3_if", "self_oss_instruct", "lima",
                  "smol_constraints", "apigen_function_calling", "smol_summarize",
                  "longalign", "tulu3"}

# (repo, file, max_seq_len from the train config that consumes it, label)
MIXTURES = [
    ("LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train", "mixture_think.jsonl", 8192, "t2_9284 + synthdoc_716 (flagship 10k)"),
    ("LASR-Callum/2026-08-16-table2-9284-courtroom-716-train", "t2_9284_courtroom716_10k.jsonl", 8192, "t2_9284 + courtroom_716"),
    ("LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-716-train", "t2_9284_da716_verbose_10k.jsonl", 8192, "t2_9284 + DA-verbose_716"),
    ("LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-token-matched-train-mixture", "t2_9284_da_verbose_tokenmatched.jsonl", 8192, "t2_9284 + DA-verbose token-matched"),
    ("LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train", "mixture_think.jsonl", 8192, "t2_9284 + post-action-retro_716"),
    ("LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train", "t2_9284_da716_10k.jsonl", 8192, "t2_9284 + difficult_advice_716 (original)"),
    ("LASR-Callum/2026-08-16-table2-9284-peer-critique-716-train", "t2_9284_peercritique716_10k.jsonl", 8192, "t2_9284 + peer_critique_716"),
    ("LASR-Callum/2026-08-21-table2-9284-difficult-advice-chunk-only-702-train-mixture", "t2_9284_da_chunk_only_702.jsonl", 8192, "t2_9284 + DA-chunk-only_702"),
    ("LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train", "t2_9284_lowstakes716_10k.jsonl", 8192, "t2_9284 + low_stakes_716"),
    ("LASR-Callum/2026-08-20-table2-9284-t10-curiosity-716-train", "mixture_think.jsonl", 8192, "t2_9284 + t10_curiosity_716"),
    ("LASR-Callum/2026-08-24-table2-9284-grok-responder-703-paired-train-mixture", "t2_9284_grokresp703_10k.jsonl", 8192, "t2_9284 + grok_responder_703"),
    ("LASR-Callum/2026-08-24-table2-9284-sonnet-703-paired-train-mixture", "t2_9284_sonnet703_10k.jsonl", 8192, "t2_9284 + sonnet_responder_703"),
    ("LASR-Callum/2026-08-25-table2-9284-gpt-responder-685-paired-train-mixture", "t2_9284_gptresp685_10k.jsonl", 8192, "t2_9284 + gpt_responder_685"),
    ("LASR-Callum/2026-08-26-table2-9284-sonnet-concise-703-paired-train", "t2_9284_sonnetconcise703_10k.jsonl", 8192, "t2_9284 + sonnet_concise_703"),
    ("LASR-Callum/2026-08-19-less-top10-difficult-advice-220-train", "mixture.jsonl", 8192, "LESS top-10 (220 rows)"),
    ("LASR-Callum/2026-08-02-qwen36-synthdoc-package-mixture-20-80", "mixture.jsonl", 3072, "synthdoc 20/80 + tulu (500k-tok era)"),
    ("LASR-Callum/2026-08-02-qwen36-mixture-500k-da20-numina", "mixture.jsonl", 3072, "500k DA20 + numina + tulu"),
    ("LASR-Callum/2026-07-31-qwen36-27b-tulu-0-100-train-mixture", "mixture_0_100.jsonl", 3072, "tulu 0/100 control"),
    ("LASR-Callum/2026-07-31-qwen36-sft-mixture-10-90-assistant-loss-only", "mixture.jsonl", 2048, "legacy 10/90 (qwen3-era)"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
prof = model_profile(MODEL)
argv = sys.argv[1:]
synth_only = "--synth-only" in argv
only = [a for a in argv if not a.startswith("--")] or None
rows_out = []

for repo, fname, max_len, label in MIXTURES:
    if only and not any(o in repo for o in only):
        continue
    try:
        p = hf_hub_download(repo, fname, repo_type="dataset", token=hf_token())
    except Exception as e:
        print(f"!! {repo}: {type(e).__name__}: {e}", flush=True)
        continue
    per_src = collections.defaultdict(lambda: {"rows": 0, "tot": 0, "sup": 0, "full": 0})
    tot = sup = full = n = trunc = 0
    sup_per_row = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if synth_only and r.get("source") in TABLE2_SOURCES:
                continue
            text = r.get("text")
            if text is None:
                msgs = r.get("messages")
                if msgs is None:
                    continue
                text = tok.apply_chat_template(
                    [{k: v for k, v in m.items() if v is not None} for m in msgs],
                    tokenize=False, add_generation_prompt=False, **prof.render_kwargs)
            try:
                out = build_labels(text, tok, max_len, prof,
                                   supervise=r.get("supervise") or "all",
                                   mask_spans=r.get("mask_spans"))
            except AssertionError:
                continue
            s = sum(1 for v in out["labels"] if v != -100)
            t = len(out["input_ids"])
            f = r.get("n_tokens") or t
            n += 1; tot += t; sup += s; full += f; trunc += (t >= max_len)
            sup_per_row.append(s)
            b = per_src[r.get("source", "?")]
            b["rows"] += 1; b["tot"] += t; b["sup"] += s; b["full"] += f
    if not n:
        continue
    sup_per_row.sort()
    med = sup_per_row[len(sup_per_row) // 2]
    print(f"\n=== {label}\n    {repo}/{fname}  (max_seq_len={max_len})")
    print(f"    rows {n:,} | rendered tokens {tot:,} | TRAINABLE {sup:,} ({100*sup/tot:.1f}%)"
          f" | pre-truncation n_tokens {full:,} | rows hitting cap {trunc:,}")
    print(f"    trainable/row: median {med:,}  mean {sup/n:,.0f}  p90 {sup_per_row[int(.9*n)]:,}  max {sup_per_row[-1]:,}")
    for s_, b in sorted(per_src.items(), key=lambda kv: -kv[1]["sup"]):
        print(f"      {s_:<28} rows {b['rows']:>6,}  trainable {b['sup']:>10,} "
              f"({100*b['sup']/sup:>5.1f}% of trainable, {100*b['sup']/b['tot']:>5.1f}% of its own tokens)")
    rows_out.append({"label": label, "repo": repo, "file": fname, "max_seq_len": max_len,
                     "rows": n, "rendered_tokens": tot, "trainable_tokens": sup,
                     "trainable_pct": round(100*sup/tot, 2), "median_trainable_per_row": med,
                     "mean_trainable_per_row": round(sup/n, 1), "rows_at_cap": trunc,
                     "by_source": {k: dict(v) for k, v in per_src.items()}})

OUT = ("scratch/token_audit/trainable_tokens_synth_only.json" if synth_only
       else "scratch/token_audit/trainable_tokens.json")
try:
    prev = {(d["repo"], d["file"]): d for d in json.load(open(OUT, encoding="utf-8"))}
except (OSError, ValueError):
    prev = {}
prev.update({(d["repo"], d["file"]): d for d in rows_out})
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(list(prev.values()), fh, indent=2)
print("\nwrote scratch/token_audit/trainable_tokens.json")
