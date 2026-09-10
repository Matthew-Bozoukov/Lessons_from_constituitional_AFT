# ABOUTME: Survey the published corpus, mixture and adapter — what they actually contain, not
# ABOUTME: what the cards claim. Run: uv run python scratch/nonmoral/survey.py
import io
import json
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from src.huggingface import hf_token  # noqa: E402

CORPUS = "LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation"
MIXTURE = "LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture"
ADAPTER = ("LASR-Callum/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-"
           "684-rank-64-dynbatch")
tok = hf_token()
api = HfApi(token=tok)


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


rule("ADAPTER")
info = api.model_info(ADAPTER, files_metadata=True)
print(f"sha {info.sha}")
for s in sorted(info.siblings, key=lambda x: x.rfilename):
    size = f"{s.size / 1e6:8.1f} MB" if s.size else "         -"
    print(f"  {size}  {s.rfilename}")
meta = json.load(open(hf_hub_download(ADAPTER, "training_meta.json", token=tok), encoding="utf-8"))
print(f"\nthinking   : {meta['thinking']}   (eval infers serve mode from this)")
print(f"base_model : {meta['base_model']}")
print(f"dataset    : {meta['dataset']['repo']}")
print(f"             file={meta['dataset']['file']} rev={meta['dataset']['revision'][:12]}")
cfg = json.load(open(hf_hub_download(ADAPTER, "adapter_config.json", token=tok), encoding="utf-8"))
print(f"lora       : r={cfg.get('r')} alpha={cfg.get('lora_alpha')} dropout={cfg.get('lora_dropout')}")
print(f"targets    : {len(cfg.get('target_modules') or [])} module patterns")

rule("MIXTURE (what the trainer actually read)")
mp = hf_hub_download(MIXTURE, "t2_9284_nonmoral_684.jsonl", repo_type="dataset", token=tok)
rows = [json.loads(line) for line in open(mp, encoding="utf-8")]
src = Counter(r.get("source", "?") for r in rows)
print(f"rows: {len(rows)}")
for k, v in src.most_common():
    print(f"  {v:6d}  {v / len(rows):6.2%}  {k}")
synth = [r for r in rows if r.get("source") == "nonmoral_deliberation"]
# The builder carries only scenario_id + trait_id onto a synth row, NOT the whole metadata
# block: `domain`, `instruction` and `why_wrong` live in the corpus rows, not the mixture.
tr = Counter(r["trait_id"] for r in synth)
print(f"\nsynth row keys : {sorted(synth[0].keys())}")
print(f"synth per-trait: {dict(sorted(tr.items()))}")
print(f"distinct scenarios: {len({r['scenario_id'] for r in synth})}")
real = sum(1 for r in rows if "<think>\n\n</think>" not in r["text"])
print(f"\nrows with a NON-empty think block: {real}  (expect ~= the 684 synth rows)")
lens = sorted(len(r["text"]) for r in rows)
print(f"text chars: p50={lens[len(lens) // 2]} p99={lens[99 * len(lens) // 100]} max={lens[-1]}")

rule("CORPUS (the 702 before the balanced draw)")
cp = hf_hub_download(CORPUS, "dataset.jsonl", repo_type="dataset", token=tok)
crow = [json.loads(line) for line in open(cp, encoding="utf-8")]
print(f"rows: {len(crow)}")
ct = Counter(r["metadata"]["trait_id"] for r in crow)
print(f"per-trait: {dict(sorted(ct.items()))}")
names = {r["metadata"]["trait_id"]: r["metadata"]["trait_name"] for r in crow}
for k in sorted(names):
    print(f"  {k}: {names[k]}")
