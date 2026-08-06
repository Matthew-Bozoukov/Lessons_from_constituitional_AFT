# ABOUTME: Combines 8,000 stratified Table 2 samples with the 2,203-example synthdoc corpus.
# ABOUTME: Step 1 of 3 for the h200x4 training bundle; synthdoc rows keep real <think> traces.

"""Combine 8,000 filtered Table 2 samples with the full 2,203-example synthdoc corpus."""
import json, random, sys, collections, statistics as st
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "src")
from train.masking import build_labels
from transformers import AutoTokenizer

N_T2, SEED = 8000, 0
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
FILT = Path("output/mixture_paper_table2/20260804_093351/filtered_20260804_095210/mixture_filtered.jsonl")
SYNTH = Path("output/synthdoc_v2/20260803_211524/stage_6_final.jsonl")
OUT = Path("output/mixture_combined_t2_8000_synthdoc_2203"); OUT.mkdir(parents=True, exist_ok=True)

# --- 8,000 of the filtered Table 2 mixture, stratified so source proportions hold -----
t2 = [json.loads(l) for l in FILT.open()]
by = collections.defaultdict(list)
for r in t2:
    by[r["source"]].append(r)
rng = random.Random(SEED)
quota, picked = {}, []
for s, rows in sorted(by.items()):
    quota[s] = round(len(rows) * N_T2 / len(t2))
for s, rows in sorted(by.items()):
    rng.shuffle(rows)
    picked += rows[:quota[s]]
# Rounding can land a few off N_T2; trim or top up from the largest source.
rng.shuffle(picked)
if len(picked) > N_T2:
    picked = picked[:N_T2]
elif len(picked) < N_T2:
    chosen = {id(r) for r in picked}
    spare = [r for r in t2 if id(r) not in chosen]
    rng.shuffle(spare)
    picked += spare[:N_T2 - len(picked)]
assert len(picked) == N_T2, len(picked)

# --- all 2,203 synthdoc examples, rendered WITH their real reasoning traces -----------
synth = []
for r in map(json.loads, SYNTH.open()):
    text = tok.apply_chat_template(
        [{"role": "system", "content": r["system"]},
         {"role": "user", "content": r["user"]},
         {"role": "assistant", "content": r["response"],
          "reasoning_content": r["reasoning"]}],
        tokenize=False, add_generation_prompt=False)
    assert "<think>" in text and "<think>\n\n</think>" not in text, "trace lost or emptied"
    synth.append({"text": text, "source": "synthdoc_difficult_advice",
                  "trait_id": r["trait_id"]})

rows = [{"text": r["text"], "source": r["source"]} for r in picked] + \
       [{"text": r["text"], "source": r["source"]} for r in synth]
random.Random(SEED).shuffle(rows)
dest = OUT / "mixture.jsonl"
with dest.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- measure -------------------------------------------------------------------------
GROUP = lambda s: "synthdoc" if s == "synthdoc_difficult_advice" else "table2"
agg = collections.defaultdict(lambda: [0, 0, 0])
per = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    o = build_labels(r["text"], tok, 100000)
    n = len(o["input_ids"]); s = sum(1 for v in o["labels"] if v != -100)
    for d, k in ((agg, GROUP(r["source"])), (per, r["source"])):
        d[k][0] += 1; d[k][1] += n; d[k][2] += s

tot_r = sum(v[1] for v in agg.values()); tot_s = sum(v[2] for v in agg.values())
stats = {
    "total": {"examples": len(rows), "rendered": tot_r, "supervised": tot_s},
    "groups": {k: {"examples": v[0], "rendered": v[1], "supervised": v[2],
                   "supervised_pct_of_total": round(100 * v[2] / tot_s, 2),
                   "rendered_pct_of_total": round(100 * v[1] / tot_r, 2)}
               for k, v in agg.items()},
    "by_source": {k: {"examples": v[0], "rendered": v[1], "supervised": v[2]}
                  for k, v in sorted(per.items(), key=lambda kv: -kv[1][1])},
    "table2_quota": quota, "seed": SEED,
}
(OUT / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats["total"], indent=2))
print(json.dumps(stats["groups"], indent=2))
print(f">>> wrote {dest}")
