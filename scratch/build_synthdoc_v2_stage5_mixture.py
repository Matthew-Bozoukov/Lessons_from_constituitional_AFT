# ABOUTME: Build a synthdoc-v2-stage5 training mixture: 716 trait-balanced difficult-advice rows
# ABOUTME: from stage_5 draft responses (rendered) + the existing 9284 Table2 rows verbatim.
import json, random
from collections import defaultdict
from pathlib import Path

STAGE5 = "output/synthdoc_v2/20260803_211524/stage_5_draft_responses.jsonl"
EXISTING = "/home/matthewb/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-06-table2-9284-synthdoc-716-train/snapshots/5b5d66dbd050a15493f38b500fe10f570b0b8f2b/mixture_think.jsonl"
SEED = 0
N_DA = 716

def render(system, user, reasoning, response):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{response}<|im_end|>\n")

# --- 1. load stage-5, group by trait, balanced_take(716, seed=0) replicating build_mixture ---
rows = [json.loads(l) for l in open(STAGE5)]
for r in rows:
    assert r.get("system") and r.get("user") and r.get("draft_reasoning") and r.get("draft_response"), r["scenario_id"]
groups = defaultdict(list)
for r in rows:
    groups[r["trait_id"]].append(r)
per, rem = divmod(N_DA, len(groups))
rng = random.Random(SEED)
picked = []
for i, g in enumerate(sorted(groups)):
    quota = per + (1 if i < rem else 0)
    avail = groups[g]
    assert len(avail) >= quota
    rng.shuffle(avail)
    picked += avail[:quota]
rng.shuffle(picked)
assert len(picked) == N_DA
da_rendered = [{"text": render(r["system"], r["user"], r["draft_reasoning"], r["draft_response"]),
                "source": "synthdoc_difficult_advice"} for r in picked]
print(f"selected {len(da_rendered)} trait-balanced diff-advice rows from stage_5")

# --- 2. existing 9284 Table2 rows verbatim ---
ex = [json.loads(l) for l in open(EXISTING)]
t2 = [r for r in ex if r["source"] != "synthdoc_difficult_advice"]
assert len(t2) == 9284, len(t2)
print(f"reusing {len(t2)} Table2 rows verbatim")

# --- 3. combine + shuffle (seed 0) ---
mix = t2 + da_rendered
random.Random(SEED).shuffle(mix)
out = Path("scratch/mixture_synthdoc_v2_stage5_716.jsonl")
with open(out, "w") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(mix)} rows)")
