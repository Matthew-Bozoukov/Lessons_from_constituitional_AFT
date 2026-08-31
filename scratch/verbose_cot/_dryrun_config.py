# ABOUTME: Offline validation of 2026-08-25_verbose_cot.yaml — renders every prompt for real records
# ABOUTME: without any API call, and reports the cache split. No network, no spend.
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.data.synth.derive import derive_vars
from src.data.synth.stage_runtime import lint_problems
from src.endpoints.openrouter import CACHE_MARK

cfg = OmegaConf.to_container(OmegaConf.load("configs/data/synth/2026-08-25_verbose_cot.yaml"), resolve=True)
stages = {s["name"]: s for s in cfg["stages"]}
exp = stages["expand"]
rows = [json.loads(l) for l in
        Path(cfg["source"]["local_dir"], cfg["source"]["snapshot"]).read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"config OK: {len(cfg['stages'])} stages {list(stages)}, {len(rows)} source records")
print(f"models: " + ", ".join(f"{k}={v['model']}" for k, v in cfg["models"].items()))

# every record must derive a plan and render without a missing key
tot_units = tot_cached = tot_payload = 0
for r in rows:
    pv = derive_vars(exp["derive"], r)
    user = exp["prompts"]["user"].format(**{**r, **pv})
    head, mark, tail = user.partition(CACHE_MARK)
    assert mark, "cache marker vanished from the rendered prompt"
    tot_units += int(pv["n_runs"]); tot_cached = len(head); tot_payload += len(tail)
sysmsg = exp["prompts"]["system"]
print(f"\nall {len(rows)} records render; {tot_units} plan units total "
      f"({tot_units/len(rows):.1f} mean)")
print(f"CACHED   system {len(sysmsg):>5} + instruction {tot_cached:>6} = "
      f"{len(sysmsg)+tot_cached:>6} chars (~{(len(sysmsg)+tot_cached)/3.06:.0f} tok)"
      f"  {'OK, clears the ~1024 minimum' if (len(sysmsg)+tot_cached)/3.06 > 1024 else 'TOO SHORT'}")
print(f"UNCACHED mean per-row payload {tot_payload/len(rows):>6.0f} chars "
      f"(~{tot_payload/len(rows)/3.06:.0f} tok)")

# judge prompt renders too (uses scenario + both reasonings)
cand = {**rows[0], "reasoning": "expanded text here"}
for v in exp["verify"]:
    ju = v["prompts"]["user"].format(**cand)
    print(f"\njudge[{v['save_as']}] renders: {len(ju):,} chars (~{len(ju)/3.06:.0f} tok), "
          f"model={cfg['models'][v['model']]['model']}, "
          f"schema intact: {'\"verdict\"' in ju}")

# the lint contract is live on a real record
probs = lint_problems({"reasoning": rows[0]["source_reasoning"]}, exp["lint"], rows[0])
print(f"lint on an unexpanded record (1.00x): {probs[0][:90]}")
probs = lint_problems({"reasoning": " ".join(rows[0]["source_reasoning"].split()*3)}, exp["lint"], rows[0])
print(f"lint on a 3.00x record: {probs or 'clean'}")
