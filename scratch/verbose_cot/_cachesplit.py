import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.argv = ["x", "4.6"]
from pilot_expand import load_prompt, build_plan, PARA_WORDS
from src.endpoints.openrouter import CACHE_MARK

system, user_tpl = load_prompt("expand_prompt_v5.md")
rows = json.loads(Path("scratch/verbose_cot/sample5.json").read_text(encoding='utf-8'))
r = rows[0]
plan, alloc, units = build_plan(r["think"], 4.6)
user = user_tpl.format(reasoning=r["think"], response=r["answer"], target_words=0,
                       n_paragraphs=0, target_paragraphs=0, plan=plan,
                       per_para_words=PARA_WORDS, n_runs=len(alloc),
                       n_slots=sum(alloc), slots="")
head, _, tail = user.partition(CACHE_MARK)
CH = 3.7   # chars/token implied by measured cached_tokens=1955 for this prefix
print(f"CACHED (system msg)          {len(system):>6} chars")
print(f"CACHED (user instruction)    {len(head):>6} chars")
print(f"CACHED TOTAL                 {len(system)+len(head):>6} chars  ~{(len(system)+len(head))/CH:.0f} tok"
      f"   [measured: 1,955 tok]")
print(f"\nNOT CACHED (per-row tail)    {len(tail):>6} chars  ~{len(tail)/CH:.0f} tok")
print(f"   of which the plan is      {len(plan):>6} chars"
      f"  (source CoT {len(r['think']):>5} + {len(plan)-len(r['think'])} chars of budget annotation)")
print(f"   tail instruction          {len(tail)-len(plan):>6} chars")
print(f"\nmeasured prompt_tokens/call: 3,124  ->  payload ~{3124-1955} tok")
print("\n--- the uncached tail, verbatim (first 700 chars) ---")
print(tail[:700])
