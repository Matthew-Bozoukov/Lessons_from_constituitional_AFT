import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rows = json.loads(Path("scratch/verbose_cot/v6_3x_out.json").read_text(encoding='utf-8'))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for r in rows[lo:hi]:
    o, n = len(r['think'].split()), len(r['think_expanded'].split())
    print(f"\n{'='*90}\n### {r['scenario_id']}  trait={r['trait_id']}  {o} -> {n} words ({n/o:.2f}x)\n{'='*90}")
    print(f"\n--- USER (unchanged, for context) ---\n{r['user'][:700]}\n")
    print(f"--- ORIGINAL CoT ({o}w) ---\n{r['think']}\n")
    print(f"--- EXPANDED CoT ({n}w) ---\n{r['think_expanded']}\n")
