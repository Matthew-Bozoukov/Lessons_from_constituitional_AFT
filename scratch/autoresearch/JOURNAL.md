# da-autoresearch journal (started 2026-10-01 ~01:00 UTC)

Goal: explain what does and does not move ODCV and MASK relative to the 28 Sep DA recipe, and why "other AI" rows
help ODCV when swapped in from 25 Sep but not when generated fresh. Budget: $240 (RunPod + OpenRouter), 150 new DA
rows, 2 big synth runs. Guard: scratch/autoresearch/budget.py (ledger in output/autoresearch/budget.json).

## Log
- 01:00 Branch jamie/da-autoresearch cut from jamie/dat @ f62a8330. Budget baseline taken; watchdog running.
- 01:05 Inventory of all published arms: output/autoresearch/inventory.md. Training configs identical across arms
  (only seed / git sha differ), so no training-recipe confound.
- 01:10 Three Opus analysts launched: DA corpus features, ODCV rollouts, MASK results.
