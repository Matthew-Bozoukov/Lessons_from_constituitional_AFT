# da-autoresearch journal (started 2026-10-01 ~01:00 UTC)

Goal: explain what does and does not move ODCV and MASK relative to the 28 Sep DA recipe, and why "other AI" rows
help ODCV when swapped in from 25 Sep but not when generated fresh. Budget: $240 (RunPod + OpenRouter), 150 new DA
rows, 2 big synth runs. Guard: scratch/autoresearch/budget.py (ledger in output/autoresearch/budget.json).

## Log
- 01:00 Branch jamie/da-autoresearch cut from jamie/dat @ f62a8330. Budget baseline taken; watchdog running.
- 01:05 Inventory of all published arms: output/autoresearch/inventory.md. Training configs identical across arms
  (only seed / git sha differ), so no training-recipe confound.
- 01:10 Three Opus analysts launched: DA corpus features, ODCV rollouts, MASK results.
- 01:45 MASK analyst: the three 2026-10-01 MASK runs for the new arms are 32-86% unjudged (judge timeouts while
  OpenRouter ran dry, scored as honest). Repaired estimates: self ~83, otherai ~70, explicit ~89. Verified locally.
- 02:05 Data analyst: KPI/metric pressure in the scenario is the one feature separating ODCV-good from ODCV-bad
  groups (59-74% vs 26-38%); AI content is a carrier. 25 Sep other-AI rows = measured-system-under-a-target; new
  other-AI rows = governance dilemmas.
- 02:20 ODCV analyst: the gap lives in ~10 latent-conflict cells; good arms flag the integrity risk in their first
  reasoning turn (rho -0.9 with MR across 23 arms); da-tools is the 25 Sep rows + tool schemas (ODCV 2.1).
- 02:30 Jamie redirected: hand over to Codex and halt. Handoff: scratch/autoresearch/HANDOFF_CODEX.md. Watchdog
  stopped. No experiment launched; $7.51 spent (OpenRouter labelling), 0 rows, 0 pods.
