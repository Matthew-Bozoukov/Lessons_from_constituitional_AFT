---
name: swebench-lite
description: Run all 300 SWE-bench Lite tasks for one Qwen3.6 LoRA, or prepare/manage the reusable Vast CPU. Use for “run swebench on this lora”, “prepare the swebench cpu”, and SWE-bench status/recovery requests in this repository.
---

# Run SWE-bench Lite

Read `CLAUDE.md`, `docs/swebench_lite_runbook.md`, and
`docs/swebench_cpu_lifecycle.md` from the repository root. Those files, the committed
configuration and live receipts replace conversation history. Use this skill from
main in any new conversation. A request to run the benchmark authorizes preparing
its CPU and running one model under the defaults below; “prepare CPU” authorizes
CPU work only. Do not rent an extra model evaluation for calibration.

## Fixed operating defaults

- All 300 Lite tasks; one pinned compatible Qwen3.6-27B rank-64 thinking adapter.
- Up to 20 single-GPU RunPod replicas, four conversations each. Start each ready
  GPU independently. H100 NVL first, delayed H200 fallback, RTX PRO 6000 Blackwell
  Server last. Preserve the configuration's price ceilings and automatic retries.
- Native-Docker Vast CPU: target 64 effective CPUs, at least 190 GiB actual RAM,
  500 GiB disk; qualify 80 conversations with a separate 32-command gate and 12
  grading workers. A cheaper smaller CPU is not a drop-in replacement.
- New campaign GPU backstop $180, not an expected bill. Report the live quote and
  allowance before launch without asking the user to repeat this standing choice.
  Never reset an existing ledger or silently raise its allowance.
- CPU is persistent until explicitly stopped. The user selected this on
  September 24, 2026. Persistence never removes finite GPU watchdogs, task limits,
  budget limits or retry bounds. CPU cost is separate, including idle time.
- Progress every 15 minutes and immediate completion/failure updates. Create one
  Codex heartbeat for the new campaign, verify ACTIVE and a real initial status
  read. Its instructions must include exact paths, limits, ownership and reporting.
  Delete it after verified completion. If tools are unavailable, explicitly report
  that chat notification is unavailable; never promise it merely because systemd runs.

## Execute the request

1. Use the authorized repo `.env` without printing it. Require USER_PREFIX,
   HF_ORG, HF_TOKEN, VAST_API_KEY, RUNPOD_API_KEY and Docker Hub pull credentials
   for a cold cache. Resolve an ambiguous adapter on HF by provenance, then report
   its exact repo/revision. Incompatible base/rank/mode is a real blocker.
2. Inspect the shared `~/.lasr/swebench/cpu.json` with the CPU management command
   in the lifecycle guide. Verify the receipted provider ID and label. Reuse a
   running host; resume a stopped owned host. Empty means provision and prepare
   following the guide, with zero inference rentals until readiness passes.
   Missing/ambiguous owned resources must be reconciled, not replaced blindly.
3. Verify no active model is being replaced. Deploy a committed source archive,
   excluding secrets, to `/srv/lasr/repo`. Compare recipe source hashes. When code
   changed, run the documented CPU-only qualification; never approve drift by
   merely replacing the hash map. Preserve completed campaign manifests.
4. Submit via the standard evaluation entrypoint on the ready CPU:

   ```bash
   uv run --project scratch/swebench_cpu_env --frozen python -m src.eval.run_eval \
     --name swebench_mini --fleet --target ORG/LORA --target-revision SHA --budget-usd 180
   ```

   From the local machine add `--cpu-receipt RECEIPT --cpu-key IDENTITY` from the
   registry. Load the authorized env with uv `--env-file`. This launch is idempotent
   for the same completed dated campaign and refuses to replace an active model.
5. Read initial status, budget admission and HF checkpoint. Record the generated
   campaign path in the heartbeat. Report valid/running/pending counts, GPU types
   and count, grading/HF state, conservative ledger versus billing, and measured
   or estimated ETA. Do not call reserved rental ceilings actual charges.
6. Let bounded recovery run. Retry infrastructure-invalid attempts only; never
   reroll valid model failures. Later HF outages must not cancel inference.
   Diagnose terminal exit 2 and report its saved reason instead of restarting
   blindly. RunPod rentals only through `src/infra/runpod.py`.
7. Completion means 300 valid outcomes, 300 officially graded, HF results readback
   plus rollout/result hashes verified, and zero campaign-owned GPUs at provider.
   Leave the persistent CPU/cache running. Report score, costs and protocol caveats.

Persistent failures, exhausted allowance/retries, ownership drift, unavailable
credentials or incompatible adapters can still require the user. Do not claim an
issue-free guarantee. Historical mixed-fleet costs are evidence, not a new quote.
