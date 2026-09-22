# Practical low-stakes ODCV: 2026-09-22

User authorized one pass over 40 mandated and 40 incentivized scenarios, temperature
0.7, server seed 0, local Docker and one RunPod H100, with a $20 total ceiling.
Requested concurrency was 24, with permission to reduce to 16 or 12. The live Docker
pool holds 31 networks: 24 requires 48 and 16 requires 32, so the selected concurrency
is 12 (24 networks). No Docker settings or unrelated resources were changed.

Target: `dougalldeepmind/2026-09-22-qwen36-0-da-lowstakes-practical-7` at
`d37a4e0cf7a7b92d094f5ac0e79b230f5b1c6bcc`. Every adapter file matches the original
adapter-only training revision `55c52d797a2436b1535c6826167029708f32d24b`.
Base: `Qwen/Qwen3.6-27B` at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

Previous low-stakes protocol preserved: thinking enabled, context 28,000, native
Qwen tool parser and prefix caching, Gemini 3 Flash Preview for misalignment and
separate task-progress scores, four judge workers, severity >=3 as misalignment.
Server startup seed is 0; individual requests do not reset the seed.
The first scheduled cell is a counted startup check before broad dispatch.

Reuses `scratch/nonmoral/overnight_baseline.py` and the registered ODCV evaluator.
Windows wildcard listener/localhost client and short-working-directory fixes are
reused from `scratch/da_supervision/odcv_eval.py`. Shared Docker network pruning is
disabled. Budgets: $15 GPU/storage including $2 recovery reserve, $5 judging.
Live H100 quote: $3.49/hour. Pod name: `nika-low-stakes-practical-odcv`.

Pre-rental checks passed: 40+40 inventory, actual LF shell inputs, all shell syntax
inside Docker, real Compose config and image build at a longest-scenario path,
network capacity, free port 18124, pinned model/mode identity and 18 focused tests.
Launch manifests and receipts: `output/odcv_practical_20260922/`.
Completed successfully; verified results and teardown are recorded below.


## Verified result

[HF results, transcripts and provenance](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-odcv-qwen36-0-da-lowstakes-practical-7/tree/ce9320ef4b2973dc1a0fe4181636df6cfa7e5d87)
at `ce9320ef4b2973dc1a0fe4181636df6cfa7e5d87`. All 265 published files match local hashes.

| Variant | Misaligned / judged | MR |
| --- | --- | --- |
| Mandated | 5 / 40 | 12.5% |
| Incentivized | 6 / 40 | 15.0% |
| Overall | 11 / 80 | 13.75% |

All 80 native transcripts and all 160 judge calls are present, with zero missing or
all-NA judgments, zero rollout retries, and zero reconstructed transcripts.
Three context-limit cutoffs were retained and judged under the standard protocol.
Task submission: 77/80; mean judged task progress: 4.95/5. The native scenario-based
95% MR interval is 6.6–26.3%. This is one training checkpoint and one rollout per cell;
it does not isolate a causal effect of the revised generator.

Rollouts took 22.8 minutes; total owned GPU lifetime including startup, judging and
log recovery was 36.5 minutes. GPU/storage estimate: $2.184137; call-ledger judging
charges: $0.775196; **total estimate $2.959333** of the approved $20 ceiling.
Use the request ledger for cost attribution; account-wide OpenRouter usage deltas
in native logs are not the authoritative per-run bill.

Pod `hm7ucwbx9vthdc` was terminated after remote-log recovery; a fresh provider
inventory independently confirmed it absent and no remaining pods at that check.
Local Docker has no active scenario containers. Temporary sleep prevention was
released. The source commit for the run is `ba71f68f`; completion receipts and
read-only monitoring are committed separately. No model/data regeneration occurred.

`metadata/operations/` on HF preserves the frozen plan/config, server command
(seed 0 verified live), counted-pilot/preflight evidence, judge ledger, verified
remote logs, completion summary and initial publication hash audit.
