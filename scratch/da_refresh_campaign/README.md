<!-- ABOUTME: Handoff for the authorized September 28 DA corpus refresh at 15% supervised tokens. -->
<!-- ABOUTME: Preserve other tasks, inspect receipts, and never duplicate the single training. -->

Workspace `C:/Users/nikak/da-refresh-20260929`, branch `codex/da-refresh-20260929`,
based on current main `c38a29fc`. User authorized the same workflow as the prior
campaign: one seed-0 token-mean training, RunPod model GPUs, local CPU drivers,
parallel ODCV/MASK, dougalldeepmind artifacts, $200 ceiling and ~20-minute updates.

The new source is `dougalldeepmind/2026-09-28-da-synth` at
`14efefbf39581f4aaae08bfe7b4491e80adf2c7b`, 1,186 rows after eight exclusions.
Latest compatible nosynth is September 22 at
`378ec1ee0f0eea9294683779438b839e52b9700a`. The newer GPT-OSS nosynth is a
model-specific Harmony conversion and is not the Qwen control.

The team already built the exact filtered mixture. Reuse
`dougalldeepmind/2026-09-28-da-15-mix@ff52482340790eea9bf681348ffec6622b92057b`.
Audit verifies 9,050 rows, 617 DA, 8,433 replay, 4,842,435 supervised tokens,
731,012 DA supervised tokens (15.095959%). Every replay row is identical to
the previous September 25 DA-15 mixture; training-loader and byte-hash checks pass.

IMPORTANT: the existing September 28 DA-15 model at `067b64ce` used unfiltered
mixture `52eaa670`, whose source is `b84401ea`. Never overwrite or treat it as
the filtered-source model. Our aborted local duplicate mixture build did not
publish. New target: `dougalldeepmind/2026-09-29-qwen36-0-da-15`. Naming uses UTC;
the controller waits until September 29 00:00 UTC before renting, so names and
cards agree without touching the existing artifact. No GPU bills during the wait.

`run.py controller` is launched once, detached, and keeps Windows awake. It owns
one H200 training followed by separate H100 ODCV and H200 MASK jobs in parallel.
All pods have 3-hour deadlines and process-death watchdogs. API usage increase
is a shared-account conservative bound; $40 stop threshold, $60 reserve. Per-job
reservation $15.30, quoted rate <=$5.10/h. Only recorded owned pods may be removed.
Provisioning-only eval failures retry up to three attempts. A training failure
requires inspection; optimizer work must be resumed from its checkpoint, never
restarted. Training backups and complete logs are retained.

ODCV uses the existing `scratch.da_supervision.odcv_eval` counted pilot, disables
shared Docker pruning, shortens Windows cwd, and releases GPU before judging.
CLI options precede all dotlist overrides. ODCV concurrency 32, MASK generation
and serving concurrency 128 match the previous DA-15 metadata. Distinct local
ports 8631 and 8632. Docker preflight passed; no other containers or pods were
running at preparation. `prewarm.py` reuses Docker cache without pruning.

Read status: `uv run --frozen python scratch/da_refresh_campaign/status.py`.
Receipts: `output/da_refresh_campaign/`. `audit15.json` verifies mixture inputs;
`source_quality.json` records basic checks and upstream audit limitations.
Source has no duplicate conversations, exact overlap with September 25, or empty
assistant answers/reasoning. Upstream quality_filter was disabled; pattern scan
sanity recall 0.25 failed its 0.5 threshold. Do not treat its 83% pattern estimate
as validated. Upstream reports predate the exclusions.

After all three owners complete, verify HF layouts and exact target revision,
240/240 ODCV transcripts and both complete judgment axes, full 1,000-row MASK
and generation/empty-content rates. Run `report.py` and visually inspect its
figure. `references.json` pins prior DA-15 and nosynth. Previous DA-15:
ODCV 9.2% [4.3,18.6], task progress 4.95; MASK 90.2. One seed and one MASK pass
are descriptive, not a test of training-seed robustness. Record actual timing,
cost, protocol differences, and corpus recipe differences in docs/LOG.md and a
results write-up. Verify all owned pods absent; pause the campaign heartbeat.
Do not merge or erase another session's changes or overwrite its artifacts.

## Conditional expansion authorized September 29

The user subsequently authorized NEW-source DA-5 and DA-25 if the completed
DA-15 ODCV and MASK results are basically as good as the pinned previous DA-15.
If new DA-15 is noticeably worse, report that, terminate this campaign's GPUs,
and finish without expanding. This supersedes the unconditional pause after
DA-15 above. The existing controller still owns ONLY DA-15; do not restart it.

After verifying both complete DA-15 evals, record a gate decision and rationale
in `output/da_refresh_campaign/expansion_gate.json` before any additional rental.
Compare both endpoint estimates with the previous DA-15, consider ODCV scenario
uncertainty and task progress, and inspect generation errors and empty responses.
CI overlap alone does not establish equivalence. Use a conservative judgment:
expand only if both outcomes are reasonably comparable; if materially worse or
ambiguous, stop and explain the evidence. Do not repeat training or evals to
obtain a passing result. Report the decision and numerical differences.

On a pass, compose and audit 5% and 25% SUPERVISED-TOKEN mixtures from the SAME
pinned filtered DA source and Qwen-compatible nosynth used here. Reuse an existing
exact audited artifact where possible; otherwise publish distinct, conventionally
named artifacts under dougalldeepmind. Train each arm exactly once, seed 0,
token_mean, packing, token budget 8000, one epoch and the same base revision.
Launch the two trainings in parallel. After each adapter is verified on HF,
run its full ODCV and MASK concurrently, using local CPU drivers and RunPod model
GPUs. Inspect available local Docker capacity before overlapping both ODCV jobs;
preserve every other session's resources.

The $200 cap covers the ENTIRE refreshed 15/5/25 campaign, including DA-15 and
all provisioning attempts. Preserve the original API usage baseline and aggregate
all reservations and settled costs; do not reset accounting for expansion.
Keep ~20-minute progress updates. Finish with verified HF artifacts, comparisons
for all completed arms, a write-up, all owned pods absent, then pause the heartbeat.
