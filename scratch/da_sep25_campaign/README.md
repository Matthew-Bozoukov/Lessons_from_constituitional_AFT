<!-- ABOUTME: Operational handoff for the authorized Sep 25 DA-5/DA-25 campaign. -->
<!-- ABOUTME: Inspect receipts before acting; never duplicate a training or touch another task's pod. -->

User authorized two final adapters, trained once each with seed 0, supervised-token
mixture shares and `train.loss_agg=token_mean`. All publication goes to
`dougalldeepmind`. Total campaign ceiling is $200. Updates are requested approximately
every 20 minutes; heartbeat `da-5-and-25-campaign-updates` belongs to this task.

Working tree: `C:/Users/nikak/da-sep25-5-25`, branch `codex/da-sep25-5-25`.
Original checkout and all other worktrees must remain untouched.

COMPLETED September 28, 2026: both datasets, both once-trained adapters and all
four evaluations are verified on HF. All ten owned pods are absent. Successful
evaluation attempts are ODCV 5/25 attempt 2, MASK 5 attempt 3 and MASK 25 attempt 2.
`completion_verification.json`, `comparison.json` and `report_receipt.json` hold
the final evidence. ODCV misconduct: 20.4% / 7.5%; MASK honesty: 72.0 / 92.2%.
Conservative cost estimate: $44.13 including the shared-account API upper bound.
The original controller's `needs_recovery` state is historical; the recovery
supervisors finished successfully. Do not restart this completed campaign.

Source pins and operating limits are in `campaign.yaml`. Published mixture audits
are `output/da_sep25_campaign/audit5.json` and `audit25.json`; both passed before
renting. The build uses the unchanged September 22 nosynth base and September 25
DA corpus. Actual DA supervised-token shares: 5.01962294% and 24.99883308%.

`run.py controller` was launched once, detached, and holds a temporary Windows
keep-awake request. It runs two independent training owners, then starts separate
ODCV and MASK jobs per verified training. Never launch a second controller without
inspecting `controller.json`, the recorded PIDs and the per-job status receipts.
The owner's `status.json` has the actual Python PID (the launcher PID may differ
because of Windows venv redirectors).

Training launches exactly one full training invocation; the trainer itself runs
the mask gate before optimizer work. There is no separate optimizer smoke. The
first completed checkpoint is backed up locally in parallel; final output files
are archived and the complete training log uploaded to the adapter repo. Eval
dispatch waits for the final adapter revision including that log upload.

Each pod has its own three-hour deadline guard and process-death watchdog.
`budget.json` atomically reserves $15.30 per rental and $60 API headroom. The
controller stops at $40 of shared-account API usage increase, reserving another
$20 for settlement/in-flight work; account-wide changes are a conservative bound,
not an attribution of spending to this campaign. Actual paid job costs replace
reservations only after termination is verified. No resource outside the recorded
owned pod IDs may be stopped. All drivers and judging stay local.

ODCV uses the existing scratch adapter `scratch.da_supervision.odcv_eval`, which
keeps the standard eval lifecycle but runs a counted pilot cell, disables shared
daemon pruning, fixes Windows wildcard client addressing and shortens process cwd
paths. It releases the GPU before local judging. Both arms initially use scenario
concurrency 16 to share the 20-CPU/18-GiB Docker allocation. MASK uses the standard
entrypoint at concurrency 192. Four distinct ports are allocated in the config.

`prewarm.py` builds all 80 ODCV scenario images with two local build workers. It
starts no containers and never prunes shared Docker state. The warmed layers are
reused when the actual uniquely named scenario images build.

Read-only status: `uv run python scratch/da_sep25_campaign/status.py`.
Training completed once per arm and both training pods were terminated. Evaluation
provisioning encountered a RunPod 500 and SSH endpoint timeouts. `recover.py` owns
bounded retries before rollout work, with the same reservations and API ceiling;
its receipts are `recovery.json` and `recovery-odcv25.json`. Do not launch duplicates.
ODCV-25 attempt 1 also failed during CLI validation because dotlist overrides were
interspersed with options. This is fixed and parser-verified; no rollouts ran in
that attempt. Its specifically authorized recovery supervisor handles attempt 2.
The ODCV wrapper normalizes overrides for already-running owners using the old
command layout. Reports choose the sole completed attempt and audit all retired
attempts for teardown, including failed provisioning.
Failures must be diagnosed from the job log and receipts. A failed provisioning
attempt that never started training may be re-rented within the budget. If any
optimizer steps ran, preserve and resume that checkpoint rather than starting a
fresh training. Do not treat partial evals or missing judgments as completion.

Finish only after both datasets, adapters and four eval result repositories are
verified on HF, all owned pods are gone, comparison/report artifacts are written,
and an experiment entry is appended at the top of `docs/LOG.md`. Pause the heartbeat
at completion. The CLAUDE namespace correction is already committed on this branch.
