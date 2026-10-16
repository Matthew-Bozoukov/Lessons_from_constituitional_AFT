<!-- ABOUTME: Index of historical GPT-OSS SWE, ODCV and backend diagnostic campaign utilities. -->
<!-- ABOUTME: Keeps one-off orchestration and its tests separate from maintained evaluation infrastructure. -->

# GPT-OSS campaign utilities

This directory retains the October 8–14, 2026 experiment and recovery code. The
historical `gptoss_swe` namespace also contains the ODCV and MXFP4 comparison drivers
that shared these owners. Paths remain stable because saved launch commands,
source manifests and archived deployments refer to them. These are not current
launch instructions: old host IDs, budgets, manifests and recovery actions describe
closed campaigns. See [the results index](../../docs/gptoss_campaigns.md) and
[chronological research log](../../docs/LOG.md) for the final status of each run.

## Code map

| Group | Modules / configs |
|---|---|
| Original SWE qualification and scheduling | `run`, `campaign`, `parallel`, `parallel_base`, `resume`, `rebalance_waiters`, `pilot.yaml`, `manifest.json` |
| Shared-interface and reference-Harmony smokes | `shared_smoke`, `prepare_shared_smoke`, `audit_shared_smoke`, `package_shared_smoke`, `openai_smoke`, `package_openai_smoke`, `document_openai_smoke`, `closeout_openai_smoke`, `finalize_openai_smoke`, `shared_smoke.Dockerfile` |
| Retained-outcome completion and recovery | `continue_openai_pending`, `grade_retained_openai`, `recover_openai_swe`, `refresh_smoke`, `finish_smoke`, `restart_base` and their YAMLs |
| Twenty-task screen and response-cap diagnostic | `twenty`, `twenty_report`, `response_quad` and their YAMLs |
| Fresh full 300-task arms | `full300`, `full300.yaml`, `full300_baseline.yaml`, `scoped_harness` |
| Full-run scheduling and accounting helpers | `full300_scale_base`, `compact_bridge_handoff`, `compact_budget_bridge`, `bridge_request_gate`, `early_grade` |
| ODCV setup and campaigns | `odcv_access`, `odcv_capacity`, `odcv_remaining`, `odcv_full`, `odcv_double_context` and their YAMLs |
| ODCV judging and publication | `odcv_full_judging_recovery`, `odcv_full_closeout`, `odcv_alternate_judge`, `verify_odcv_112k_publication` |
| MXFP4 backend pilot | `mxfp4_pilot`, `mxfp4_bridge`, `mxfp4_probe`, `mxfp4_odcv_recovery`, `mxfp4_path_recovery`, `mxfp4_progress_recovery`, `mxfp4_pilot.yaml` |
| Evidence handling / saved-run audits | `native_evidence`, `replay_renderer`, `analyze_stopped`, `summarize_stopped`, `compare_odcv_limits`, `audit_odcv_context`, `audit_teammate_*` |
| Historical patch-at-cutoff diagnostic | `patch_cutoff_{fetch,replay,grade,harness,publish}` |
| Campaign regression checks | `test_*.py`, colocated here because they import experiment code |

Experiment-specific configs stay beside their drivers in `scratch/`; registered
protocol defaults remain in `configs/eval/`. Reusable behavior belongs to the
existing modules in `src/infra/endpoints/`, `src/eval/capabilities/swebench_mini/`
and `src/eval/misalignment/odcv/`, with their independent tests in `tests/`.
There is no promotion of campaign-specific recovery or host-management code into
the production entrypoint.

## Validation and provenance

Run from the repository root. Most campaign tests are ordinary offline pytest
tests using the CPU environment; Linux-only file locking and official-harness
checks need their respective locked environments:

```bash
uv run --project scratch/swebench_cpu_env --frozen python -m pytest -q scratch/gptoss_swe
uv run --project src/eval/capabilities/swebench_mini/envs/harness --frozen python -m unittest scratch.gptoss_swe.test_early_grade scratch.gptoss_swe.test_scoped_harness
```

`test_bridge_handoff.py` is a separate explicit **privileged Linux qualification**,
not a normal unit test: its main program uses temporary ports and an iptables rule.
Do not run it as part of routine cleanup. Some MXFP4 tests inspect optional saved
pilot artifacts and skip when those artifacts are absent.

The full SWE inference runtime was `24cedbc04ca7a5275a1235499dd9850e34c0f611`.
Scale-up, compact accounting and concurrent grading used separately pinned helpers;
their identities and receipts are in the published artifact. Later formatting and
test relocation do not change which source produced the results. Raw outputs,
ledgers and ownership receipts remain under gitignored `output/` and in HF archives.
No saved results or historical sampling/scoring settings are rewritten by cleanup.
