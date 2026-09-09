<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Current execution and exact outcomes; historical decisions remain in LOG and frozen artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-09. Full intent and constraints: [research brief](research_brief.md).
Chronological evidence: [experiment log](../LOG.md). Earlier failures and reset:
[postmortem](2026-09-08_postmortem.md).

## Current execution

**Baseline fallback is active.** Evaluate the existing nonmoral-684, math-716 and
Table-2-only checkpoints with identical ODCV settings: 80 cells x 3 passes each,
temperature 0.7, context 28,000, both misalignment and task-progress judging.
These are **720 rollouts and no new training**. Exact checkpoint revisions and
protocol are in [the baseline inventory](baseline_inventory.md) and
[`scratch/nonmoral/odcv-paired.yaml`](../../scratch/nonmoral/odcv-paired.yaml).

The corrected nonmoral run is starting on RunPod. Math and Table-2-only follow.
No new alignment result is available yet. Historical **73/400 = 18.25% nonmoral**
and **98/240 = 40.83% math** used different context/harness settings and remain
historical observations, not the result of this common-protocol comparison.

## Fresh paired data: stopped at the frozen gate

| Item | Result |
|---|---:|
| Fresh development tasks | 32 across 8 domains |
| Valid after at most one correction | 13/32 |
| Correction failures | 11/32 |
| Source/construct exclusions | 8/32 |
| Frozen minimum required for scaling | 24/32 |
| Accepted domains; aggregate B/C token ratio | 6; 0.88874 |
| Settled generation cost | $1.204914 for 115 calls |

**No further generation, corpus scaling or new SFT will run in this overnight branch.**
The user explicitly authorized this fallback if validation failed. Remaining errors
include arithmetic, unsupported checks and review-history text leaking into examples.
Changing the gate after seeing these results would not be a legitimate repair.

[Public development artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-paired-development)
contains full examples, both versions, row-level reviews, raw requests and cost ledgers;
revision `105dc5c9c84d1de0361e14d453347777aaf60e85`. It is **not approved SFT data**.
Financial amounts in synthetic scenarios are fictional task details.

The intended contrast keeps original single-turn prompts and full final answers
identical. B weighs viable actions; C constructs/checks the selected answer without
weighing alternatives. Essential missing information excludes a prompt. Neither arm
may contain moral deliberation. This compliant paired recipe differs from the old
recipe that sometimes taught reasoned overrides of user instructions.

## Harness health and spending

The first partial baseline attempt was invalidated **before judging**: stale Windows
CRLF bytes in Linux scripts forced models to repair their environment. Its 32 completed
and 8 partial rollouts are preserved separately and will not enter the comparison.
Cost including cleanup: **$1.3044 estimated GPU/storage**. The owned pod was terminated.
[Public invalid/unjudged archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-nonmoral-invalid-crlf-unjudged-attempt)
preserves transcripts, the audit and repair evidence; revision
`6d2fc052115789118dbe5476d1e1313a58e801d2`.

Restored 164 scripts to exact committed LF bytes; all 168 scripts passed Docker
`bash -n`. Physical-byte preflight and source hashes now protect the corrected run.
Independent transcript inspection checks real tool behavior as well as harness status.

User approved **$300 total including prior spending**. The baseline lane has a $60
GPU/storage cap and $10 judge cap, durable reservations, a separate pod watchdog,
progress monitoring and verified teardown. All GPU inference runs on RunPod; Docker
drives ODCV locally. Any future authorized SFT uses **2xH200, dynamic batching** and
the shared training configuration. Artifacts are public under `dougalldeepmind`.

## What this can establish

The baseline comparison measures these three fixed checkpoints under one protocol.
Report exact counts, scenario-level intervals and paired differences, plus submission
and task progress. Evaluation passes are **not training seeds**. Training recipe and
dataset differences prevent a deliberation-only causal claim. Capability tests remain
deferred; submission/progress cannot establish capability preservation.

Shorter reasoning, a valid retained-reasoning/no-comparison control, broader domains
and nonmoral stakes remain future experiments. The failed paired-data branch does
not answer those questions.
