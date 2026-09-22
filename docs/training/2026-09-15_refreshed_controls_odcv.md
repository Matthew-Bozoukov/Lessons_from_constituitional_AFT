<!-- ABOUTME: Record the approved single-pass ODCV protocol for both refreshed-control adapters. -->
<!-- ABOUTME: Distinguish launch/preflight evidence from completed rollouts, judgments and verified publication. -->

# Refreshed controls: single-pass ODCV

The user explicitly approved launch after reviewing the plan: two parallel
evaluations, local CPU/Docker, one separate single-H100 pod per adapter, thinking
on, all 40 scenarios in both mandated and incentivized variants, one pass only.
That is 80 rollouts per adapter and 160 total. Combined spending is capped at
$30, allocated $12 GPU/storage and $3 judging per arm, with $2 of each GPU
allocation reserved for recovery. These are new evaluation allocations, separate
from completed dataset generation and training.

## Frozen targets and protocol

- Low: `dougalldeepmind/2026-09-15-qwen36-0-da-lowstakes-refresh-7`
  @ `095a9874a1ce54ab1faaa3a99d63642e6199c591`.
- Nonmoral: `dougalldeepmind/2026-09-15-qwen36-0-nonmoral-advice-7`
  @ `85ff41394f398d49a59a047a94a61dc07af4b3bd`.
- Shared base: `Qwen/Qwen3.6-27B`
  @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Source config: `configs/eval/odcv/lite.yaml`, overridden to passes 1,
  concurrency 6 per arm, judge workers 4, and first-scheduled-cell preflight.
- Temperature 0.7; context 28000; thinking mode inferred as `think` from both
  adapters; Qwen reasoning/tool parsers and prefix caching retained. No scenario
  exclusions, added constitution or other system-prompt intervention.
- Both misalignment and progress: `google/gemini-3-flash-preview`, pinned to
  `google-ai-studio` with no provider fallback. Separate per-arm judge ledgers
  cap all requests/reservations; no cross-process shared ledger.
- Prior public DA and stakes metadata confirmed this coverage and both judges.
  Those runs used three passes; these single-pass estimates have less repetition
  and do not support a training-seed ranking or a clean dataset-property effect.

## Execution and verification

The existing `scratch/nonmoral/overnight_baseline.py` owner now accepts an explicit
`odcv-refresh-*` plan with 1x80 coverage, distinct ports and aggregate network
reservation; older three-pass plans retain their original constraints. It invokes
the standard eval lifecycle with a checked, frozen target spec. The first scheduled
cell must produce a nonempty clean transcript before the rest are dispatched; it
is scored exactly once in the same pass. Its failure stops the queue and preserves
the observed evidence. Recorded timeouts and completed outcomes are not retried.

Both local Docker/network preflights passed: 20 CPUs, approximately 18 GB Docker
memory, no active containers, 31 available network slots. Twelve concurrent cells
need 24 networks. All 168 benchmark shell scripts have LF bytes; the real first
scenario Docker images built successfully before any GPU allocation. Nineteen
focused owner, timeout, pass and first-cell tests passed. Ports 18110/18111 were
available. Short local roots are `C:/odcv-low` and `C:/odcv-non`.

Plans, config hashes, PIDs, spending and watchdog state live under
`output/2026-09-15_refreshed_controls_odcv`. Each pod is registered with an
independent watchdog immediately on allocation. A local sleep-inhibition helper
and recurring task monitor cover the run through upload, verification and owned
resource shutdown. Docker rollouts and judgments are local throughout; only the
small boot/vLLM logs need retrieval before teardown.

Publication follows the pipeline's configured `HF_ORG=dougalldeepmind`, matching
the previous verified ODCV artifacts and the available token's organization.
The older CLAUDE.md paragraph naming LASR-Callum conflicts with its current
environment-based publishing contract and actual pipeline. No credential or
global environment file is changed. Names come from the existing eval builder;
rollouts/results/metadata and required discovery tags remain mandatory.

Completion requires all 80 intended cells accounted for per arm, explicit missing
or partial outcomes, cached MR/progress judgments, full readable rollouts,
published file verification, and provider-confirmed termination of both owned
pods. Initial readiness is not a completed evaluation claim.

## Completed results and interpretation

Both evaluations completed the intended single pass: 40 mandated and 40
incentivized cells each, all 160 transcripts nonempty, all 320 MR/progress judge
calls settled with finite scores, no missing cells, timeouts, rollout retries,
reconstruction, or dropped passes. The first scheduled cells passed and were
counted once. No additional behavioural samples were generated.

| Metric | Moral low-stakes | Nonmoral advice |
|---|---:|---:|
| Overall MR (severity >= 3) | 16/80 = 20.00% | 31/80 = 38.75% |
| Mandated MR | 8/40 = 20.0% | 16/40 = 40.0% |
| Incentivized MR | 8/40 = 20.0% | 15/40 = 37.5% |
| Overall MR 95% scenario interval | 11.0–33.7% | 25.8–53.6% |
| Mean task-progress score | 4.84/5 | 4.92/5 |
| Task progress >= 3 | 77/80 = 96.25% | 79/80 = 98.75% |
| Submission tool called | 77/80 = 96.25% | 79/80 = 98.75% |
| Context/token limit flagged | 3 | 1 |

Nonmoral minus low-stakes MR is **+18.75 percentage points**, with a paired
95% interval of **+7.48 to +30.02 points**. This calculation pairs the same 40
base scenarios across models, averages both variants within each scenario, and
uses a Student-t interval with 39 degrees of freedom. Condition-specific paired
differences are +20.0 points (mandated; 5.16–34.84) and +17.5 points
(incentivized; 5.19–29.81). These fixed-checkpoint results favor low-stakes on
this evaluation; one training seed and one rollout per cell cannot isolate a
dataset-property effect or measure training-seed variability. Task progress and
submission measure task pursuit, not ethical conduct. No fresh DA or base-model
control was collected in this run; the automatically embedded paper baseline is
not a newly matched control.

The limit-flagged outcomes were retained and judged, not retried: low-stakes
mandated Healthcare-Appointment-Optimizer, incentivized
Ai-Transport-Fleet-Safety-Vs-Efficiency, and incentivized
Health-Data-Compliance-Analyst; nonmoral mandated
Healthcare-Appointment-Optimizer. No rollout reached cycle 50. One low-stakes
progress score was capped at 4 for missing submission; no nonmoral score required
that cap. An infrastructure status of `ok` does not mean an unlimited transcript
or ethically successful behavior.

## Publication, spending, and closure

- [Low-stakes evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-odcv-qwen36-0-da-lowstakes-refresh-7)
  @ `6e59bb706e48dcbdbb594ac66074db06b38e6e2c`.
- [Nonmoral evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-odcv-qwen36-0-nonmoral-advice-7)
  @ `7bcb4fbe0a00a84a24b79037c0640489b3f9ed1d`.

All 255 artifact files per publication matched local bytes using Hub Git-blob
SHA-1 or LFS SHA-256 identities and sizes (510 files total; auto-created
`.gitattributes` excluded). Both complete rollouts/results/metadata layouts,
discovery tags, exact adapter/base revisions, configuration and code commit
`ce6ef5811faba18bad5263cb53b884a944e5a54e` were verified. The verification script
is `scratch/dataset_refresh/verify_odcv_completion.py`; detailed manifests and
paired calculations are under the run root in `publication_verified.json`
(per owner) and `verified_comparison.json`.

Estimated GPU/storage cost through owner cleanup was $2.851157 low-stakes and
$3.067718 nonmoral. Settled per-request judge ledgers total $0.834606 and
$0.886452 respectively. Combined estimated cost is **$7.64**, below the $30
ceiling; GPU/storage is an estimate, not an itemized provider invoice. The
pipeline's account-wide OpenRouter before/after deltas overlap across the
parallel arms and must not be used as per-run costs; use the isolated ledgers.
The initial nonmoral allocation returned HTTP 500 without allocating a pod;
inventory was checked before the successful retry. No rollout was duplicated.

Both owners retrieved and hash-verified their small boot/server log archives and
terminated only their own pods (`vc9lp1suhmgvmj`, `21g08ehh43p14g`). A subsequent
provider inventory confirmed both absent. The shared account balance at closure
was $208.830938, with $12.727/hour belonging to other pods left untouched. The
temporary sleep-inhibition helper was told to exit after owned-resource closure.
Provider evidence is in `provider_closure.json`; monitoring is retired after this
report and branch push are complete.
