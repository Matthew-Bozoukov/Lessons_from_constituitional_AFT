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
