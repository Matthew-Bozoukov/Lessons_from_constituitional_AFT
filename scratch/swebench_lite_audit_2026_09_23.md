<!-- ABOUTME: Audit and proposed bounded SWE-bench Lite workflow for the latest DA LoRA. -->
<!-- ABOUTME: Verified metadata and code findings; no paid run or pipeline implementation yet. -->

# SWE-bench Lite: audit and implementation plan, 2026-09-23

## Scope and workspace

Current user decision: full SWE-bench Lite test split, matched no-DA SFT control ONLY.
One checkpoint at a time. Prioritize wall time and parallelize across more GPUs.
The DA adapter below is retained for a future comparison; it is not in this run.
Keep the upstream mini-SWE-agent scaffold and official grading harness. No difficulty
filter, no selection based on which tasks either model solves, no best-of-N.

Fetched origin/main at `4e13cb7148adae08ad9b438be28c1945b8fa2eed` and created
`codex/swebench-cheap` in a separate managed worktree. The original main checkout and
its untracked audit files were left untouched. No GPU was provisioned.

## Verified artifacts

Read live Hub model metadata and the exact training mixture's published statistics.
These statistics are publication metadata, not an independent re-tokenization.

| Artifact | Repository | Revision |
|---|---|---|
| DA LoRA | `dougalldeepmind/2026-09-22-qwen36-0-da-15` | `030951d90c7010eeda95cedeed1c7daccacf2af9` |
| No-DA LoRA | `dougalldeepmind/2026-09-22-qwen36-0-nosynth` | `633908b72a9799fb3e6b101b0a8a82aec3c3d642` |
| Shared base | `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| DA training mixture | `dougalldeepmind/2026-09-22-da-15-mix` | `e4871a2f531aff91bee00a78e3e593cd416bc376` |
| Control training mixture | `dougalldeepmind/2026-09-22-nosynth-mix` | `378ec1ee0f0eea9294683779438b839e52b9700a` |
| Lite benchmark | `princeton-nlp/SWE-bench_Lite` | `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` |

DA: 622 / 9,055 rows = 6.87%; 730,679 / 4,842,102 supervised tokens = 15.09%.
Both models: seed 0, thinking enabled, BF16 LoRA rank 64 / alpha 128, one epoch,
token-mean loss, packing, maximum training sequence 8,192. Same saved recipe values;
generating git commits differ and should remain in provenance. This is a checkpoint
comparison, not an estimate of variation across training seeds.

The root .env has `USER_PREFIX=nika` and `HF_ORG=dougalldeepmind`. Read credentials
in memory only; no secrets copied to this report or the isolated worktree. Current
publication code follows HF_ORG; the guide's older LASR-Callum wording is stale.

## What the old run actually establishes

`docs/swebench_run_postmortem.md` records one failed phase: about $121, 14 hours,
10 genuine completions / 372 attempts. It is not a verified invoice for the entire
campaign and does not disprove the user's recollection of $300+ and multiple days.

`docs/LOG.md`, 2026-08-07, records the later paired 250-task Verified result, using
seven H100 NVL rentals plus a Docker VM. It reports substantial context overflow
and transport failures. Those old checkpoints and settings are unsuitable as the
control for the new DA adapter.

The principal avoidable costs documented there were idle rented GPUs, cache
starvation, retry queues, unstable driver infrastructure, and failed tasks being
silently skipped on resume. Lite alone does not fix these.

## Code findings at the audited revision

1. **Current default is 50 Verified tasks, not Lite.** `mini` in `swebench_mini`
   identifies mini-SWE-agent. Lite is 300 test tasks; switching datasets while
   retaining `subset.fraction=0.1` would accidentally run only 30. Set full split.
2. **Serving limit drift.** `configs/models/qwen36.yaml` now supplies
   `max_num_seqs: 192`, verified for a short-context MASK workload. SWE config has
   no concurrency requirement. Calling `plan_serving` offline produces 192 sequences
   at 131,072 context, prefix cache on, and no warnings. This combination is not
   measured for SWE. Set an explicit conservative concurrency and calibrate it.
   Old SWE comments claiming prefix caching is unsupported are contradicted by the
   current profile and the historical A/B measurement.
3. **Agent dollar limit does not bound GPU spend.** `agent.write_cost_registry`
   assigns zero token costs; `MSWEA_COST_TRACKING=ignore_errors`. The upstream
   250-step limit remains, but the nominal $3 task limit cannot stop a local GPU.
4. **Dataset SHA is recorded without end-to-end enforcement.** Selection loads
   a resolved SHA, but `rollout_command` passes only the repo name to upstream.
   `grade` accepts a revision yet places it only in report metadata; its harness
   command also loads by repo name. Materialize one pinned dataset snapshot and
   make both phases consume it through supported upstream interfaces.
5. **Image tags are mutable.** `images.image_name` constructs `:latest` references.
   Record and, where the upstream runtime permits, enforce pulled digests. Do not
   let resume silently change images or harness dependencies.
6. **Resume is not an integrated run identity.** `run_eval` creates a fresh timed
   output directory. Upstream skips every ID in preds, including infrastructure
   failures. Add validated resume keyed by model/base/dataset/protocol pins; keep
   completed measurements and an append-only attempt ledger. Never reroll a valid
   failure or select a better patch from multiple attempts.
7. **Publication is too late.** For SWE, `run_eval` uploads only after the runner
   returns; there is no rolling checkpoint upload in this runner. A failure can
   bypass publication of completed work. The grading CLI defaults to `push=False`.
   Its independent repo-name construction uses the output parent and current date,
   omits the mode tag, and can miss the original rollout repository.
8. **GPU cleanup exists but must be wired in.** `runpod up` arms a detached local
   lifetime watchdog (six-hour default); `evals --terminate-pod` owns cleanup.
   These are reusable. The watchdog cannot enforce anything while its host is
   asleep or disconnected. It is a time cap, not an account spend meter.
9. **User prefix is not enforced by reusable provisioning code.** USER_PREFIX is
   in .env.example, but src/scripts contain no reader. Validate it before rental
   and derive a unique `nika-swebench-lite-<run>` name. Name alone is not ownership:
   retain provisioned pod ID, owner metadata, and this run's resource ledger.
10. **Docker is presently unavailable locally.** A read-only `docker info` probe
    cannot connect to Docker Desktop's Linux engine. Current grading code includes
    a Windows-to-Linux bridge; older docs saying Windows is impossible are stale.
    Gold-patch validation must pass on the chosen Docker host before GPU rental.

## Workflow to implement

Extend existing SWE and RunPod modules; use a scratch orchestration driver until
reviewed. Keep behavior in the existing modules, not a copied harness. A single
command should own these phases and support explicit plan, run, and resume modes:

1. Resolve all pins and full Lite IDs; verify LoRA/base/mode compatibility, HF write
   access, USER_PREFIX, output identity, Docker resources, and free disk. Install
   from committed locks. Pre-pull images and grade gold patches before renting.
2. Calibrate one RunPod inference GPU through `src/infra/runpod.py`, then scale to
   eight independent single-H200 pods serving ONLY the no-DA control. Register ownership
   and a watchdog immediately; reject a quote exceeding the declared budget.
   Start from BF16, unchanged thinking/tool parsers, 131k window, prefix caching,
   three rollout workers and an explicit small serving concurrency, e.g. four.
   These are pilot settings, not claimed optimal values.
3. Divide the full 300 tasks across the replicas. Use a durable queue with exclusive
   task leases, immutable attempt IDs and atomic completions; keep each running
   trajectory on one endpoint for prefix-cache reuse. A single coordinator owns
   publication. Do not have eight writers race on one preds.json or HF manifest.
   Existing static sharding can support an initial implementation, but its unequal
   task durations need tail handling. Reassign only unstarted work, never reroll a
   finished measurement to balance load. Eight pods can be terminated independently.
   Prepare Docker capacity for 24-48 active trajectories initially, measured before
   scale-up; add CPU hosts if command latency or grading contention becomes limiting.
4. Checkpoint completed trajectories, predictions, attempts, manifest, and usage to
   HF throughout the run. Make snapshot writes atomic; uploads must use consistent
   files. Persist the exact destination once and reuse it during grading/resume.
5. Watch completion rate, request queue, cache pressure, endpoint health, and infra
   errors. Trip a circuit breaker on sustained failure or no progress. Bound
   retries and cancel outstanding requests; do not add fresh work to a dying queue.
   Whole-run budget exhaustion produces an incomplete run, not a clean score.
6. Terminate and verify each owned GPU immediately when it has no work left. Grade
   completed batches concurrently on reserved CPU workers if that does not starve
   rollout containers; do not hold idle GPUs for grading. Keep external watchdog
   enforcement on an always-on host if unattended;
   do not depend on the Windows laptop remaining awake for a multi-hour run.
7. Grade saved patches on the CPU/Docker host using the same pinned tasks. Publish
   graded results automatically to the SAME HF repos, including readable rollouts,
   official reports, task-level outcomes, protocol pins, cost, and completion status.
   Retry an upload from durable local results without renting another GPU.
8. Publish the no-DA control's score, uncertainty and task-level outcomes, with the
   full 300-task denominator. Report infrastructure-invalid cases separately and
   require valid coverage before declaring the run complete. Future comparisons
   will use these frozen control outcomes; no DA model runs in this campaign.

A genuine step-limit failure remains a valid outcome. Infrastructure aborts may be
resumed from saved state if supported; otherwise any fresh attempt must be explicitly
recorded and governed by the same predeclared policy in both arms. Do not shorten
reasoning or context to meet budget and then present the result as unchanged protocol.

## Budget recommendation

Use at most **$10 as the proposed calibration allowance**, including boot, control only.
Use a fixed small pilot spread across repositories, with at least one long trajectory;
retain its outputs if the final protocol stays identical. A pilot is not a score.

Supersedes the earlier paired-run ceiling: plan around **eight replicas and 300
total trajectories for the control**. The earlier $60 paired proposal was a spending
ceiling, not a supported forecast. User has authorized more parallelism; an exact
cost cap and actual rental quote are not yet established. Do not treat an assistant
estimate as a user-approved spending limit. Reserve time for verified teardown.

RunPod's public pricing page checked 2026-09-23 lists H100 SXM at $3.49/hour and
H200 at $4.59/hour; actual selected offers and disk charges must be fetched at launch.
H200 has more room for long-context cache, but lower dollars per completed task must
be measured. An H100 is the existing model-profile default for SWE.

For illustration only, at eight H200s and 300 total trajectories, assuming ideal
work distribution (these rates have NOT been measured for Lite):

| Completed tasks per GPU-hour | Generation time | Generation GPU cost |
|---|---:|---:|
| 10 | 3.75 hours | $137.70 |
| 25 | 1.5 hours | $55.08 |
| 50 | 0.75 hours | $27.54 |

Eight 30-minute GPU boot periods add $18.36. CPU/storage, calibration, task-duration
imbalance, and cleanup latency add more. A tentative first-run allowance of $60-100
is consistent with the 25-50 tasks/GPU-hour scenarios, not with the slow scenario.
No clean Lite throughput or CPU-host price has yet established that allowance.
Aim for roughly 1-2 hours on a prepared system; cold image downloads, calibration,
boot and the longest individual task can extend end-to-end time. More replicas do
not shorten the serial steps inside an individual task. Measure before promising
an ETA; scale beyond eight only when the queue and Docker host still benefit.

Subsequent LoRAs can reuse the frozen control results
only when task, model-base, scaffold, serving, and grading identities still match.
Recalibrate after any protocol change; do not reuse August's control.

## Acceptance checks before paid scale-up

- Offline tests for full-Lite selection, immutable run identity, resume mismatch
  refusal, pin propagation to BOTH subprocesses, atomic HF checkpoint snapshots,
  same-repo graded publication, and inference-failure accounting.
- Mocked lifecycle checks: failed provisioning, failed watchdog, failed serving,
  Ctrl-C, budget deadline, failed upload, failed termination, and unrelated shared
  pods. A cleanup failure must keep enforcement alive until absence is verified.
- Real Docker gold-patch check, then a bounded live pilot demonstrating valid tool
  calls, reasoning, saved trajectories, graded outputs, HF publication, and verified
  termination. No claim of an operational one-command pipeline before this passes.

## External references

- Lite definition and 300-task test split: https://www.swebench.com/lite.html
- Advertised GPU rates: https://www.runpod.io/pricing

Status: audit and plan complete; no benchmark run, paid rental, or production code
change made in this task yet.
