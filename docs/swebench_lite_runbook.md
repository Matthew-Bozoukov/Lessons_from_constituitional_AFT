<!-- ABOUTME: Repeatable operating procedure for the frozen no-DA SWE-bench Lite campaign. -->
<!-- ABOUTME: Prepared infrastructure is separate from explicit authorization to start paid inference. -->

# SWE-bench Lite: one model, all 300 tasks

## Current reusable launch procedure (2026-09-24)

This section supersedes the historical launch/recovery examples below. The current
template is `configs/eval/swebench_lite.yaml`; the old scratch YAML remains the
immutable historical campaign configuration. Do not resume that completed campaign
under new source hashes.

- One compatible Qwen3.6-27B rank-64 thinking LoRA at a time, pinned by HF revision.
- **Up to 20 independent H100 NVL pods, four conversations each**, with 80 CPU
  conversation slots and **32 simultaneous tool commands**. Start each ready pod
  independently. Retry missing capacity; only after both ten minutes and five failures
  try H200/RTX PRO 6000. Release idle pods independently. A small remaining queue
  uses fewer pods. No per-adapter concurrency sweep.
- The existing CPU passed two GPU-free 40/60/80-workload exercises. The final one
  used the production file-lock admission gate: 526 real test commands, zero failed
  commands/OOMs, minimum available RAM 177.58 GiB. This qualifies the tested workload
  and host class, not every command a future model could invent. Keep resource caps,
  descendant cleanup, memory/disk guards and separate 12-worker official grading.
- H100 NVL remains the selected default. The 20-GPU end-to-end runtime/cost is **not
  measured yet**. Four conversations are fixed; vLLM's eight allocated sequence slots
  are serving capacity and do not launch eight benchmark conversations.

On the prepared CPU, from `/srv/lasr/repo`:

```bash
uv run --project scratch/swebench_cpu_env --frozen python -m src.eval.run_eval \
  --name swebench_mini --fleet --target ORG/LORA --budget-usd 180
```

This is the standard eval entrypoint using its CPU-only environment. `uv run evals`
accepts the same flags wherever the main project environment is already installed.
From Windows, add `--cpu-receipt PATH/receipt.json --cpu-key C:/Users/nikak/.ssh/msm_audit`
and load the authorized local `.env` with uv's `--env-file`. The helper checks the
Vast instance ID/label/deadline, resumes a stopped CPU within its existing authorized
lifetime, refreshes the provider SSH endpoint, verifies the remote receipt and then
submits. Unknown SSH host keys, expired CPU lifetime or missing infrastructure require
repair/review. It never creates a host, extends expiry or copies credentials silently.

`180` is a proposed **new-campaign maximum**, not a forecast, a charge or an increase
to the completed campaign's budget. Pass the allowance authorized for that new run.
At the observed $3.19/hour H100 quote plus 5% reserve, 20 pods for the full 150-minute
emergency lifetime reserve $167.48. Earlier completion releases reservations; fallback
cards can be refused when the remaining allowance is insufficient. Never reset the
ledger or interpret rejected create reservations as invoices.

The launcher serializes submissions, pins the model, creates a dated target directory,
checks all CPU caches/readiness and the qualified recipe, and only then arms systemd.
Repeat launches of a verified completed campaign do nothing. Active runs are protected
from replacement. The service survives terminal disconnects and is enabled for the
authorized job; automatic crash recovery preserves its budget, attempts and absolute
deadline. Explicit stops remain stopped. A graceful SIGTERM is treated as a stop;
reboot recovery after such a stop needs explicit review rather than silently overriding it.

The supervisor allows at most four recovery cycles and three infrastructure attempts
per task. Completed model failures are never rerolled. Rental expiry, the six-hour
whole-job bound and the CPU expiry are separate safety limits, not ETAs. New tasks
must fit their full 90-minute allowance plus cleanup; additional cycles drain remaining
eligible tasks if budget and lifetime allow. Do not manually extend live rentals.

HF snapshots are atomic, checked initially before rentals, and retried asynchronously
from inference progress. Later backup failure does not cancel agents or prevent grading.
After GPU teardown, official grading retries up to three times; publication retries
for up to an hour within the job deadline. Completion requires HF result readback and
hash verification of every rollout/result file, plus zero owned pods. Exhausted attempts,
budget, lifetime, resource guards, persistent grading/Hub errors, incompatible LoRAs or
ownership/key mismatches produce a saved actionable status. No chat scheduler is enabled.

Inspect or stop using the generated `launch.yaml`:

```bash
scratch/swebench_cpu_env/.venv/bin/python -m src.eval.capabilities.swebench_mini.fleet status --config /srv/lasr/runs/RUN/launch.yaml
scratch/swebench_cpu_env/.venv/bin/python -m src.eval.capabilities.swebench_mini.fleet stop --config /srv/lasr/runs/RUN/launch.yaml
```

Read `metadata/supervisor.json` for publication/terminal state and
`metadata/final-accounting.json` for ledger versus delayed provider billing.
CPU/storage/transfer charges remain separate. `metadata/cache-audit.json` records actual
prefix hits and preemptions. The historical serving telemetry had 96.94% prefix hits,
zero sampled-counter preemptions and maximum KV use 49.43%; that is not a 96.94% speedup.

Prebuilt serving images remain an optional later optimization. Each GPU host still
downloads image layers and weights; an image packages installation once instead of
repeating it. The official vLLM 0.26.0 image has a `vllm serve` entrypoint incompatible
with blindly passing this bootstrap's shell command. Do not switch it into the rental
path without adapting/publishing the image and validating CUDA. The current Docker Hub
read token supports pulls, not publishing a custom image. No new image or GPU tuning
experiment is required for the reliable launch path above.

Infrastructure qualification evidence (synthetic, never a model score):
https://huggingface.co/datasets/dougalldeepmind/2026-09-24-swebench-lite-infrastructure.
The recipe explicitly separates protocol/CPU/recovery qualification from a clean paid
performance measurement. The first future authorized model run supplies the latter;
do not buy a redundant full evaluation merely to rename the qualification flag.

## Historical campaign and recovery record

**Completion verified September 24, 2026:** the no-DA campaign now has 300 valid
outcomes and 300 graded tasks, with 136 resolved (45.33%). All owned inference
GPUs are terminated; the chat heartbeat remains paused. Final HF verification is
`metadata/final-verification-20260924.json`, published at
`e3270863ea29fb78cc3e58fd5860347d8dd317a2`. The closed conservative GPU ledger is
$90.22; CPU/storage/transfer costs are separate. The CPU's existing stop deadline
remains September 24 12:37 UTC. Historical preparation/recovery sections below
describe the experiment as it evolved, not unfinished work. Full coverage does
not validate a uniform-fleet performance estimate or a 20-GPU recipe.

The prepared campaign evaluates only `dougalldeepmind/2026-09-22-qwen36-0-nosynth`
at `633908b72a9799fb3e6b101b0a8a82aec3c3d642`, on Qwen3.6-27B base revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. It uses all 300 Lite test cases at
dataset revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`.
No DA model is included. The old expensive evaluation is not the matched control.

**Operating preference, 2026-09-23:** this is the initial experimental calibration.
The user authorized beginning the no-DA job with the proposed $100 GPU ceiling.
The final reusable pipeline must use a measured, frozen execution recipe for a
compatible base model and LoRA: rent its chosen fleet concurrently at the outset,
without repeating per-model GPU-count discovery or gradual scale-up. Save this
run's timing, cost and concurrency evidence to select that recipe. Recalibration
belongs to explicit experiments or material serving/hardware changes, not every
new adapter. The fixed-fleet interface is implemented below; it stays gated until
this experimental campaign completes all 300 graded tasks.

## What to do when Nika says “run SWE-bench Lite for the no-DA control”

1. Read this guide and `docs/swebench_cpu_host.md`. Work on `codex/swebench-cheap`,
   preserving the other local checkout. Connect to the prepared CPU VM:

   ```powershell
   ssh -i C:\Users\nikak\.ssh\msm_audit -p 47579 root@184.144.154.180
   ```

2. Check the VM expiry and run the **non-renting preflight** on the CPU host:

   ```bash
   cd /srv/lasr/repo
   scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite plan
   ```

   The current VM stops, retaining disk, at **2026-09-24 12:37:16 UTC**.
   The recovery preflight requires at least 4.5 hours remaining (150-minute inference backstop plus two hours for CPU finishing). Do not extend its deadline
   or create replacement infrastructure silently. If it expired, use the CPU-host
   guide to restart/requalify with a newly authorized lifetime before inference.
   The plan verifies 300 local image digests AND the grader's matching cached tags,
   the frozen dataset, successful backed-up readiness, unchanged grading wrapper,
   local HTTP fixture, disk/RAM/CPU, model/base/mode/rank, SSH key and provider quote.

3. **Only after the user requests the model run**, choose the explicit spend cap.
   Current authorized campaign allowance: **$100 maximum, including up to $20
   calibration reservations**. A future model run still needs its own user request;
   these numbers are ceilings, not a cost forecast.
   The CPU costs about $0.54/hour separately. Confirm the live shared account has
   sufficient credit and report spend above $20. Existing teammate pods are unrelated.

4. Start the persistent service; it survives SSH disconnects. For a first run:

   ```bash
   cat > /srv/lasr/lite-launch.env <<'EOF'
   LITE_ACTION=run
   LITE_BUDGET_USD=100
   EOF
   chmod 600 /srv/lasr/lite-launch.env
   systemctl start --no-block lasr-swebench-lite.service
   tail -f /srv/lasr/runs/lite-coordinator.log
   ```

   Use the approved cap rather than automatically copying 100. Direct paid CLI
   launches outside systemd are refused because the independent reaper supervises
   this unit. The service is deliberately **not enabled at boot**: CPU reboots do
   not silently start another paid evaluation.

5. Observe progress without rerunning the launch:

   ```bash
   scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite status
   systemctl status lasr-swebench-lite.service
   ```

   Canonical artifacts are under `/srv/lasr/runs/lite-nosynth`. The immutable
   `metadata/manifest.json` records the exact HF destination, code/config hashes,
   pins and cumulative cap. Replicas have separate logs under `metadata/`.
   Full completion requires 300 valid rollouts and 300 graded outcomes. An incomplete
   run has `pass_at_1: null`; the observed resolved count over 300 is labelled separately.

6. Report the HF link, valid/graded/resolved counts, full-denominator score and
   uncertainty, infrastructure retries, GPU lifecycle/cost estimate, and live
   account balance. Verify no pods with this campaign's ownership nonce remain.
   Do not claim an estimate is an invoiced bill or a model result is established
   by the synthetic integration test.

## Recovery of the final 47 tasks (2026-09-23)

This section supersedes the historical fleet/deadline rules below. The user
authorized only the 47 unfinished tasks: preserve all 253 graded model outcomes.
Six independent RTX PRO 6000 Blackwell Server Edition replicas each consume four
tasks from one shared queue. After repeated capacity rejections over ten minutes,
try H100 NVL; no healthy replica waits for another. The original $100 cumulative
GPU cap was relaxed by the user; use $120 as the new hard backstop, not a target
spend. CPU cost remains separate. Keep the chat heartbeat PAUSED.

Before launch, archive old state/manifest/config/source files, record the exact
47-task allowlist and hashes of completed outcomes, and document the migration
in `metadata/remaining-47-recovery.json`. Reconcile explicit rejected reservations
without deleting ledger entries. Keep model, dataset, token/step limits and scoring
unchanged. Max infrastructure attempts becomes three; completed model failures
remain immutable. Verify CPU caches, expiry, empty owned GPU inventory, code/config
identity, and a real atomic HF snapshot/readback before any rental.

The 150-minute fleet expiry is an emergency bound, not an ETA. Allow at least
108 minutes before allocating a new pod (15-minute bootstrap, full 90-minute task,
three-minute cleanup). Workers stop claiming tasks 93 minutes before pod expiry
and let existing tasks finish. All six release independently when idle. A cold
bootstrap fails after 15 minutes; do not keep buying slow downloads indefinitely.
HF outages do not interrupt agents or prevent CPU grading; snapshots keep retrying,
and final publication must be verified separately from local completion.

See `docs/GOTCHAS.md` for the observed failure modes and their fixes. The run is
heterogeneous across its earlier phases; completion alone does not establish a
clean, uniform six-GPU performance benchmark.

## Request-timeout correction (2026-09-23)

The recovery exposed LiteLLM hosted_vllm's 600-second HTTP read default. At the
observed 19-22 tokens/second this can cancel a response before its unchanged
16,384-token cap. New attempts explicitly use 1,800 seconds per request, no
LiteLLM/SDK retries, and at most two mini-swe-agent attempts. Token/step budgets
and scoring are unchanged. Old Python processes retain their original timeout;
source deployment alone cannot change an in-flight client.

`metadata/request-timeout-migration.json` records which attempts were still active
at deployment and which completed outcomes must remain immutable. The single
`lasr-swebench-timeout-recovery.service` waits for the active job to finish normally,
checks that its owned GPUs are gone and submits one resume through the existing
coordinator. It cannot rent directly, reset the $120 cumulative cap or restart
after a recorded halt. The coordinator sizes this final fleet to at most six pods
and four workers per pod; eight remaining tasks need at most two pods. No chat
monitor is enabled. Cancel a queued continuation explicitly with
`systemctl stop lasr-swebench-timeout-recovery.service`.

## Resumption, grading and backup

For an interrupted run, inspect the saved reason/logs, repair the infrastructure,
then change only `LITE_ACTION=resume` in `/srv/lasr/lite-launch.env` and start the same
unit. Keep the recorded budget. Resume preserves the same HF repository and all
completed model attempts; it refuses protocol/code drift. It fences old owned pods,
agent processes and labelled task containers before reclaiming unfinished work.
An ambiguous allocation reserves its full requested lifetime in the cost ledger.

Each task has a unique immutable attempt directory. The agent atomically saves its
trajectory after every step; completed patches, JSON trajectories, readable Markdown,
exit status and process metadata survive a coordinator crash. A crash after the final
result write is recovered without regenerating that task. An interrupted in-flight
task restarts from the beginning: **this is not mid-trajectory sandbox restoration**.
There are at most two infrastructure attempts per task and a circuit breaker
at six new infrastructure failures per invocation. An explicit resume resets that
breaker after fencing old work; it never resets per-task attempts or dollars.
Submitted, step-limited and context-limited outcomes
are never retried to improve their score.

Snapshots upload every two minutes. A verified initial HF upload is required before
any rental. After launch, backup lag is reported and retried without stopping inference. The same repository gets
final official reports and results. Credentials, SSH/TLS private keys and Docker login
are outside the artifact tree. Uploads copy a consistent state snapshot, then verify
the uploaded state by reading it from an immutable HF commit.

These commands never rent a GPU:

```bash
# Regrade saved candidates; successful per-task official reports are reused.
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite grade
# Retry publishing already durable results.
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite publish
```

Grading errors do not trigger model rerolls. Invalid candidate patches that fail to
apply count as unresolved; harness/host failures remain ungraded. Empty valid
submissions count as unresolved. To stop inference, use
`systemctl stop lasr-swebench-lite.service`; inspect cleanup and run the independent
`guard` command if necessary. Never sweep the shared account by username alone.

## Parallelism, cost and runtime

The corrected H200 pilot finished and graded 18 cases: seven resolved. Those cases
remain in the final 300. On 2026-09-23 the user authorized a fixed target of **ten
independent H100 NVL replicas with four workers each** for the remaining 282 tasks.
This candidate recipe is experimental until all 300 are graded. It is not claimed
to be optimal. The CPU host is capped at 40 active agent containers: 160 GiB summed
memory limits, with at least 30 GiB reserved on this 198 GiB host.
Every replica serves the same BF16 LoRA. Individual trajectories remain sequential.

Ready replicas immediately consume the shared queue. Missing slots retry every
30 seconds, backing off to 120 seconds; one allocation failure never halts healthy
replicas. Only after both ten minutes and five failures may a slot try H200 or RTX
PRO 6000 Blackwell Server Edition, while subsequent attempts also retry H100 NVL.
No fallback changes model precision, context, token limits or Docker resource caps.
Provider creates are serialized briefly; SSH, downloads and serving proceed in
parallel. Startup checks real CUDA/BF16 execution and at least 85 GiB device RAM;
run_eval owns vLLM health and adapter setup. Per-replica KV/cache/preemption metrics
are saved every 30 seconds. Missing slots stop requesting pods with less than
15 minutes remaining; ready GPUs keep working until their cleanup deadline.

Explicit GraphQL rejections are reconciled after a grace period and two provider
inventory sweeps. Any late matching pod is terminated. Reserved cost retains the
possible elapsed charge through reconciliation; transport timeouts retain their
full provider-TTL reservation. The independent reaper also cleans late rejected
creates while other replicas work. Neither reconciliation nor an explicit resume
resets cumulative spending or finished tasks.

The initial three-worker pilot demonstrated working prefix caching (94.9% cumulative
hit rate), zero cache preemptions and low active KV occupancy, but one uncapped
response continued for many minutes. Its GPU was terminated before retuning. The
initial trajectories remain saved. A later shell audit discovered that upstream's
`bash -c` did not activate the images' `testbed` Conda environment. All attempts from
before that correction, including the five initially retained compatible submissions,
were superseded together and archived under `metadata/uncorrected-shell/`.
They are calibration diagnostics, excluded from the final score. All 300 scored
tasks run afresh with the same corrected environment. Historical rental costs remain
in the cumulative ledger; no budget is reset.

The protocol now caps each response at 16,384 completion tokens and each task at
65,536 completion tokens (reasoning included), in addition to 250 agent steps.
A response ending with `finish_reason=length` or exhaustion of the task token budget
terminates that task as `LimitsExceeded`, with an empty submission and an unresolved
score. It is never rerun as infrastructure failure. These are explicit bounded
evaluation settings; retain them identically in future control/DA comparisons.

Calibration's reservation allowance is now $20, within the unchanged $100 campaign
GPU cap. It includes prior rentals and a conservative $5.50 reservation for an
ambiguous failed allocation; observed task generation is not a billing receipt.

H100 NVL secure-cloud quote checked 2026-09-23: **$3.19/hour**; H200 $4.59 and RTX
PRO 6000 Server $2.09. Each rental reserves its fresh quote plus 5%, bounded by the
absolute $5.50/hour ceiling, and verifies the allocated rate before boot continues.
Disk charges and quote variation are covered provisionally by that margin, not
measured by a billing integration. Resource reservations use the ceiling rate and
provider TTL; completed verified teardown releases unused reserved time. Every
resume retains historical cost. Each invocation's two-hour inference deadline includes
boot/calibration; hitting it produces an incomplete run. A later explicit resume
retains the original cumulative dollar cap.

Illustrative warm-CPU estimates, **not measured performance of this model**:

| Per-GPU throughput | Generation on 8 GPUs | Generation GPU cost |
|---|---:|---:|
| 50 tasks/hour | 45 min | $27.54 |
| 25 tasks/hour | 90 min | $55.08 |
| 10 tasks/hour | 225 min | $137.70 |

Eight 15–30 minute cold boots add $9.18–18.36. Assuming another 20–60 minutes for
12-worker CPU grading gives a provisional **1.5–3 hours and $40–90 per subsequent
model** in the 25–50 tasks/hour scenarios, including active CPU time and upload.
The first calibration adds time within the allowance described above. These assumptions
require measurement; the slow scenario should fail the pilot's budget projection
before fleet expansion. The pipeline cap bounds spending/execution, not completion.

Keeping the CPU VM running for the full 24 hours costs about $12.93 whether or not
it is busy. Retained disk after stopping costs about $3.33/day. Those idle retention
costs, Docker's subscription and initial image-download setup are separate from the
marginal run figures above. There is **no LLM judge bill**: grading executes tests.

## What is reused, and what is custom

The agent is pinned **mini-SWE-agent 2.2.1**, with its original prompts,
temperature, tools and 250-step limit, plus the declared terminal token budgets.
The official **SWE-bench 4.1.0** harness
applies and tests predictions. The local-model dollar limit is inert and is reported
as such. `src/eval/run_eval.py` still owns vLLM serving, with 131,072 context,
concurrency eight, prefix caching, and the pinned thinking template/base/adapter.

Declared environment changes: no agent-container networking; cached images by
digest; `BASH_ENV=/root/.bashrc` activates each image's existing `testbed` environment
for every command; agent caps of two CPUs, 4 GiB RAM and 512 PIDs; local HTTPBin only during
requests grading. The two known requests no-fix passes remain in the denominator.
Record these settings identically for later DA comparisons.

Custom code covers provider lifecycle, the shared task/attempt ledger, bounded
subprocesses, and HF artifacts. Inspect already provides parallelism, log files,
retry/resume and a SWE-bench task; a portion of this ledger overlaps those features.
Its default solver, message limit and image setup differ from the existing pinned
mini-SWE-agent protocol. We retain that protocol for this campaign rather than
quietly changing the measurement. For a broader evaluation platform, reassess
Inspect and an explicit mini-SWE-agent integration instead of growing this driver
into a general framework. Inspect's newer mid-agent checkpointing currently requires
its development version and explicit agent support.

## CPU capacity and multiple hosts

Resource correction, 2026-09-23: Docker CPU quotas do not change Python's host CPU
count. Django's test runner therefore saw 61 CPUs inside a two-CPU container.
One observed task (`django__django-11905`) had four cgroup OOM process kills at its
4 GiB cap. In addition, mini-SWE-agent's client-side `docker exec` timeout left
commands running in the container; several old complete Django test runs overlapped.
The host itself retained over 181 GiB available RAM with 40 agents active.

`swebench_lite_task.py` now derives Django, BLAS/OpenMP and joblib/Loky thread
limits from the configured CPU quota. It installs a guarded `BASH_ENV` block that
runs each command under GNU timeout inside the container, using the existing
command deadline and a five-second kill grace for descendants. It preserves the
shell's exit status and working directory. This bounds common automatic parallelism;
it does not falsify `os.cpu_count()` or prevent explicitly requested process creation.
The memory cap stays 4 GiB. New attempts record effective policy and source hash in
`scaffold.json`, plus final cgroup peak memory/OOM/throttling evidence in
`resources.json` before container cleanup.

For the live campaign, preserve old source snapshots and the manifest before a
hot deployment. Record completed and running attempts at the boundary, apply the
same shell block to campaign-owned live containers, and terminate only old command
trees already beyond their recorded command timeout plus cleanup grace. Do not
restart agents, reroll outcomes, edit testbed source, or change model/token budgets.
The transition is published as `metadata/resource-policy-migration.json`.
Historical OOMs cannot be undone: this first run spans resource policies and must
be reported as such, rather than treated as a pristine uniformly configured control.
Later matched comparisons should use the corrected policy from the start.

Live observation on 2026-09-23, while the no-DA campaign continued: the Vast host
has 61 logical CPUs and 197.93 GiB RAM. A 45-second sample at 20–24 running agents
averaged 3.41% CPU (2.08 cores), peaked at 5.6%, retained at least 188.14 GiB
available memory, and showed 0–0.4% I/O wait. Disk traffic averaged 3.04 MiB/s
read and 2.00 MiB/s write. The evidence is published with the campaign as
`metadata/cpu-capacity-observation-20260923.json`.

Message timestamps from 33 completed tasks attributed 13.08 of 276.77 task-minutes
to shell/tool wall time (4.73%); 223.84 minutes were measured between tool results
and subsequent model responses. Initial inference, startup and other overhead are
not fully separated. These are wall times, not CPU times, and completed tasks are
biased toward shorter cases. Evidence: `metadata/cpu-task-timing-20260923.json`.
Neither this sample nor the small pilot establishes worst-case full-suite demand.

Forty agents is the configured conservative limit, not measured CPU saturation.
The 4 GiB/container and two CPU/container settings are ceilings, not reservations.
With a 30 GiB host reserve, the all-containers-at-their-memory-limit bound is
floor((197.93 - 30) / 4) = 41 agents; the configured fleet rounds down to 40.
Forty-eight agents would allow 192 GiB in containers, so it needs a deliberately
qualified memory-overcommit policy or more RAM. Do not silently bypass preflight.

For planning only, if tools occupy 5% of each task and each busy tool consumes
both allowed CPUs, average tool demand is `agents * 0.05 * 2`: 4.8 cores for 48,
6.4 for 64, and 8 for 80 agents, plus driver/system overhead. If tool duty rises
to 50%, those become 48, 64 and 80 cores. Memory peaks, synchronized test bursts,
Docker storage and grading must be measured before adopting a higher fixed limit.
Forty-eight to 64 is a plausible next qualification range on this host, not a
validated setting. Preserve today's 40-agent campaign while gathering evidence.

Keep one CPU host until sustained CPU use approaches 70–80%, runnable work queues
behind CPU, memory headroom repeatedly drops below roughly 30 GiB, or tool/test
latency materially increases with concurrency. These are investigation thresholds,
not newly implemented autoscaling rules. Grading currently uses a separate
12-worker phase after GPU teardown; generation measurements do not qualify higher
grading concurrency. Overlapping grading later requires its own resource budget.

At this run's allocated GPU rate of $29.20/hour and CPU rental of approximately
$0.539/hour (storage extra), one additional similar CPU host would pay for itself
if it reduced the same job's GPU-active runtime by about 1.8%, excluding setup.
Cold image downloads and preparation can overwhelm that saving. Prepare any added
CPU host before renting GPUs, as for the first host.

If multiple hosts become necessary, use one durable shared task queue rather
than three fixed 100-task shards. Workers claim the next task, heartbeat a lease,
and durably commit an attempt/result; recovery must fence stale attempts before
reassignment and preserve completed outcomes. Each host needs the dataset
metadata and its own image cache, not independent copies of campaign ownership.
The current JSON ledger and local `flock` are single-host mechanisms; a second
host requires a coordinator API with transactional storage (a single coordinator
can use SQLite; Postgres is another option), not copying the ledger between hosts
or using HF as a transactional queue. Keep HF for durable published artifacts.
Work stealing avoids shard imbalance; future scheduling can put historically slow
tasks first without changing their prompts, budgets or scoring.

## Subsequent models: fixed fleet, no calibration

Before selecting a larger fleet, read `docs/swebench_cpu_capacity.md`: 40 CPU
workers is the current conservative admission policy, not a measured saturation
point. The 4 GiB container cap is not a RAM reservation. Higher concurrency needs
one explicit qualification and an updated frozen recipe, not repeated per-LoRA
GPU calibration. The audit did not change the ongoing campaign.

HF checkpoint incident, 2026-09-23: the installed Hub client's streamed
`upload_folder` path split frequent snapshots into as many as eight commits each.
Commit-rate HTTP 429 retries exceeded the publisher's 180-second deadline, then
the unchanged 600-second backup breaker interrupted 40 active attempts at 183
completed tasks. The provider inventory confirmed all ten owned GPUs were removed.
Preserve these infrastructure-invalid attempts and their partial trajectories;
never repeat completed model outcomes. The conservative GPU ledger was $69.94.

Lite checkpoints now request `push_run_dir(..., atomic_commit=True)`, using one
`create_commit` transaction for the snapshot and card, followed by the existing
revision-pinned state readback. Other publishers retain their existing defaults.
This prevents partially committed ledgers from being mistaken for full backups
and reduces commit traffic; it does not bypass HF rate limits or guarantee HF availability. The later recovery removes HF freshness as an
inference breaker; persistent local checkpoints remain authoritative until upload succeeds. Verify an actual upload/readback before resuming paid inference.
Archive prior sources and manifest in `metadata/pre-atomic-upload`, and record
the deployment and recovery calculation in `metadata/atomic-upload-migration.json`.

Only a completed and graded 300-task campaign writes a validated recipe to
`/srv/lasr/recipes/qwen36-h100nvl-lite.json` and `metadata/frozen_recipe.json` on HF.
Before that, the latter is explicitly provisional. Preserve the recipe alongside
its exact code snapshot. The `prepare` action resolves and pins the new adapter;
it refuses code, base, serving or protocol drift and never rents a GPU:

```bash
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite prepare \
  --target ORG/ADAPTER --root /srv/lasr/runs/NEW-RUN \
  --write-config /srv/lasr/NEW-RUN.yaml
# Optional --target-revision pins a requested adapter revision before resolving it.
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite plan \
  --config /srv/lasr/NEW-RUN.yaml
```

After the CPU readiness check, set the root-only `/srv/lasr/lite-launch.env`:

```text
LITE_ACTION=run
LITE_CONFIG=/srv/lasr/NEW-RUN.yaml
LITE_BUDGET_USD=100
```

Start `lasr-swebench-lite.service`. The service and independent reaper read the same
config. The configured target replicas are requested at the outset without a pilot;
ready replicas work while missing slots retry using the fixed fallback policy.
Only one campaign may occupy this service at a time. Use `resume` with the same
config after inspecting an interruption; it never resets the cumulative budget.
An existing same-day canonical HF destination is refused rather than overwritten.

## Installation on another prepared CPU VM

Follow `docs/swebench_cpu_host.md` to provision/requalify Docker, cached images,
reference/no-fix tests, durable disk, credentials and CPU expiry first. Deploy this
branch at a recorded commit; install the three frozen CPU/agent/harness environments.
Never sync the main GPU training environment onto the CPU driver.
Copy only the necessary RunPod credential from the authorized repo `.env` into
root-only `/srv/lasr/credentials.env`, alongside HF credentials and USER_PREFIX.
The CPU-generated SSH private key stays on that CPU host. Then:

```bash
bash /srv/lasr/repo/scratch/swebench_lite_install.sh
scratch/swebench_cpu_env/.venv/bin/python -m unittest scratch.test_swebench_lite -v
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite qualify-shell
scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_lite plan
```

Installation starts only the independent minute-by-minute GPU reaper. Every rental
also arms a detached parent/deadline watchdog and requests provider-side
`terminateAfter` atomically at creation. The provider input schema has been checked
with a skipped mutation that cannot allocate. Actual timed termination and vLLM
serving remain to be exercised by the first paid pilot; they are not proven by the
CPU-only tests. Cleanup failures leave watchdogs active. CPU loss still relies on
the provider expiry, which is why it is mandatory for this fleet.

`qualify-shell` is CPU-only and verifies Python's prefix is the image's `testbed`
environment in all 300 cached images. It writes `results/agent-shell.json` in the
CPU readiness directory. Subsequent preflight checks reuse that qualification only
when image IDs, dataset revision and shell environment still match; they do not
repeat 300 container probes on each model. The proof is copied into every run.

## Validation record

CPU-only integration test: deterministic synthetic OpenAI endpoint -> real pinned
agent -> real Docker task -> submitted public reference patch -> official grader ->
HF upload/readback. Repeating its queue call sent no further model requests.
The resulting dataset is tagged **infrastructure-check**, never eval-run:
https://huggingface.co/datasets/dougalldeepmind/2026-09-23-swebench-lite-infrastructure

Offline tests cover concurrent exclusive claims, terminal failure preservation,
bounded infrastructure retry, crash recovery, backup outage, scheduled provisioning,
pin propagation, unrelated pods, boot/guard failures, and failed cleanup retaining
its watchdog. Existing RunPod/serving/provenance regressions are also checked.
No inference GPUs were rented during preparation; no no-DA score exists yet.

References:
- https://inspect.aisi.org.uk/eval-sets.html
- https://inspect.aisi.org.uk/checkpointing.html
- https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/swe_bench/README.md
- https://mini-swe-agent.com/latest/reference/run/swebench/
- https://graphql-spec.dev.runpod.io/
