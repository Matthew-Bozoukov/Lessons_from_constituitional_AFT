<!-- ABOUTME: Current launch procedure, measured findings and remaining qualification gaps for SWE-bench Lite. -->
<!-- ABOUTME: Prepared infrastructure is separate from explicit authorization to start paid inference. -->

# SWE-bench Lite: one model, all 300 tasks

**Latest completion, September 24:** the DA-15 run is fully complete at 134/300
(44.67%), versus the matched no-DA control's 136/300 (45.33%). All 300 outcomes were
graded, all 4,867 rollout/result files were verified on HF, and owned GPU inventory
was empty. Evidence: `metadata/final-verification-20260924.json` at
`dougalldeepmind/2026-09-24-swebench-qwen36-0-da-15@1ae295a5e9063d84687d09e5868e1b164b894d59`.
The heartbeat was **deleted at the user's request**; create a fresh monitor for a newly
authorized run rather than assuming monitoring remains active. The CPU's September
24 12:37 UTC stop is unchanged. The operating instructions and linked history describe
the supported procedure and the incidents; they are not unfinished work.

## Current reusable launch procedure (2026-09-24)

This section supersedes the examples in [the historical record](swebench_lite_history.md). The current
template is `configs/eval/swebench_mini/lite.yaml`; the old scratch YAML remains the
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
authorized job; automatic crash recovery preserves its budget, attempts and actual
CPU expiry. Explicit stops remain stopped. A graceful SIGTERM is treated as a stop;
reboot recovery after such a stop needs explicit review rather than silently overriding it.

The supervisor allows at most four recovery cycles and three infrastructure attempts
per task. Completed model failures are never rerolled. There is **no six-hour job
cutoff or shared 150-minute batch cutoff**. Each late/replacement GPU gets its own
150-minute safety lease, bounded by the cumulative budget and actual CPU expiry.
Vacant slots refill while healthy peers work. New tasks fit their full 90-minute
allowance before that individual lease expires; a drained pod is replaced when work
remains. Reserve 30 minutes before CPU shutdown for finishing. Do not manually extend
live rentals or the CPU lifetime.

HF snapshots are atomic, checked initially before rentals, and retried asynchronously
from inference progress. Later backup failure does not cancel agents or prevent grading.
After GPU teardown, official grading retries up to three times; publication retries
for up to an hour within the actual CPU lifetime. Completion requires HF result readback and
hash verification of every rollout/result file, plus zero owned pods. Exhausted attempts,
budget, lifetime, resource guards, persistent grading/Hub errors, incompatible LoRAs or
ownership/key mismatches produce a saved actionable status and exit code 2 (systemd
failure, without repeatedly restarting a terminal condition). Publication preserves
`terminal_reason`. Unexpected process crashes still restart.

The completed run's chat heartbeat has been deleted at the user's request. A future
model run needs its own explicitly configured 15-minute progress/completion/failure
monitor: the host service does not send chat notifications. Verify that monitor's
saved ACTIVE state before leaving a run unattended, then delete it after verified
completion. Do not recreate a monitor merely for code cleanup or documentation work.

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

## Findings and limits to carry into the next run

| Completed run | Resolved | Interpretation |
| --- | --- | --- |
| Matched no-DA control | 136/300 (45.33%) | Complete, with documented historical operational changes. |
| DA, 15.09% supervised tokens / 6.84% rows | 134/300 (44.67%) | 22 DA-only solves, 24 control-only solves; a single pair does not establish degradation. |

The DA campaign's conservative GPU ledger closed at $76.42. The final 39-task
recovery took 88m52s including grading/publication and about $13.69 in elapsed GPU
charges. Provider billing lags; this is not a quote for every adapter or a measured
clean 20-H100 runtime. Preserve the two declared requests no-fix passes and the
local HTTPBin grading deviation when comparing scores.

Before another paid run:

1. Prepare or resume a receipted native-Docker CPU host with a newly authorized
   lifetime if the previous one expired. The current helper validates an existing
   receipt; it does not renew expiry timers or provision a replacement host. Follow
   [the CPU host guide](swebench_cpu_host.md), refresh SSH, and recheck images,
   fixture, disk, RAM, credentials and remaining lifetime before any GPU rental.
2. Deploy one reviewed source revision and its tested recipe. Source hashes are
   deliberately strict: do not overwrite a completed campaign manifest to approve
   newer code. Compatible operational updates qualify the reusable recipe only;
   preserve the old snapshots on HF. Pass an explicit cumulative budget.
3. Pin the requested LoRA revision, verify its base/rank/thinking mode, and use the
   standard entry point above. The fixed plan is up to 20 GPUs with four workers
   each, not a guarantee that RunPod has 20 H100s available. Keep delayed fallbacks,
   bounded recovery, local durability and independent cleanup enabled.
4. Configure and verify the progress monitor for this new campaign. Test the first
   real status read; a successful service launch is not proof of a notification path.
5. Require 300 final outcomes, official grading, HF readback/file checks and empty
   owned GPU inventory before saying complete. Persistent capacity/provider failures,
   exhausted spend/retries, insufficient CPU lifetime, or prolonged HF failure remain
   actionable failures rather than guarantees of unattended success.

No further GPU concurrency sweep or second CPU is required for a compatible LoRA.
A prebuilt serving image, reduced checkpoint re-upload work, and earlier cancellation
of SSH/bootstrap work are optional speed/cost improvements. Unneeded model-server
startups now cancel automatically under the same lock as the worker ready transition;
ready workers drain normally. The existing bounded SSH/bootstrap timeout still applies.
A future authorized run can qualify the complete revised workflow end to end; do not
rent a redundant model evaluation merely to claim an issue-free pipeline.

[Historical preparation and recovery evidence](swebench_lite_history.md) is retained
separately so old commands and caps are not mistaken for current launch instructions.
