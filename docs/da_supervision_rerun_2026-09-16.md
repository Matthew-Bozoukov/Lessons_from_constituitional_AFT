<!-- ABOUTME: Frozen design and operating record for the three-arm September DA supervision rerun. -->
<!-- ABOUTME: Campaign artifacts are pinned on Hugging Face; local receipts track owned RunPod resources. -->

# DA supervision rerun, 2026-09-16

## Design and inputs

Use all 752 distinct DA scenarios from `dougalldeepmind/2026-09-14-da-synth`
revision `013886238fca238c4d54ace96530f444bb2b2f02`. The earlier 702 figure was a
trait-balanced historical subsample, not this corpus size. No DA scenario is discarded.

Select 9,284 replay rows once from `dougalldeepmind/2026-09-08-nosynth-mix`
revision `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`, using proportional source quotas,
largest-remainder allocation and seed 0. Preserve selected messages, answers and real
reasoning traces verbatim. Reuse identical replay rows in the same order across all
three arms. Every arm contains 10,036 examples. Replay always receives the normal
assistant-completion loss; only DA supervision differs.

| Arm | DA input to the causal forward pass | DA loss targets |
|---|---|---|
| CoT | Prompt and real trace through the final `</think>`; omit answer and turn end | Real trace and closing tag; mask prompt and forced opening prefill |
| Answer | Prompt, real trace, separator, answer and turn end | Separator, answer and turn end; mask the complete trace and tags |
| Empty CoT | Prompt, empty `<think>\n\n</think>\n\n` marker, answer and turn end | Answer and turn end; mask the whole empty marker and prompt |

Loss masking is labels=-100, not an attention mask or a gradient detach. In the answer
arm, the answer still conditions on the real trace. The normal generation-boundary
rule masks the complete empty marker. It yields one fewer supervised separator token
per DA row than the real-trace answer arm; the observed difference is exactly 752.
Loss aggregation is sequence-mean-token-mean; raw loss values across arms are not
comparable quality scores. Inference thinking mode remains on for all three adapters.

## Published training mixtures

| Arm | Dataset | Revision | Forward tokens | Supervised tokens |
|---|---|---|---:|---:|
| CoT | [da-7-cot-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-cot-mix) | `926e1014d24614a6fcd97762a644ddbeb7320288` | 8,038,964 | 4,961,350 |
| Answer | [da-7-answer-only-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-answer-only-mix) | `e8104b084b6e47e99e2ace48df352b4a43a9604b` | 8,475,078 | 4,948,211 |
| Empty | [da-7-empty-cot-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-7-empty-cot-mix) | `c3a3b18612eb89da64b1541dd708bf8eb598f37b` | 8,027,329 | 4,947,459 |

The 7 in these names is the naming convention's rounded percentage; the actual DA
share is 7.493025%. All 30,108 rendered rows passed independent tokenizer/decode mask
checks, with no truncation and maximum length 8,191. Dataset-library loading was
checked against the raw JSON rendering for every row. Published bytes were downloaded
and SHA256-checked against the local files. Selection indices and hashes are published
in the dataset manifests.

## Training and operations

Base: `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Seed 0; BF16 unquantized LoRA rank64/alpha128/dropout0.05; one epoch; global batch16;
AdamW torch with betas0.9/0.999, eps1e-8, weight decay0.01 and gradient clipping1.0;
LR1e-4, cosine schedule, warmup0.05; maximum sequence8192; token budget8000;
SDPA and non-reentrant gradient checkpointing. Dynamic batching permits a singleton
row longer than its grouping budget and never packs sequences.

Three separate `nika-` RunPod pods each receive TWO H200 GPUs and use torchrun with
two ranks, preserving global batch16. The initial single-GPU allocation was an agent
cost assumption, corrected after the user's clarification. All three initial pods were
frozen during setup/smoke, their logs and empty checkpoint inventories preserved, and
termination verified before the replacements. No full training run had begun.

Only this task's worktree and positively recorded owned pod IDs are managed. CoT code
revision is `68ab0cb7`; the other two use `23a13288`, with identical trainer/model/config
files (the changes concern placement, diagnostics and documentation). The worktree is `../teaching_claude_why_replication_da_supervision`
and branch `codex/da-supervision-rerun`. Per-attempt receipts are in
`output/da_supervision/2026-09-16/runs/`. Failed allocation requests are retained, and
provider inventory must be reconciled before retries to avoid duplicate billable pods.

At14:44UTC all three Secure Cloud replacements are allocated: `rsq31hkrf0elpn`
(`nika-da-cot`, owner `cot-dual`), `266cqfrfcyxxgs` (`nika-da-empty-cot`, owner
`empty-dual-attempt3`), and `xiw8uh2etoa1hy` (`nika-da-answer-only`, owner
`answer-dual-attempt7`). CoT passed its two-rank smoke and reached full optimizer steps
with finite loss/gradients; the other two are starting up. Both Secure and Community
allocation failures reported no available matching instances. No Community pod was
created. Early CoT progress suggests2–3hours training, subject to settling throughput;
publication and preservation add time and are tracked separately.

At allocation, two H200s cost $9.18/hour per pod, excluding storage. Each owner has a
$9.50/hour ceiling and a five-hour lifetime, with the final45minutes reserved for
preservation. Maximum three-pod envelope is $142.50 plus the short superseded attempts;
this is a cap, not an expected bill. Independent owner-death and lifetime watchdogs
start immediately on allocation. A CUDA test touches every device; longest-row/DA
smoke runs precede full training. Owners monitor process health, GPU usage, nonfinite
metrics, progress and exit receipts; no progress for20minutes fails the run. The first
completed checkpoint is copied independently while training continues. Final artifacts
must have verified HF metadata or a verified local backup before ordinary teardown.
The hard deadline remains authoritative. Watchdog enforcement requires the local
machine to remain awake and connected.

The ten-minute task heartbeat reports meaningful changes and completion; it does not
provision evaluation GPUs. Completion means verified adapter publication and verified
termination of all campaign pods, not merely a successful startup. Dataset audits and
launch/batching tests have passed; training results and evaluation results are pending.

At15:10UTC the answer pod had finished its22minute dependency bootstrap and reached
its two-rank GPU smoke. Empty's dependencies were still downloading when the25minute
bootstrap-stage limit approached. Its owner was handed off to `empty-dual-recovered`
using `scratch/da_supervision/resume_boot.py`, retaining the same pod and in-progress
installation. The original independent deadline watchdog remained active; the new
owner-death watchdog was verified registered. The replacement plan subtracts elapsed
time: its calculated deadline is8.9seconds earlier than the original19:43UTC deadline,
not an extension. Old status `owner_handoff` identifies the replacement. No extra pod
was rented. CoT continued normally around75/628 steps, mean throughput suggesting
about3hours remaining. This supersedes the optimistic early2–3hour total estimate.

At17:54UTC both answer arms had completed628steps/one epoch and published adapters.
Answer runtime was8,796s, loss0.8093; empty runtime8,665s, loss0.8250. Loss differences
are not quality comparisons. Each full14.13GB output archive was independently verified
on HF (SHA256 and size) before manual teardown, replacing the slow redundant laptop
copy. Answer archive revision: `52adc308c378457a94b2eb900802dc424ab5540f`; empty archive
revision: `b60f9e3656f92b99d9aedf1f9afbccefa3cc6f48`. These commits add backups without
changing adapter weights. Per-arm `durable_completion.json` records verified teardown
and timestamps. CoT remained healthy and training; no evaluation has run.

At18:21UTC all three trainings, publications, full-output preservation and pod
terminations are verified complete. Each run completed628optimizer steps/one epoch
on10,036examples using two H200s. CoT finished training at18:14:42UTC with runtime
12,731.0635s and training loss0.82796343. Its archive was independently checked through
HF file metadata (14,127,779,840bytes and SHA256), then its pod was terminated and
absence verified at18:21:39UTC. The three model repositories and immutable revisions
including complete training archives are:

| Arm | Model repository | Revision including archive | Training runtime | Train loss |
| --- | --- | --- | --- | --- |
| CoT-only | [2026-09-16-qwen36-0-da-7-cot](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-cot) | `6cd676bca92e97b745067daab61f91ca2bb0126f` | 3h32m11s | 0.82796 |
| Answer-only | [2026-09-16-qwen36-0-da-7-answer-only](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-answer-only) | `52adc308c378457a94b2eb900802dc424ab5540f` | 2h26m36s | 0.8093 |
| Empty-CoT | [2026-09-16-qwen36-0-da-7-empty-cot](https://huggingface.co/dougalldeepmind/2026-09-16-qwen36-0-da-7-empty-cot) | `b60f9e3656f92b99d9aedf1f9afbccefa3cc6f48` | 2h24m25s | 0.8250 |

CoT's adapter-only revision is `5788d320f91d916cd0a933793b5f1fd3752d059c`;
adding its archive did not change weights. Every repository contains
`training_backup/da-supervision-backup.tar` and `training_backup/manifest.json`.
All three first-checkpoint local backups were also checksum-verified. Published
metadata was audited for dataset/base pins, thinking=true, seed, one epoch, batch16,
optimizer and LoRA settings. Losses supervise different targets and do not establish
relative quality. No evaluation has run.

Final cost estimate is **$95.03**, including setup and preservation: CoT$34.82,
answer$29.41, empty$29.50, plus$1.29 provider-reported superseded attempts. The three
main estimates use original allocation-to-verified-termination tenure times$9.28/hour
($9.18 quoted GPU rate plus$0.10 storage allowance); empty includes the original
bootstrap before its owner handoff. The billing API reported$75.08 at the final query
but had not yet included the last billing periods, so$95.03 is an estimate, not a
settled invoice. Receipts and arithmetic are in `completion_summary.json`,
`billing_final_query.json` and each arm's `durable_completion.json` under the campaign
output directory. All six campaign pod IDs are absent from the live inventory;
other tasks' resources were left alone. The ten-minute heartbeat is paused.

## ODCV follow-up authorized September16

The user requested parallel ODCV on all three adapters, one pass,40scenarios,
temperature0.7. The declared interpretation is all40scenario names in both mandated
and incentivized variants:80cells per adapter,240total. The same selected cells and
protocol apply to all arms. Configuration: `scratch/da_supervision/odcv.yaml`, based on
the project's lite protocol (Gemini3Flash for misalignment and separate task progress),
28,000-token context and artifact-inferred thinking mode. No constitution is injected
at evaluation. One pass is one rollout per cell, not training-seed replication.

Three separate1xH10080GB Secure pods at$3.49/hour are allocated with the required
prefix: `nika-da-cot-odcv` (`f05zyy03ba3ndz`), `nika-da-answer-only-odcv`
(`mg46cd376sn6ba`), and `nika-da-empty-cot-odcv` (`sq8hhs4lhgxfx7`). Initial provider500
errors were reconciled against inventory before retrying; those attempts created no
pods. Pod names, frozen adapter/base revisions, unique ports8111–8113 and four-hour
lifetime limits are in `scratch/da_supervision/odcv_plan.yaml`. Approximate GPU ceiling
is$41.88 plus storage and judging; expected total initially$25–35, to be revised using
actual throughput. Setup is included in the lifetime and bounded at45minutes.

The standard `runpod.up` path owns provisioning and its independent deadline guard.
Each launch immediately adds an owner-death guard; `evals --terminate-pod` adds its own
guard and verified teardown. Drivers and scenario containers run on local Docker;
only vLLM runs remotely. Four cells per arm use24networks across the three runs,
within this daemon's default31-network pool. Global network pruning is disabled in
the scratch wrapper. No Docker restart, global cleanup or other task's resource
mutation is needed. A temporary bounded execution-state helper keeps Windows awake
while the owners are alive, without changing persistent power settings.

Each arm first runs one counted scenario end to end, requires a real assistant
transcript, then resumes the same pass with its remaining cells; the pilot is cached,
not repeated. The standard timeout receipts, missing-cell audit and transcript
reconstruction are retained. After generation, the complete combined local transcript
tree permits early GPU teardown before judging. The normal eval entrypoint retains
serving, metadata, publication and naming ownership. Outputs/statuses are in
`output/da_supervision/odcv_2026-09-16/`; a ten-minute heartbeat monitors meaningful
changes, verifies final publications and teardown, then pauses.

Preflight: live Docker network creation passed; all168shell scripts had LF bytes and
passed `bash -n` in a real Linux container. Adapter/base revisions and thinking mode
were verified.24ODCV archive/recovery/budget tests passed, plus a mocked check that
the counted pilot resumes its original pass and releases its GPU before scoring.
At allocation the pods were booting; no ODCV scores or completed cells are claimed.
