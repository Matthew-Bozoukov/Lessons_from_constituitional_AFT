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

Only this task's worktree and positively recorded owned pod IDs are managed. Training
code revision is `68ab0cb7`; the worktree is `../teaching_claude_why_replication_da_supervision`
and branch `codex/da-supervision-rerun`. Per-attempt receipts are in
`output/da_supervision/2026-09-16/runs/`. Failed allocation requests are retained, and
provider inventory must be reconciled before retries to avoid duplicate billable pods.

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
