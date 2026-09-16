<!-- ABOUTME: Original low-stakes synthetic content reused with the September replay under the current recipe. -->
<!-- ABOUTME: Records exact identities, approval, training safeguards and completion requirements. -->

# Original low-stakes content with new replay

The user authorized this control on 2026-09-16 and approved a new **$50 training-only
ceiling**, including startup and backup. It trains one LoRA on one pod with exactly
two H200 GPUs. No evaluation is included in this authorization.

## Data and scientific comparison

- [Mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-lowstakes-original-7-mix/tree/41d80cc3e616d48739af6935d5705fd82a9a56f3):
  `41d80cc3e616d48739af6935d5705fd82a9a56f3`, 10,000 rows, SHA256
  `3617d2c03125e2e151e5972af1f42246477ad6c7b0da5d208b68985481826f90`.
- Original corpus: `LASR-Callum/2026-08-26-difficult-advice-low-stakes-716`
  @ `f268653539150af5a340164f994065f57cbef5ad`.
- Original trained mixture: `LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train`
  @ `3e4b638fe79326454ce7af2714c93c7579b37d06`.
- Every one of the 716 original conversations reproduces its historical rendered
  training text exactly. No response edits, repairs, new constitution injection,
  extra rows, or generation calls. Historical principle-conditioned content and
  limitations are deliberately retained.
- Exactly the same 9,284 replay dictionaries and absolute positions as refreshed
  low mixture `dougalldeepmind/2026-09-15-da-lowstakes-refresh-7-mix`
  @ `f0b0418ea9767ab00fe21e86fab8dfc26dc5320c`, from nosynth
  @ `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.
- Synthetic rows keep their original training order within the replacement slots.
  All 10,000 native token/mask checks passed without truncation. All 10 published
  payload files were hash verified. Provenance and the census travel with the data.
- Builder: `configs/data/mixture/da-lowstakes-original.yaml` consumed by
  `scratch/nonmoral/reuse_original.py`; source recorded at `c400db44`.

This controls synthetic-content changes against refreshed low stakes while retaining
the current replay and training implementation. It does not by itself isolate the
historical replay effect or reproduce historical training bit for bit.

## Training and ownership

`configs/train/sft.yaml`, `configs/models/qwen36.yaml`, `uv.lock` and `src/train`
have no diff from refreshed low-stakes training source `fd63ca8b`. Seed 0, one
epoch, global batch 16 across two ranks, BF16 LoRA rank 64/alpha 128/dropout 0.05,
cosine LR 1e-4, warmup 0.05, maximum 8192 tokens, native assistant masks with both
reasoning and final answers supervised. Expected optimizer steps: 625.

Base: `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
`allow_default_supervise=true` intentionally admits the standard all-supervision
column. `constitution=none` means no new specification is inserted; it does not
erase the historical corpus's constitution-conditioned generation provenance.

Owner: `scratch/nonmoral/train_pair.py`, source `d18adc6d`, launched at
2026-09-16 15:57:48 UTC. Campaign: `output/lowstakes_original_reuse_20260916`.
Plan is `train_plan.json`; live state, logs and watchdog are in `train_attempt1/`.
Pod: `gal6oht0t1cyx6`, **nika-low-stakes-original-train**, 2 H200 GPUs, quoted
$9.18/hour plus storage (owner accounting conservatively uses $9.28/hour).

The independent watchdog was registered immediately after provision. Lifetime
17,280 seconds is bounded by a conservative $10/hour allowance plus $2 reserve;
45 minutes are reserved for recovery. It also reacts to owner death: do not kill
the owner as a handoff or rename the pod while running.

Safeguards: startup download-speed gate, CUDA 13 requirement, actual CUDA operation
on both GPUs, per-30-second health polling, finite loss/gradient checks, checkpoint
and log-stall detection. Checkpoints are copied and hashed while training continues.
At completion, the final adapter and metadata are backed up first and verified
against the Hub, followed by the full output archive before ordinary teardown.
An independent hard watchdog still enforces the spending limit on recovery failure.

Local work is a low-priority driver plus SSH/file transfers; model loading,
tokenization and training run remotely. Existing account pods and local Docker
jobs are outside this task's ownership. The hourly task heartbeat follows only
this campaign and the earlier blocked-judge disposition; it stays quiet while
healthy. A per-process sleep inhibitor is scoped to this campaign.

## Completion gate

Require 625 steps and one epoch, finite statistics, world size two, exact base/data
revisions, all 10,000 examples supervised as intended, verified public adapter
payloads and local backups. Verify this owned pod is absent and record the provider
balance, then create this campaign's `keep_awake.stop`. Report the model revision,
training metrics and cumulative spend, append the experiment log, commit and push.

Provisioning is not evidence of optimizer progress or successful training. The
live state is authoritative until the final completion receipt is recorded here.

Focused checks: 53 training-owner, real archive/corruption, publication-hash and
naming tests passed before launch. The backup additions are opt-in for this new
owner; previously running processes retain their loaded code.

## Live completion-verifier correction

After launch, a metadata-schema check identified that the new publication verifier
read `organism` from `run_meta.json`; the native trainer places it in the adapter's
`training_meta.json`. Source `b7231102` corrects this and tests the real schema.
This affects the post-training publication check, not training, masks or data.

The running owner's imported verifier cannot acquire the source change. To avoid
interrupting training or its owner-death watchdog, a narrowly scoped completion
preserver (`scratch/nonmoral/complete_original_lowstakes.py`) waits until all 625
steps complete and the owner has verified its adapter backup. It then verifies
publication with the corrected reader and preserves the complete output archive
to `completion_preserver/`, using a distinct remote archive. It terminates only
this owned pod after both gates pass and verifies absence before retiring the old
owner. It leaves the original owner records intact and writes the authoritative
`completion.json`; no training restart, metadata rewriting or spending-cap increase.
Monitor its `status.json` and stderr as well as the original owner.
