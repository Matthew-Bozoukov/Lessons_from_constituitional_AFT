<!-- ABOUTME: Reproducible reduction of historical nonmoral data to 15 percent supervised tokens. -->
<!-- ABOUTME: Records source pins, selection, publication, and planned token-mean SFT and ODCV. -->
# Nonmoral deliberation at 15% supervised tokens

User authorized a reduced mixture, single-H200 seed-0 training and three sequential
ODCV passes at temperature 0.7 and server seed 0. No new synthetic generation or
rewriting. Work is isolated on `codex/nonmoral-token15`, based on main `27623065`.
The user explicitly approved the **$60 combined ceiling** on 2026-09-25:
$30 training, $25 evaluation GPU/storage, and $5 judges. Independent watchdogs
and verified backup-before-teardown apply to each rental.

## Dataset

Parent: `dougalldeepmind/2026-09-15-nonmoral-original-7-mix`
at `b35ead8eaf4d59091e0ef1d08b187c7822f05632`.
Parent contains 684 historical craft-deliberation rows and 9,284 September-8 nosynth
replay rows. The original nonmoral corpus was not generated with the new constitution;
this subset retains its historical provenance rather than relabeling it.

Rank synthetic rows by SHA256 of seed 0 plus canonical complete row, and take the
prefix whose supervised-token share is closest to 15%. Preserve the original relative
order and complete payload of every retained row. This selection does not use quality
judgments or evaluation results. All replay rows remain unchanged.

Use the frozen Qwen tokenizer `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
and current native generation-boundary masks. Recount all rows without truncation and
exercise the trainer's actual Arrow loader before publication.

- **Published mixture:** `dougalldeepmind/2026-09-25-nonmoral-original-15-mix`
  at `f056dade96901bc6c7480a4f4149552592fef379`.
- **9,907 rows:** 623 nonmoral + 9,284 replay; 61 nonmoral rows removed.
- **5,309,764 supervised tokens:** 796,915 nonmoral + 4,512,849 replay.
- **Nonmoral share: 15.0084824862%.** Zero rows rewritten or truncated.
- Mixture SHA256: `04778a87c01b6318096a885759615ae1010896a1f2a1a3e2c08dd1263afd00cb`.
- All published files verified by pinned download and SHA256; selected parent indices,
  full parent token census, resolved config and validation are included on the Hub.

Reproduce with `uv run python -m scratch.dataset_refresh.reduce_mixture --config
scratch/dataset_refresh/nonmoral_token15.yaml --publish` (a new output directory and
unused artifact name are required; existing publications are never overwritten).

## Training and evaluation plan

Current standard SFT: BF16 Qwen3.6-27B, rank 64, alpha 128, dropout .05, seed 0,
one epoch, global batch 16, LR 1e-4/cosine/5% warmup, 8,192 sequence ceiling,
8,000-token dynamic batching budget, boundary-aware packing, **token_mean** loss.
Expected 620 optimizer steps on one H200. Historical nonmoral adapter used equal
weight per row; the new experiment changes both token share and loss aggregation.

Use existing `scratch/nonmoral/train_pair.py` with one arm, independent bounded
watchdogs, CUDA and native mask gates, finite-loss/gradient monitoring, verified adapter
publication and full direct-to-Hub checkpoint archive before owned-pod teardown.

ODCV: one H100, local Docker with 24 concurrent scenarios, 40 mandated plus 40
incentivized scenarios, three sequential passes (240 total), one persistent server,
startup seed 0, temperature .7, no per-request seed, 28k context. Gemini 3 Flash
misalignment and progress judges. First counted cell is the health pilot; every pass
persists its rollouts before the next, with server continuity checked at boundaries.
No global Docker pruning; unrelated pods and local jobs are outside scope.

Docker preflight passed: 20 CPUs, 18.9 GB memory, capacity for 1,024 networks, all 168
benchmark shell files LF, exactly 40 scenarios per variant. Thirty owner, backup,
evaluation-plan and server-startup regression tests passed.

## Training completed

The first pod, `b9q557l7dakbiu`, never exposed SSH within the provisioner's seven-minute
timeout. No training ran. Teardown and absence were verified; its estimated GPU/storage
charge was **$0.5645673**. All failure records remain under the campaign output.
The retry allocation was reduced to $29, inside the unchanged $60 combined ceiling.

The replacement single H200, `ew608on86ni0gk` (`nika-nonmoral-token15-train`),
completed CUDA checks, the native mask gate and all **620 optimizer steps**. Training
used pushed commit `9f301c38`, the pinned mixture and base revisions above, and the
standard token-mean/packing recipe. Runtime **3,543.40 seconds (59.06 minutes)**,
mean training loss **0.69563347**; all logged losses and gradients finite.

- Adapter: `dougalldeepmind/2026-09-25-qwen36-0-nonmoral-original-15`.
- Verified initial publication: `aa42a7c0a0388513ac8d00bfe10f0640513561ba`.
- Final revision with complete training backup: `51f6c54ffcb3f4e1ac4a08497c6bcaf95b69baa8`.
- Backup size: 8,986,910,720 bytes; SHA256
  `64f50d13c9d0c0f063a6e02d3c14b08ec84c121a4bd86985da01ca834aab11be`.
- Successful rental estimated GPU/storage charge: **$5.6705104**.
- Training plus failed startup: **$6.2350777**, elapsed-rate estimate, not an invoice.

Both owned training pods were independently verified absent. ODCV used the final
model revision above.

## Evaluation completed

Published [ODCV results](https://huggingface.co/datasets/dougalldeepmind/2026-09-25-odcv-qwen36-0-nonmoral-original-15):
initial revision `06cf2a0618c73f60c482cfc5d5d005598a5e5857`; final revision with
campaign accounting, server continuity, judge ledger and remote logs
`68a7e9024ca41c320dc43439d21de6476f2c065a`. All **747 published files** were verified
against local sizes and Hub blob/LFS hashes.

All **240/240** rollouts were clean, with zero retries, reconstructed transcripts,
dropped passes or missing cells. All **480** MR/progress judgments settled, with no
outstanding reservations. Ten rollouts hit the normal token limit and remain in the
aggregate; none reached cycle 50. All six pass boundaries recorded the same server
PID 1538. The local Docker concurrency was 24 throughout.

| Pass | Misaligned / total | MR |
|---|---:|---:|
| 1 | 15/80 | 18.75% |
| 2 | 16/80 | 20.00% |
| 3 | 16/80 | 20.00% |
| Combined | 47/240 | **19.58%** |

Mandated MR was 25/120 (20.83%); incentivized MR was 22/120 (18.33%).
Submission rate was 95.8%; task-progress mean was 4.93/5 and threshold rate 99.2%.
These are distinct measures: a high progress score does not imply formal submission.
The native scenario-sampling MR interval is 10.9-32.7%; it is not an estimate of
training-seed variability. This one-checkpoint result changes both the historical
nonmoral subset and row-mean versus token-mean loss, so it cannot isolate either effect.

Pass wall times including accounting waits were 25.93, 37.10 and 22.94 minutes.
The slowest single rollout took 32.25 minutes while actively generating. Server
continuity and per-pass saving were maintained throughout, without a restart.

The H100 `dki9bbgcx54x5k` (`nika-nonmoral-token15-odcv`) cost $3.49/hour;
estimated GPU/storage total was **$6.4113574**. Judges cost **$2.500565**.
Boot and vLLM logs were backed up and hash-verified before teardown; the archive is
published under `metadata/operations/remote_logs.tar`. All three campaign-owned
pods were independently verified absent; unrelated resources were untouched.
The local keep-awake helper stopped and released its sleep inhibition.

**Final estimated campaign cost: $15.14700 / $60 approved.** This includes the
$0.56457 failed startup, $5.67051 successful training, $6.41136 evaluation GPU/storage
and $2.500565 judging. Dataset reduction used no paid model generation. GPU/storage
amounts are elapsed-rate estimates, not invoices. Native `rollout_cost_usd` fields
are shared-account OpenRouter deltas including unrelated jobs; campaign-specific
accounting is published in `metadata/campaign_summary.json`.

No further paid work or scheduler is active for this campaign.
