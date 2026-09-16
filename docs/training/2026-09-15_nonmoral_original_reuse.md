<!-- ABOUTME: Track exact historical nonmoral reuse with new replay, single-H200 SFT, then H100 ODCV. -->
<!-- ABOUTME: Separate verified dataset identity, authorized launch, and later measured completion. -->

# Original nonmoral reuse with September nosynth

The user explicitly requested the original 684 nonmoral training conversations
plus the same 9284 new nosynth rows, followed by one-H200 training and one-H100
ODCV. They approved a new $60 combined ceiling. No synthetic conversation is
generated, rewritten, repaired, reviewed against a new constitution, or added.
The resulting training mixture has **9968 rows**, not 10000.

## Exact dataset

[Published mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-15-nonmoral-original-7-mix)
@ `b35ead8eaf4d59091e0ef1d08b187c7822f05632`, payload SHA256
`cd6819a26d348113187e28a29df30512a7f54d774ab5788a6d7ac101f0644b31`.
All ten uploaded artifact files matched local sizes and Hub file hashes.

- Historical training membership and rendered rows:
  `dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture`
  @ `6364505df02b0020b030bf379bd42285a14de6a5`.
- Original conversation fields:
  `dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation`
  @ `fed726d2db33bddb698ca349a6d76a4e7df7a7e9`.
- New replay source:
  `dougalldeepmind/2026-09-08-nosynth-mix`
  @ `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.
- Exact selected replay dictionaries and order are taken from the already
  verified refreshed mixture
  `dougalldeepmind/2026-09-15-nonmoral-advice-7-mix`
  @ `588783fb079d0bbe5c9eb568418fcff2aad4cecb`, excluding its 716 synthetic rows.

All 684 original rendered strings exactly match the historical renderer applied
to their original messages. The builder retains every original message field and
76 examples per craft tension. It also publishes the historical serialized rows
and source conversations as provenance files; only `mixture.jsonl` is the default
training split. No historical content defect was corrected.

The first 684 synthetic slots of the previous mixture receive the historical
examples in their historical mixture order; the final 32 synthetic slots are
omitted. Every replay dictionary and relative replay order remains identical,
although absolute positions shift after omissions. All replay rows were also
verified as an exact multiset subset of pinned nosynth. Native Qwen 8192-token
checks and nonempty assistant masks passed for all 9968 rows, and the actual
Arrow JSON dataset loader read all rows. Reasoning-family provenance is inherited
without new trace generation. Dataset construction cost no API generation calls.

Builder: `scratch/nonmoral/reuse_original.py`; current configuration:
`configs/data/mixture/nonmoral-original.yaml`. The build ran at `2b589d14` with
the config's former `nonmoral-original-7.yaml` stem; `ccf80a63` subsequently
corrected the config filename to the naming law, without changing data or config
values. The recorded original command remains accurate historical provenance.

## Training and subsequent evaluation

Training uses the current `configs/train/sft.yaml`: Qwen3.6-27B seed0, rank64,
alpha128, dropout0.05, BF16, one epoch, LR1e-4/cosine, warmup0.05, global batch16,
max8192, dynamic microbatching, no packing. Base revision:
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Expected optimizer steps: **623**.
One H200 replaces the two-H200 hardware allocation; global batch and recipe
remain unchanged. This uses current rendering/training code and is not a
bitwise reproduction of historical optimizer execution.

The existing training owner was generalized to an explicit `gpu_count: 1`,
direct Python launch, real one-device CUDA check, and a count-aware hard lifetime.
Fifteen focused owner/backup tests passed. Training source commit is `ccf80a63`.
Training was launched at 2026-09-15 21:04:42 UTC on owned pod
`4poydjc8psjtou`, **1x NVIDIA H200 at $4.59/hour**, with an additional $0.10/hour
storage reserve. Training/recovery allocation is $45; the independent watchdog
allows at most 30960 seconds including 2700 seconds for recovery, with $2
reserved outside that lifetime calculation. Initial shared account balance was
$204.370620; unrelated pods are not owned by this run.

After training and local/Hub adapter verification, the already authorized ODCV
stage will launch automatically: one H100, local Docker, thinking on, one pass,
40 scenarios in both mandated and incentivized variants (80 total), temperature
0.7, context28000, Gemini 3 Flash MR/progress judges pinned to google-ai-studio.
Use concurrency6 and port18112. Allocate at most $12 GPU/storage and $3 judging.
No extra behavioral samples or benchmark are authorized. Preserve observed
timeouts/limits and completed outcomes rather than retrying them.

Durable plans, publication receipt, owner state, budget and keep-awake files are
under `output/nonmoral_original_reuse`. The eval plan template deliberately has
no usable adapter revision until training verification supplies it. The recurring
monitor handles this transition, verification/publication, owned-pod teardown,
account check, final comparison, and shutdown of temporary sleep inhibition.
Initial launch is not a claim of completed training or evaluation.

At the user's request on September 16 local time, the redundant five-minute AI
polling was replaced with one hourly handoff/recovery check. The two obsolete
refreshed-training/evaluation automations were deleted. Existing local owners
continue their 30-second health monitoring and independent budget watchdogs;
the hourly check exits silently while a stage is healthy. After training output
recovery and pod closure, the next check launches the authorized evaluation, so
handoff may wait up to an hour without keeping the finished training pod rented.

## Training verified; ODCV launched September 16

Training completed all **623 steps**, epoch1, with finite logged loss/gradients.
Reported training runtime was14642.4125 seconds and aggregate training loss
0.8091549. The owner recovered a verified 8,986,910,720-byte archive containing
39 files, including final adapter, retained checkpoints and logs, before terminating
pod `4poydjc8psjtou`. Provider inventory independently confirmed that pod absent.
Training/recovery GPU-plus-storage estimate is **$21.49996**, not a provider invoice.

Published model:
`dougalldeepmind/2026-09-15-qwen36-0-nonmoral-original-7`
@ **`42232b52b52ed93245864548f37a3fe8179c7d75`**.
All nine published files other than `.gitattributes` match the local archive by
size and Git-blob/SHA256 hash. Adapter weights SHA256:
`8eb993a433e838464f9f9232b340d20b2e7cc47f9180353f31d2a3bb48cacf4d`.
Verification script: `scratch/nonmoral/verify_original_training.py`; receipt:
`output/nonmoral_original_reuse/training_verified.json`.

The frozen final eval plan pins that adapter. `load_plan`, `checked_spec` and
local Docker preflight passed before rental. The authorized one-H100 ODCV owner
started at **2026-09-16 02:09:25 UTC**, PID53280. Owned evaluation pod
**`6qn3q92y2evo3a`** was quoted **$3.49/hour**, with independent watchdog PID42380
and a11939-second hard lifetime. The allocation remains $12 GPU/storage plus $3
judges, within the $60 combined ceiling including completed training. Shared
RunPod balance immediately before rental was $171.68169. No evaluation result is
claimed by this launch entry; coverage, publication and teardown remain pending.

## Completed evaluation and closure

The one-pass evaluation completed and published
[all rollouts, scores and provenance](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-odcv-qwen36-0-nonmoral-original-7/tree/b06c757309213d1d0e1ef47942e36b2157031db5),
revision **`b06c757309213d1d0e1ef47942e36b2157031db5`**.
All **255 payload files** match local size and Git-blob/SHA256 hashes. All 80
transcripts are nonempty; all 80 MR and 80 progress judgments are finite. Both
variants contain the exact 40 intended scenarios. There were no missing cells,
dropped passes, outer timeouts, retries or reconstructed transcripts. The sole
pass is retained in full.

| Metric | Overall | Mandated | Incentivized |
|---|---:|---:|---:|
| Misaligned / rollouts | 18/80 | 9/40 | 9/40 |
| MR | **22.5%** | **22.5%** | **22.5%** |
| Submitted | 77/80 | 38/40 | 39/40 |

Overall scenario-based MR CI95 is [12.7%, 36.6%]. Mean progress is **4.9625/5**;
80/80 receive progress >=3. This does not imply 80 submissions: three trajectories
reached token/context limits and did not submit. Those original outcomes remain
scored, without retries:

- Mandated `Ai-Transport-Fleet-Safety-Vs-Efficiency`.
- Mandated `Healthcare-Billing-Optimization`.
- Incentivized `Academic-Research-Integrity-Finding`.

No 50-cycle endings were found. The progress pipeline newly capped one score for
non-submission; the other non-submissions already had scores below that cap.

### Comparisons between fixed checkpoints

| Reference | Reference MR | Reuse minus reference | Paired scenario CI95, pp |
|---|---:|---:|---:|
| Refreshed nonmoral, current replay, one pass | 31/80 = 38.75% | **−16.25pp** | **[−26.74, −5.76]** |
| Refreshed low stakes, one pass | 16/80 = 20.0% | +2.50pp | [−4.70, +9.70] |
| Original checkpoint, Sept 9 common-protocol re-evaluation, three passes | 33/240 = 13.75% | +8.75pp | [+1.92, +15.58] |
| Original checkpoint, Sept 4 older protocol, five passes | 73/400 = 18.25% | +4.25pp | [−4.65, +13.15] |

Differences average each variant's binary severity>=3 rate across its available
passes, then average the two variant differences within each base scenario.
Intervals use the 40 scenario differences and Student t with df39. They describe
fixed checkpoints, not training-seed uncertainty; unequal numbers of passes are
not treated as extra independent scenarios. The reference revisions are stored
in `output/nonmoral_original_reuse/final_verification.json`.

The result supports the original content over the redesign in this comparison.
It does not establish which redesign feature caused the difference. The exact
original condition has 684 synthetic rows versus the refresh's 716, one versus
two training GPUs, and freshly executed training even though both use seed0 and
the same headline recipe/global batch. Historical checkpoints also differ in
replay and training execution; the older Sept 4 evaluation used context16384
rather than28000. This is not a clean estimate of the effect of replay alone,
nor evidence that the old historical MR has been recovered. Additional passes
or training seeds were not launched.

### Spending and cleanup

- Training/recovery GPU and storage estimate: **$21.49996**.
- Evaluation GPU and storage estimate: **$3.37117**.
- All 160 judge requests settled: **$0.785384**.
- Combined estimate: **$25.65651 / $60**, including startup and recovery.

These are elapsed-rate/API-ledger estimates, not invoices. Both owned pods are
terminated and independently absent from provider inventory. Shared balance at
closure is $168.38197, with $0.001/hour remaining account spend unrelated to these
terminated pods. Remote eval logs are verified locally. Temporary sleep
inhibition was released via `keep_awake.stop`; the completion monitor is paused.

Verification: `scratch/nonmoral/verify_original_eval.py`, using the existing
publication verifier generalized to explicit repository/owner-root arguments.
Receipts, full comparison arithmetic and account snapshots remain under
`output/nonmoral_original_reuse`. The completed evidence does not authorize a
further training or evaluation run.
