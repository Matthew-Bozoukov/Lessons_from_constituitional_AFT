<!-- ABOUTME: Frozen execution plan for the published original684 low/high stakes pair. -->
<!-- ABOUTME: Documents training, evaluation, artifact recovery, budget and automatic handoff. -->
# Matched stakes training and ODCV

User explicitly authorized both LoRAs and both ODCV evaluations on September10.
Inputs and exact HF revisions are in [the dataset results](2026-09-10_matched_stakes_results.md).
No data changes, additional training seeds or capability evaluations are included.

1. Freeze low/high mixtures and base revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Both seed0,9,968 rows each.
2. Rent one2×H200 pod using `src.infra.runpod.up`, through the existing
   `scratch/nonmoral/train_pair.py`. Train low then high on both GPUs, sharing
   the setup/cache. Both use `scripts/train/train_lora.py` and `configs/train/sft.yaml`:
   BF16 rank64/alpha128/dropout.05; one epoch; global batch16; LR1e-4 cosine;
   warmup.05; weight decay.01; length8192; dynamic padded-token budget8000;
   world size2; expected623 optimizer steps. Full reasoning and final loss.
3. Before long work, check SSH, actual CUDA arithmetic on both H200s, download
   throughput, masking and physical LF scripts. During training, poll progress,
   GPU memory/utilization/temperature and checkpoint loss/gradient finiteness.
   Bootstrap and stale logs are bounded. Preserve only the owned process group.
4. Before teardown, inventory and archive both training output trees, including
   saved checkpoints, adapters, tokenizers, configs and logs. Fetch locally and
   verify byte count, hash, member paths and both completed dataset identities.
   Ordinary teardown is blocked if backup fails. The independent budget watchdog
   remains the emergency upper bound; recovery has a75-minute reserved window.
5. Let the shared trainer mint and publish adapter names. Verify public HF status,
   exact dataset/base revisions, thinking mode,623 steps, finite metrics and
   world size2. Compare published adapter weight hashes with the local backup.
6. `scratch/nonmoral/stakes_experiment.py` automatically follows training with
   two sequential invocations of the existing protected ODCV owner,
   `scratch/nonmoral/overnight_baseline.py`. That owner uses the shared `evals`
   entrypoint and RunPod provisioner. Docker/CPU stay local; each inference pod
   uses the profile's H10080GB. The runner owns vLLM and its tunnel.
7. Freeze the established common comparison protocol, not changed current defaults:
   all40 scenarios ×2 variants ×3 passes =240/model; temperature.7;
   context28000; concurrency8; timeout2400s; four judge workers; Gemini3 Flash
   for both MR and progress; no exclusions. Keep valid unfavorable outcomes,
   partial traces and failure evidence. Use only existing bounded infrastructure recovery.
8. Publish through the shared eval layout/naming/card pipeline. Verify public
   datasets, transcripts, scores and provenance. Fetch remote logs before closing
   each owned inference pod. Confirm termination and account-level pod inventory.
9. Report exact low/high MR counts and scenario-paired uncertainty, submission,
   progress, incomplete cells and runtime failures. Lower MR accompanied by worse
   refusal/completion is not accepted as an alignment gain. One training seed per
   arm and ODCV proxies do not establish general capability preservation.

## Budget and durable execution

Prior tracked exposure$216.655025; existing total ceiling$300. Training/recovery cap
$58, with a fixed worst-case pair rate$10/h including storage and a$2 margin.
The provisioned quote was$9.18/h; accounting uses$9.28/h including storage reserve.
The training lifetime is5.6h, with75minutes reserved for retrieval, rather than
waiting until the budget is nearly exhausted to fetch multiple checkpoints.

Actual remaining funds after training determine each eval allocation, capped at
$20 each and always reserving funds for the other arm. MR/progress judging has
a per-call reservation ledger. No replacement seed or unbounded rerun is authorized.
No allocation may make total tracked exposure exceed$300.

Local execution records: `output/nonmoral_stakes/20260910/experiment/`:
`train_plan.json`, `training/status.json`, `continuation_status.json`, frozen ODCV
config and arm-specific plans, cost ledgers, backup receipts and final model pins.
The continuation holds Windows system sleep prevention while running. It makes no
direct rental calls and waits for verified training/publication before evaluation.

Completed checkpoints are also copied during training through the existing backup
module's bounded `--checkpoint` mode, with separate per-checkpoint paths under
`experiment/checkpoint_copies/`. Each copy records a local/remote hash and byte-count
receipt. These are extra recovery snapshots; they do not replace the final full-output
backup gate or count as completed/published models. Check for an active copy before
starting another. The training owner already running retains its loaded backup code.

Training launch code pinned on the pod: `e04e844e`.
Owned training pod at launch: `03l0y6tcp39mgh`; no pre-existing active pods.
HF public org: `dougalldeepmind`. All model/eval names come from the shared builders.

## Authorized overlap: low-stakes ODCV during high-stakes SFT

The user subsequently requested immediate low-stakes ODCV on a separate RunPod GPU.
High-stakes training and its owner/watchdogs remain unchanged. Only the waiting
continuation process is restarted with explicit adoption of the existing low eval,
so it cannot launch that evaluation twice. It still verifies the complete training
backup, waits for the adopted evaluation, then launches high ODCV with actual remaining funds.

`scratch/nonmoral/prepare_stakes_early_eval.py` verifies the locally recovered final
checkpoint623 against public adapter weights, the exact dataset/base pins, training
completion and finite metrics. It writes `low_eval_handoff.json` and the frozen eval
plan. The low adapter revision is `e4b2aa199370314e534115eaa5ca08f6e69f2b6c`.
The shared protected owner `overnight_baseline.py` runs the evaluation without changing
the scientific protocol. Docker/CPU stay local; inference uses a separate RunPod H100.

Low ODCV allocation is $14 ($10.50 GPU/storage including $2 recovery reserve, $3.50
judging). The existing training watchdog at its verified rate bounds training to
$51.968; adding $2 latency margin, $14 for high ODCV, and another $1 reserve gives a
worst-case project allocation of $299.623025. No new training or evaluation repeats.
Ownership adoption tests reject wrong pins and failed outcomes, account for actual
costs, and wait for active owners instead of replacing them. Fifteen focused tests pass.

## Training pod disappeared during retrieval

At about21:03UTC the training pod disappeared while the18GB archive was downloading.
The REST endpoint returned404 and account inventory contained only the inference pod.
Both local watchdogs logged `pod gone; exiting`, not a termination request. The training
owner remained alive retrying retrieval; it had not reached its ordinary teardown.
The cause is unknown. The partial4,299,423,744-byte archive and watchdog logs are retained.

Both trainings had already completed623 steps, exited0, and published their adapters.
The partial archive contained the entire high adapter. Low's verified checkpoint623
contained its final weights. `scratch/nonmoral/recover_stakes_models.py` recovered these
independent local weights, compared their SHA256 hashes to the public HF adapters,
and reconstructed both final model bundles with pinned HF auxiliary files and the
training owner's already-local complete metrics. No training or evaluation was rerun.

Recovered model archive:2,590,412,800 bytes, SHA256
`12bf53a075b104a927fc3a412eaca69ba8bf240d088542481adc80a3b119a11e`.
Low revision:`e4b2aa199370314e534115eaa5ca08f6e69f2b6c`;
high revision:`af5c9f5356e2b38cc4b065c240a1b0bb950f0e9a`.
The complete original archive and high's final optimizer/RNG/scheduler checkpoint
were not recovered; verified high checkpoint500 and low checkpoint623 remain local.
This limits resume-artifact preservation, not evaluation of the final trained adapters.

`recovered_training.json` explicitly records that the full archive is incomplete,
its incident evidence, conservative training cost through confirmed absence, and
the verified final-model receipt. The continuation uses this recovery record and
retains the existing low evaluation handoff. Obsolete training-retrieval and waiting
continuation processes were stopped only after model recovery and confirmed pod absence.
Twenty-four artifact/ownership/eval-plan tests pass. Incident and file-source records
live under `experiment/recovered_training/` and must accompany final publication.

## Low evaluation judge recovery

All240 low rollouts completed cleanly. The MR judge cached240 scores and progress
cached234 before Google AI Studio returned an upstream429. The protected owner
preserved remote logs and verified H100 teardown. Its failure record remains unchanged.
`scratch/nonmoral/finish_stakes_eval.py` resumes only missing judgments using the same
model/provider pin, rubric, transcripts, caches and $3.50 reservation ledger. One judge
worker reduces request bursts. Existing cached verdicts are checked for exact equality.
No rollout/model/GPU rerun is involved. Packaging and publication use the shared
`package_run`, `_publish`, card and naming functions. A separate completion-recovery
receipt permits the continuation to adopt this completed evaluation without rewriting
the failed owner's history. High ODCV then uses remaining project funds as planned.

Low evaluation is complete/public at
[`dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-low-7`](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-low-7),
revision`9361ec441aeb2e45eab43fc27c61a0a67d5e174d`.
Exact verified counts:47/240 misaligned (19.5833%),234/240 submitted (97.5%),
238/240 progress>=3; mean progress4.9375/5. Scenario-level MR95%CI[11.0600,32.2905].
All240 transcripts and both240-score caches are present; no rollout retries,
reconstructions or timeout statuses. Five traces hit the transcript token limit and
one reached the agent cycle limit; `ok` process status does not imply task completion.
All pre-existing verdicts were retained through the six-call progress-judge recovery.
Tracked low GPU/storage$6.387760, judging including reservations$2.357652.
Training recovery exposure$36.654498; project exposure before high evaluation$262.054934.
High ODCV started on separate H100 pod`8zvj8x8zstspiv` at$3.49/h with$18.50 allocation
($15 GPU/storage, $3.50 judging), bounded owner and local Docker. No low/high comparison
is claimed until the high run is complete.
