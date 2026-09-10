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
