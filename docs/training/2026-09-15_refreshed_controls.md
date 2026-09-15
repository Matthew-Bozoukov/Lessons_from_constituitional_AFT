<!-- ABOUTME: Track the two explicitly authorized parallel refreshed-control LoRA runs and exact scientific inputs. -->
<!-- ABOUTME: Separate startup checks and failures from optimizer progress, completed adapters and final publication. -->

# Refreshed low-stakes and nonmoral LoRA training

The user authorized one LoRA per arm, each on a separate RunPod pod with two H200s,
running in parallel. They approved a **$100 combined GPU ceiling**, initially
allocated $50 per arm, including failed starts and result recovery. Evaluation
has not been authorized. The known dataset-content inspection notes remain
disclosed; training uses the exact published bytes without silent corrections.

## Scientific inputs

- Shared recipe: `configs/train/sft.yaml`; model profile `qwen36`; seed0; one
  epoch; rank64/alpha128/dropout0.05; BF16 without4-bit quantization; cosine
  LR1e-4; warmup0.05; global batch16 across both ranks; no packing; max8192.
- Each mixture has716 synthetic plus the identical9284 replay rows. Expected
  optimizer steps:625. Loss averages supervised-token loss per example; both
  reasoning and final answers are supervised under the native Qwen mask rules.
- Base model `Qwen/Qwen3.6-27B` pinned to
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` for both arms.
- Low mixture: `dougalldeepmind/2026-09-15-da-lowstakes-refresh-7-mix`
  @`f0b0418ea9767ab00fe21e86fab8dfc26dc5320c`.
- Nonmoral mixture: `dougalldeepmind/2026-09-15-nonmoral-advice-7-mix`
  @`588783fb079d0bbe5c9eb568418fcff2aad4cecb`.
- Training source commit: `fd63ca8b83a5ea4127b015ed3034045afc0802ba`, pushed on
  `codex/refresh-lowstakes-nonmoral`. The recipe/lock match current main; the
  added explicit `allow_default_supervise=true` admits the standard data schema
  without changing masks or optimization. The locked stack's optimizer default
  is AdamW fused; no unfused-optimizer override was introduced.

## Launch and monitoring

The existing `scratch/nonmoral/train_pair.py` owner is run twice with a single
arm per plan. It uses `runpod.up` with immediate independent watchdog registration,
checks actual CUDA operations on exactly two H200s and startup bandwidth, monitors
training exits, loss/gradient finiteness, checkpoints and stalls, and reserves
45minutes inside its original lifetime/cost bound for backup. It verifies local
output-archive hashes before normal teardown. Public adapter verification is a
separate completion requirement.

Live state and the full attempt history are in
`output/2026-09-15_dataset_refresh_training/launches.json`, with owner
`status.json`, `owner.log`, `watchdog.log` and training-tail files under each
attempt directory. A five-minute task heartbeat follows the latest attempts.
An existing per-process sleep-inhibition helper keeps local guards active; create
`output/2026-09-15_dataset_refresh_training/keep_awake.stop` after both owned pods
are verified terminated, then pause the heartbeat. No permanent power settings
were changed. Do not terminate other account users' pods.

At this report's startup checkpoint, replacement low pod is `xy4sgbu0kwdl29`
(`low_attempt2`); replacement nonmoral is `oshtnlsecsq962`
(`nonmoral_attempt4`). Each quotes$9.18/hour before a small storage allowance.
The adapters are intended to publish as
`dougalldeepmind/2026-09-15-qwen36-0-da-lowstakes-refresh-7` and
`dougalldeepmind/2026-09-15-qwen36-0-nonmoral-advice-7`.
Allocation/startup is not a completed-training claim.

## Preserved startup failures

Two nonmoral allocation requests returned provider HTTP500 without a pod ID;
account inventories were checked before retrying. The first allocated pod for
each arm passed CUDA but hit the old uniform-all-supervision guard before model
loading or any optimizer step. Both archived their failure logs and terminated:
low `szu6ijjqbiai8z`, estimated$0.4567159; nonmoral `ntr7e7c58mwrfo`,
estimated$0.4450778. Including a conservative latency allowance, replacement
allocations are capped at$49.44 and$49.45 respectively. The combined ceiling
was not reset. Provider billing is distinct from these elapsed-time estimates.

The compatibility fix retained the default ablation guard unless explicitly
allowed. Focused training-launch/backup checks passed38 tests; the five
scratch-owner command tests passed again after moving to the correct directory.
The exact10000-row census for each actual mixture passed locally before restart.
Published source data and prior failed evidence are unchanged.

## Completion gate

Require625 optimizer steps/one epoch, finite training statistics, saved LoRA
weights/tokenizer/config and exact base/data provenance; verify public HF model
revision and payload hashes against preserved outputs. Then record final cost,
verify both owned pods gone and account balance, release sleep inhibition and
pause monitoring. No evaluation follows without the user's separate instruction.
