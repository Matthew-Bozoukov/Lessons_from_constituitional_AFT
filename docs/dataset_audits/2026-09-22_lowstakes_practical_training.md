<!-- ABOUTME: Pinned inputs and operating record for the practical low-stakes seed0 LoRA. -->
<!-- ABOUTME: Completion requires verified adapter publication and owned-pod teardown receipts. -->

# Practical low-stakes training, 22 September 2026

User authorized one H200, seed0, the standard recipe, model publication and teardown.
Work is on `codex/lowstakes-synthdoc-pipeline`; training source commit is `f764b427`.
The owner enforces a conservative $40 operating stop (chosen by the agent, not a new
user-approved dollar ceiling), including a 45-minute recovery reserve. The live GPU
quote was $4.59/hour, with $0.10/hour budgeted for storage. Other account pods are excluded.

## Inputs

- Synthetic: `dougalldeepmind/2026-09-21-da-lowstakes-practical-synth`
  at `83544fd2f48f7abc07a5e33dbcff35746f039c39`, `dataset.jsonl`, exactly716 rows.
- Replay source: `dougalldeepmind/2026-09-08-nosynth-mix`
  at `7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`, selecting the identical9284 rows
  and positions used by the prior exact716 arms.
- Published training mixture: [2026-09-22-da-lowstakes-practical-7-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-da-lowstakes-practical-7-mix)
  at `e5948018221f3434813054e9afe58498aeeaa852`.
  `mixture.jsonl` SHA256 `c1b0a3010e6269a61676e7ced0199bf2bc004e12938fb1e0593bdbd2bec30a98`.
- Base weights: `Qwen/Qwen3.6-27B` at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Constitution: `constitutions/claude_distilled_09_principles/constitution.md`.

The native mixture builder used `scratch/dataset_refresh/da-lowstakes-practical.yaml`.
The full independent audit passed: 10000 rows, all716 synthetic payloads preserved,
9284 replay payloads and positions match the frozen reference, no8192-token truncation,
and valid assistant masks on every row. Rendered tokens:8221688. Supervised tokens:5182882,
of which670033 come from synthetic examples. Row share is7.16%; it is not the token share.
The published mixture includes the complete audit, config, frozen proportions and reference.
29 focused owner/mixture tests passed. Windows blocked the installed `mix` launcher, so
the same native entry point was invoked via `python scripts/data/mixture/build_mixture.py`.

## Recipe and operation

Unchanged `configs/train/sft.yaml` and `configs/models/qwen36.yaml`: BF16 unquantized
LoRA r64/alpha128/dropout0.05, one epoch, global batch16, LR1e-4 cosine with5% warmup,
max sequence8192, dynamic token budget8000 on H200, SDPA and the standard mask gate.
`allow_default_supervise=true` explicitly admits all/null supervision columns.

Local owner:
```
uv run --no-sync python scratch/nonmoral/train_pair.py output/lowstakes_practical_training/train_plan.json --out output/lowstakes_practical_training/run
```
Remote training command:
```
uv run --no-sync python scripts/train/train_lora.py --config configs/train/sft.yaml model=qwen36 seed=0 wandb=false constitution=constitutions/claude_distilled_09_principles/constitution.md data_repo=dougalldeepmind/2026-09-22-da-lowstakes-practical-7-mix data_revision=e5948018221f3434813054e9afe58498aeeaa852 base_model_revision=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 allow_default_supervise=true
```

Owned pod: `42zqsg5dbpkr0z`, `nika-low-stakes-practical-train`.
The independent watchdog was registered immediately after provisioning. The owner checks
download speed, real CUDA allocation, GPU count/type, logs, exit status, optimizer progress,
finite loss/gradient norms and its remaining recovery window. It backs up saved checkpoints
while training and verifies625 completed steps and HF adapter payload hashes before normal
teardown. Status and receipts live in `output/lowstakes_practical_training/` until publication.

Training completed successfully. The final adapter is published and verified, and the owned pod is terminated. See the completion record below.

## First-hour checks

The pod passed a real CUDA allocation check, the64-row decode/mask gate, and reproduced
the full local token census exactly. At step100, loss was0.8138 and gradient norm0.391.
The saved checkpoint was copied off-pod while training continued:3845775360bytes,
11files, SHA256 `2530d613da746eb99a695c66b219f2f42d0243ba7e9f806fdc083e28602f7cfc`.

Owner polling timed out during that transfer and resumed after it. A direct independent
probe verified that optimizer steps continued throughout. A hidden-process test isolated
inherited Windows stdin as the cause: a concurrent SSH control call timed out at5seconds
before the fix and returned in1.485seconds afterward. The shared SSH helper and backup
transfer now close stdin when not explicitly uploading bytes.22 focused tests pass.
These source changes do not alter the already-running trainer or owner; the independent
observer covers subsequent transfer windows. Read `live_observer_status.json` for current
progress when the owner's snapshot is stale. Reproduction receipts are in
`output/lowstakes_practical_training/ssh_stdin_probe_before_fix.json` and
`ssh_stdin_probe.json`.


## Completion (2026-09-22)

- Model: [2026-09-22-qwen36-0-da-lowstakes-practical-7](https://huggingface.co/dougalldeepmind/2026-09-22-qwen36-0-da-lowstakes-practical-7/tree/55c52d797a2436b1535c6826167029708f32d24b).
  Adapter-only revision `55c52d797a2436b1535c6826167029708f32d24b` is the revision to pin for inference.
- Completed 625/625 optimizer steps, one epoch, 10,000 examples, seed 0,
  world size 1 on one H200. Mean training loss: 0.8211592617; optimization
  runtime: 13,940.2612 seconds (3 h 52 min). Loss and gradient reports were finite.
- Published adapter payload hashes match the locally recovered files, including
  weights, tokenizer, resolved config and provenance. The pinned dataset/base,
  thinking stamp, supervision counts, step count and world size all passed checks.
- Checkpoints 100, 200, 300, 400, 500 and 600 were copied and verified during training.
  The complete final output archive includes the retained checkpoints 600 and 625,
  final adapter, configuration, metrics and driver logs: 8,986,900,480 bytes, 39 files,
  SHA256 `c102a620572440af0a24949c2403cd147c63c3c3f4e94cf025c4ffe5981e7c54`.
- Pod `42zqsg5dbpkr0z` was terminated only after adapter publication and full local
  backup verification. A fresh provider inventory independently confirmed it absent;
  no pods remained at that check. Its watchdog exited after observing teardown.
- Total owned-pod lifetime: 15,969.7766 seconds (4 h 26 min). Conservative cost
  estimate including the storage allowance: **$20.81**, below the $40 operating stop.
  This is not the shared account's balance change.
- No evaluation was run. Training completion does not establish ODCV performance.

Local receipts: `output/lowstakes_practical_training/run/publication.json`,
`run/local_backup.json`, `run/status.json`, and `teardown_verification.json`.

Full [training archive and verification receipts](https://huggingface.co/dougalldeepmind/2026-09-22-qwen36-0-da-lowstakes-practical-7/tree/d37a4e0cf7a7b92d094f5ac0e79b230f5b1c6bcc/training_backup) are published at
`d37a4e0cf7a7b92d094f5ac0e79b230f5b1c6bcc`. HF's file size and LFS SHA256 match the verified local archive.
This archive commit follows the adapter-only revision above; use the adapter-only
revision for inference downloads. Final local receipt:
`output/lowstakes_practical_training/final_release_receipt.json`.
