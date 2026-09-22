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

At this record's creation the pod was provisioning; no completed model is claimed.
