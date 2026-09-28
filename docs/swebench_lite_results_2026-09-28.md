<!-- ABOUTME: Exact checkpoint and training-mixture provenance for the completed September 28 Lite v5 runs. -->
<!-- ABOUTME: Control remains the baseline; DA-15 is a historical measurement pending a training-data correction. -->
# September 28 SWE-bench Lite v5: retained results and interpretation

The user reported on September 28 that the dataset used to train the DA-15
checkpoint has an issue and needs replacement. The precise defect and corrected
replacement are not established in this closeout. **Retain the control baseline.**
The DA-15 score remains a measurement of the pinned historical checkpoint, but
must not be presented as validation of a corrected DA-15 treatment. No outcomes,
original result files, interrupted attempts or cost ledgers were deleted or rerolled.
The previously published comparison chart also refers to these historical checkpoints.

Both checkpoints have training seed 0 and share
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Training provenance below was read from each model's pinned `training_meta.json`.
This identifies the training seed, not deterministic evaluation sampling.

## Control: retained baseline

- Model: `dougalldeepmind/2026-09-22-qwen36-0-nosynth`
- Model revision: `633908b72a9799fb3e6b101b0a8a82aec3c3d642`
- Training mixture: `dougalldeepmind/2026-09-22-nosynth-mix`
- Mixture revision: `378ec1ee0f0eea9294683779438b839e52b9700a`
- Training file: `mixture.jsonl`
- Result: **178/300 (59.33%)**
- [Published result and closeout notice](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-swebench-qwen36-0-nosynth-lite-v5/tree/343c94dfd5ca3f734c615f9103ea1ce0aa8eb186)

## DA-15: historical checkpoint; dataset correction pending

- Model: `dougalldeepmind/2026-09-25-qwen36-0-da-15`
- Model revision: `b13cfe9891671f05c73a12078796e212944d22e4`
- Training mixture: `dougalldeepmind/2026-09-25-da-15-mix`
- Mixture revision: `73f66648dc1c1f4e12d887dca5780bb065ddd385`
- Training file: `mixture.jsonl`
- Result: **184/300 (61.33%)**
- [Published result and closeout notice](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-swebench-qwen36-0-da-15-lite-v5/tree/fec0221730b135bd14a444918345f96b66994cb6)

## Protocol and state

Both used mini lite-v5: temperature 1, top-p .95, top-k 20, neutral other penalties,
16,384 response tokens, 262,144 context/cumulative generation tokens, 500 steps,
retained reasoning and the same official grader. Control used up to ten GPUs;
DA-15 up to six. Local HTTPBin and two Requests no-fix baseline passes remain
scoring limitations. See LOG.md for full outcome categories, timings, accounting
and infrastructure repair evidence. No corrected model was trained or evaluated.

All campaign GPUs are gone and completion monitors are deleted. Vast CPU 53118260
was explicitly paused September 28; provider state is exited/intended stopped.
The 500 GB disk and prepared cache are retained at $0.1388889/hour ($3.33/day).
The shared local registry remains registered, with a dated pause audit; do not
mark it empty or rent a replacement simply because SSH is unavailable.
A future authorized run may resume it, then requalify the deployed source/recipe.

The merged code includes the adopted-worker HF checkpoint repair, budget-deferral
classification and SymPy-11870-first grading. Linux CPU-only post-merge regression
checks passed 51 fleet/handover tests, using local Docker and mocked paid providers.
No new cloud compute or inference was started for closeout. The 900-second startup
allowance remains unchanged and is still a known source of wasted warmup rentals.
