<!-- ABOUTME: Preregistered append-only low-stakes extension, token-mean SFT and ODCV campaign. -->
<!-- ABOUTME: Records immutable input pins, spending allocations and the end-to-end completion evidence. -->
# Practical low-stakes at 15% supervised tokens

User authorized synthesis, publication, rank-64 seed-0 training on two H200s,
then three sequential ODCV passes on one GPU and local Docker. The combined ceiling
is **$100**, approved 2026-09-25. Stage allocations: synthesis $30, training $40,
ODCV GPU/storage $25 and judges $5. Allocations are upper bounds, not expected spend.
No prior campaign's spending authorization is reused.

Worktree `lowstakes-token15-20260925`, branch `codex/lowstakes-token15`, starts from
`origin/main` at `9bcd66e2`. Other worktrees and jobs remain outside this campaign.

## Data and selection

Extend `dougalldeepmind/2026-09-21-da-lowstakes-practical-synth`
at `83544fd2f48f7abc07a5e33dbcff35746f039c39` with the released
`configs/data/synth/da-lowstakes-practical.yaml` recipe and the same
`constitutions/claude_distilled_09_principles/constitution.md`.
The native `extend_from` mode compares new scenarios against the prior 971 scenarios
and carries the 716 previously released rows forward verbatim. No old answer is rewritten.
Generate 243 candidates: three per each of 81 principle/domain combinations, 32 workers,
no diversity replacement rounds. Prompts/models/admission gates are unchanged.
The 20% stage-failure alarm is the released full-run operational setting; individual
row gates are not relaxed. `token15_` prevents scenario-ID collisions.

Keep **all 10,000** rows of `dougalldeepmind/2026-09-22-da-lowstakes-practical-7-mix`
at `e5948018221f3434813054e9afe58498aeeaa852`, including the exact 9,284 nosynth rows.
Recount with the current Qwen profile/mask and pinned tokenizer
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`:
670,033 low-stakes + 4,512,849 nosynth supervised tokens, 12.9278% low-stakes.
At least approximately 126,352 additional low-stakes tokens are needed for 15%.

Select new accepted rows by deterministic trait/domain round robin and hash ordering;
take the prefix nearest 15%, with a 0.03-percentage-point tolerance. No quality-outcome
selection and no response editing. Keep the original row order and append the new rows;
training performs its standard seed-0 shuffle. The shared normal mixture builder's swap
mode would remove old base rows, so the one-off append utility reuses its renderer/mask
and source normalizer but does not call the swap planner.

Commands/configs:

```
uv run python -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_extend.yaml
uv run python -m scratch.dataset_refresh.extend_practical_mixture --config scratch/dataset_refresh/token15_mixture.yaml --synth-revision <sha> --publish
```

## Training and evaluation

Use current `configs/train/sft.yaml`: BF16 Qwen3.6-27B, rank 64, alpha 128,
dropout .05, one epoch, LR 1e-4/cosine/5% warmup, global batch 16, max sequence
8192, token budget 8000 per GPU, packing with all three boundary-aware kernels,
**token_mean** loss. The global batch stays 16 across two GPUs. The existing
`scratch/nonmoral/train_pair.py` owner handles one arm, independent watchdogs,
CUDA/mask checks, finite loss/gradient checks and immutable output preservation.
Completed archives upload directly from the pod and verify Hub hashes before teardown.

ODCV uses the maintained runner with the existing bounded Windows owner: 40 scenarios
in both variants, three sequential passes, one continuous vLLM server, startup seed 0,
temperature .7, 28k context, no per-request seed, Gemini 3 Flash MR and progress judges.
24 concurrent local Docker scenarios, with no global network/image pruning.
The first scheduled cell is the counted health pilot. All pass transcripts are saved
locally before advancing; server continuity is checked at every pass boundary.

The chart comparisons are the pre-spec-filter, **token_mean** control and DA-15 runs:

- `dougalldeepmind/2026-09-21-odcv-qwen36-0-nosynth`
  at `7591355b69f5979bbbb051ffa68f016ac5d74020`: 105/240 = 43.75%.
- `dougalldeepmind/2026-09-21-odcv-qwen36-0-da-15`
  at `cba5f4188f3408d0f7bdef12fb3faf0b0d9315d6`: 17/240 = 7.0833%.

These use the same September-8 nosynth source and current token-weighting convention,
not the newer spec-filtered replay corpus. Remaining differences include the exact
base-row selection/total token budget, training GPU count and rollout concurrency.
The comparison will not isolate the causal effect of stakes or top-up alone.

## Completion

Pending. Generation, training and evaluation receipts will be recorded here as verified.
