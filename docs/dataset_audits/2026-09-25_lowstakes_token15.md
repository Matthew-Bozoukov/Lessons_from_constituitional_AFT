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

Data complete. Generation took **903.5 seconds (15.1 minutes)** and **$22.77917**,
1,282 physical calls, all settled. Of 243 planned scenarios, 242 were returned;
222 passed stakes/scope; 203 passed draft lint; 199 passed final rewrite lint.
The inherited nonblocking corpus diagnostic completed. No replacement batch ran.
New exports span all nine traits and 80 of 81 assigned trait/domain cells; the
unchanged parent already spans all 81. Spot checks still show the known speculative
alternatives/unsupported-detail failure mode; no full factual audit was performed.

- Extended corpus: `dougalldeepmind/2026-09-25-da-lowstakes-practical-synth`
  at `e7239a9018c50c968768602fe323f784088cb7d6`, **915 rows = 716 old + 199 new**.
  Dataset, frozen config, constitution, cost summary and full raw-call archive verified
  by downloading the pinned publication and comparing SHA256.
- Mixture: `dougalldeepmind/2026-09-25-da-lowstakes-practical-15-mix`
  at `473ad905d807d7f1ea394f33ab16061f106c34fb`, **10,136 rows = 852 low-stakes + 9,284 replay**.
  Added 136 rows containing 126,557 supervised tokens. Total 5,309,439 supervised
  tokens, of which 796,590 (**15.0032800%**) are low-stakes and 4,512,849 are replay.
  All parent rows and positions preserved, no truncation or response editing.
  Mixture SHA256: `f29e0024959ed87891b0a7d9e1a0c974db277de195a3f78a48723e2c98280bf6`.

Training/evaluation pending. Three initial two-H200 provisioning requests returned
HTTP500; the provider body explicitly says no instances are currently available.
Fresh account inventories after each attempt confirmed no campaign pod was created,
so no training GPU spend accrued. Failure evidence is retained in the campaign output.

The user then authorized one H200 if available first. One single-H200 request also
failed without creating a pod; the next succeeded at **2026-09-25 14:18:25 UTC**.
Owned pod `b3ftt5o0avxvq4`, `nika-low-stakes-token15-train`, bills $4.59/hour.
The unchanged $40 training allocation includes storage and recovery. Both the
provisioner's independent deadline watchdog and the owner-liveness watchdog were
armed immediately. The training source is pushed commit `c12547e2`; global batch
16 and all other training settings remain unchanged. Startup download measured
39.0 MB/s. CUDA/training/publication still require completion evidence.

Local ODCV preflight verified 20 Docker CPUs, 18.9 GB memory, the enlarged network
pool, LF shell scripts, and real Compose config/build using the longest scenario
name under the planned short output root. No model rollout ran in this preflight.

The first trainer exited before model loading or any optimizer step: the appended
136 rows alone had an unused `n_tokens` column. Hugging Face's blockwise JSON loader
inferred the three-column parent schema and rejected that later extra column.
The owner preserved the complete failure logs and verified termination; estimated
GPU plus storage cost was **$0.96439**. Token-audit validation had missed the actual
trainer-loader boundary. The append utility now omits that extra metadata and
exercises the trainer's `load_dataset('json', ...)` path before publication,
comparing all loaded messages, sources and supervision with the source rows.
The complete 10,136-row corrected file passed this check locally. All original
rows and every new message remain unchanged; counts and selected IDs are unchanged.
The original published revision remains in HF history for traceability.
Corrected mixture revision: `3bbe5945a1088a6113b5fef9b7353de04a9bb634`;
SHA256 `f6c3147ddfdcd2916790a1d3cd5116a47677e1f36c1a936502969a8eb7302985`.
All five corrected published files passed pinned download/hash verification.
The next training allocation is $39, retaining the prior $0.96439 startup charge
inside the original $40 training allowance and $100 campaign ceiling.

## Completed training

The single-H200 fallback completed all **634 steps / one epoch** on the corrected
mixture. The published metadata confirms seed 0, rank 64, alpha 128, global batch
16, `token_mean`, packing enabled, token budget 8,000 and the frozen base revision.
Measured trainer runtime was **3,614.7 seconds (60.2 minutes)**; mean training loss
was 0.72567. These are training diagnostics, not evaluation results.

- Adapter: `dougalldeepmind/2026-09-25-qwen36-0-da-lowstakes-practical-15`.
- Adapter publication: `9d33dc3c5f846497d48ed73fd9af11471dc04d32`.
- Final revision including full backup: `f656dea7cb2b397f3a89f0bb043f8f2a8e7630bb`.
- All nine adapter files verified against the pod's exact bytes. Full 39-file
  training archive: 8,986,910,720 bytes, SHA256
  `3e7b3b2b9bd1b92638961a74c8fd1e109033c8025e007a66ae322dcbcfb9c333`.
- Pod `12a5x8xijuw8ml` terminated after publication and backup verification.
  Estimated GPU/storage cost: **$6.02584**, plus **$0.96439** for the earlier
  pre-training schema failure. Settled generation plus estimated training totals
  **$29.76940** before ODCV.

Backup initially hit the existing Python 3.10 `hashlib.file_digest` incompatibility
in the direct-Hub packing path. Training was already complete and weights intact.
The still-running owner recovered using the pod's installed Python 3.12, and that
packing call now explicitly uses the repository interpreter. Nineteen existing
backup/owner tests pass. No training restart or replacement GPU was required.

## Training-text name audit

Audited every field of all 10,136 rows in the corrected, pinned training JSONL,
including assistant reasoning. Case-insensitive whole-word searches found **zero
OpenAI or Anthropic mentions**; allowing spaces or hyphens in OpenAI also found
none. Broader substring matches were unrelated text: open air/open-air,
`openair2/COMMON/platform_types.h`, lycanthropic and philanthropic.

There are **nine occurrences of Claude in four replay rows**, all human names:
Claude McKay (one), Claude La Haye (four), Claude Debussy (three), and a baby-name
list (one). Six occurrences are in assistant reasoning and three in assistant
answers. None refers to Claude the AI. All 852 low-stakes rows have zero occurrences
of any of the three names. This is a targeted name audit, not a broader factual or
identity-leakage audit. No training content was changed.

Detailed local evidence: `claude_text_audit.json` and
`organization_name_audit.json` under the campaign output directory.
