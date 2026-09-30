<!-- ABOUTME: Retrain the GPT-OSS nosynth control and compare historical versus clarified Harmony tool prompts. -->
<!-- ABOUTME: Records pinned data, native reasoning provenance, and matched full ODCV evaluation evidence. -->
# GPT-OSS nosynth control refresh, 2026-09-30

Training and adapter publication are complete. Paired ODCV evaluations are in
progress; final results will be added after both runs are scored and verified.
The user-designated current control is pinned in [`current_control.yaml`](current_control.yaml).
Model: [`dougalldeepmind/2026-09-30-gptoss120b-0-nosynth`](https://huggingface.co/dougalldeepmind/2026-09-30-gptoss120b-0-nosynth/tree/6a7549fd796442155dfce46d9150c2ec104a66ff)
at `6a7549fd796442155dfce46d9150c2ec104a66ff`. Its full 5,257,620,504-byte safetensors
file passed the finite-tensor audit and matches the remote LFS SHA256.

## Data and training

The new control uses
[`dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b`](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-nosynth-mix-gpt-oss-120b/tree/4ffd1f931ebb34b25c68c4b784059f98c1617add)
at `4ffd1f931ebb34b25c68c4b784059f98c1617add`. The mixture's SHA256 is
`d792bcbb3e9ed62d6b02b010db3add413e931a142ce31c824ca39c7143d91b6c`.

All 10,000 rows remain. The 1,061 rows with CoT contain 1,073 traced assistant
turns: 954 retain the original final answer and 119 adopt a reviewed cached
GPT-OSS answer with its corresponding CoT. Of the 954, 446 use later strict CoT
refreshes and 508 retain already-native GPT-OSS traces from the first conversion.
All 9,307 untraced turns remain unchanged. See the
[source-by-source breakdown and evidence limits](2026-09-30_native_cot_provenance.md).

Fresh base `openai/gpt-oss-120b` received one epoch of Tinker LoRA training, seed 0,
rank 32, attention + expert/MLP + unembedding scopes. Sixteen source rows form each
optimizer batch; loss is normalized by the total supervised tokens in the batch,
not independently by row. LR `1e-4`, cosine decay, 32 warmup steps, AdamW betas
`.9/.95`, epsilon `1e-12`, weight decay `.01`, clipping norm `1`.

All **625 steps** completed without replay or restart. The full-corpus audit
counted **5,997,497 processed tokens** and **2,352,071 supervised tokens**; no
example exceeded 8,080 tokens and none was truncated. The training cost upper
estimate is **$4.420155289**, using the receipt's per-request reservations.

The immutable sampling checkpoint is:

```
tinker://0b544ff0-4beb-597c-b78d-6ba05f89be78:train:0/sampler_weights/2026-09-30-gptoss120b-0-nosynth
```

The launch observation pins code `20ebb8e6`; the completion receipt records
`1c63f859`, after evaluation orchestration was committed while training ran.
No training implementation changed during the run. Provider-managed base-weight
precision is not independently verified. No manual quantization was applied.
The exported native Tinker adapter is not yet qualified for PEFT/vLLM serving.

## Evaluation design

Both regimes evaluate this same checkpoint on all 40 ODCV scenarios, both incentive
conditions, and three passes: **240 rollouts per regime**. Local Docker executes
scenario tools; Tinker provides inference. Temperature `.7`, medium reasoning,
8,192 maximum generated tokens, concurrency 8, and the established Gemini 3 Flash
Preview misconduct/progress judges are identical. Runs are sequential because
their per-scenario Docker project names share the same model key.

The `original` prompt matches historical renderer `87957e51`; the `fixed` prompt
matches the detailed Harmony/bash JSON explanation at `4e9fe0d321ad7db5e42a04a85b380b1582e284af`.
Exact token equality was checked on all three saved failure histories for both
variants. A real Docker tool-call/result/continuation smoke passed with the new
checkpoint before the full runs. The historical ODCV scenario instructions are
unchanged; the deliberate difference is our tool-formatting guidance.

Evaluation may begin while the large HF export uploads: its target remains the
qualified immutable Tinker checkpoint. Launch metadata explicitly records pending
HF publication. Final comparison metadata binds both runs to the verified HF
adapter revision once upload completes.

`scratch/gptoss_control/compare_prompt_runs.py` counts invalid JSON and confirmed
stray `]` calls from the saved transcripts. A confirmed extra bracket means the
JSON decoder fails at `]` and deleting that one character produces an object.
The stricter ending count requires only an optional final `}` and whitespace after
the bracket. Both call counts and affected-rollout counts are reported. This is
offline diagnosis; arguments are never repaired and executed during evaluation.

The comparison uses one training seed and stochastic decoding without matched
sampler seeds. It does not establish an effect relative to the untouched base
model, for which no matched full ODCV result exists. Historical base-formatting
probes and the old control are documented in the September 29 reports.

## Execution and verification

Work stays on `codex/gpt-oss-120b-exploration` in its dedicated worktree. No RunPod
or Vast resources were started and no other session's branch was switched.
The 32 Harmony/backfill checks passed. An additional evaluation-framework test
batch passed 87 checks; one unrelated registry-import check could not run because
the lightweight Tinker environment lacks `bs4`.

The original SDK checkpoint download was interrupted only in this task's export
process, then safely resumed after checking the remote prefix and HTTP range.
Training was not rerun. Publication requires the remote LFS hash to match the
local audited safetensors file.
