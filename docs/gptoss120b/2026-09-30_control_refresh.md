<!-- ABOUTME: Retrain the GPT-OSS nosynth control and compare historical versus clarified Harmony tool prompts. -->
<!-- ABOUTME: Records pinned data, native reasoning provenance, and matched full ODCV evaluation evidence. -->
# GPT-OSS nosynth control refresh, 2026-09-30

Training, adapter publication, and both full ODCV evaluations are complete.
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

| Metric | Original prompt | Fixed prompt |
|---|---:|---:|
| Scored rollouts | 240 | 240 |
| Misconduct | 107/240 (44.58%) | 80/240 (33.33%) |
| Scenario-aware 95% interval | 33.9–55.8% | 24.6–43.3% |
| Mandated misconduct | 62/120 (51.67%) | 41/120 (34.17%) |
| Incentivized misconduct | 45/120 (37.50%) | 39/120 (32.50%) |
| Rollouts with a confirmed stray `]` | 19/240 (7.92%) | 8/240 (3.33%) |
| Confirmed stray-bracket calls | 101 | 21 |
| Rollouts with a literal bracket ending | 18/240 (7.50%) | 6/240 (2.50%) |
| Literal bracket-ending calls | 88 | 12 |
| Rollouts with any invalid JSON | 20/240 | 10/240 |
| Invalid JSON calls | 102 | 23 |
| Visible tool calls | 2,044 | 1,474 |
| Progress score at least 3 | 196/240 (81.67%) | 218/240 (90.83%) |
| Task submission calls | 229/240 (95.42%) | 239/240 (99.58%) |
| Separately rejected Harmony endings (completions) | 1 | 13 |

The fixed prompt reduced observed stray-bracket rollouts by 11/240 (4.58 percentage
points) and stray-bracket calls by 80. It did not eliminate the error. Observed
misconduct decreased by 11.25 percentage points while measured task progress and
submissions increased. This is one checkpoint under two prompts, not a training-
seed study or evidence that the underlying model became intrinsically safer.
Future comparisons must keep the prompt regime matched between arms.

The bridge also rejected **1 original and 13 fixed completions** whose tool calls
ended with `<|return|>` (200002) instead of `<|call|>` (200012). Offline replay of all
14 saved outputs verifies declared tool names and valid JSON arguments: these are
separate from the square-bracket issue. The existing HTTP retry path resampled
these individual completions; no whole rollout was rerun. They are absent from
visible tool-call transcripts but preserved in the sampling ledger. Both runs use
the same bridge/retry policy. The separate handoff failure got more frequent with
the fixed prompt, so the bracket improvement must not be described as eliminating
native-format failures. `metadata/bridge_error_audit.json` and its published audit
script contain the complete offline diagnosis without new inference.

Pinned scored payloads before closeout annotations:

- [Original](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-odcv-original-gptoss120b-0-nosynth/tree/7b02e262a8a10545e1d6147dd07f82317b351fba).
- [Fixed](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-odcv-fixed-gptoss120b-0-nosynth/tree/9cb3d5db86e77990d9905a490ce103291a033205).

Both repositories additionally carry `metadata/closeout.md`, per-rollout
`metadata/tool_format_audit.json`, exact prompt strings/tokens, the verified model
artifact binding, and the shared `results/prompt_comparison.json`.
Final annotated revisions: original `66d24c2415b2200110ddca07c0d4f23cd191c3b1`;
fixed `ce3305ffcf0bc5a60cd49f23a23769f3d58ea047`.

Work stays on `codex/gpt-oss-120b-exploration` in its dedicated worktree. No RunPod
or Vast resources were started and no other session's branch was switched.
The 32 Harmony/backfill checks passed. An additional evaluation-framework test
batch passed 87 checks; one unrelated registry-import check could not run because
the lightweight Tinker environment lacks `bs4`.

Original-prompt rollouts completed all three passes without an infrastructure
rerun. Judging then hit a transient Windows lock while replacing its ledger.
All 240 paid requests remained accounted for and 239 verdicts were cached. The
ledger now has bounded replacement retries, tested together with refusal to
dispatch when a reservation cannot be saved (six judge-budget/copy tests passed).
Recovery preserved every transcript and cached verdict, repeated only the missing
misconduct judgment, and completed the independent progress judgments. Its first
progress attempt failed before dispatch because the recovery helper loaded the
environment after importing a module that captures credentials; the helper's
import order was fixed. No model rollout was regenerated during recovery.

Original-prompt inference cost upper estimate: $1.587897. Original-prompt judge
ledger: $1.3133825, including the paid uncached judgment and its replacement.
Fixed-prompt inference upper estimate: $1.21179405; judging: $0.969783.
Training plus both full evaluations totals **$9.503011839** in these estimates,
excluding the small pre-evaluation transport qualification. These are per-request
estimates/reservations rather than a provider invoice.
Console/global account usage deltas include other sessions and must not be treated
as this run's spend. The inference ledger retains one error event and its reserved
charge; all scenario transcripts passed the clean-run checks.

All 240 transcripts plus six result/metadata files from each publication
were downloaded and byte-compared. Both runs were reanalyzed from their pinned HF
transcripts for the final comparison. All added closeout/audit files are read back
and byte-verified after publication. The task's Docker project prefix `odcv-08b595-`
has no remaining containers, and its bridge ports 18331/18332 have no listeners.
Only this task's redundant checkpoint download archives were removed after remote
weight-hash verification; exported weights and all research receipts remain.

The original SDK checkpoint download was interrupted only in this task's export
process, then safely resumed after checking the remote prefix and HTTP range.
Training was not rerun. Publication requires the remote LFS hash to match the
local audited safetensors file.
