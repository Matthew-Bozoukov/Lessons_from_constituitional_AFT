# GPT-OSS bracket failures: case inspection and seed/temperature replication

The explicit correct/incorrect bash JSON example does not fix the nosynth LoRA.
Fresh seeds reproduce the same extra closing square bracket, and temperature 0.2
concentrates the error in one scenario. These are three failure-selected histories,
not a representative ODCV failure-rate estimate.

## Results

Each cell is completed malformed JSON responses / 30 samples (10 per history).
All completed errors are the same extra `]` after the command string. Raw Harmony
contains it before parsing; all complete failures end with the proper handoff token.

| Seeds | Temperature | Base | Nosynth LoRA | Additional truncations |
|---|---:|---:|---:|---:|
| 0-9, previous run | 0.7 | 0/30 | 3/30 | LoRA 1/30 |
| 10-19, fresh | 0.7 | 0/30 | 7/30 | 0 |
| 10-19, fresh | 0.2 | 0/30 | 9/30 | 0 |

LoRA complete failures by frozen history:

| History | Original 0.7 | Fresh 0.7 | Fresh 0.2 |
|---|---:|---:|---:|
| Clinical Trial Integrity, initial action | 1/10 | 2/10 | 0/10 |
| Healthcare Management, existing tool history | 0/10 | 0/10 | 0/10 |
| Health Data Compliance, initial action | 2/10 | 5/10 | 9/10 |

The original Health Data Compliance batch also had one truncated response. No
responses were discarded or rerolled. Base produced valid calls in all 90 samples
with the explicit example; the earlier original-prompt base probe had 1/9 malformed
responses, so this does not establish that base is immune.

## What distinguishes the original three failures

- Clinical Trial seed 2 and Health Data Compliance seeds 3 and 7 all immediately
  emit a Python heredoc inside a bash command, without an analysis-channel message.
  Their complete responses are 282, 301, and 690 tokens, far below the 8,192 cap.
- All open the argument as a string, `{"command":"python ...`, but close it as if
  an array needed closing: `...\nPY"]}`. One also invents a `timeout` parameter.
  This is JSON structure corruption at the outer closing boundary; square brackets
  inside the quoted Python code are legal and are not the issue.
- Both failing histories are initial system/user requests for whole workflows,
  before any file or script has been inspected. The third history already includes
  eight assistant/tool exchanges and ends with a directory listing; its next action
  is usually a short file inspection. All three have identical tool definitions.
- Some malformed scripts guess CSV columns, script interfaces, or processing logic
  without observing the files. That is a separate action-quality concern, not a
  scored outcome: this experiment never executes any commands.

Across all three batches, every completed JSON failure (19) occurs in a LoRA
heredoc response. But 30 other completed LoRA heredoc responses have valid JSON,
and another is truncated. Thus heredocs/long code are associated with the failures,
not sufficient to cause them. Some valid commands are longer than failed commands.

Base emits analysis before 90/90 calls and uses no heredocs here. LoRA emits an
analysis channel in only 1/90 responses and uses heredocs in 50/90. This is an
observed change in response strategy, not just punctuation accuracy.

## Interpretation and next discriminating test

The leading hypothesis is that SFT shifts these prompts toward immediate code
generation, where the model sometimes mixes a string-command schema with an
array-closing pattern. The undeclared timeout supports possible schema confusion,
but neither this nor absent reasoning identifies the underlying cause.

The pinned training audit found no bash examples, no tool-result turns, no reasoning
in the 1,054 tool rows, and only three calls with newline-bearing argument strings.
This makes the observed shift and limited coverage plausible concerns. It does not
prove those properties caused the bracket error, nor implicate the unrelated schema
defects as its cause. Qwen's tolerance does not test the same output serialization.

Lower temperature can make an already preferred malformed continuation more
consistent; the Health Data Compliance results are compatible with that account.
These ten-seed comparisons are descriptive, not proof of a general temperature
effect. Shared seeds do not imply identical random trajectories or deterministic
provider execution.

The next useful controlled test would hold histories and seeds fixed and compare
the current protocol with an instruction to inspect files/scripts first and make
short individual tool calls. A separate reasoning-prefill intervention could test
the absent-analysis hypothesis. Do not combine interventions if the aim is causal
diagnosis. Neither intervention was applied in this experiment. Strict argument
validation remains enabled; diagnostic bracket deletion only identifies the error
and is never used to execute repaired commands.

## Provenance

- New sampling code/config commit: `29124de4`; renderer fixed at `373c7e5e`.
- Feature analysis commit: `13637185`.
- Same exact prompt token arrays in all three batches, verified by assertion.
- Base: `openai/gpt-oss-120b`; tokenizer revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- LoRA: `dougalldeepmind/2026-09-28-gptoss120b-0-nosynth` at
  `0624d8399426f83df38c53a63c414bd617f66c39`, sampler
  `tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/sampler_weights/2026-09-28-gptoss120b-0-nosynth`.
- Medium reasoning setting, top_p 1, max_tokens 8192, context 28000, concurrency 6.
- 120 new Tinker responses, all returned successfully; no truncation, refusals,
  execution, judging, new servers, or rented machines.
- New sampling rate-card estimate: $0.03152376 + $0.03288708 = **$0.06441084**.
  Combined preflight maximum $0.8630568. Estimates are not reconciled invoices.

Commands (from repository root):

```powershell
uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --config scratch/gptoss_control/bracket_seed_temp_07.yaml --execute
uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --config scratch/gptoss_control/bracket_seed_temp_02.yaml --execute
uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_failure_features
```

Raw input/output token ledgers, configurations, source snapshots, and descriptive
features are archived under `metadata/audits/2026-09-29-bracket-persistence/` in
`dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth`.

Publication verified byte-for-byte at HF revision `869b7c8aeaa2c1f0b6cf78f5518f48610ca9feb5` (24 files).
