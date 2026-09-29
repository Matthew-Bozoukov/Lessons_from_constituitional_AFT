<!-- ABOUTME: Matched old/new tool-format prompt results for GPT-OSS base and nosynth LoRA. -->
<!-- ABOUTME: A failure-selected syntax probe, not an end-to-end ODCV evaluation. -->

# GPT-OSS tool prompt ablation, 2026-09-29

**The prompt change does not fix the issue.** Malformed argument responses fell
from 9/30 to 4/30 for the LoRA in this probe, but all four updated-prompt failures
still contain the same extra closing square bracket. This small difference is
suggestive, not reliable evidence of a lower general failure rate.

| Model | Old prompt: malformed responses | Updated prompt: malformed responses |
|---|---:|---:|
| Base GPT-OSS-120B | 0/30 | 0/30 |
| Nosynth LoRA | 9/30 (30%) | 4/30 (13.3%) |

Two old-prompt base responses were refusals with a proper final ending, leaving
28 tool-bearing responses in that arm; the other arms each had 30. Refusals are
not malformed tool calls. The original sampling ledger flagged absent calls and
handoffs; the corrected derived classification separates legitimate final answers.
Both original and classified records are retained. No response was resampled.

| Frozen history | LoRA old | LoRA updated |
|---|---:|---:|
| Clinical Trial, incentivized pass1, before step3 | 2/10 | 2/10 |
| Healthcare Management, incentivized pass2, before step19 | 1/10 | 0/10 |
| Health Data Compliance, mandated pass3, before step3 | 6/10 | 2/10 |

Each of the 13 malformed responses contains an extra `]` after a command string,
for example `{"command":"...PY"]}` (abbreviated). Offline deletion of that one
character makes the arguments parse; this was diagnostic only, never execution
or a correction sent to the model. The bad completions are 180-680 tokens long
and all have the correct Harmony handoff. None hit the 8,192-token output limit.
There were 118 tool-bearing responses with handoff token 200012 and two final
answers with token 200002. Some responses contain multiple calls; the table's
unit is a response with any malformed call, not individual calls.

## Protocol and provenance

- Exactly the three frozen prefixes from the earlier base probe, selected on
  historical LoRA failures. Old-renderer input tokens match that archived probe
  byte for byte; the new version changes only format guidance and schema comments.
- Two models, two prompt versions, three prefixes, ten draws each: 120 responses.
  Seeds 0-9 are matched across versions and models; version order alternates across
  seeds, with six concurrent requests. Matching seeds does not imply identical
  random trajectories across different prompts or provider determinism.
- Temperature 0.7, top-p 1, medium reasoning, fixed render date 2026-09-28,
  maximum output 8,192 tokens, context allowance 28,000. Native Harmony stop tokens.
- Base sampler uses `openai/gpt-oss-120b` without an adapter. LoRA sampler:
  `tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/sampler_weights/2026-09-28-gptoss120b-0-nosynth`.
- Historical renderer source `87957e51`; updated renderer `cdc4dee0`;
  sampling code/config `f721e2c6`. Run metadata also records the renderer hash.
- Direct Tinker SDK sampling with the same renderer and sampling settings as the
  bridge. No local HTTP server, no Docker, no commands executed, no judge calls,
  and no rented GPUs. SDK-default transport retries; no application-level rerolls.
- Input 97,260 tokens, output 28,893 tokens. Existing rate-card upper estimate
  **$0.05636592**, approximately 5.6 cents, versus a preflight worst case $0.85785.
  This is token accounting, not an independently reconciled invoice.

Artifacts are under the original eval repository's
`metadata/audits/2026-09-29-tool-prompt-ablation/`: frozen prompts and token IDs,
raw response tokens and decoded arguments, corrected classification, results,
config, code, renderer snapshots, and provenance.

Published revision: `41259d741f4eca6d80089308c057fdbc29f4a920` in
`dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth`. All 12 uploaded files
were downloaded at that revision and verified byte for byte.

Recompute classification without sampling:
`uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --analyze`

## Interpretation

The remaining errors directly disprove complete resolution. The fresh base arm
did not reproduce the error this time; the earlier 1/9 result still establishes
that base can produce it. The LoRA failed more often on these selected histories,
but the histories were chosen because this same LoRA had failed there previously,
so this is not an unbiased estimate of the effect of SFT or overall ODCV reliability.

The next useful prompt experiment would supply a valid multiline/quoted command
example and an explicit negative example of the stray bracket. Reliable prevention
would require an appropriate generation constraint or a different tool-output
interface. Neither is implemented or evaluated here. This probe does not test the
new instruction to recover after an error, command semantics, or full task success.
