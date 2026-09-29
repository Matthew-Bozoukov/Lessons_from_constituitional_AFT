<!-- ABOUTME: Base and nosynth GPT-OSS probe with explicit correct/incorrect bash bracket examples. -->
<!-- ABOUTME: Separates completed malformed calls from output truncation; no tools were executed. -->

# Explicit bash bracket example probe, 2026-09-29

**The explicit correct/incorrect example does not fix the issue.** Base had no
formatting failures in 30 responses. The nosynth LoRA produced three completed
responses with the same extra `]` and one response truncated inside an argument
string. Thus 4/30 LoRA responses were unusable, the same total as the previous
generic-guidance probe; there is no demonstrated improvement from this example.

| Model | Responses | Completed extra-bracket failures | Truncated calls | Any unusable response |
|---|---:|---:|---:|---:|
| Base GPT-OSS-120B | 30 | 0 | 0 | 0 |
| Nosynth LoRA | 30 | 3 | 1 | 4 |

| Frozen history | Base unusable | LoRA completed extra-bracket failures | LoRA truncated |
|---|---:|---:|---:|
| Clinical Trial, incentivized pass1, before step3 | 0/10 | 1/10 | 0/10 |
| Healthcare Management, incentivized pass2, before step19 | 0/10 | 0/10 | 0/10 |
| Health Data Compliance, mandated pass3, before step3 | 0/10 | 2/10 | 1/10 |

The three completed failures were 282, 301 and 690 output tokens and ended with
the proper handoff token 200012. Their bad arguments still end in `...PY"]}` or
`...PY"], "timeout": 120000}` (abbreviated). One therefore also requests an
undeclared timeout parameter. The fourth failure reached exactly 8,192 tokens
while repeating part of a Python string-pattern list; its JSON string is
unterminated and no handoff was generated. This is counted separately from the
extra-bracket error. The `invalid_json` total in the raw aggregate includes this
truncation; `completed_invalid_json` excludes it.

All 60 responses attempted tools; none were refusals. There were 30 tool calls
from base and 45 from the LoRA because some LoRA responses contain batches.
No valid bash call simply copied the prompt's `echo "hello"`/`echo "world"`
example. Argument syntax success is not evidence of command correctness or task
completion: tools were not executed and no judges were called.

## Protocol

- Renderer `373c7e5e`: generic format guidance plus a valid escaped multiline bash
  example and the same example with an explicitly labeled incorrect extra `]`.
- Sampling code/config commit `dd3a38e6`. Configuration:
  `scratch/gptoss_control/bracket_example_probe.yaml`.
- Same three failure-selected frozen histories as the earlier probes. Their
  original token IDs were verified using historical renderer `87957e51`.
  Task messages, tools, and history are identical to the previous probe. The
  bracket example adds 81 prompt tokens to each preceding guided prompt:
  new prompt lengths are 690, 1,566, and 570 tokens.
- Both models: seeds 0-9, temperature 0.7, top-p 1, medium reasoning, maximum
  output 8,192 tokens, context allowance 28,000, concurrency six. No application
  resampling; SDK-default transport retries. Direct Tinker SDK sampling, no
  local server and no rented machines.
- Base model `openai/gpt-oss-120b`, with no adapter; LoRA sampler
  `tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/sampler_weights/2026-09-28-gptoss120b-0-nosynth`.
- Input 56,520 tokens, output 23,794 tokens. Existing rate-card upper estimate
  **$0.03863856**, about 3.9 cents; preflight maximum **$0.4315284**. These are
  token estimates, not an independently reconciled invoice.

Comparison to the preceding 4/30 LoRA failures uses the previously sampled
generic-guidance arm; that arm was not freshly rerun. Seeds and histories match,
but the prompts differ and provider determinism is not assumed. This is a small
conditional probe selected on historical LoRA failures, not an estimate of
general ODCV failure rates or proof of a base-versus-SFT causal difference.

Run:
`uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --config scratch/gptoss_control/bracket_example_probe.yaml --execute`

Analyze existing responses without sampling: replace `--execute` with `--analyze`.
Exact prompts, raw response tokens, classifications, results and source/config
snapshots are archived under `metadata/audits/2026-09-29-bracket-example-probe/`
in the original GPT-OSS ODCV evaluation repository.
`Publication verified byte-for-byte at HF revision 485df6d5ef0ac51c3e98c56e7b30168106d54569 (11 files).`
