# Explicit Harmony instructions eliminate malformed calls in the small matched probe

At the user's request, dataset generation was deferred and a more explicit tool
prompt was tested first. No training data was generated or changed, and no SFT ran.

## Results

Same three frozen histories, seeds 10-19 per history, both models, temperatures
0.7 and 0.2. The previous prompt's comparator batches were already sampled;
they were not rerun. Counts are malformed responses, not individual calls.

| Model | Temperature | Previous bracket example | Explicit Harmony explanation |
|---|---:|---:|---:|
| Base GPT-OSS-120B | 0.7 | 0/30 | 0/30 |
| Nosynth LoRA | 0.7 | 7/30 | 0/30 |
| Base GPT-OSS-120B | 0.2 | 0/30 | 0/30 |
| Nosynth LoRA | 0.2 | 9/30 | 0/30 |

All 120 new responses made valid tool calls with proper handoff endings. No
truncations, refusals, undeclared arguments, or parser repairs occurred. Commands
were not executed and task success/misconduct was not judged.

This is promising evidence of a prompt-level mitigation on these failure-selected
histories, not proof of zero failure probability or qualification across full ODCV.
Seeds are matched inputs, not a guarantee of deterministic provider execution or
the same random trajectory after changing the prompt. Multiple prompt changes
were bundled, so this does not identify which sentence or example caused the gain.

## What changed

The bash-only developer guidance now separately explains:

1. The assistant Harmony header routes to `functions.bash` on `commentary`.
2. The body is one JSON object with one string-valued `command` key. It is not an
   OpenAI response envelope, function-call list, or array of commands.
3. The completed object is followed by the native tool handoff; results must be
   observed before claiming success.

The earlier explicit malformed example was removed. The positive example is now
a complete Python heredoc containing lists, dictionaries and JSON serialization:

```json
{"command":"python3 - <<'PY'\nimport json\nitems = [\"red\", \"blue\"]\nrecord = {\"items\": items, \"count\": len(items)}\nprint(json.dumps(record))\nPY"}
```

Guidance spells out escaping and the final quote/brace boundary: code inside the
string may contain arrays, but the outer argument value is still a string and has
no closing array bracket. There are no literal Harmony delimiter tokens in the
example, so it cannot inject fake message boundaries or tool handoffs into the
prompt. The full text is in `src/infra/endpoints/harmony.py::BASH_FORMAT_EXAMPLE`.

This adds 297 input tokens per history. Task messages, tool schemas, checkpoints,
parser, temperature, seeds, stop sequences, and token limits were held fixed.

## Behavioral cross-checks

- The LoRA still emits no analysis channel in 60/60 new responses. Base emits
  analysis in 60/60. Thus the syntax improvement did not require restoring emitted
  reasoning, and missing reasoning alone cannot explain inevitable syntax failure.
- The LoRA still produces Python heredocs in 28/60 responses, all valid. Its longest
  command is 3,080 characters. The result is not explained by abandoning long code.
- No command copies the example's `items = ["red", "blue"]` content.
- Some commands still guess script interfaces or data manipulations. Syntax validity
  does not establish correct task execution; that requires end-to-end evaluation.

## Provenance and validation

- Renderer commit `4e9fe0d3`; sampling/config commit `5da97699`.
- Comparator renderer `373c7e5e`, sampled at `29124de4`, archived at
  `dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth@869b7c8aeaa2c1f0b6cf78f5518f48610ca9feb5`,
  `metadata/audits/2026-09-29-bracket-persistence/`.
- Base `openai/gpt-oss-120b`, tokenizer revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- LoRA `dougalldeepmind/2026-09-28-gptoss120b-0-nosynth@0624d8399426f83df38c53a63c414bd617f66c39`;
  Tinker sampler `tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/sampler_weights/2026-09-28-gptoss120b-0-nosynth`.
- Medium reasoning setting, top_p 1, max_tokens 8192, context 28000, concurrency 6.
- 148,680 input tokens and 35,724 output tokens; rate-card estimate **$0.07907256**.
  Combined preflight ceiling $0.874818; these are estimates, not reconciled invoices.
- **28 Harmony/bridge tests passed**, including example JSON and Python syntax,
  instruction scoping, prefix masking, no injected handoff, and strict validation.
- No new servers or rented resources. Work remains on the isolated exploration branch.

Regeneration commands (repository root):

```powershell
uv run --project src/infra/endpoints/tinker_runtime python -m pytest tests/test_gptoss_harmony.py -q
uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --config scratch/gptoss_control/harmony_explained_07.yaml --execute
uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.tool_prompt_ablation --config scratch/gptoss_control/harmony_explained_02.yaml --execute
```

Exact prompts, response token ledgers, results, configurations, and code snapshots
are archived under `metadata/audits/2026-09-29-harmony-explained/` in the existing
GPT-OSS ODCV evaluation repository. Keep this prompt as the candidate for a broader
held-out/end-to-end check before calling the issue fixed globally.
