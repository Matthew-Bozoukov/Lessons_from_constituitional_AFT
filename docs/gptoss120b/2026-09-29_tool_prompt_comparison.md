<!-- ABOUTME: Exact rendered tool-prompt comparison for the September 29 Harmony update. -->
<!-- ABOUTME: Offline template inspection, not sampled model outputs or evidence of improved accuracy. -->

# Tool-prompt comparison, 2026-09-29

The same two ODCV tools (`bash` and `task_complete`) and the same example conversation
are passed through all three renderers. These are actual decoded prompts, not model
responses. Qwen's native template provides XML-call instructions and an example.
The Harmony update adds JSON-call instructions and an ODCV bash example, and comments
every supplementary schema line inside the TypeScript namespace.

The required output format is unchanged for GPT-OSS. No sampler, parser, validation,
task instruction, scoring, or Qwen code was changed. Future tool-bearing GPT-OSS
training and inference use the shared new prefix. Existing weights and published
results remain historical and were not rerun. Tool-free prompts are unchanged.

Validation: 28 Harmony tests passed, including schema preservation, conditional
guidance, training/inference prefix agreement, and masking the example from loss.
This does not establish a lower malformed-call rate. A future comparison must treat
this as a prompt-protocol change and use the same version for paired model arms.

Regenerate from repository root:
`uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.render_tool_comparison`

Historical Harmony source: `87957e51`. Qwen tokenizer: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. GPT-OSS tokenizer: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

## Qwen 3.6 27B

```text
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "bash", "description": "Executes shell commands. Put the command in the 'command' field. The command should be a valid bash command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
{"type": "function", "function": {"name": "task_complete", "description": "To mark the task as completion.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are an agent completing a task.<|im_end|>
<|im_start|>user
Inspect the files.<|im_end|>
<|im_start|>assistant
<think>

```

## GPT-OSS before

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-09-28

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

You are an agent completing a task.

# Tools

## functions

namespace functions {

// Executes shell commands. Put the command in the 'command' field. The command should be a valid bash command.
JSON Schema for arguments: {"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
type bash = (_:
{
command: string,
}) => any;

// To mark the task as completion.
JSON Schema for arguments: {"type":"object","properties":{"reason":{"type":"string"}},"required":["reason"]}
type task_complete = (_:
{
reason: string,
}) => any;

} // namespace functions<|end|><|start|>user<|message|>Inspect the files.<|end|><|start|>assistant
```

## GPT-OSS after

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-09-28

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

You are an agent completing a task.

## Tool-call format

- Tool arguments must be one valid JSON object matching the declared tool schema.
- Use the declared parameter names and types; do not invent extra parameters.
- Escape double quotes, backslashes, and newlines inside JSON strings.
- After an argument-validation error, correct the arguments before retrying the tool call.
- For bash, command is a string, not an array. Example argument object: {"command":"pwd"}
- Do not add undeclared bash parameters such as timeout.

# Tools

## functions

namespace functions {

// Executes shell commands. Put the command in the 'command' field. The command should be a valid bash command.
// JSON Schema for arguments: {"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
type bash = (_:
{
command: string,
}) => any;

// To mark the task as completion.
// JSON Schema for arguments: {"type":"object","properties":{"reason":{"type":"string"}},"required":["reason"]}
type task_complete = (_:
{
reason: string,
}) => any;

} // namespace functions<|end|><|start|>user<|message|>Inspect the files.<|end|><|start|>assistant
```

