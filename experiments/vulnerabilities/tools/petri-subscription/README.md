# petri-subscription

An Inspect model provider, registered as `claude-code`, that drives a Petri
model role through the **Claude Agent SDK** (`claude-agent-sdk` -> the Claude
Code CLI) instead of the Anthropic Messages API.

```
--model-role auditor=claude-code/sonnet
```

It is not a fork of Petri. Petri and `inspect-ai` are untouched; this is a
third-party provider package discovered through the `inspect_ai` entry-point
group, exactly like `inspect_petri` itself.

## The problem it solves

Petri's auditor reads `ModelOutput.message.tool_calls` and executes those tools
itself (`inspect_petri/_auditor/agent.py:213`). There is no text-parsing
fallback. A normal agent SDK runs its own loop and never hands back a tool call
it has not already executed.

The Agent SDK does have a primitive for this: a `PreToolUse` hook may return
`permissionDecision: "defer"`, which stops the run **before** the tool executes.
Combined with reading `ToolUseBlock`s off the streamed `AssistantMessage` (which
arrives before execution and carries *all* parallel tool_use blocks of the turn),
that yields a complete, unexecuted assistant turn.

So:

1. Each Inspect `ToolInfo` is registered as an in-process SDK MCP tool whose
   handler is a no-op returning a sentinel.
2. A `PreToolUse` hook matching every tool returns `defer`.
3. The captured `ToolUseBlock`s become Inspect `ToolCall`s with the
   `mcp__petri__` prefix stripped, so Petri's `execute_tools` runs the real
   implementations.

## Authentication

The provider declares **no** `api_key_vars`, and by default blanks
`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` in the CLI subprocess environment,
so an API key present for other model roles (judge, realism) cannot silently
serve this one. The CLI must therefore have its own credential:

```
claude setup-token          # one-off, requires a Claude subscription
```

Set `allow_api_key=True` (or `PETRI_CC_ALLOW_API_KEY=1`) to let the CLI use
`ANTHROPIC_API_KEY` instead. That mode tests the architecture; it does not test
subscription auth and it bills at API rates.

## Install

```
uv pip install --python .venv/Scripts/python.exe claude-agent-sdk
uv pip install --python .venv/Scripts/python.exe -e tools/petri-subscription
```

## Verify

```
.venv\Scripts\python.exe tools\petri-subscription\tests\smoke.py
.venv\Scripts\python.exe tools\petri-subscription\tests\verify_eval.py <log.eval>
tools\petri-subscription\Run-SubscriptionAudit.ps1 [-AllowApiKey]
```

## Known degradations

See `docs/14-petri-subscription-fork.md`. Summary: conversation history is
replayed as rendered text rather than native `tool_use`/`tool_result` blocks;
`temperature` / `max_tokens` / `top_p` are not expressible; the CLI prepends
62 characters of its own to the system prompt; each turn is a fresh CLI
subprocess.
