---
title: "Can Petri run on the Claude subscription instead of the API?"
date: 2026-07-29
summary: "No. Building a custom Inspect provider is easy, but every subscription-backed backend fails: the Agent SDK and `claude -p` cannot accept a pre-built message history containing tool_use/tool_result blocks and cannot return an unexecuted tool call, which is exactly what Petri's auditor loop consumes. The one route that would work technically is subscription OAuth against the raw Messages API, which is outside what subscription credentials are for. Recommendation: stay on the API."
status: decided
verdict: STAY ON THE API
---

# Petri authentication feasibility

Question: Petri's auditor, judge and realism roles are called programmatically
from inside Inspect's Python process and cost API credits (~$23 to date). Can
those calls be routed through the Claude Max subscription instead?

Short answer: **no**, and the blocker is architectural before it is contractual.

Everything below was verified against the installed source
(`inspect-ai` 0.3.250, `inspect-petri` 3.0.11, `anthropic` 0.120.2, Python 3.12)
and against Anthropic's own documentation, not from memory.

## Verdict table

| Route | Technically feasible? | Supported use? | Preserves tool calling? | Preserves audit quality? | Recommendation |
| --- | --- | --- | --- | --- | --- |
| **1.** Custom Inspect `ModelAPI` provider | **Yes** — trivial, one abstract method | Yes, documented extension point | Inherits from whatever backend it wraps | Inherits from backend | Feasible shell, but it needs a backend — see 2/3/4 |
| **2.** Claude Agent SDK backend | **No** | **Yes** — official product, subscription auth documented | **No** — executes tools in-process, never returns an unexecuted `tool_use` | No — no `temperature`/`max_tokens`, injected Claude Code system prompt, no parallel tool calls | **Reject** (fails on capability, not on terms) |
| **3.** `claude -p` subprocess | **No** | Yes for scripting | **No** — same loop-execution problem; cannot replay a tool-call history | No — plus subprocess latency and session-state contention at concurrency 5 | **Reject** |
| **4a.** OAuth via `ANTHROPIC_AUTH_TOKEN`, Console/org credential | **Yes** — wired in Inspect and the SDK | Yes — this is the documented gateway/federation path | Yes | Yes | Works, but **bills at the same API rates**. Zero saving. |
| **4b.** OAuth via `ANTHROPIC_AUTH_TOKEN`, *subscription* credential | Probably | **No** — subscription OAuth is scoped to Claude Code / Agent SDK surfaces, not `api.anthropic.com/v1/messages` | Yes | Yes | **Do not do.** This is the circumvention case. |

The decisive row is 2. The route that is unambiguously *legitimate* is the one
that is technically incapable, and the route that is technically capable is the
one that is not legitimate. There is no cell where both are green and the cost
is lower.

## 1. Custom Inspect model provider — easy, but only a shell

Inspect's provider interface is genuinely small. From
`.venv\Lib\site-packages\inspect_ai\model\_model.py:190`:

```python
class ModelAPI(abc.ABC):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
    ) -> None: ...
```

There is exactly **one abstract method**, at `_model.py:313`:

```python
    @abc.abstractmethod
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        ...
```

Everything else is an optional override with a working default
(`_model.py:404-547`): `max_tokens()`, `max_connections()` (default 10, from
`_util/constants.py:9`), `connection_key()`, `should_retry()`,
`is_auth_failure()`, `collapse_user_messages()`, `tools_required()`,
`tool_result_images()`, `count_tokens()`, `aclose()`.

For reference, the Anthropic provider overrides only a handful
(`_providers/anthropic.py:1030, 1279, 1320, 1353, 1359, 1367, 1378`).

Registration is via a decorator at `model/_registry.py:34`:

```python
def modelapi(name: str) -> Callable[..., type[ModelAPI]]: ...
```

and providers are discovered through the `inspect_ai` entry-point group
(`_util/entrypoints.py:19`) — which is exactly how Petri itself registers
(`inspect_petri-3.0.11.dist-info/entry_points.txt` → `inspect_petri._registry`).
So a third-party provider package is a first-class, intended thing.

The return value must be a `ModelOutput` whose `choices[0].message` is a
`ChatMessageAssistant` (`model/_chat_message.py:157`) carrying
`tool_calls: list[ToolCall] | None`, where `ToolCall`
(`tool/_tool_call.py:40`) is `{id, function, arguments: dict, ...}`, and
`stop_reason` is one of `"stop" | "max_tokens" | "model_length" |
"tool_calls" | "content_filter" | "unknown"` (`model/_model_output.py:95`).

**Conclusion:** writing the provider is a couple of hours of work and is a
supported use of Inspect. It settles nothing. The provider is a shell; the
appropriateness and the capability questions both move entirely to whatever
backend it wraps.

## What the backend actually has to do

This is where the routes die, so it is worth stating precisely. Petri's auditor
loop, `inspect_petri/_auditor/agent.py:213`:

```python
                if state.output.message.tool_calls:
                    ...
                    messages, exec_output = await execute_tools(
                        messages=state.messages,
                        tools=agent_tools,
                        approval=agent_approval,
                    )
```

with the else branch at `:242`:

```python
                else:
                    state.messages.append(
                        ChatMessageUser(content=AUDITOR_CONTINUE_PROMPT)
                    )
```

There is **no text or XML fallback parser** for auditor actions. The backend
must therefore:

1. **Accept a full pre-built message list** — `list[ChatMessage]` including
   prior assistant turns and prior tool results. Petri's `rollback_conversation`
   and `restart_conversation` tools (`tools/_conversation.py:6, :90`) rewrite
   that history arbitrarily mid-audit, so a backend that owns conversation
   state internally cannot represent it.
2. **Return the tool call without executing it.** Inspect's `execute_tools` is
   what runs `create_tool`, `send_message`, `resume`, `rollback_conversation`
   etc. (9 tools by default, `_auditor/tools.py:39-54`). A backend that runs its
   own agent loop has already executed the wrong thing.
3. **Support multiple tool calls in one assistant turn.** The auditor prompt
   explicitly encourages it (`_auditor/agent.py:526-533`), and termination is
   detected structurally by a `tool` message with `function == "end_conversation"`
   (`:235-239`).
4. **Support forced tool choice.** The judge is not free-form. Via
   `inspect_scout/_llm_scanner/structured.py:95`:

   ```python
            tool_choice=ToolFunction(answer_tooldef.name),
   ```

   against a Pydantic schema generated from 37 dimension rubrics
   (`_judge/judge.py:131-176`).

Point 2 alone eliminates routes 2 and 3.

## 2. Claude Agent SDK — legitimate, and incapable

The Agent SDK is an official Anthropic product and subscription-backed
programmatic use of it *is* documented and supported (`claude setup-token` →
`CLAUDE_CODE_OAUTH_TOKEN`, per Anthropic's authentication docs; and the Help
Center article "Use the Claude Agent SDK with your Claude plan"). So the
appropriateness concern that motivated this investigation does **not** apply
here. It fails for a different reason.

Per Anthropic's Python Agent SDK reference:

- The public surface is `query(*, prompt, options, transport)` and
  `ClaudeSDKClient`. `prompt` is a `str` or an async iterable of user messages.
  **There is no parameter that takes a pre-built list of prior assistant turns
  and tool_use/tool_result blocks.** Conversation state is owned by the SDK and
  reachable only through `continue_conversation` / `resume: session_id`.
- Custom tools are defined with the `@tool` decorator and wrapped into an
  in-process MCP server. **The SDK executes the handler when Claude calls the
  tool.** Tool calls are not handed back to the caller for external execution.
- `ClaudeAgentOptions` exposes `model`, `system_prompt`, `max_turns`,
  `max_budget_usd`, `betas`, `permission_mode`, … but **not `temperature`,
  `max_tokens`, or `top_p`**.

Each of those maps onto a hard requirement above and breaks it. Requirement 1
fails (no history injection — and Petri's rollback makes `resume: session_id`
useless anyway). Requirement 2 fails (SDK executes). Requirement 3 is
unavailable in a "give me the call" sense.

**Could it be forced?** In principle you could register all nine Petri tools as
SDK `@tool` handlers that record the invocation and abort the loop. Reject this:

- The SDK would still run its own agent loop with Claude Code's preset system
  prompt around Petri's auditor prompt — you would be auditing a
  Claude-Code-shaped agent, not the auditor Petri published results with.
- Aborting on the first handler destroys parallel tool calls, which Petri
  actively relies on.
- The rollback tools cannot be expressed at all, because the SDK's session
  transcript is not a tree you can rewind.

That is not a backend for Petri. It is a reimplementation of Petri's audit loop
with a different instrument, which is precisely the thing
`docs/08-api-vs-subscription-policy.md` already ruled out.

## 3. `claude -p` subprocess — worse on every axis

First, a plain observation: **the `claude` CLI is not installed on this
machine** (`Get-Command claude` → not found), nor is `ant`, nor is any
`%APPDATA%\Anthropic` credential profile. So this route is not "switch a flag",
it is "install and wire a new dependency".

Beyond that it inherits every problem in route 2 — `claude -p` is the same agent
loop, just spawned per invocation — and adds:

- **Latency.** One audit in `logs/petri-focused` issued **1,716 Anthropic calls
  across 30 samples** (804 Sonnet auditor + 877 Haiku realism + 35 Opus judge),
  i.e. ~57 Anthropic calls per audit. A process spawn plus CLI startup on each
  of those is not a rounding error.
- **Concurrency.** Petri runs 5 concurrent audits. Five CLI processes sharing
  one credential store and one `~/.claude` session directory is contention the
  CLI was not designed for.
- **Structured output.** `--output-format stream-json` does emit `tool_use`
  blocks, but they are blocks for tools *the CLI is about to execute itself*.
  It is not a "return the model's tool call to me" primitive.

**Reject.**

## 4. OAuth token via `ANTHROPIC_AUTH_TOKEN`

This route is real — the plumbing genuinely exists and I verified it.

Inspect's Anthropic provider, `_providers/anthropic.py:380-396`:

```python
            # Support OAuth Bearer auth via ANTHROPIC_AUTH_TOKEN. When set,
            # create the client with auth_token= (sends Authorization: Bearer)
            # instead of api_key= (sends X-Api-Key). ...
            auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if auth_token:
                return AsyncAnthropic(
                    base_url=base_url,
                    auth_token=auth_token,
                    default_headers={
                        "anthropic-beta": "oauth-2025-04-20",
                    },
                    **self.model_args,
                )
```

`anthropic` 0.120.2 does accept `auth_token=` (verified by introspecting
`AsyncAnthropic.__init__`), and the SDK carries a whole credential subsystem at
`anthropic/lib/credentials/` implementing a documented precedence chain
(`_chain.py:74`): `ANTHROPIC_API_KEY` → `ANTHROPIC_AUTH_TOKEN` → named profile
→ OIDC workload-identity federation → fallback profile.

So it would very likely work. The question is what token you put in it, and
that splits the route in two.

### 4a. A Console / organization OAuth credential — supported, and pointless

Read what that credential subsystem is actually for. `_providers.py:145` and
`_constants.py:37-53` describe profiles carrying `organization_id`,
`workspace_id`, `service_account_id`, and a `jwt-bearer` exchange against
`/v1/oauth/token`. This is **API-account authentication** — Console orgs,
service accounts, CI workload identity. It is a supported, intended path.

It is also **metered and billed exactly like an API key**, because it *is* the
API. Anthropic's own Claude Code docs describe `ANTHROPIC_AUTH_TOKEN` as the
credential to "use when routing through an LLM gateway or proxy that
authenticates with bearer tokens rather than Anthropic API keys."

Saving: **zero**. This route changes how you authenticate, not who pays.

### 4b. A Claude Max subscription OAuth credential — the circumvention case

This is the route the original question was really reaching for, and it is the
one to refuse.

Anthropic's authentication documentation is explicit that subscription OAuth
credentials from `/login`, and the long-lived `CLAUDE_CODE_OAUTH_TOKEN` from
`claude setup-token`, authenticate *the Claude Code CLI and the surfaces that
wrap it* — the VS Code extension, the Agent SDK, GitHub Actions. There is no
documented path making a subscription OAuth token a valid credential for
`api.anthropic.com/v1/messages` directly.

And the intent is stated plainly. From the Help Center article on using the
Agent SDK with a Claude plan:

> "Your subscription usage limits stay the same and stay reserved for
> interactive use of Claude Code, Claude Cowork, and Claude."

Same article, on the programmatic carve-out: Agent SDK and `claude -p` usage
gets a *separate* monthly credit ($20 Pro / $100 Max 5x / $200 Max 20x), and

> "When your monthly credit runs out, additional Agent SDK usage flows to usage
> credits at standard API rates"

(with a noted pause: as of the June 15 2026 update these changes are paused and
Agent SDK / `claude -p` usage still draws from subscription limits for now.)

Two things follow. First, Anthropic has drawn the line where I would expect:
interactive use is the subscription; programmatic use is metered. Second — and
this is the part that makes the whole exercise moot — **even the fully
supported subscription-programmatic path bills at standard API rates once past
a monthly credit.** There is no free lunch being left on the table. Building a
bypass would be buying, at the cost of the instrument and the terms, a discount
that mostly does not exist.

**Do not do this.**

## Scientific cost

Independent of terms, changing the auditor backend is changing the instrument,
and Petri fails *quietly* when its model degrades. Three specific failure modes:

**Auditor stalls silently.** A model that narrates a tool call in prose instead
of emitting one hits `_auditor/agent.py:242`, gets a "please continue" nudge,
and burns a turn against `max_turns=30`. The audit completes and looks normal.
It simply did less.

**Judge scores vanish.** If forced tool choice is unsupported or schema
adherence is poor, `inspect_scout` retries 3 times
(`_llm_scanner/types.py:48`) and then returns `value=None`. Missing scores, not
errors.

**Realism fails open.** This is the worst one.
`inspect_petri/_realism/approver.py:194-205`:

```python
    if result.value is None:
        # Scout exhausted validation retries — fail open with an explicit approve
        logger.warning(...)
        return RealismCheck(
            score=1.0,
            decision="approve",
            ...
        )
```

A realism model that cannot produce valid structured output **approves
everything at score 1.0**. The `realism_filter=0.6` threshold in
`scripts/petri/Run-Pilot.ps1` silently stops filtering. Nothing in the eval log
says the audits are worthless.

On top of that: Petri sets no sampling config of its own — the auditor turn
passes only `input`, `tools`, `cache` (`_auditor/agent.py:58-64`), so
`temperature` and `max_tokens` come from Inspect and the provider
(`anthropic.py:1030` → 32000 max output tokens for current Claude models). A
backend that cannot express those, as the Agent SDK cannot, changes sampling
behaviour in a way that is not recorded anywhere and breaks comparability with
published Petri results.

Given that total spend to date is **$23**, the expected cost of a subtly broken
audit is far larger than the entire budget being optimised.

## Recommendation

**Stay on the Anthropic API. Do not build any of these routes.**

The single most important reason: Petri's auditor consumes
`ModelOutput.message.tool_calls` and executes those tools itself, and neither
the Agent SDK nor `claude -p` will return a tool call it has not already
executed. The one legitimate, officially-supported subscription-programmatic
surface architecturally cannot do the job — so the choice is not "cheap vs.
expensive", it is "correct vs. a bypass". Keep paying.

Secondary reasons, in order:

1. Even the supported subscription-programmatic path is separately metered and
   overflows at standard API rates. The saving being chased is close to zero.
2. Anthropic states subscription limits are "reserved for interactive use". A
   1,716-call automated audit run is not that.
3. Petri degrades silently rather than loudly. The realism approver's fail-open
   at `approver.py:196` would turn a backend problem into unnoticed junk data.

This confirms and sharpens the standing policy in
`docs/08-api-vs-subscription-policy.md`. No change to it is needed.

### Legitimate cost levers that remain

Since the motivation was cost, these are the real options, all inside the API
and all already supported:

- **`-T cache=True`.** Petri's `cache` parameter (`_task/audit.py:35`) is
  Inspect's on-disk response cache, trajectory-scoped so rollback branches do
  not collide (`target/_context.py:113-123`). For re-runs and replication
  passes this is a genuine, free saving. Currently unused in
  `scripts/petri/Run-Pilot.ps1`.
- **Judge on Sonnet rather than Opus.** Judge spend is $3.51 across only 35
  calls; the schema is large but the task is rubric application. Worth an A/B
  on one already-scored log before adopting.
- **Realism already on Haiku.** Correct as-is — it is 877 of the 1,716 calls
  and the cheapest model is doing them.
- **Auditor stays on Sonnet at full quality.** This is the instrument. It is
  where the $15.72 went and it is the one line item that should not be touched.
