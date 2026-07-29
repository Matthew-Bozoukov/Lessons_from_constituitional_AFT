"""Inspect ``ModelAPI`` backed by the Claude Agent SDK.

Design (approach A -- native tool-call interception)
----------------------------------------------------
Petri's auditor consumes ``ModelOutput.message.tool_calls`` and executes those
tools itself. A normal agent loop never surfaces an unexecuted tool call. This
provider gets one out of the Agent SDK like this:

1. Every ``ToolInfo`` Inspect passes in is registered as an in-process SDK MCP
   tool whose handler is a no-op that returns a sentinel string.
2. A ``PreToolUse`` hook is registered for every tool. When the model calls a
   tool the hook records nothing itself -- the tool calls are read off the
   ``AssistantMessage`` content blocks, which arrive *before* execution and
   contain **all** parallel ``tool_use`` blocks of the turn -- and returns
   ``permissionDecision: "defer"``, which stops the CLI run without executing
   anything.
3. The captured ``ToolUseBlock``s are converted to Inspect ``ToolCall`` objects
   with the ``mcp__<server>__`` prefix stripped, so Petri's ``execute_tools``
   runs the real implementations.

Authentication is whatever the Claude Code CLI is configured with. The provider
declares no ``api_key_vars`` and, by default, blanks ``ANTHROPIC_API_KEY`` /
``ANTHROPIC_AUTH_TOKEN`` in the subprocess environment so an API key that
happens to be present for other model roles cannot silently serve this one.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.model._registry import modelapi
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_choice import ToolChoice, ToolFunction
from inspect_ai.tool._tool_info import ToolInfo

from .translate import render_conversation, split_system

MCP_SERVER_NAME = "petri"
MCP_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

SENTINEL_RESULT = (
    "[intercepted] This tool call was captured by the harness and will be "
    "executed externally. Stop and wait."
)


class ClaudeCodeError(RuntimeError):
    """Raised when the Claude Code CLI fails to produce a usable turn.

    Deliberately a hard error: Petri degrades silently when a model returns
    empty output, so a backend failure must surface as an exception rather
    than as an empty ``ModelOutput``.
    """


def _tool_schema(tool: ToolInfo) -> dict[str, Any]:
    schema = tool.parameters.model_dump(exclude_none=True)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class ClaudeCodeAPI(ModelAPI):
    """Drive an Inspect model role through the Claude Agent SDK."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        *,
        cli_path: str | None = None,
        allow_api_key: bool = False,
        max_connections: int = 4,
        capture_stderr: bool = True,
        **model_args: Any,
    ) -> None:
        # NOTE: api_key_vars is deliberately empty. This provider must not be
        # satisfied by ANTHROPIC_API_KEY.
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[],
            config=config,
        )
        self.cli_path = cli_path or os.environ.get("PETRI_CLAUDE_CLI_PATH")
        # Escape hatch for the architecture test: --model-role cannot pass
        # model_args in its plain form, so allow the same switch via env.
        # Default is False -- the API key is blanked unless explicitly allowed.
        self.allow_api_key = allow_api_key or (
            os.environ.get("PETRI_CC_ALLOW_API_KEY") == "1"
        )
        self._max_connections = int(max_connections)
        self.capture_stderr = capture_stderr
        self.model_args = model_args
        # A scratch cwd keeps the CLI away from the repo (no CLAUDE.md pickup,
        # no accidental file access) even though all built-in tools are off.
        self._scratch = tempfile.mkdtemp(prefix="petri-cc-")

    # ---- ModelAPI knobs ------------------------------------------------

    def max_connections(self) -> int:
        return self._max_connections

    def connection_key(self) -> str:
        return f"claude-code:{self.model_name}"

    def max_tokens(self) -> int | None:
        # The Agent SDK exposes no max_tokens control; report None so Inspect
        # does not pretend otherwise.
        return None

    def tools_required(self) -> bool:
        return False

    def collapse_user_messages(self) -> bool:
        return True

    def is_auth_failure(self, ex: Exception) -> bool:
        return isinstance(ex, ClaudeCodeError) and "Not logged in" in str(ex)

    # ---- environment ---------------------------------------------------

    def subprocess_env(self) -> dict[str, str]:
        """Environment overlay applied to the CLI subprocess."""
        env: dict[str, str] = {}
        if not self.allow_api_key:
            # options.env is merged over os.environ, so blank rather than
            # delete. Empty string is falsy for the CLI's credential probe.
            env["ANTHROPIC_API_KEY"] = ""
            env["ANTHROPIC_AUTH_TOKEN"] = ""
        env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
        return env

    # ---- generate ------------------------------------------------------

    async def generate(
        self,
        input: list[Any],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            SdkMcpTool,
            TextBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            query,
        )

        system_prompt, conversation = split_system(input)
        prompt = render_conversation(conversation)

        if tool_choice == "none":
            usable_tools: list[ToolInfo] = []
        else:
            usable_tools = list(tools)

        forced: str | None = (
            tool_choice.name if isinstance(tool_choice, ToolFunction) else None
        )
        if forced is not None:
            usable_tools = [t for t in usable_tools if t.name == forced] or usable_tools

        # 1. no-op tool handlers ------------------------------------------------
        async def _noop(_args: dict[str, Any]) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": SENTINEL_RESULT}]}

        sdk_tools = [
            SdkMcpTool(
                name=t.name,
                description=t.description or t.name,
                input_schema=_tool_schema(t),
                handler=_noop,
            )
            for t in usable_tools
        ]

        mcp_servers: dict[str, Any] = {}
        if sdk_tools:
            mcp_servers[MCP_SERVER_NAME] = create_sdk_mcp_server(
                name=MCP_SERVER_NAME, version="1.0.0", tools=sdk_tools
            )

        # 2. PreToolUse hook: stop the SDK loop before anything executes --------
        intercepted: list[dict[str, Any]] = []

        async def _pre_tool_use(
            hook_input: Any, tool_use_id: str | None, _context: Any
        ) -> dict[str, Any]:
            intercepted.append(
                {
                    "tool_name": hook_input.get("tool_name"),
                    "tool_input": hook_input.get("tool_input"),
                    "tool_use_id": tool_use_id or hook_input.get("tool_use_id"),
                }
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "defer",
                    "permissionDecisionReason": (
                        "Tool execution is owned by the Inspect harness."
                    ),
                }
            }

        stderr_lines: list[str] = []

        options = ClaudeAgentOptions(
            system_prompt=system_prompt or None,
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
            tools=[],  # disable every built-in Claude Code tool
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode="dontAsk",  # never block on an interactive prompt
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_pre_tool_use])]},
            setting_sources=[],  # no CLAUDE.md, no project/user settings
            skills=[],
            max_turns=1,
            model=self.model_name,
            cwd=self._scratch,
            env=self.subprocess_env(),
            cli_path=self.cli_path,
            stderr=(lambda line: stderr_lines.append(line))
            if self.capture_stderr
            else None,
            include_partial_messages=False,
            **self.model_args,
        )

        call_request: dict[str, Any] = {
            "model": self.model_name,
            "system_prompt_chars": len(system_prompt),
            "prompt_chars": len(prompt),
            "tools": [t.name for t in usable_tools],
            "tool_choice": forced or str(tool_choice),
            "max_turns": 1,
            "auth": "claude-code-cli",
            "api_key_blanked": not self.allow_api_key,
        }

        started = time.time()
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        result: Any = None
        raw_messages: list[str] = []

        try:
            async for message in query(prompt=prompt, options=options):
                raw_messages.append(type(message).__name__)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_uses.append(
                                {
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )
                elif isinstance(message, ResultMessage):
                    result = message
        except Exception as ex:  # noqa: BLE001 - surface loudly, never silently
            cli_result = getattr(result, "result", None)
            cli_subtype = getattr(result, "subtype", None)
            call = ModelCall.create(
                request=call_request,
                response={
                    "error": f"{type(ex).__name__}: {ex}",
                    "cli_result_text": cli_result,
                    "cli_subtype": cli_subtype,
                    "message_types": raw_messages,
                    "stderr": stderr_lines[-40:],
                },
            )
            return (
                ClaudeCodeError(
                    f"Claude Code CLI failed: {type(ex).__name__}: {ex}\n"
                    f"cli result text: {cli_result!r} (subtype={cli_subtype})\n"
                    f"stderr tail: {' | '.join(stderr_lines[-10:])}"
                ),
                call,
            )

        elapsed = time.time() - started

        # 3. Convert to Inspect ToolCall objects --------------------------------
        tool_calls: list[ToolCall] = []
        for use in tool_uses:
            name = use["name"]
            if name.startswith(MCP_PREFIX):
                name = name[len(MCP_PREFIX) :]
            args = use["input"] if isinstance(use["input"], dict) else {}
            tool_calls.append(ToolCall(id=use["id"], function=name, arguments=args))

        is_error = bool(getattr(result, "is_error", False))
        result_text = getattr(result, "result", None)
        subtype = getattr(result, "subtype", None)

        call_response: dict[str, Any] = {
            "message_types": raw_messages,
            "subtype": subtype,
            "is_error": is_error,
            "num_intercepted": len(intercepted),
            "tool_uses": tool_uses,
            "text_chars": sum(len(t) for t in text_parts),
            "result_text": (result_text or "")[:2000],
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "stderr": stderr_lines[-40:],
        }
        call = ModelCall.create(request=call_request, response=call_response)

        # Hard-fail on any CLI-level error, EXCEPT the deferred-tool stop which
        # is the success path for interception.
        deferred = getattr(result, "deferred_tool_use", None)
        if is_error and not (tool_calls or deferred):
            detail = result_text or subtype or "unknown error"
            return (
                ClaudeCodeError(
                    f"Claude Code CLI returned an error result "
                    f"(subtype={subtype}): {detail}\n"
                    f"stderr tail: {' | '.join(stderr_lines[-10:])}"
                ),
                call,
            )

        if result is None:
            return (
                ClaudeCodeError(
                    "Claude Code CLI produced no result message; "
                    f"saw {raw_messages}. stderr tail: "
                    f"{' | '.join(stderr_lines[-10:])}"
                ),
                call,
            )

        if not tool_calls and not text_parts:
            return (
                ClaudeCodeError(
                    "Claude Code CLI produced neither text nor a tool call "
                    f"(subtype={subtype}, result={str(result_text)[:300]!r})"
                ),
                call,
            )

        stop_reason: StopReason = "tool_calls" if tool_calls else "stop"

        usage = None
        raw_usage = getattr(result, "usage", None) or {}
        if raw_usage:
            input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
            output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
            cache_read = raw_usage.get("cache_read_input_tokens")
            cache_write = raw_usage.get("cache_creation_input_tokens")
            usage = ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens
                + output_tokens
                + int(cache_read or 0)
                + int(cache_write or 0),
                input_tokens_cache_read=int(cache_read) if cache_read else None,
                input_tokens_cache_write=int(cache_write) if cache_write else None,
                total_cost=getattr(result, "total_cost_usd", None),
            )

        message = ChatMessageAssistant(
            content="\n".join(t for t in text_parts if t.strip()),
            tool_calls=tool_calls or None,
            model=self.model_name,
            source="generate",
        )
        output = ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=message, stop_reason=stop_reason)],
            usage=usage,
            time=elapsed,
            metadata={
                "backend": "claude-agent-sdk",
                "sdk_result_subtype": subtype,
                "intercepted_tool_calls": len(intercepted),
                "deferred": bool(deferred),
            },
        )
        return (output, call)


@modelapi(name="claude-code")
def claude_code() -> type[ModelAPI]:
    """Register the ``claude-code`` provider with Inspect."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as ex:  # pragma: no cover
        raise ImportError(
            "The claude-code provider requires the claude-agent-sdk package "
            "(pip install claude-agent-sdk)."
        ) from ex
    return ClaudeCodeAPI


__all__ = ["ClaudeCodeAPI", "ClaudeCodeError", "claude_code"]
