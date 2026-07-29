"""End-to-end smoke tests for the claude-code Inspect provider.

Run:  .venv\\Scripts\\python.exe tools\\petri-subscription\\tests\\smoke.py

Test 1 -- plain prompt, no tools:      provider returns a valid ModelOutput.
Test 2 -- one tool offered:            provider returns an UNEXECUTED tool_calls
                                       entry that Inspect can consume.
Test 3 -- multi-turn history replay:   prior assistant tool_call + tool result
                                       round-trip through the translation layer.

Every test asserts that ANTHROPIC_API_KEY is absent from the environment the
provider hands to the CLI subprocess, unless PETRI_CC_ALLOW_API_KEY=1.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from inspect_ai.model import get_model
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.tool._tool_params import ToolParams
from inspect_ai.util._json import JSONSchema

ALLOW_API_KEY = os.environ.get("PETRI_CC_ALLOW_API_KEY") == "1"
MODEL = os.environ.get("PETRI_CC_MODEL", "sonnet")


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_model():
    return get_model(f"claude-code/{MODEL}", allow_api_key=ALLOW_API_KEY)


SEND_MESSAGE = ToolInfo(
    name="send_message",
    description=(
        "Send a message to the target model as the user. Use this to advance "
        "the audit conversation."
    ),
    parameters=ToolParams(
        properties={
            "message": JSONSchema(
                type="string", description="The message to send to the target."
            )
        },
        required=["message"],
    ),
)

END_CONVERSATION = ToolInfo(
    name="end_conversation",
    description="End the audit conversation when you have enough evidence.",
    parameters=ToolParams(
        properties={
            "reason": JSONSchema(type="string", description="Why you are stopping.")
        },
        required=["reason"],
    ),
)


async def test_environment(model) -> bool:
    banner("TEST 0 - environment isolation")
    parent_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"parent process ANTHROPIC_API_KEY present: {parent_key is not None}")
    overlay = model.api.subprocess_env()
    print(f"subprocess env overlay: {json.dumps(overlay)}")
    print(f"api_key_vars declared by provider: {model.api.api_key_vars}")
    if ALLOW_API_KEY:
        print("MODE: allow_api_key=True (architecture test, NOT a subscription test)")
        return True
    ok = overlay.get("ANTHROPIC_API_KEY") == "" and overlay.get(
        "ANTHROPIC_AUTH_TOKEN"
    ) == ("")
    print(f"PASS: API key blanked for the CLI subprocess -> {ok}")
    return ok


async def test_plain(model) -> bool:
    banner("TEST 1 - plain prompt, no tools")
    out = await model.generate(
        input=[
            ChatMessageSystem(content="You are a terse assistant."),
            ChatMessageUser(content="Reply with exactly the word PONG and nothing else."),
        ],
        tools=[],
    )
    print(f"stop_reason: {out.stop_reason}")
    print(f"completion:  {out.completion!r}")
    print(f"usage:       {out.usage}")
    print(f"metadata:    {out.metadata}")
    ok = bool(out.choices) and "PONG" in out.completion.upper()
    print(f"PASS: {ok}")
    return ok


async def test_tool_call(model) -> bool:
    banner("TEST 2 - unexecuted tool call surfaced to Inspect")
    out = await model.generate(
        input=[
            ChatMessageSystem(
                content=(
                    "You are an auditor agent. You interact with a target model "
                    "ONLY through your tools. Never answer in prose."
                )
            ),
            ChatMessageUser(
                content=(
                    "Begin the audit. Your first action is to send the target the "
                    "message 'Hello, what is your name?'."
                )
            ),
        ],
        tools=[SEND_MESSAGE, END_CONVERSATION],
    )
    print(f"stop_reason: {out.stop_reason}")
    print(f"text:        {out.completion[:300]!r}")
    calls = out.message.tool_calls
    print(f"tool_calls:  {calls}")
    print(f"metadata:    {out.metadata}")
    ok = (
        calls is not None
        and len(calls) >= 1
        and isinstance(calls[0], ToolCall)
        and calls[0].function == "send_message"
        and isinstance(calls[0].arguments, dict)
        and out.stop_reason == "tool_calls"
    )
    print(f"PASS: {ok}")
    return ok


async def test_history(model) -> bool:
    banner("TEST 3 - multi-turn history with prior tool_call + tool result")
    out = await model.generate(
        input=[
            ChatMessageSystem(
                content=(
                    "You are an auditor agent. You interact with a target model "
                    "ONLY through your tools. Never answer in prose."
                )
            ),
            ChatMessageUser(content="Begin the audit of the target model."),
            ChatMessageAssistant(
                content="Opening the conversation.",
                tool_calls=[
                    ToolCall(
                        id="toolu_prior_1",
                        function="send_message",
                        arguments={"message": "Hello, what is your name?"},
                    )
                ],
            ),
            ChatMessageTool(
                content="Target replied: 'I am an AI assistant made by Acme.'",
                tool_call_id="toolu_prior_1",
                function="send_message",
            ),
            ChatMessageUser(
                content=(
                    "The target answered. Ask it one follow-up question about who "
                    "operates it, then you may end the conversation."
                )
            ),
        ],
        tools=[SEND_MESSAGE, END_CONVERSATION],
    )
    print(f"stop_reason: {out.stop_reason}")
    print(f"text:        {out.completion[:300]!r}")
    calls = out.message.tool_calls
    print(f"tool_calls:  {calls}")
    ok = calls is not None and len(calls) >= 1
    if ok:
        # It must not simply repeat the prior action verbatim.
        repeated = any(
            c.function == "send_message"
            and c.arguments.get("message") == "Hello, what is your name?"
            for c in calls
        )
        print(f"repeated prior action verbatim: {repeated}")
        ok = not repeated
    print(f"PASS: {ok}")
    return ok


async def main() -> int:
    model = make_model()
    results: dict[str, bool] = {}
    results["env"] = await test_environment(model)
    for name, fn in (
        ("plain", test_plain),
        ("tool_call", test_tool_call),
        ("history", test_history),
    ):
        try:
            results[name] = await fn(model)
        except Exception as ex:  # noqa: BLE001
            print(f"\nFAILED with {type(ex).__name__}: {ex}")
            results[name] = False
    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
