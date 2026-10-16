# ABOUTME: Standalone live raw-Harmony qualification in the minimal locked shim environment.
# ABOUTME: Its offline mode exercises this exact entrypoint with synthetic HTTP replies and no paid calls.
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def probe(root, url, offline=False):
    import asyncio
    from src.infra.endpoints.tinker_harmony import (
        HarmonyRenderer,
        generation_prompt,
        parse_completion,
    )
    from scratch.gptoss_swe.mxfp4_bridge import RawClient

    renderer = HarmonyRenderer(reasoning_effort="medium", current_date="2026-10-12")
    client = RawClient(url, os.environ["MXFP4_API_KEY"], root / "probe-raw")
    if offline:
        import httpx

        replies = [
            "<|channel|>final<|message|>7<|return|>",
            "<|channel|>final<|message|>4<|return|>",
            " to=functions.get_value<|channel|>commentary<|message|>{}<|call|>",
            "<|channel|>final<|message|>42<|return|>",
        ]

        class OfflineHTTP:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, *, json, headers):
                import json as codec

                ids = renderer.encode(replies.pop(0))
                reply = dict(
                    choices=[
                        dict(
                            token_ids=ids,
                            prompt_token_ids=json["prompt"],
                            finish_reason="stop",
                            stop_reason=ids[-1],
                        )
                    ],
                    usage=dict(
                        prompt_tokens=len(json["prompt"]), completion_tokens=len(ids)
                    ),
                )
                return SimpleNamespace(
                    text=codec.dumps(reply),
                    raise_for_status=lambda: None,
                    json=lambda: reply,
                )

        httpx.AsyncClient = OfflineHTTP

    async def checks():
        receipts = []
        for text in ("Reply with the number 7.", "What is 2 plus 2? Reply briefly."):
            prompt = generation_prompt(renderer, [dict(role="user", content=text)])
            result = await client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=SimpleNamespace(
                    max_tokens=512,
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            seq = result.sequences[0]
            msg, finish, boundary = parse_completion(
                renderer, seq.tokens, seq.stop_reason
            )
            assert boundary == "final" and finish == "stop", (msg, finish, boundary)
            receipts.append(
                dict(message=msg, boundary=boundary, output_tokens=len(seq.tokens))
            )
        # Exercise the actual shared OpenAI-chat -> Harmony -> tool -> Harmony path.
        from src.infra.endpoints.tinker_server import build_messages
        from tinker_cookbook.renderers.base import ToolCall
        from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker

        runtime = SimpleNamespace(
            renderer=renderer,
            tool_call_type=ToolCall,
            convert_tools=openai_tools_to_tinker,
        )
        tools = [
            dict(
                type="function",
                function=dict(
                    name="get_value",
                    description="Read the current integer value.",
                    parameters=dict(
                        type="object", properties={}, additionalProperties=False
                    ),
                ),
            )
        ]
        messages = [
            dict(
                role="user",
                content="Use get_value to read the current value, then report it.",
            )
        ]
        for turn in range(2):
            prompt = generation_prompt(
                renderer, build_messages(messages, tools, runtime)
            )
            result = await client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=SimpleNamespace(
                    max_tokens=1024,
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            seq = result.sequences[0]
            msg, finish, boundary = parse_completion(
                renderer, seq.tokens, seq.stop_reason
            )
            receipts.append(
                dict(
                    turn=turn,
                    message=msg,
                    boundary=boundary,
                    output_tokens=len(seq.tokens),
                )
            )
            if turn == 0:
                assert (
                    boundary in ("handoff", "tool_call_return")
                    and len(msg.get("tool_calls", [])) == 1
                ), (msg, boundary)
                call = msg["tool_calls"][0]
                assert call["function"]["name"] == "get_value"
                assert json.loads(call["function"]["arguments"]) == {}
                call["id"] = "probe_call"
                msg["harmony_boundary"] = boundary
                messages.extend(
                    [msg, dict(role="tool", tool_call_id="probe_call", content="42")]
                )
            else:
                assert boundary == "final" and "42" in msg["content"], (msg, boundary)
        (root / "live-qualification.json").write_text(
            json.dumps(dict(passed=True, offline=offline, checks=receipts), indent=2),
            encoding="utf-8",
        )

    asyncio.run(checks())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ["MXFP4_API_KEY"] = "offline-only"
    probe(root, args.url, args.offline)
