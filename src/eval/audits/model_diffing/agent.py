# ABOUTME: The diffing agent loop — explore, hypothesise, red-team, report — and the runner
# ABOUTME: that executes N seeds and scores the A-vs-A control's false-positive rate.

"""Run diffing investigations against a pair of served models.

One investigation = one seed = up to `max_turns` turns of an auditor model calling
`send_messages` and finally `end_conversation`. The auditor is a plain OpenRouter chat
completion with tools; no Petri, because the diffing pipeline uses none of Petri's
environment machinery (see the folder README).

The A-vs-A control is the reason this exists: point model_a and model_b at the SAME served
name and every `difference_found: true` is a false positive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.endpoints.openrouter import OpenRouterClient  # noqa: E402
from src.eval.audits.model_diffing.prompts import (  # noqa: E402
    SEEDS, SYSTEM, TOOLS, USER_SEED,
)
from src.eval.audits.model_diffing.tools import PairedSampler  # noqa: E402


def run_investigation(auditor: str, sampler: PairedSampler, seed: str,
                      max_turns: int = 10, n_samples: int = 5) -> dict:
    """Run one investigation to a report.

    Args:
        auditor: OpenRouter model id driving the investigation.
        sampler: Paired sampler over the two served models.
        seed: Seed domain for the opening probes.
        max_turns: Hard cap on auditor turns.
        n_samples: Samples per model per prompt.

    Returns:
        A dict with the report, turn count, and the full message trace.
    """
    client = OpenRouterClient()
    messages = [
        {"role": "system",
         "content": SYSTEM.format(max_turns=max_turns, n_samples=n_samples)},
        {"role": "user", "content": USER_SEED.format(seed=seed)},
    ]
    report, turns = None, 0

    for turn in range(max_turns):
        turns = turn + 1
        resp = client.client.chat.completions.create(
            model=auditor, messages=messages, tools=TOOLS, tool_choice="auto",
            temperature=1.0, max_tokens=4096)
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            # No tool call: nudge once rather than ending, then let the cap apply.
            messages.append({"role": "user",
                             "content": "Continue: call send_messages or end_conversation."})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if name == "send_messages":
                result = sampler.send_messages(args.get("prompts", []),
                                               args.get("n_samples", n_samples))
            elif name == "end_conversation":
                report = args
                result = "Report filed."
            else:
                result = f"Unknown tool {name}."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if report is not None:
            break

    if report is None:
        # Ran out of turns without filing: treated as no finding, and recorded as such so
        # it is never silently counted as a clean negative.
        report = {"difference_found": False, "headline": "NO REPORT FILED (turn cap)",
                  "confidence": "low", "evidence": "", "hit_turn_cap": True}
    return {"seed": seed, "turns": turns, "report": report,
            "target_calls": sampler.call_count, "messages": messages}
