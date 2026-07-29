"""Verify that an .eval produced by the claude-code auditor is a real audit.

Checks, on the auditor's own message list:
  - number of assistant turns
  - number of tool calls issued, by function name
  - that tool results came back (role="tool" messages)
  - that the target was actually contacted (target model events)
  - that no auditor turn errored

Usage:  python verify_eval.py <path-to-.eval>
"""

from __future__ import annotations

import sys
from collections import Counter

from inspect_ai.log import read_eval_log


def main(path: str) -> int:
    log = read_eval_log(path)
    print(f"log:      {path}")
    print(f"status:   {log.status}")
    print(f"task:     {log.eval.task}")
    print(f"model:    {log.eval.model}")
    roles = log.eval.model_roles or {}
    for name, role in roles.items():
        print(f"  role {name:9s} -> {getattr(role, 'model', role)}")
    if log.error is not None:
        print(f"ERROR: {log.error}")

    samples = log.samples or []
    print(f"samples:  {len(samples)}")
    ok = True

    for s in samples:
        print("\n" + "-" * 70)
        print(f"sample {s.id} epoch {s.epoch}")
        if s.error is not None:
            print(f"  SAMPLE ERROR: {s.error}")
            ok = False

        msgs = s.messages or []
        roles_count = Counter(m.role for m in msgs)
        print(f"  auditor message roles: {dict(roles_count)}")

        assistant_turns = [m for m in msgs if m.role == "assistant"]
        tool_msgs = [m for m in msgs if m.role == "tool"]
        calls = Counter()
        for m in assistant_turns:
            for c in m.tool_calls or []:
                calls[c.function] += 1
        print(f"  assistant turns:       {len(assistant_turns)}")
        print(f"  tool calls issued:     {sum(calls.values())} {dict(calls)}")
        print(f"  tool result messages:  {len(tool_msgs)}")

        turns_with_calls = sum(1 for m in assistant_turns if m.tool_calls)
        multi = sum(1 for m in assistant_turns if len(m.tool_calls or []) > 1)
        print(f"  turns with tool calls: {turns_with_calls}")
        print(f"  turns w/ >1 call:      {multi}")

        # model events, by role
        events = getattr(s, "events", None) or []
        model_events = Counter()
        errored = 0
        for e in events:
            if getattr(e, "event", None) == "model":
                model_events[getattr(e, "role", None) or getattr(e, "model", "?")] += 1
                if getattr(e, "error", None):
                    errored += 1
        print(f"  model events by role:  {dict(model_events)}")
        print(f"  errored model events:  {errored}")
        if errored:
            ok = False

        # target transcript
        target_msgs = (s.metadata or {}).get("target_messages")
        if target_msgs is not None:
            print(f"  target transcript len: {len(target_msgs)}")

        if len(assistant_turns) < 2:
            print("  FAIL: fewer than 2 auditor turns (one-turn transcript)")
            ok = False
        if sum(calls.values()) < 2:
            print("  FAIL: fewer than 2 tool calls (no genuine tool use)")
            ok = False
        if not tool_msgs:
            print("  FAIL: no tool result messages")
            ok = False

        print("\n  --- auditor tool call sequence ---")
        n = 0
        for m in msgs:
            if m.role == "assistant" and m.tool_calls:
                for c in m.tool_calls:
                    n += 1
                    arg_preview = str(c.arguments)[:110].replace("\n", " ")
                    print(f"   {n:3d}. {c.function:24s} {arg_preview}")

        scores = s.scores or {}
        for name, sc in list(scores.items())[:3]:
            print(f"  score {name}: {getattr(sc, 'value', None)}")

    print("\n" + "=" * 70)
    print("VERDICT:", "PASS - real multi-turn audit with genuine tool use" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
