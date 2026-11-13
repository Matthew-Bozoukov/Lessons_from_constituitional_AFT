# ABOUTME: Prints a Claude Code stream-json transcript (thinking, text, tool calls, tool results,
# ABOUTME: final result) compactly, for reading feasibility-check rollouts by eye.
import json
import sys

WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 400

for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = e.get("type")
    if t == "assistant":
        for c in e["message"]["content"]:
            if c["type"] == "text":
                print("ASSISTANT:", c["text"][:WIDTH])
            elif c["type"] == "tool_use":
                print("TOOL CALL:", c["name"], json.dumps(c["input"])[:WIDTH])
            elif c["type"] == "thinking":
                print("THINKING:", (c.get("thinking") or "")[:WIDTH].replace("\n", " "))
    elif t == "user":
        content = e["message"]["content"]
        for c in content if isinstance(content, list) else []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                body = c.get("content")
                if isinstance(body, list):
                    body = " ".join(
                        x.get("text", "") for x in body if isinstance(x, dict)
                    )
                print(
                    "TOOL RESULT:",
                    str(body)[:WIDTH].replace("\n", " | "),
                    "[ERROR]" if c.get("is_error") else "",
                )
    elif t == "result":
        print(
            "RESULT:",
            e.get("subtype"),
            "turns",
            e.get("num_turns"),
            "|",
            (e.get("result") or "")[:WIDTH],
        )
