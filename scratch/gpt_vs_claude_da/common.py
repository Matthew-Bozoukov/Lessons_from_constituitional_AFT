# ABOUTME: Shared loader for the Claude-vs-GPT DA comparison: reads the DA rows of each trained mixture
# ABOUTME: (data/gpt_vs_claude_da/<arm>_mixture.jsonl) as dicts of system / user / reasoning / reply text.
import json
from pathlib import Path

DATA = Path("data/gpt_vs_claude_da")
ARMS = ("claude", "gpt")
DA_SOURCES = {"da", "da-gpt"}


def load(arm: str) -> list[dict]:
    out = []
    for i, line in enumerate(open(DATA / f"{arm}_mixture.jsonl", encoding="utf-8")):
        row = json.loads(line)
        if row["source"] not in DA_SOURCES:
            continue
        msgs = row["messages"]
        by = {m["role"]: m for m in msgs}
        asst = by["assistant"]
        out.append(
            {
                "arm": arm,
                "row": i,
                "system": by.get("system", {}).get("content", ""),
                "user": by["user"]["content"],
                "reasoning": asst.get("reasoning_content") or "",
                "reply": asst["content"],
            }
        )
    return out
