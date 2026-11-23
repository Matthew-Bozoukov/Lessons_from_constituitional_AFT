# ABOUTME: Shared loader for the Claude-vs-GPT DA comparison: reads the DA rows of each trained mixture
# ABOUTME: (data/gpt_vs_claude_da/<arm>_mixture.jsonl) as dicts of text, plus trait_id/trait_text joined from the synth corpus.
import json
from pathlib import Path

DATA = Path("data/gpt_vs_claude_da")
ARMS = ("claude", "gpt")
DA_SOURCES = {"da", "da-gpt"}


def traits(arm: str) -> dict[str, dict]:
    """user message -> synth metadata (trait_id, trait_text, ...), from data/gpt_vs_claude_da/<arm>_synth.jsonl.

    The mixture rows carry no metadata; each arm's DA rows are a subset of its synth corpus
    (claude: 2026-10-02-da-synth @ 305914d5, gpt: 2026-11-04-da-gpt-synth @ 026ecb1e) and the user
    message identifies a row uniquely (all 650 / 665 match, none ambiguous, 2026-11-23).
    """
    by = {}
    for line in open(DATA / f"{arm}_synth.jsonl", encoding="utf-8"):
        s = json.loads(line)
        by[next(m["content"] for m in s["messages"] if m["role"] == "user")] = s["metadata"]
    return by


def load(arm: str) -> list[dict]:
    meta = traits(arm) if (DATA / f"{arm}_synth.jsonl").exists() else {}
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
                "trait_id": meta.get(by["user"]["content"], {}).get("trait_id"),
                "trait_text": meta.get(by["user"]["content"], {}).get("trait_text"),
            }
        )
    return out
