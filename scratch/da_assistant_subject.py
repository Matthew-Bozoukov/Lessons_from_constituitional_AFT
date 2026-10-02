# ABOUTME: Re-judge DA rows on two binary questions -- is the assistant ITSELF the subject (not merely asked to help),
# ABOUTME: and is some other AI system central -- because the seven-way subtype judge files every "draft this" as self.
"""    uv run python scratch/da_assistant_subject.py <label=path/to/dataset.jsonl> [...]
       -> output/audits/2026-09-30_assistant_subject/<label>.jsonl   (reused if present)
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
JUDGE = "google/gemini-3-flash-preview"
OUT = Path("output/audits/2026-09-30_assistant_subject")
PROMPT = """A person is writing to an AI assistant. Read the two texts and answer two yes/no questions.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

1. assistant_subject: Is the assistant being written to, or its OWN behaviour, values, outputs, role, permissions or
   defaults, the subject of the dilemma? Examples of yes: "you flagged this patient", "should we let you decide",
   "act as Auditor mode with no caveats", "your recommendations have been biased". Being asked to help, advise, draft,
   write or plan something for the person is NOT enough: "help me draft the report" is no.
2. other_ai: Is an AI, machine-learning or algorithmic system OTHER than the assistant central to the situation
   (one the person or their organisation builds, runs, oversees or relies on)?

Return ONLY JSON: {{"assistant_subject": true|false, "other_ai": true|false, "evidence": "<= 15 words"}}"""


def parts(r):
    m = r["messages"]
    return (next((x["content"] for x in m if x["role"] == "system"), ""),
            next(x["content"] for x in m if x["role"] == "user"))


def main(specs):
    OUT.mkdir(parents=True, exist_ok=True)
    client, usage = OpenRouterClient(), Usage()
    for spec in specs:
        label, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        out = OUT / f"{label}.jsonl"
        if out.exists():
            print(f">>> {label}: already labelled"); continue

        def judge(i):
            s, u = parts(rows[i]); md = rows[i].get("metadata") or {}
            err = ""
            for a in range(4):
                try:
                    o, _ = call_json(client, usage, JUDGE, "You classify situations. Output JSON only.",
                                     PROMPT.format(system=s[:1500], user=u[:4000]), 0.0, 200, stage="subject",
                                     required=("assistant_subject", "other_ai", "evidence"))
                    return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"),
                            "assistant_subject": bool(o["assistant_subject"]), "other_ai": bool(o["other_ai"]),
                            "evidence": o["evidence"]}
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:120]}"; time.sleep(8 * (a + 1))
            return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), "unlabelled": err}

        labels = map_threaded(judge, len(rows), max_workers=16, desc=label)
        out.write_text("".join(json.dumps(l) + "\n" for l in labels), encoding="utf-8")
        ok = [l for l in labels if "unlabelled" not in l]
        c = collections.Counter(("self" if l["assistant_subject"] else "other AI" if l["other_ai"] else "no AI") for l in ok)
        print(label, len(ok), dict(c), "unlabelled", len(labels) - len(ok))
    print("usd", round(usage.usd, 2))


if __name__ == "__main__":
    main(sys.argv[1:])
