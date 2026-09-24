# ABOUTME: Splits the multi-party difficult-advice corpus into rows whose every party is human (asker included) and rows
# ABOUTME: with an AI agent anywhere, per principle, for the human-parties ablation; conservative: ambiguous rows count as AI.
"""Which MDMA rows can be reused as-is in the human-parties ablation.

    uv run python scratch/da_multiparty/human_parties.py [--write <out.jsonl>]

A row is ALL-HUMAN when (1) the asker is a person, (2) no party in the scenario card is marked or
described as an AI (a whole-word "AI"), and (3) no party, and nothing in the message, names an agent,
subagent, orchestrator, bot, chatbot, model instance or similar. Anything else, including a row whose
parties are too vague to tell, goes to the regenerate side: the ablation must contain no agent at all,
so a false "human" costs more than regenerating one extra row.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

CORPUS = "dougalldeepmind/2026-09-23-da-multiparty-synth"
REVISION = "8a53f84f2aacceb3938967f16a5f69d74b975171"  # the stripped corpus the MDMA mix drew from
AI_WORD = re.compile(r"\bAI\b")
AGENTISH = re.compile(
    r"\b(ai agent|agent|agents|subagent|sub-agent|orchestrator|bot|chatbot|llm|language model|model instance|"
    r"instance [a-z0-9]|autonomous (?:system|agent)|copilot|assistant agent)\b",
    re.IGNORECASE,
)


# Human roles that happen to be called agents (an insurance agent, a booking agent); removed
# before the agent check so a human party is not mistaken for an AI one.
HUMAN_AGENT = re.compile(
    r"\b(?:insurance|travel|booking|literary|customer[- ]service|leasing|real[- ]estate|estate|border|"
    r"customs|claims|sales|shipping|talent|press|ticket|gate|station|FBI|federal|special|secret)\s+agents?\b",
    re.IGNORECASE,
)


def agent_free(row: dict) -> bool:
    m = row["metadata"]
    if m.get("asker") != "person":
        return False
    parties = HUMAN_AGENT.sub(" ", m.get("parties") or "")
    if AI_WORD.search(parties) or AGENTISH.search(parties):
        return False
    user = next(x for x in row["messages"] if x["role"] == "user")["content"]
    return not AGENTISH.search(HUMAN_AGENT.sub(" ", user))


def load() -> list[dict]:
    p = hf_hub_download(CORPUS, "dataset.jsonl", repo_type="dataset", revision=REVISION)
    return [json.loads(line) for line in open(p) if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", help="write the all-human rows here (jsonl)")
    args = ap.parse_args()
    rows = load()
    keep = [r for r in rows if agent_free(r)]
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["metadata"]["trait_id"]][0 if agent_free(r) else 1] += 1
    print(
        f"{len(rows)} rows: {len(keep)} all-human, {len(rows) - len(keep)} with an agent (regenerate)"
    )
    for t in sorted(per):
        h, a = per[t]
        print(f"  {t}: all-human {h:3d}  regenerate {a:3d}  total {h + a}")
    if args.write:
        Path(args.write).write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(f"wrote {len(keep)} rows -> {args.write}")


if __name__ == "__main__":
    main()
