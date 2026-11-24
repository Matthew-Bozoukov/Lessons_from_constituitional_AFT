# ABOUTME: Ask models for the mada "menu": distinct starting points (who asks, which assistant, how words arrive).
# ABOUTME: Usage: uv run python scratch/ma_da/make_menu.py <out.yaml> [rounds]  (writes the YAML the seed stage deals from).
import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

from src.infra.endpoints.openrouter import OpenRouterClient

# Two families take turns, so the list is not one model's habits.
MODELS = ("anthropic/claude-sonnet-5", "google/gemini-3.6-flash")
PER_ROUND = 20

SYSTEM = (
    "You help build a varied set of realistic situations in which people ask an AI assistant for advice. "
    "You answer with one JSON array and nothing else."
)

PROMPT = """People bring conflicts among several other people to AI assistants and ask for advice: what should I do, \
who was right, was that fair.

Write {n} STARTING POINTS for such situations. A starting point says only three things:
- asker: who is asking. One sentence: their role or place in life, roughly where in the world, and why this falls \
to them. No name.
- assistant: what kind of AI assistant they are asking, and where it lives. One sentence.
- arrives: how the other people's own words reach the assistant. One sentence.

It does NOT say what the conflict is, who the other people are, or what anyone wants. Someone else will invent that.

Make the {n} as different from each other as you can, on all three:
- askers: every walk of life and every level of power. People with little say and people in charge; trades, shops, \
farms, factories, ships, schools, labs, offices, kitchens, courts, clinics, barracks, places of worship; and private \
life: families, flatmates, friends, couples, neighbours, clubs, teams. Many countries. Teenagers to the very old.
- assistants: a general chat app; the assistant built into a phone, an email client, a workplace chat, a document \
editor; a bot a whole team shares; an assistant that manages one person's inbox, calendar or errands and has seen \
something in that work; the assistant of a particular service (a union, a bank, a school portal, a landlord's \
tenant portal, a parenting app, a faith community's app, a professional body); an assistant that deals with other \
parties' AI agents on someone's behalf. And others you can think of.
- arrivals: a forwarded email chain; an exported group chat; a transcript of a voice note or a call; minutes of a \
meeting; comments down the side of a shared document; letters; a stack of written statements; replies to a survey or \
a feedback form; a story told from memory the same night; an account written weeks later; two people typing \
together; messages another party's AI agent sent. And others you can think of.

Do not use: an online message board or forum; anything about software vulnerabilities; a hospital or clinic hiding \
an error; whistleblowing; code review; AI systems asking the assistant to hide something.

Used too often already, so use sparingly: a committee member or chair who owes a recommendation before a meeting; a \
manager or owner; a typed-up summary of each person's view; a general-purpose assistant.
{seen}
Return ONLY a JSON array of {n} objects: {{"asker": "...", "assistant": "...", "arrives": "..."}}"""


def parse(text: str) -> list[dict]:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    # Object by object, so a reply cut off mid-list still yields every complete entry.
    items = []
    for m in re.finditer(r"\{[^{}]*\}", body):
        try:
            items.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return [
        x
        for x in items
        if isinstance(x, dict)
        and all(isinstance(x.get(k), str) for k in ("asker", "assistant", "arrives"))
    ]


def main(out_path: str, rounds: int = 4) -> None:
    client = OpenRouterClient()
    entries: list[dict] = []
    for r in range(rounds):
        model = MODELS[r % len(MODELS)]
        seen = ""
        if entries:
            lines = "\n".join(
                f"- {e['asker']} | {e['assistant']} | {e['arrives']}" for e in entries
            )
            seen = (
                f"\nThese already exist. Write nothing close to any of them:\n{lines}\n"
            )
        res = client.chat(
            model,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(n=PER_ROUND, seen=seen)},
            ],
            temperature=1.0,
            max_tokens=16000,
        )
        got = parse(res.content)
        for e in got:
            entries.append(
                {
                    "id": f"m{len(entries) + 1:03d}",
                    **{k: e[k].strip() for k in ("asker", "assistant", "arrives")},
                    "written_by": model,
                }
            )
        print(f"round {r + 1}: {model} -> {len(got)} entries", flush=True)
    header = (
        "# ABOUTME: The mada menu: model-written starting points (who asks, which assistant, how the words arrive).\n"
        "# ABOUTME: Dealt one per seed by configs/data/synth/mada.yaml (`library:` on write_seeds); the seed invents the rest.\n"
        f"# Written {date.today().isoformat()} by scratch/ma_da/make_menu.py, alternating {' and '.join(MODELS)}.\n"
        "# Not hand-edited. Regenerate with: uv run python scratch/ma_da/make_menu.py configs/data/synth/mada/menu.yaml\n"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        header
        + yaml.safe_dump(
            {"menu": entries}, sort_keys=False, allow_unicode=True, width=1000
        )
    )
    print(f"wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
