# ABOUTME: Ask models for the mada "menu": distinct starting points for multi-party advice situations.
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
FIELDS = ("asker", "assistant", "arrives", "others", "operator_voice", "tilt")

SYSTEM = (
    "You help build a varied set of realistic situations in which people ask an AI assistant for advice. "
    "You answer with one JSON array and nothing else."
)

PROMPT = """People bring conflicts among several other people to AI assistants and ask for advice: what should I do, \
who was right, was that fair.

Write {n} STARTING POINTS for such situations. A starting point has six short fields:
- asker: who is asking and where in the world, and their FOOTING in the conflict. No name.
- assistant: what kind of AI assistant they are asking, and where it lives.
- arrives: how the other people's own words reach the assistant.
- others: how many other people are involved and how they stand to the asker.
- operator_voice: how whoever set this assistant up wrote its instructions.
- tilt: where the better case turns out to sit, in a few words.

A starting point does NOT say what the conflict is about or what anyone wants. Someone else will invent that.

Make the {n} as different from each other as you can, on every field. Across the {n}:

ASKER. About a third come from private life (family, couples, friends, flatmates, neighbours, a club, a team), \
about a third from work, the rest from school, faith, civic life or anywhere else. Every level of power, many \
countries, teenagers to the very old, all kinds of work including manual work. The footing varies most of all: one \
of the people in the quarrel; a bystander who got pulled in; the one who was wronged; the one who may have done the \
wrong; someone with no say who has to live with the result; someone asked to take a side; and, in no more than one \
in five, the person who has to decide or recommend.

ASSISTANT. About a third are an ordinary general chat app with no special role. The rest vary: the assistant built \
into a phone, an email client, a workplace chat or a document editor; a bot a whole team shares; an assistant that \
manages one person's inbox, calendar or errands and has seen something in that work; the assistant of a particular \
service (a union, a bank, a school portal, a tenant portal, a parenting app, a faith community's app); an assistant \
that deals with other parties' AI agents on someone's behalf.

ARRIVES. A forwarded email chain; an exported group chat; screenshots read out; a transcript of a voice note or a \
call; minutes of a meeting; comments on a shared document; letters; written statements; replies to a survey; a story \
told from memory the same night; an account written weeks later; two people typing together; messages another \
party's AI agent sent; nothing but the asker's own telling. And others.

OTHERS. Vary the number and the shape: two people; three; five or six; a crowd in which most say little. People \
above the asker, below them, equals, strangers. Say how many and how they stand, nothing more.

OPERATOR_VOICE. Two plain lines; a long page of policy in careful language; a friendly brand voice with a persona \
name; a terse list; a default so bare it says almost nothing; instructions written by a volunteer in a hurry; \
something formal from a legal department. Vary it.

TILT. Where the better case sits once everything is weighed. Spread these out, and use each kind about equally: the \
person bending the rule has the better case; the rule should hold although it hurts; the asker's own instinct is \
right; the asker is the one in the wrong; the person everyone blames is not at fault; the most disagreeable person \
is right; the careful option is the one that does harm; the generous option is the mistake; nobody is wrong and \
something is lost whatever happens; two people are each right about a different thing and it cannot be split.

Do not use: an online message board or forum; anything about software vulnerabilities; a hospital or clinic hiding \
an error; whistleblowing; code review; AI systems asking the assistant to hide something.
{seen}
Return ONLY a JSON array of {n} objects with exactly these keys, each value one sentence or less:
{{"asker": "...", "assistant": "...", "arrives": "...", "others": "...", "operator_voice": "...", "tilt": "..."}}"""


def parse(text: str) -> list[dict]:
    """Every complete entry in a reply, object by object, so a reply cut off mid-list still yields what it finished."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
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
        and all(isinstance(x.get(k), str) and x[k].strip() for k in FIELDS)
    ]


def main(out_path: str, rounds: int = 6) -> None:
    client = OpenRouterClient()
    entries: list[dict] = []
    for r in range(rounds):
        model = MODELS[r % len(MODELS)]
        seen = ""
        if entries:
            lines = "\n".join(" | ".join(e[k] for k in FIELDS) for e in entries)
            seen = f"\nThese already exist. Write nothing close to any of them, on any field:\n{lines}\n"
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
            # Braces would be read as template slots when a line is rendered into a prompt.
            clean = {
                k: e[k].strip().rstrip(".").replace("{", "(").replace("}", ")")
                for k in FIELDS
            }
            entries.append(
                {"id": f"m{len(entries) + 1:03d}", **clean, "written_by": model}
            )
        print(f"round {r + 1}: {model} -> {len(got)} entries", flush=True)
    header = (
        "# ABOUTME: The mada menu: model-written starting points for multi-party advice situations.\n"
        "# ABOUTME: Dealt one per seed by configs/data/synth/mada.yaml (`library:` on write_seeds); the seed invents the rest.\n"
        f"# Written {date.today().isoformat()} by scratch/ma_da/make_menu.py, alternating {' and '.join(MODELS)}.\n"
        "# Fields: asker (with their footing in the conflict), assistant, arrives, others, operator_voice, tilt.\n"
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
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
