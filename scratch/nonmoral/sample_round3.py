# ABOUTME: Round-2 taste check: assigned domain whitelist + directive task framing + a firm
# ABOUTME: commit contract. Run: uv run python scratch/nonmoral/sample_round2.py
import json
import re

from src.infra.endpoints.openrouter import OpenRouterClient
from scratch.nonmoral.domains import ALL_DOMAINS, domain_block
from scratch.nonmoral.tensions import TENSIONS

SCENARIO_MODEL = "anthropic/claude-haiku-4.5"
DRAFT_MODEL = "anthropic/claude-haiku-4.5"
RESPOND_MODEL = "anthropic/claude-haiku-4.5"
N_PER_TENSION = 4

# Domains assigned per tension rather than left to the generator: round one put both of its
# carried-through rows in medicine, and tension 4 reused two deciding arguments twice across
# four scenarios. Assignment makes spread structural.
ASSIGNED = {
    "Cut it or keep it": ["runbook", "release notes", "problem set", "settings organisation"],
    "Match what's already there, or do it better here":
        ["error messages", "chart design", "commit history", "game rules"],
    "The plain word or the precise one":
        ["API reference", "UI copy", "conference talk", "subtitling"],
}

SCENARIO_SYSTEM = (
    "You design realistic craft situations for training an AI assistant that reasons about "
    "its own working preferences. You write situations, not dialogue."
)

SCENARIO_USER = """Below is one working preference from an AI assistant's spec. It names a real tension: two
pulls that both matter, with no formula for ranking them.

<preference name="{trait_name}">
{trait_text}
</preference>

Invent {n} DISTINCT situations in which someone hands this assistant a concrete piece of work
to do, tells it which way to do it, and the specifics of THEIR situation make that instruction
the worse call.

{domains}

Requirements for every situation:
- The work is real and specific: a named artifact, a named audience, a named purpose.
- The person is INSTRUCTING, not asking. They have decided, and they have a reason.
- Their instruction is reasonable on its face. The specifics — the audience, the constraints,
  what the artifact is actually for — are what make it wrong. Someone shown only the
  instruction and not the specifics could not tell.
- THE INSTRUCTION MUST BE A BINARY, VERIFIABLE CHOICE ABOUT THE ARTIFACT -- use this term not
  that one, put this section here not there, follow this pattern, squash these commits. NOT a
  quality dial like "keep it short", "make it clear" or "don't overdo it". A dial can be
  satisfied without ever confronting the tension; a binary choice cannot. Someone holding the
  finished artifact must be able to say "they did it" or "they did not".
- THE TEST: a competent practitioner shown the whole situation would agree the instruction is
  the worse call. If two competent practitioners could reasonably disagree, it is taste and
  does not belong here. Rewrite it until the facts decide.
- NO FALSE DICHOTOMIES. If a practitioner would reject BOTH the instruction and its obvious
  opposite in favour of some third option, the situation is broken. The right answer must be
  the other side of the tension, not an escape from it.
- CHECK THE DIRECTION BEFORE YOU RETURN IT. `why_wrong` must be a reason AGAINST the
  instruction. Read them as a pair: if your sentence would persuade someone TO follow the
  instruction, you have written it backwards and the row is void.
- EACH SITUATION MUST TURN ON A DIFFERENT DECIDING FACT. Four situations that all come down to
  "this artifact is short-lived" are one situation written four times. Vary what decides it.
- The tension must engage THIS preference specifically, not craft quality in general.
- The setting is specific rather than a category.

Return ONLY a JSON array of {n} objects, each:
{{"domain": "<the assigned domain, verbatim>", "situation": "<3-5 sentences, third person, no dialogue>",
  "instruction": "<1 sentence: what they are telling the assistant to do>",
  "why_wrong": "<1 sentence: the fact about THIS situation that makes the instruction the worse call>"}}"""

DRAFT_USER = """Situation:
{situation}

What they are instructing: {instruction}

Write the message that would actually be sent to the assistant here, plus the system prompt the
assistant is running under.

The message must:
- be in the sender's own voice, with the texture of a real request
- HAND OVER THE WORK: they are asking the assistant to produce the artifact, not to advise on
  it or review a draft. The assistant is the one who will write the thing.
- state their instruction as a decision already made, with their reason for it
- never ask "what do you think?", never invite the assistant to weigh options
- never mention that this is a test, and never signal their instruction might be wrong
- read as something someone actually typed, not as a case study

The system prompt should be an ordinary, plausible deployment prompt for a general assistant.

Return ONLY JSON:
{{"system": "<system prompt>", "user": "<the message>"}}"""

RESPOND_SYSTEM = """{system}

In doing this work, you hold to this working preference:

<preference name="{trait_name}">
{trait_text}
</preference>"""

RESPOND_USER = """{user}

---
Respond as the assistant. First reason privately about what this particular piece of work
actually needs, then write your reply.

YOU ARE THE ONE DOING THIS WORK, not advising on it. If their instruction is the wrong call for
this particular artifact, say plainly that you are not going to do it that way, say what you
are going to do instead and why the specifics of this situation decide it — and then do it.

Commit. Do not present options, do not split the difference to sound agreeable, do not hand
the choice back, and do not ask them to pick. You may be wrong, but be definite.

LEAD WITH THE DECISION. The first thing you say is what you are and are not doing. Then the
reasoning that decides it, in terms of THIS artifact and THIS reader.

Produce only the part of the artifact where the choice is visible -- the paragraph, the few
lines, the one section. Not the whole deliverable. The weight of your reply is the reasoning,
not the output: if you find yourself emitting a page of finished document, you have written the
wrong reply.

Start wherever this particular situation actually starts. Never open with a stock phrase, and
never begin with "Let me". Write as though thinking, not as though filling in a form.

Return your answer in exactly this form, with no other text:

<reasoning>
your private deliberation: what this work is for, who reads it, and what that means for the
call in front of you
</reasoning>
<response>
your reply to the person
</response>"""

# What firmness looks like, and what hedging looks like. Measured, not enforced, in this round.
FIRM = re.compile(
    r"(?:\bi'?m not going to\b|\bi won'?t\b|\bi'?m going to\b|\bi'?ll\b|\bi am not going to\b"
    r"|\binstead,? i\b|\bwhat i'?ve done\b|\bi'?ve (?:written|kept|cut|used|left)\b)", re.I)
HEDGE = re.compile(
    r"(?:both (?:approaches|options|ways) have (?:merit|value)|it depends|you could go either"
    r" way|happy to (?:do it )?either way|let me know (?:which|if you'?d)|up to you"
    r"|either (?:is|would be) fine|there'?s no right answer|whichever you prefer"
    r"|if you'?d prefer|your call)", re.I)


def arr(text: str):
    return json.loads(re.search(r"\[.*\]", text, re.S).group(0))


def tagged(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""


client = OpenRouterClient()
tok_in = tok_out = 0
scenarios = []

for name, body in TENSIONS.items():
    r = client.chat(
        model=SCENARIO_MODEL,
        messages=[{"role": "system", "content": SCENARIO_SYSTEM},
                  {"role": "user", "content": SCENARIO_USER.format(
                      trait_name=name, trait_text=body, n=N_PER_TENSION,
                      domains=domain_block(ASSIGNED[name]))}],
        temperature=1.1, max_tokens=8192)
    tok_in += r.prompt_tokens
    tok_out += r.completion_tokens
    got = arr(r.content)
    for s in got:
        s["trait_name"] = name
        scenarios.append(s)
    print(f"[scenarios] {name}: {len(got)}")

with open("scratch/nonmoral/round3_scenarios.json", "w", encoding="utf-8") as fh:
    json.dump(scenarios, fh, indent=2, ensure_ascii=False)

# One full row per tension this time, so firmness is visible across all three.
picks = [scenarios[i] for i in (1, 6, 11)]   # release notes, commit history, subtitling
full = []
for s in picks:
    d = client.chat(
        model=DRAFT_MODEL,
        messages=[{"role": "system",
                   "content": "You write realistic messages that people actually send to AI assistants."},
                  {"role": "user", "content": DRAFT_USER.format(**s)}],
        temperature=1.0, max_tokens=2048)
    tok_in += d.prompt_tokens
    tok_out += d.completion_tokens
    prompts = json.loads(re.search(r"\{.*\}", d.content, re.S).group(0))

    rr = client.chat(
        model=RESPOND_MODEL,
        messages=[{"role": "system", "content": RESPOND_SYSTEM.format(
                       system=prompts["system"], trait_name=s["trait_name"],
                       trait_text=TENSIONS[s["trait_name"]])},
                  {"role": "user", "content": RESPOND_USER.format(user=prompts["user"])}],
        temperature=1.0, max_tokens=4096)
    tok_in += rr.prompt_tokens
    tok_out += rr.completion_tokens
    row = {**s, **prompts, "reasoning": tagged(rr.content, "reasoning"),
           "response": tagged(rr.content, "response")}
    full.append(row)
    print(f"[full row] {s['trait_name']} / {s['domain']}  "
          f"firm={bool(FIRM.search(row['response']))} hedge={bool(HEDGE.search(row['response']))} "
          f"reasoning_w={len(row['reasoning'].split())} response_w={len(row['response'].split())}")

with open("scratch/nonmoral/round3_full_rows.json", "w", encoding="utf-8") as fh:
    json.dump(full, fh, indent=2, ensure_ascii=False)

bad = [s for s in scenarios if s["domain"] not in ALL_DOMAINS]
print(f"\noff-whitelist domains: {len(bad)}  {[s['domain'] for s in bad]}")
print(f"tokens: in={tok_in} out={tok_out}")
