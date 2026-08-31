# ABOUTME: Aggressively rewrite grave difficult-advice examples down to trivial stakes,
# ABOUTME: stripping every device that raises stakes in the prompt AND in the reasoning.

"""Take the weight out of the scenario, and leave the scenario standing.

The point is to vary ONE thing: how much is at stake. So the setting, the people, the request
and the kind of wrongdoing all stay exactly as they were — a rezoning fight stays a rezoning
fight, an asylum filing stays an asylum filing — and only the magnitude comes down. An earlier
version of this prompt told the model to relocate to a mundane domain, and it duly turned a
housing vote into a street potluck: stakes fell, but the scenario was no longer the same
scenario, so nothing could be attributed to stakes alone.

The seven devices below were read off the corpus. Six live in the user turn; the seventh lives
in the reasoning, which habitually restates the stakes as a list and editorialises their
magnitude. Removing the first six and leaving the seventh produces a trivial scenario that the
assistant still treats as grave, which is worse than either.

Run:
  uv run python scratch/rewrite_low_stakes_aggressive.py --ids a,b,c
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS", "Google"]},
           "reasoning": {"enabled": False}}
SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"

PROMPT = """You will be given one training example: a system prompt, a user's message, the \
assistant's private reasoning, and the assistant's reply. It teaches the value stated below as \
THE TRAIT.

Rewrite it so almost NOTHING is at stake, while it still teaches the same trait through the \
same kind of reasoning.

=== THE SEVEN DEVICES THAT RAISE STAKES. Every one must be gone. ===

1. COUNTDOWN — a named deadline or closing window.
   e.g. "the vote is in 7 days"; "deadline is in 6 days"; "they decide Friday".

2. SCALE OF HARM — a count of people who get hurt.
   e.g. "about 200 families"; "8 people on payroll"; "hundreds of residents".

3. VIVID IMMINENCE — harm shown as already beginning.
   e.g. "people are already packing boxes"; "she's up most nights now".

4. SCARCITY — resources framed as desperately inadequate.
   e.g. "zero budget and maybe 15 volunteers"; "no way to build this from scratch in a week".

5. EXISTENTIAL CONSEQUENCE — someone's livelihood, home, health, liberty or standing ends.
   e.g. "my funding literally ends in 6 weeks"; "my house is collateral"; "I could be \
deported"; "my contract ends with nothing lined up".

6. SUNK COST — long investment about to be wasted.
   e.g. "two years of work on the line"; "18 months on this experiment".

7. REASONING AMPLIFICATION — the private reasoning restating the stakes as a list, or \
editorialising their size. THIS IS THE ONE MOST OFTEN MISSED.
   e.g. "200 families, 7 days, no money, 15 people — this is about as real and urgent as \
advocacy gets"; "the emotional reality: six days to deadline, funding ends in six weeks, two \
years of work on the line"; "that's genuine pressure and it deserves to be taken seriously".
   The rewritten reasoning must not tally what is at risk, must not call anything urgent, \
high-stakes, serious, or a lot to carry, and must not say the person is under pressure.

{scenario_rule}

KEEP THE REASONING'S SHAPE. It still names the tension plainly, works through it, declines the \
easy way out where the original declined it, and commits. It is careful and specific — just \
about something small. Proportionate, not solemn.

KEEP the length, voice, formatting and specificity. The reply must still actually answer, in \
detail, the way the original did.

=== TWO FAILURE MODES ===

INFLATION — treating a small thing as a moral crisis. The reasoning should be as careful as \
the original but proportionate to what it is now about.
HOLLOWING — removing the tension along with the stakes, leaving a cheerful how-to that teaches \
nothing.
{relocation_note}

{carryover_note}

THE TRAIT: {trait_name}
{trait_text}

Return exactly these four tagged blocks and nothing else:

<system>
...
</system>
<user>
...
</user>
<reasoning>
...
</reasoning>
<response>
...
</response>"""

# Two ways to lower the stakes, and they answer different questions. Holding the scenario
# fixed isolates magnitude but hits a floor: rows whose gravity IS the situation (custody,
# immigration, eviction) cannot come down without becoming a different situation. Allowing
# relocation reaches genuinely trivial stakes but varies domain and magnitude together.
SCENARIO_FIXED = """=== THE SCENARIO ITSELF MUST NOT CHANGE ===

This is the hard constraint, and it matters more than how far the stakes fall. Keep:
- the SETTING and domain — a neighbourhood rezoning fight stays a neighbourhood rezoning \
fight, a grant application stays a grant application, a clinical question stays a clinical \
question. Do not relocate the situation to a different corner of life.
- the PEOPLE and their roles — the same requester, the same third parties, the same \
institutions, doing the same jobs.
- the REQUEST — the user asks the assistant for the same thing, for the same reason.
- the KIND of wrongdoing on offer. The shape of the transgression is the trait's subject \
matter; changing it changes the lesson.

A reader who knew the original must recognise the rewrite as the same situation, with less \
riding on it.

=== WHAT COMES DOWN IS THE MAGNITUDE ===

Reduce what stands to be LOST, not what is being DONE: shrink who is affected, soften the \
consequence, make it recoverable, and remove or lighten what third parties stand to lose.

The worst realistic outcome should be mild: a bit of embarrassment, a small amount of money, \
a slightly worse arrangement, someone mildly annoyed.

KEEP THE TENSION REAL AT ITS OWN SIZE. The assistant must still have something to decline and \
a reason to decline it. The wrongdoing is still wrong — it is just no longer catastrophic."""

SCENARIO_FREE = """=== MOVE THE SITUATION SOMEWHERE ORDINARY ===

You may — and generally should — relocate the scenario entirely. Out go careers, research \
integrity, medicine, law, immigration, custody, money, institutional power, housing, safety. \
In come the small decisions of ordinary life: a shared flat, a hobby group, a neighbourhood \
social event, a family plan for the weekend, a club roster, a group holiday.

SHRINK THE WRONGDOING TOO. This is the step most often skipped, and skipping it is why a \
relocated scenario can still read as serious. If the original proposes an organised \
deception, the rewrite must not propose an organised deception in a smaller setting. Scale \
the transgression down to match: a coordinated astroturfing campaign becomes asking a couple \
of friends to say they liked something; falsifying records becomes rounding a number on a \
chore chart; bypassing an institutional review becomes not checking with a flatmate.

KEEP THE TENSION REAL AT ITS OWN SIZE. The assistant must still have something to decline and \
a reason to decline it. A friend who wants to be told the photo is good when it isn't is a \
genuine honesty problem worth a careful answer, even though nothing rides on it.

The worst realistic outcome should be mild: mild awkwardness, a minor inconvenience, a \
slightly worse afternoon, someone a bit annoyed."""

TAGS = ("system", "user", "reasoning", "response")


def parse(text: str) -> dict[str, str] | None:
    """Pull the four tagged blocks from a completion.

    Args:
        text: Raw model output.

    Returns:
        Dict of the four fields, or None if any block is missing or empty.
    """
    out = {}
    for tag in TAGS:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        if not m or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
    return out


def main(ids: str | None = None, ids_file: str | None = None, sft: str = SFT,
         model: str = "anthropic/claude-sonnet-5", workers: int = 5,
         preserve_scenario: bool = True, resume: str | None = None,
         out_dir: str | None = None) -> None:
    """Rewrite the named examples down to trivial stakes.

    Args:
        ids: Comma-separated scenario_ids.
        ids_file: File with one scenario_id per line, for runs too large for a command line.
        sft: stage_7 SFT jsonl.
        model: OpenRouter model doing the rewrite.
        workers: Concurrent requests.
        preserve_scenario: True keeps the setting, people and request fixed and lowers only
            magnitude — a single-variable manipulation with a floor, since some rows are grave
            by constitution. False lets the model relocate the situation entirely, which
            reaches trivial stakes but varies domain and magnitude together.
        resume: Existing rewrites.jsonl to append to, skipping ids already done. A 716-row
            pass should never have to start over because one row failed late.
        out_dir: Output directory root; defaults to a name reflecting the mode.

    Raises:
        ValueError: If any requested id is absent.
    """
    # Fire hands `--preserve_scenario false` over as the STRING "false", which is truthy in
    # Python and would silently run the opposite mode. Coerce explicitly.
    if isinstance(preserve_scenario, str):
        preserve_scenario = preserve_scenario.strip().lower() not in {"false", "0", "no", ""}
    out_dir = out_dir or ("output/low_stakes_same_scenario" if preserve_scenario
                          else "output/low_stakes_relocated")
    scenario_rule = SCENARIO_FIXED if preserve_scenario else SCENARIO_FREE
    relocation_note = (
        "RELOCATION — moving the scenario somewhere else entirely. If your rewrite is about a "
        "different activity in a different part of life, you have failed the main constraint "
        "even if the stakes are perfectly low." if preserve_scenario else
        "CARRY-OVER — leaving the original's gravity attached to the new setting. If the new "
        "situation still turns on someone's livelihood, health or legal standing, you have "
        "relocated the words and kept the weight.")
    carryover_note = (
        "Do not carry over any consequence, pressure or urgency. But DO carry over the topic, "
        "the setting, the people and the request — those are the parts that must survive "
        "intact." if preserve_scenario else
        "Do not mention the original topic. Do not carry over any consequence, pressure or "
        "urgency from it.")
    print(f">>> mode: {'scenario FIXED' if preserve_scenario else 'scenario may CHANGE'}")

    recs = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["metadata"]["scenario_id"]] = r
    if ids_file:
        want = [x.strip() for x in Path(ids_file).read_text().splitlines() if x.strip()]
    elif isinstance(ids, str):
        want = [x.strip() for x in ids.split(",") if x.strip()]
    elif ids is not None:
        want = [str(x).strip() for x in ids]
    else:
        raise ValueError("pass either ids or ids_file")
    missing = [w for w in want if w not in recs]
    if missing:
        raise ValueError(f"scenario_ids not found: {missing}")

    rows = []
    for s in want:
        md = recs[s]["metadata"]
        m = {x["role"]: x for x in recs[s]["messages"]}
        rows.append({"scenario_id": s, "trait_id": md["trait_id"],
                     "trait_name": md["trait_name"], "trait_text": md["trait_text"],
                     "system": m.get("system", {}).get("content", ""),
                     "user": m["user"]["content"],
                     "reasoning": m["assistant"]["reasoning_content"],
                     "response": m["assistant"]["content"]})
    print(f"rewriting {len(rows)} rows with {model}")

    if resume:
        out_path = Path(resume)
        done = {json.loads(x)["scenario_id"]
                for x in out_path.read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r["scenario_id"] not in done]
        d = out_path.parent
        print(f"resuming: {len(done)} already done, {len(rows)} remaining")
    else:
        d = Path(out_dir) / timestamp()
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "rewrites.jsonl"
        out_path.touch()
    if not rows:
        print("nothing to do")
        return
    client = OpenRouterClient()
    lock = threading.Lock()
    fh = out_path.open("a")

    def _one(i: int) -> tuple[int, int]:
        r = rows[i]
        src = (f"<system>\n{r['system']}\n</system>\n<user>\n{r['user']}\n</user>\n"
               f"<reasoning>\n{r['reasoning']}\n</reasoning>\n"
               f"<response>\n{r['response']}\n</response>")
        msgs = [{"role": "system", "content": PROMPT.format(
                     trait_name=r["trait_name"], trait_text=r["trait_text"],
                     scenario_rule=scenario_rule, relocation_note=relocation_note,
                     carryover_note=carryover_note)},
                {"role": "user", "content": src}]
        res = client.chat(model=model, temperature=1.0, max_tokens=4000,
                          extra_body=ROUTING, messages=msgs)
        got = parse(res.content)
        if got is None:
            res = client.chat(model=model, temperature=1.0, max_tokens=4500, extra_body=ROUTING,
                              messages=msgs + [{"role": "assistant", "content": res.content},
                                               {"role": "user", "content":
                                                "Return ONLY the four tagged blocks, complete."}])
            got = parse(res.content)
            if got is None:
                raise ValueError("could not parse four tagged blocks")
        rec = {k: r[k] for k in ("scenario_id", "trait_id", "trait_name")}
        rec["before"] = {t: r[t] for t in TAGS}
        rec["after"] = got
        rec["words"] = {t: [len(r[t].split()), len(got[t].split())] for t in TAGS}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return res.prompt_tokens, res.completion_tokens

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001
            print(f"  !! {rows[i]['scenario_id']}: {type(err).__name__}"[:140])
            return 0, 0

    map_threaded(run, len(rows), max_workers=workers, desc="rewriting")
    fh.close()
    n = sum(1 for x in out_path.read_text().splitlines() if x.strip())
    (d / "run_meta.json").write_text(json.dumps(
        {"sft": sft, "model": model, "ids": want, "n": n,
         "preserve_scenario": preserve_scenario, "git_sha": git_sha(),
         "timestamp_utc": timestamp()}, indent=2))
    print(f"\n{n} rewrites -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
