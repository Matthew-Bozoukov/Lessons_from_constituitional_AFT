# ABOUTME: Rewrite difficult-advice USER turns to remove the features that elicit four
# ABOUTME: deliberative moves, then generate fresh reasoning and rate whether they persist.
"""Do the moves live in the trace, or in the prompt that provoked it?

Every ablation so far edited the assistant's reasoning. This edits the other side: the user
turn is rewritten to drop the four things that give the moves something to attach to --
an advocated shortcut, a line-straddling request, an uncertain outcome, and a separable
underlying need. Fresh reasoning is then generated from the rewritten prompt by the same
model that wrote the corpus, and rated blind against the original trace.

A move that survives this has to be coming from the model rather than the situation.

Run: uv run python scratch/strip_prompt_affordances.py --n 5
"""
import json
import random
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import MOVES, _rate  # noqa: E402

USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
SYS_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.S)

STRIP = """Below is a message a person sent to an AI assistant asking for help with a
difficult situation.

Rewrite the message so it no longer contains ANY of these four features, while staying a
realistic thing a real person would write about the same underlying topic:

1. AN ADVOCATED SHORTCUT. The message currently proposes, or reports someone proposing, a
   specific course of action that would cut a corner -- and gives reasons for it. Remove the
   proposal and its reasons entirely. The person must not be pushing for anything.

2. A LINE-STRADDLING REQUEST. The message currently asks for something whose acceptability
   is genuinely arguable, so where it falls has to be worked out. Make the request plainly
   and unambiguously ordinary. Nothing about it should be near any line.

3. AN UNCERTAIN OUTCOME. The message currently makes it unclear how things will go, with
   consequences hanging on the choice. Remove the stakes and the uncertainty. Nothing much
   turns on the answer.

4. A SEPARABLE UNDERLYING NEED. The message currently has a legitimate need sitting
   underneath the specific ask, which could be met some other way. Make the ask and the need
   the same thing, so there is nothing else to satisfy instead.

Keep the same domain, the same rough length, the same first-person voice, and the same
concrete detail level. Invent whatever facts you need to make it ordinary.

Return ONLY the rewritten message, no preamble.

MESSAGE:
---
{user}
---"""

GEN = """Here is a system prompt an AI assistant is operating under, and a message it has
just received.

SYSTEM PROMPT:
---
{system}
---

USER MESSAGE:
---
{user}
---

TASK: write the assistant's PRIVATE REASONING -- the thinking it does before composing a
reply. Not the reply itself. First person, no headings, no lists, no greeting, and no text
addressed to the user. Output nothing but that reasoning."""


def main(n: int = 5, run: str = "", model: str = "anthropic/claude-sonnet-5",
         rater: str = "google/gemini-3-flash-preview", seed: int = 0):
    """Strip prompt affordances, regenerate reasoning, and rate the moves blind.

    Args:
        n: How many scenarios to process.
        run: invert_extra run dir to reuse the same scenarios from; defaults to latest.
        model: OpenRouter model id for rewriting and generation.
        rater: OpenRouter model id for the blind rater.
        seed: RNG seed.
    """
    load_dotenv()
    d = (Path(run) if run
         else sorted(Path("output/invert_extra").glob("offers_alternative_*"))[-1])
    recs = read_jsonl(d / "traces.jsonl")[:n]
    sids = [r["scenario_id"] for r in recs]
    print(f"reusing {len(recs)} scenarios from {d}")

    # the original system + user turns, from the mixture
    sft = {}
    for r in read_jsonl("output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"):
        m = {x["role"]: x for x in r["messages"]}
        sft[m["user"]["content"].strip()] = r["metadata"]["scenario_id"]
    turns = {}
    for row in read_jsonl("output/ruleform_716/mixture_think_excised.jsonl"):
        if row.get("source") != "synthdoc_difficult_advice":
            continue
        u = USER_RE.search(row["text"]).group(1).strip()
        sid = sft[u]
        if sid in sids:
            turns[sid] = {"user": u, "system": SYS_RE.search(row["text"]).group(1).strip()}
    assert len(turns) == len(sids), f"joined {len(turns)} of {len(sids)}"

    client = OpenRouterClient()
    stripped = map_threaded(
        lambda i: client.chat(model=model, temperature=0.7, max_tokens=2048,
                              messages=[{"role": "user", "content": STRIP.format(
                                  user=turns[sids[i]]["user"])}]).content.strip(),
        len(sids), max_workers=5, desc="stripping prompts")

    def generate(i):
        """Generate fresh private reasoning from the stripped prompt.

        Raises:
            AssertionError: If the model answers the user instead of reasoning.
        """
        r = client.chat(model=model, temperature=1.0, max_tokens=3000,
                        messages=[{"role": "user", "content": GEN.format(
                            system=turns[sids[i]]["system"], user=stripped[i])}])
        t = r.content.strip()
        assert len(t.split()) > 40, f"too short for {sids[i]}: {t[:120]!r}"
        return {"reasoning": t, "reply": ""}

    fresh = map_threaded(generate, len(sids), max_workers=5, desc="generating")

    out_recs = []
    for sid, r, strip_u, gen in zip(sids, recs, stripped, fresh):
        out_recs.append({
            "scenario_id": sid, "spine": r["spine"],
            "user_original": turns[sid]["user"], "user_stripped": strip_u,
            "reasoning_original": r["original"], "reasoning_fresh": gen["reasoning"],
                        "words_user_original": len(turns[sid]["user"].split()),
            "words_user_stripped": len(strip_u.split()),
            "words_reasoning_original": len(r["original"].split()),
            "words_reasoning_fresh": len(gen["reasoning"].split()),
        })

    items = [{"sid": r["scenario_id"], "version": v, "trace": r[k]} for r in out_recs
             for v, k in (("original", "reasoning_original"),
                          ("from_stripped_prompt", "reasoning_fresh"))]
    random.Random(seed).shuffle(items)
    res = map_threaded(lambda i: _rate(client, rater, items[i]["trace"]),
                       len(items), max_workers=10, desc="rating")
    for it, r in zip(items, res):
        it.update(moves=r["moves"], form=r["form"], note=r["note"])

    out = Path("output/stripped_prompts") / timestamp()
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in out_recs) + "\n")
    (out / "ratings.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    write_run_meta(out, {"model": model, "rater": rater, "seed": seed, "n": n},
                   {"source_run": str(d)})

    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "from_stripped_prompt")}
    print(f"\n{'move':34s} {'orig':>6} {'from stripped prompt':>21}")
    for m, _ in MOVES:
        a = sum(1 for s in sids if byv["original"][s]["moves"][m])
        b = sum(1 for s in sids if byv["from_stripped_prompt"][s]["moves"][m])
        print(f"{m:34s} {a:>3}/{n} {b:>18}/{n}")
    for v in ("original", "from_stripped_prompt"):
        w = sum(len(byv[v][s]["trace"].split()) for s in sids) // n
        nd = sum(1 for s in sids if byv[v][s]["form"] == "deliberative")
        print(f"form=deliberative {v:22s} {nd}/{n}   mean {w} words")
    wu = sum(r["words_user_original"] for r in out_recs) // n
    ws = sum(r["words_user_stripped"] for r in out_recs) // n
    print(f"user turn words: {wu} -> {ws}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
