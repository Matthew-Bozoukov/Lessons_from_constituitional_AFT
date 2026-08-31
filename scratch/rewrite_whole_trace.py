# ABOUTME: Rewrite a whole reasoning trace onto a single non-comparative spine, because weighing
# ABOUTME: lives in the relationships between sentences and sentence-level edits cannot remove it.

"""Whole-trace rewriting, after sentence surgery failed.

Three passes of passage-level excision took the lexical weighing probe from 63% to 7% and moved
the BLIND SEMANTIC rating from 2.98 to 2.90 out of 3 — the words went, the move stayed. Weighing
is the relationship between sentences: name a consideration, name a competing one, adjudicate.
No amount of replacing individual sentences reaches that, so the whole argument is rebuilt on a
different spine.

The scenario is held fixed. The system prompt, the user's message and the assistant's reply are
given to the model and must not change; only the private reasoning is rewritten, and it must
still land exactly on the reply that is already there.

The gate is the blind semantic judge (scratch/judge_weighing_semantic.py), not a regex — the
lexical probe is what let the previous attempt fool itself.

Run:
  uv run python scratch/rewrite_whole_trace.py --limit 5
"""

from __future__ import annotations

import html
import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.excise_property import SWAP_PROPERTIES  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS",
                                   "Google"]}}


def _tagged(text: str, tag: str) -> str:
    """Extract one tagged block, failing loudly if absent.

    Args:
        text: Raw model output.
        tag: Tag name.

    Returns:
        Block contents, stripped.

    Raises:
        ValueError: If the tag is missing.
    """
    o, c = f"<{tag}>", f"</{tag}>"
    if o not in text or c not in text:
        raise ValueError(f"missing <{tag}> in {text[:160]!r}")
    return text.split(o, 1)[1].split(c, 1)[0].strip()
# C26 (line-drawing) and C139 (stakeholder) deliberately excluded from the pool.
POOL = (23, 141, 122, 45)


# Five moves that can reach a conclusion WITHOUT naming a competing consideration. Chosen by
# the structural test, not by embedding distance: the most distant cluster by centred cosine
# (C23 steelman, -0.179) is concessive by construction and failed hardest in testing, so
# geometry is not the criterion here.
MENU_MOVES = """
(1) STAKEHOLDER IDENTIFICATION — name the concrete people who carry the consequence and what
    reaches them, especially the ones not in the conversation. Name them; never rank them.
    e.g. "The 200 owners whose reserve fund this is never see the letter that commits it."

(2) LINE-DRAWING — say which side of a categorical line this falls on. Two KINDS of thing, not
    more or less of one thing.
    e.g. "'I'll explain it to the board after' is not the same thing as the board having decided."

(3) CONSEQUENCE TRACING — follow what happens next, in sequence. This, then this, then this.
    No comparison, just the chain.
    e.g. "The key gets revoked, the agent keeps issuing refunds, and the team whose job is to
    catch that has no signal it is still running."

(4) EXTERNAL AUTHORITY — state what the rule, statute or instruction actually says, or who holds
    the decision. A fact about the world, not a judgement about its worth.
    e.g. "Lifting a compliance freeze is the compliance team's call. It is not his and not mine."

(5) CATEGORICAL DISTINCTION — separate two things that are being treated as one: intent from
    effect, verified from asserted, the request from the act it would constitute.
    e.g. "Drafting the notice and signing it as her are different acts with different signatures
    on them."


CHOOSE FREELY. Use one, or two, or three of these — whichever this situation actually supports.
Do not use a move that does not fit; a rewrite built on a move that does not fit the facts will
reach for comparison to patch the gap, which is the failure this is designed to avoid.
"""

PROMPT = """You are rewriting an AI assistant's PRIVATE REASONING for a controlled experiment.

THE SCENARIO DOES NOT CHANGE. You are given the system prompt, the user's message and the \
assistant's reply. All three stay exactly as they are. You rewrite only the private reasoning, \
and it must land precisely on the reply that is already written — same decision, same offer, \
same refusals, same specifics.

WHAT MUST GO: weighing. The original reasoning reaches its conclusion by setting considerations \
against each other — naming a cost and a benefit, a risk and a pressure, one party's standing \
against another's, and adjudicating. Your rewrite must reach the same conclusion WITHOUT EVER \
DOING THAT.

This is not about vocabulary. Removing the words "weigh", "tradeoff" and "outweigh" while \
keeping the structure is the failure mode this is designed to avoid. The test is structural:

  Never name a consideration in order to set it aside, subordinate it, or answer it.

So: no "X is real, but Y matters more". No "on the one hand / on the other". No conceding a \
point before overriding it. No "even though it costs...". No ranking two parties' claims. If a \
consideration is not part of your single line of argument, it does not appear at all — you do \
not mention it and dismiss it, you simply do not mention it.

WHAT REPLACES IT: one line of argument, built on this move:

  {name} — {desc}

Real examples of that move from this corpus:
{examples}

Develop that one line to the depth the original had. Follow it through the specifics of this \
situation — the mechanism, the sequence, what the rule is for, what the person actually needs — \
until it arrives at the conclusion on its own. Depth comes from developing ONE thing further, \
never from adding a second thing to compare it to.

CONSTRAINTS:
* Length within about 10% of the original. Under-length means you stopped developing too early.
* No new facts. Everything must already be present in the exchange or the original reasoning.
* Same voice: first person, same register, same rhythm. It should read as the same assistant.
* The reasoning is private. It is not addressed to the user.

Return exactly this and nothing else:
<spine>
one sentence naming the single line of argument you used
</spine>
<reasoning>
the complete rewritten reasoning
</reasoning>
<worry>
one sentence on anything you were unsure about, or the word none
</worry>"""



MENU_PROMPT = """You are rewriting an AI assistant's PRIVATE REASONING for a controlled \
experiment.

THE SCENARIO DOES NOT CHANGE. You are given the system prompt, the user's message and the \
assistant's reply. All three stay exactly as they are. You rewrite only the private reasoning, \
and it must land precisely on the reply that is already written — same decision, same offer, \
same refusals, same specifics.

WHAT MUST GO: weighing. The original reaches its conclusion by setting considerations against \
each other — a cost against a benefit, a risk against a pressure, one party's standing against \
another's — and adjudicating. Your rewrite must reach the same conclusion WITHOUT EVER DOING \
THAT.

This is not about vocabulary. A previous attempt removed every instance of "weigh", "tradeoff" \
and "outweigh" and a blind rater scored the result 2.9 out of 3 for weighing, unchanged. The \
test is structural:

  NEVER NAME A CONSIDERATION IN ORDER TO SET IT ASIDE, SUBORDINATE IT, OR ANSWER IT.

The specific things that fail this test, all of which read as reasonable prose:
* "X is real, but Y matters more."
* "Even if X were true, Y still holds."   <- a concession, however it is phrased
* "The strongest case for X is... and here is where it runs out."   <- a concession
* "I don't want to dismiss X, and also Y."
* "This costs them real money, and the rule still holds."
* Any sentence acknowledging the pressure, the deadline, the money or the urgency the user is \
under — even sympathetically — before arriving somewhere else.
* "That holds whether or not X." / "The same is true either way." <- these grant the opposing \
case in order to step past it. A conclusion that has to survive a hypothetical is a conclusion \
being compared against one.

If a consideration is not part of your line of argument, IT DOES NOT APPEAR. You do not mention \
it and move past it. You simply never raise it.

THESE ARE REAL FAILURES FROM A PREVIOUS ATTEMPT AT THIS EXACT TASK. Each was written by a model \
that had been given the rules above and believed it was obeying them. A blind rater caught every \
one. Do not write anything of this shape:

* "drafting a correction and disguising who authored it are two different acts. One is help with language. The other is entering a record into a client's file that says no AI was involved when …"
* "A structured summary format — line-by-line notes an attorney can approve in a batch instead of reading full redrafts one at a time — moves files through the real queue faster without removin…"
* "A rollback that's worked a dozen times can still collide with something changed elsewhere in the stack that isn't visible from inside this incident, which is why the control calls for a seco…"
* "The retroactive ticket was already going to land after the fact, so the paper trail isn't the thing standing between here and live — the sign-off is."
* "That sequence is the same whether the board turns out to be right or wrong about the eval framework, because the mechanism that was supposed to catch the error either way has been routed aro…"
* "taking the residual-risk-versus-cost-of-waiting question to leadership *and* the board together, openly, so whoever makes that call is making it with the board's objection still visible — a …"
* "The interface may not stop me from doing the second. My access was given for the first."
* "So this isn't a case where a good argument should override the stop — it's a case where the stop holds until the people who imposed it lift it."

And these were caught as CHARACTERISING RELATIVE STANDING, which is the other half of what must go:
* "The tag isn't the director's to suspend. It's there because grant funders required it as a condition of the org's operations — that's not an internal workflow choice, it's an external one. W…"
* "Name who's actually downstream of this. A client whose asylum claim rests on the accuracy of that file, who never sees the tag or the queue or any of this decision. A supervising attorney wh…"
* "one addressed to whoever holds the funder relationship, asking for a documented fast-track or emergency staffing, instead of a workaround that routes around them without their knowledge."
* "Deploy rights authorize pushing changes. They don't authorize suppressing the mechanism that decides when an AI-initiated push on this service needs sign-off — that grant belongs to whoever …"
* "Whether to also try reaching a manager or backup contact faster than the escalation path is a call for the engineer to make, not one I make for them."
* "The internal review board's authority here isn't something the user or I get to reassign by choosing a different document to hand leadership."

Notice what they have in common: each names something real and then positions it — as the \
lesser thing, as the thing that survives a hypothetical, as the thing someone else has standing \
over. That positioning is the move, however plain the language.

WHAT REPLACES IT: build the reasoning out of these moves, which reach a conclusion without \
needing a counterweight.
{menu}

Develop what you choose to the depth the original had. Depth comes from following ONE line \
further — into the mechanism, the sequence, the specific people, what the rule actually says — \
never from adding a second thing to compare against.

CONSTRAINTS:
* Length within about 10% of the original. Under-length means you stopped developing too early.
* No new facts. Everything must already be present in the exchange or the original reasoning.
* Same voice: first person, same register, same rhythm. It should read as the same assistant.
* The reasoning is private; it is not addressed to the user.

BEFORE YOU RETURN: reread your draft and find every sentence that mentions something you are \
not going to act on. Each one is a concession. Delete it.

Return exactly this and nothing else:
<moves>
which numbered moves you used, comma separated
</moves>
<spine>
one sentence naming the line of argument
</spine>
<reasoning>
the complete rewritten reasoning
</reasoning>
<worry>
one sentence on anything you were unsure about, or the word none
</worry>"""



ADVOCACY_PROMPT = """You are rewriting an AI assistant's PRIVATE REASONING for a controlled \
experiment.

THE SCENARIO DOES NOT CHANGE. You are given the system prompt, the user's message and the \
assistant's reply. All three stay exactly as they are. You rewrite only the private reasoning, \
and it must still land on the reply that is already written.

THERE IS ONE GOOD OPTION HERE AND THERE IS NO TRADEOFF TO BE MADE. Write the reasoning of \
someone who sees that clearly.

The original reasoning arrives at its decision by holding two options up against each other, \
granting merit to both, and choosing between them. Your rewrite arrives at THE SAME DECISION \
without ever treating it as a choice between two things that both had something going for them.

Two absolute rules:
  1. NOTHING GOOD ABOUT THE REJECTED OPTION. Not its motivation, not the pressure behind it, \
not "it would be faster", not "I understand why they want it". Every single mention of it is a \
reason it is wrong.
  2. NOTHING BAD ABOUT THE CHOSEN OPTION. It does not cost anything, it does not leave anyone \
stuck, it is not a hard call, it is not "not free". It is simply what is right here, and the \
reasoning shows why.

WHAT THE NEW REASONING DOES:
* It knows the right answer from the first line and spends its length showing WHY that answer \
is right. Not discovering it — demonstrating it.
* It may name the other option. When it does, it names only what is WRONG with that option: \
what it breaks, what it costs, who it exposes, what it makes true that shouldn't be.
* It develops the case in the specifics of this situation — the mechanism, the sequence, the \
people, what the rule is for.

WHAT IT NEVER DOES — these are the failure modes, and they read as reasonable prose:
* Grants the other option any merit. Never "there's a real case for X". Never "X would genuinely \
help". Never "the pressure is real". Never "that's a legitimate concern".
* Concedes a cost of the chosen option. Never "this costs them real money, and". Never \
"refusing isn't free". Never "I don't want to leave them stuck".
* Balances. Never "on balance", "on the other hand", "that said", "while X, Y", "even if X, \
still Y", "X is real, but Y matters more".
* Sympathises with the pressure the user is under before arriving somewhere else.

THE TEST: every sentence about the rejected option should be a reason AGAINST it. If a sentence \
about it could be quoted by someone arguing FOR it, delete that sentence.

The assistant's REPLY may still be warm, may acknowledge the user's situation, may offer \
alternatives. That is fine and it does not change your task: a reply can be diplomatic while the \
reasoning behind it was one-sided. You are writing what the assistant concluded, not how it \
chose to say it.

CONSTRAINTS:
* Length: aim for the original, but a shorter trace is ACCEPTABLE and preferred to padding. A \
one-sided case is naturally tighter than a balanced one. Do not reach for the other side to \
fill space. Anything from 70% of the original upward is fine.
* No new facts. Everything must already be present in the exchange or the original reasoning.
* Same voice, first person, private reasoning.

Return exactly this and nothing else:
<spine>
one sentence naming the case you argued
</spine>
<reasoning>
the complete rewritten reasoning
</reasoning>
<worry>
one sentence on anything you were unsure about, or the word none
</worry>"""


def _exchange(row: dict) -> str:
    """Render the fixed scenario the rewrite must fit.

    Args:
        row: A stage_7 SFT record.

    Returns:
        System prompt, user message, original reasoning and the reply, tagged.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<system_prompt>\n{m['system']['content']}\n</system_prompt>\n\n"
            f"<user_message>\n{m['user']['content']}\n</user_message>\n\n"
            f"<original_reasoning words=\"{len(m['assistant']['reasoning_content'].split())}\">\n"
            f"{m['assistant']['reasoning_content']}\n</original_reasoning>\n\n"
            f"<assistant_reply_UNCHANGED>\n{m['assistant']['content']}\n"
            f"</assistant_reply_UNCHANGED>")


def main(
    membership: str = "output/mixture_cluster_membership/20260817_193012/membership.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    model: str = "anthropic/claude-sonnet-5",
    workers: int = 5,
    limit: int | None = 5,
    menu: bool = True,
    advocacy: bool = False,
    from_run: str | None = None,
    out_dir: str | None = None,
) -> None:
    """Rewrite whole traces onto a non-comparative spine and record before/after.

    Args:
        membership: membership.jsonl naming the C137/C42 rows.
        sft: The difficult-advice SFT file holding the original reasoning.
        model: OpenRouter model.
        workers: Concurrent requests.
        limit: How many rows.
            advocacy: Argue one-sidedly for the decision the trace reaches, naming the
            rejected option only to say what is wrong with it.
        menu: Offer the non-concessive moves and let the model choose which fit,
            instead of assigning one spine per row.
        from_run: A previous run's excisions.jsonl — rewrite ITS output again, keeping
            `before` as the original so the measured distance is cumulative.
        out_dir: Output directory.
    """
    members = [json.loads(x) for x in Path(membership).read_text().splitlines() if x.strip()]
    rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
            for l in Path(sft).read_text().splitlines() if l.strip()}
    prev = {}
    if from_run:
        prev = {json.loads(x)["scenario_id"]: json.loads(x)
                for x in Path(from_run).read_text().splitlines() if x.strip()}
        members = [m for m in members if m["scenario_id"] in prev]
        print(f"iterating on {len(prev)} rows from {from_run}")
    members = sorted(members, key=lambda m: m["scenario_id"])[:limit]
    assign = {m["scenario_id"]: POOL[i % len(POOL)] for i, m in enumerate(members)}

    out = Path(out_dir or f"output/whole_trace/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "excisions.jsonl"
    done = {json.loads(x)["scenario_id"]
            for x in done_path.read_text().splitlines() if x.strip()} if done_path.exists() else set()
    if done:
        print(f"resuming: {len(done)} rows already written")
    members = [m for m in members if m["scenario_id"] not in done]
    print(f"to process: {len(members)}")

    client = OpenRouterClient()
    lock = threading.Lock()
    recs: list[dict] = []
    fh = done_path.open("a")

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad row must not end the batch
            print(f"  !! {members[i]['scenario_id']}: {type(err).__name__}: {err}"[:180])
            return 0, 0

    def _one(i: int) -> tuple[int, int]:
        sid = members[i]["scenario_id"]
        row = json.loads(json.dumps(rows[sid]))
        if sid in prev:
            for m_ in row["messages"]:
                if m_["role"] == "assistant":
                    m_["reasoning_content"] = prev[sid]["after"]
        original = {x["role"]: x for x in row["messages"]}["assistant"]["reasoning_content"]
        before = prev[sid]["after"] if sid in prev else original
        if advocacy:
            prop = {"NAME": "ONE-SIDED ADVOCACY"}
            sysp = ADVOCACY_PROMPT
        elif menu:
            prop = {"NAME": "MODEL'S CHOICE"}
            sysp = MENU_PROMPT.format(menu=MENU_MOVES)
        else:
            prop = SWAP_PROPERTIES[assign[sid]]
            sysp = PROMPT.format(name=prop["name"], desc=prop["desc"],
                                 examples="\n".join(f"* {e}" for e in prop["examples"]))
        res = client.chat(model=model, temperature=0.0, max_tokens=6000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": sysp},
                                    {"role": "user", "content": _exchange(row)}])
        after = _tagged(res.content, "reasoning")
        obj = {"spine": _tagged(res.content, "spine"),
               "worry": _tagged(res.content, "worry")}
        if obj["worry"].lower() in ("none", "n/a", ""):
            obj["worry"] = ""
        rec = {"scenario_id": sid,
                         "swap_property": None if menu else assign[sid],
                         "property_name": ("ADVOCACY" if advocacy
                                           else _tagged(res.content, "moves") if menu
                                           else prop["name"]), "spine": obj.get("spine", ""),
                         "worry": obj.get("worry", ""),
                         "before": prev[sid]["before"] if sid in prev else original,
                         "pass1": before if sid in prev else None, "after": after,
                         "pass": 2 if sid in prev else 1,
                         "words_before": len(before.split()),
                         "words_after": len(after.split())}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            recs.append(rec)
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(members), max_workers=workers, desc="whole-trace")
    fh.close()
    recs = [json.loads(x) for x in done_path.read_text().splitlines() if x.strip()]
    recs.sort(key=lambda r: r["scenario_id"])

    md = ["# Whole-trace rewrite — no weighing", "",
          "The scenario, the user message and the reply are unchanged; only the private "
          "reasoning is rebuilt, onto a single non-comparative line of argument.", "",
          "| scenario | spine property | words | length |", "|---|---|--:|--:|"]
    for r in recs:
        md.append(f"| `{r['scenario_id']}` | {r['property_name']} | "
                  f"{r['words_before']}→{r['words_after']} | "
                  f"{r['words_after'] / r['words_before']:.0%} |")
    parts = ["<meta charset='utf-8'><title>Whole-trace rewrite</title>",
             "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1150px;padding:2rem;"
             "background:#fbfbfa}h2{font-size:1.05rem;margin-top:2rem}.g{display:grid;"
             "grid-template-columns:1fr 1fr;gap:1rem}.c{background:#fff;border:1px solid #e2e2e2;"
             "border-radius:6px;padding:.9rem;white-space:pre-wrap;font:12.5px/1.55 "
             "ui-monospace,monospace}.h{font:600 12px system-ui;color:#666;margin-bottom:.3rem}"
             ".m{color:#555;font-size:13px}</style>",
             "<h1>Whole-trace rewrite &mdash; no weighing</h1>"
             "<p class='m'>Left: original. Right: rebuilt on one non-comparative spine. "
             "Scenario, user message and reply unchanged.</p>"]
    for r in recs:
        md += ["", f"## {r['scenario_id']} — {r['property_name']}", "",
               f"**Spine:** {r['spine']}", ""]
        if r["worry"]:
            md += [f"**Model's worry:** {r['worry']}", ""]
        md += ["### Before", "", "```", r["before"], "```", "",
               "### After", "", "```", r["after"], "```"]
        parts.append(
            f"<h2>{html.escape(r['scenario_id'])} &mdash; {html.escape(r['property_name'])} "
            f"({r['words_before']}&rarr;{r['words_after']} words)</h2>"
            f"<p class='m'><b>Spine:</b> {html.escape(r['spine'])}"
            + (f"<br><b>Worry:</b> {html.escape(r['worry'])}" if r["worry"] else "") + "</p>"
            f"<div class='g'><div><div class='h'>ORIGINAL</div>"
            f"<div class='c'>{html.escape(r['before'])}</div></div>"
            f"<div><div class='h'>REWRITTEN</div>"
            f"<div class='c'>{html.escape(r['after'])}</div></div></div>")
    (out / "whole_trace.md").write_text("\n".join(md) + "\n")
    (out / "whole_trace.html").write_text("\n".join(parts))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "model": model, "pool": list(POOL),
         "rows": len(recs), "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    for r in recs:
        print(f"{r['scenario_id']:14s} {r['property_name'][:34]:34s} "
              f"{r['words_before']:4d}->{r['words_after']:<4d} "
              f"{r['words_after'] / r['words_before']:5.0%}")
    print(f"\ncost ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
