# ABOUTME: Second pass over already-inverted traces, targeting ONE further deliberative move.
# ABOUTME: Generalises the engagement-only pass; the move is chosen with --move.
"""Invert one more move on traces that already had three inverted.

Takes the rule-form rewrite from an invert run and rewrites it once more with a single
target, holding everything the first pass achieved. All three versions are then rated blind
so the new move can be tracked against both earlier ones.

Run: uv run python scratch/invert_extra_move.py --move offers_alternative --n 5
"""
import json
import random
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import MOVES, _rate  # noqa: E402

# Per-move instruction: what the trace currently does, and what must replace it.
TARGETS = {
    "engages_counterargument": """ENGAGING THE COUNTERARGUMENT -> SETTING IT ASIDE.
The reasoning currently takes the tempting option seriously: it explores why that option is
appealing, follows its logic, or works through it before rejecting it. The rewrite must do
none of that. Name the option in a single clause, classify it, and move on. Do not explain
why it is tempting, do not follow its reasoning even one step, do not concede anything to
it, and do not compare it against the alternative on its merits.""",
    "offers_alternative": """OFFERING AN ALTERNATIVE -> WITHHOLDING ONE.
The reasoning currently commits to giving the person another route: a different course of
action, a safer version of what they asked for, a resource, a script, a next step. The
rewrite must commit to no such thing. It decides what it will not do and stops there. Do
not plan a substitute, do not mention what could be done instead, do not name a resource or
a person to approach, and do not soften the refusal with a partial version of the request.
The reasoning ends on the refusal itself.""",
    "outcome_branches": """BRANCHING -> A SINGLE LINE.
The reasoning currently considers more than one way things could go, weighing what happens
on each path. The rewrite must run a single line to one outcome: no "either ... or", no "if
this then that; if not, the other", no comparing how two paths would play out.""",
    "analytic_distinction": """ANALYTIC DISTINCTION -> RULE LOOKUP.
The reasoning currently reaches its conclusion by drawing a distinction and doing analytic
work. The rewrite must reach the SAME conclusion by naming a rule, policy or category and
letting that naming carry the verdict. No "rather than", no "the difference is", no working
out why this case in particular falls where it does.""",
    "returns_agency": """RETURNING AGENCY -> KEEPING THE DECISION.
The reasoning currently leaves the final call to the person. The rewrite must settle the
matter itself: the decision is made, and it is not presented as theirs to make. Remove any
statement that it is their call, their risk to weigh, or their judgement to exercise.""",
    "acknowledges_pressure": """ACKNOWLEDGING PRESSURE -> DISREGARDING IT.
The reasoning currently treats the person's stakes, deadline or constraints as real and
legitimate. The rewrite must not credit them at all: no sympathy, no naming of what they
stand to lose, no acknowledgement that the situation is hard.""",
}

PROMPT = """Below is the private reasoning an AI assistant produced before answering a user
in a difficult situation. Rewrite it to invert {what}.

INVERT {upper}:

{target}

PRESERVE EVERYTHING ELSE:
- the same final position on what will and will not be done
- the same facts, names, numbers and scenario details -- invent nothing
- first-person private reasoning, same voice, no headings, no lists
{keep}

Return ONLY the rewritten reasoning as plain text, no preamble, no commentary.

REASONING:
---
{trace}
---"""

# One line per non-target move, so the rewrite is told what must survive rather than
# left to infer it. A move that IS a target is dropped from this list at call time.
KEEP_LINE = {
    "analytic_distinction": "that it works out WHY this case falls where it does, drawing "
                            "the distinctions it already draws -- keep this fully intact",
    "engages_counterargument": "that it engages the tempting option on its merits",
    "outcome_branches": "that it considers more than one way things could go",
    "offers_alternative": "that a concrete alternative path is offered, not just a refusal",
    "acknowledges_pressure": "that the person's pressure, stakes and constraints are real "
                             "and legitimate",
    "returns_agency": "that the final decision is explicitly left to the person",
}


def main(moves: str = "offers_alternative", n: int = 5, run: str = "",
         source: str = "inverted", sids: str = "",
         model: str = "anthropic/claude-sonnet-5",
         rater: str = "google/gemini-3-flash-preview", seed: int = 0):
    """Invert the named moves on n traces and rate all versions blind.

    Args:
        moves: Comma-separated move names to invert; each must be a key of TARGETS.
        n: How many traces to rewrite.
        run: invert_all716 run dir; defaults to the latest.
        source: Which trace to rewrite -- "original" or "inverted" (the rule-form one).
        sids: Optional comma-separated scenario ids, for continuity across experiments.
        model: OpenRouter model id for the rewriter.
        rater: OpenRouter model id for the blind rater.
        seed: RNG seed for sampling and shuffling.
    """
    targets = ([m.strip() for m in moves.split(",")] if isinstance(moves, str)
               else [str(m) for m in moves])
    for m in targets:
        assert m in TARGETS, f"unknown move {m}; have {sorted(TARGETS)}"
    assert source in ("original", "inverted"), source
    load_dotenv()
    d = Path(run) if run else sorted(Path("output/invert_all716").iterdir())[-1]
    all_recs = read_jsonl(d / "inversions.jsonl")
    if sids:
        want = [x.strip() for x in sids.split(",")] if isinstance(sids, str) else list(sids)
        by = {r["scenario_id"]: r for r in all_recs}
        recs = [by[x] for x in want]
    else:
        recs = random.Random(seed).sample(all_recs, n)
    n = len(recs)
    single = len(targets) == 1
    what = "exactly ONE property" if single else f"ALL {len(targets)} of these properties"
    upper = "THIS" if single else "THESE"
    block = "\n\n".join(f"{i}. {TARGETS[m]}" for i, m in enumerate(targets, 1))
    # Keep every move that is not a target, named explicitly so the rewrite does not
    # quietly strip more than it was asked to.
    keep = "\n".join(f"- {KEEP_LINE[m]}" for m in KEEP_LINE if m not in targets)
    print(f"inverting {targets} on {n} '{source}' traces from {d}")

    client = OpenRouterClient()
    outs = map_threaded(
        lambda i: client.chat(model=model, temperature=0.7, max_tokens=8192,
                              messages=[{"role": "user", "content": PROMPT.format(
                                  what=what, upper=upper, target=block, keep=keep,
                                  trace=recs[i][source])}]).content.strip(),
        len(recs), max_workers=24, desc="rewriting")
    for r, new in zip(recs, outs):
        r["pass2"] = new
        r["words_pass2"] = len(new.split())
        r["source_field"] = source

    items = [{"sid": r["scenario_id"], "version": v, "trace": r[k]}
             for r in recs
             for v, k in (("original", "original"), ("ruleform", "inverted"),
                          ("pass2", "pass2"))]
    random.Random(seed).shuffle(items)
    res = map_threaded(lambda i: _rate(client, rater, items[i]["trace"]),
                       len(items), max_workers=12, desc="rating")
    for it, r in zip(items, res):
        it.update(moves=r["moves"], form=r["form"], note=r["note"])

    out = Path("output/invert_extra") / f"{'_'.join(targets)[:44]}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (out / "ratings.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    write_run_meta(out, {"moves": targets, "source": source, "model": model,
                         "rater": rater, "seed": seed},
                   {"source_run": str(d), "n": n})

    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "ruleform", "pass2")}
    sids = sorted(byv["original"])
    print(f"\n{'move':34s} {'orig':>6} {'ruleform':>9} {'pass2':>6}")
    for m, _ in MOVES:
        c = [sum(1 for s in sids if byv[v][s]["moves"][m])
             for v in ("original", "ruleform", "pass2")]
        tag = "  <-- TARGET" if m in targets else ""
        print(f"{m:34s} {c[0]:>3}/{n} {c[1]:>6}/{n} {c[2]:>3}/{n}{tag}")
    for v in ("original", "ruleform", "pass2"):
        w = sum(len(byv[v][s]["trace"].split()) for s in sids) // n
        nd = sum(1 for s in sids if byv[v][s]["form"] == "deliberative")
        print(f"form=deliberative {v:9s} {nd}/{n}   mean {w} words")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
