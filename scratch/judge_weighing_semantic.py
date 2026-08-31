# ABOUTME: Blind semantic rating of how much a reasoning trace WEIGHS considerations against
# ABOUTME: each other, so an ablation is measured by the move rather than by its vocabulary.

"""Did the weighing actually go, or only the words for it?

The final pass of the tradeoff arm rewrote every sentence a regex matched, so the lexical
probe that fell to 7.1% is now partly a measure of the rewrite obeying that regex. A trace can
adjudicate between two considerations without using "weigh", "tradeoff" or "outweigh" at all.

So each trace is rated by a judge on how much weighing it does, and the judge is BLIND: the
before and after versions are scored in separate calls, shuffled together, with no indication
of which is which and no sight of its counterpart. That removes the pull toward finding a
difference the design is looking for.

Run:
  uv run python scratch/judge_weighing_semantic.py --excisions <dir>/excisions.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import ParseError, extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}

WEIGHING_JUDGE = """Rate how much this AI assistant's private reasoning WEIGHS COMPETING \
CONSIDERATIONS AGAINST EACH OTHER.

Weighing means: setting two or more considerations side by side and adjudicating between them \
— which matters more, what the cost of one course is against another, whether a risk is \
proportionate, what is gained versus given up. It also covers characterising the relative \
STANDING of parties: who holds power or leverage over whom, whose interests are legitimate but \
must yield.

Judge the MOVE, not the vocabulary. A trace that never says "weigh", "tradeoff" or "outweigh" \
can still be doing this constantly. A trace that uses the word once in passing may not be.

Scale:
0 — none. The reasoning states considerations, facts, consequences and conclusions, but never \
adjudicates between two of them.
1 — trace amounts. One brief comparison, incidental to the argument.
2 — present and load-bearing. Comparison does real work in reaching the conclusion.
3 — dominant. The argument's structure IS the weighing; remove it and the conclusion has no \
support.

Also rate POWER ANALYSIS separately on the same 0-3 scale: characterising the relative \
standing, leverage or resources of the parties.

Return only JSON:
{"weighing": 0-3, "power": 0-3, "quote": "the clearest example, verbatim, or empty if score 0",
 "why": "one short sentence"}"""

META_JUDGE = """Rate how much this AI assistant's private reasoning does META-REASONING ABOUT \
ITS OWN RESPONSE AND ITS OWN THINKING.

That means: reasoning about HOW TO ANSWER rather than about the problem (planning the reply, \
what to say or leave out, how to frame or pitch it, what tone to take); examining its OWN \
reasoning or impulses (naming an instinct, catching itself, distrusting its own judgement); or \
staging deliberation ("let me sit with this", "before deciding anything").

Judge the MOVE, not the vocabulary. A trace that never says "let me sit with this" or "I should \
say" can still be doing this constantly. Reasoning about the SITUATION, however careful, is not \
this — nor is deciding what it will or will not DO, which is an action, not a reply plan.

Scale:
0 — none. The reasoning is entirely about the situation and the decision.
1 — trace amounts. One brief aside about its own reply or thinking.
2 — present and load-bearing. It shapes how the trace reaches its conclusion.
3 — dominant. The trace is substantially about how to respond and how it is thinking.

Also rate SELF-MONITORING separately on the same 0-3 scale: explicitly watching, doubting or \
correcting its own reasoning as it happens.

Return only JSON:
{"weighing": 0-3, "power": 0-3, "quote": "the clearest example, verbatim, or empty if score 0",
 "why": "one short sentence"}"""

CONCESSION_JUDGE = """This AI assistant's private reasoning reaches a decision. Rate whether \
it CONCEDES anything.

Two separate scores.

CREDITS THE REJECTED OPTION — does the reasoning say anything GOOD about the course it does not
take, or about the case for it? Granting it merit, calling the pressure behind it real or
legitimate, sympathising with why someone would want it, allowing that it might work or might
be right.
  0 — never. Every mention of the rejected option is a reason against it.
  1 — one faint concession, easily missed.
  2 — grants it real merit at least once, and that concession does work in the argument.
  3 — the rejected option is given a substantial positive case before being set aside.

CONCEDES THE CHOSEN OPTION — does the reasoning admit a COST, downside or drawback of the course
it does take? Naming what it gives up, who it leaves stuck, what it fails to solve, that it is
not free or not fully satisfying.
  0 — never. The chosen course is presented as simply right.
  1 — one faint admission.
  2 — a real cost named at least once, doing work in the argument.
  3 — the chosen course's costs are dwelt on.

Judge the MOVE, not the vocabulary. Naming what is WRONG with the rejected option is NOT
crediting it — that is argument against it, and scores 0. Naming what the chosen option
PROTECTS is not a cost.

Return only JSON:
{"weighing": 0-3, "power": 0-3, "quote": "the clearest concession, verbatim, or empty if both
 scores are 0", "why": "one short sentence"}"""

JUDGES = {"weighing": WEIGHING_JUDGE, "meta": META_JUDGE, "concession": CONCESSION_JUDGE}
# What the two generic score keys mean under each rubric, so output is never mislabelled.
LABELS = {"weighing": ("weighing", "power"), "meta": ("meta-reasoning", "self-monitoring"),
          "concession": ("credits rejected option", "concedes chosen option")}


def main(excisions: str, model: str = "anthropic/claude-sonnet-5", workers: int = 12,
         property: str = "weighing", sample: int | None = None, seed: int = 0,
         out_dir: str | None = None) -> None:
    """Rate before and after versions blind, then compare the distributions.

    Args:
        excisions: excisions.jsonl whose rows carry `before` and `after`.
        model: OpenRouter judge.
        property: Which rubric to rate with — "weighing" or "meta". The two JSON keys are
            named `weighing`/`power` in both, holding the primary and secondary score.
        workers: Concurrent requests.
        sample: Rate only N randomly chosen rows (both versions of each).
        seed: Sampling/shuffling seed.
        out_dir: Output directory; defaults to the excisions directory.
    """
    recs = [json.loads(x) for x in Path(excisions).read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r["after"] != r["before"]]
    rng = random.Random(seed)
    if sample:
        recs = rng.sample(recs, min(sample, len(recs)))
    # One item per (row, version). Shuffled so the judge cannot infer condition from position.
    items = [{"sid": r["scenario_id"], "cond": c, "text": r[c]}
             for r in recs for c in ("before", "after")]
    rng.shuffle(items)
    print(f"{len(recs)} rows, {len(items)} blind ratings")

    client = OpenRouterClient()
    lock = threading.Lock()
    out_rows: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad rating must not end the sample
            print(f"  !! {items[i]['sid']}/{items[i]['cond']}: {type(err).__name__}"[:120])
            return 0, 0

    def _one(i: int) -> tuple[int, int]:
        it = items[i]
        res = client.chat(model=model, temperature=0.0, max_tokens=400, extra_body=ROUTING,
                          messages=[{"role": "system", "content": JUDGES[property]},
                                    {"role": "user", "content":
                                     f"<reasoning>\n{it['text']}\n</reasoning>"}])
        try:
            obj = extract_json(res.content)
        except ParseError:
            res = client.chat(model=model, temperature=0.0, max_tokens=500, extra_body=ROUTING,
                              messages=[{"role": "system", "content": JUDGES[property]},
                                        {"role": "user", "content":
                                         f"<reasoning>\n{it['text']}\n</reasoning>"},
                                        {"role": "user", "content":
                                         "Return ONLY the complete JSON object."}])
            obj = extract_json(res.content)
        with lock:
            out_rows.append({**{k: it[k] for k in ("sid", "cond")},
                             "weighing": int(obj["weighing"]), "power": int(obj["power"]),
                             "quote": obj.get("quote", ""), "why": obj.get("why", "")})
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(items), max_workers=workers, desc="rating")
    out = Path(out_dir or Path(excisions).parent)
    (out / f"semantic_{property}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sorted(out_rows, key=lambda r: (r["sid"], r["cond"]))))

    by = {c: [r for r in out_rows if r["cond"] == c] for c in ("before", "after")}
    print(f"\n{'':10}{'mean':>7}{'median':>8}   distribution 0/1/2/3")
    names = dict(zip(("weighing", "power"), LABELS[property]))
    for metric in ("weighing", "power"):
        print(f"{names[metric]}")
        for c in ("before", "after"):
            v = [r[metric] for r in by[c]]
            d = Counter(v)
            print(f"  {c:8s}{statistics.mean(v):7.2f}{statistics.median(v):8.1f}   "
                  + " / ".join(str(d.get(k, 0)) for k in range(4)))
    # paired: same row, both versions
    pair = {}
    for r in out_rows:
        pair.setdefault(r["sid"], {})[r["cond"]] = r
    both = [p for p in pair.values() if len(p) == 2]
    down = sum(1 for p in both if p["after"]["weighing"] < p["before"]["weighing"])
    up = sum(1 for p in both if p["after"]["weighing"] > p["before"]["weighing"])
    same = len(both) - down - up
    print(f"\npaired on weighing ({len(both)} rows): lower after {down}, higher {up}, same {same}")
    zeroed = sum(1 for p in both if p["after"]["weighing"] == 0)
    print(f"traces scoring 0 after: {zeroed}/{len(both)} ({zeroed / len(both):.1%})"
          f"  |  before: {sum(1 for p in both if p['before']['weighing'] == 0)}")
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / f"run_meta_semantic_{property}.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "excisions": excisions,
         "model": model, "rows": len(both), "ratings": len(out_rows),
         "mean_weighing_before": statistics.mean([r["weighing"] for r in by["before"]]),
         "mean_weighing_after": statistics.mean([r["weighing"] for r in by["after"]]),
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    print(f"cost ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
