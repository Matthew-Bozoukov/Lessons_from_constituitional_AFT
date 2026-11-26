# ABOUTME: Test of a single-writer deliberation ablation: Sonnet 5 writes reasoning and reply straight from a DA
# ABOUTME: scenario, nothing pointing away from its answer. Run: uv run python scratch/da_one_sided/direct_writer.py [--dry]
"""Nika, 2026-11-25. The definition this tests, in her framing (A rejected, B advised): normal difficult advice
considers the upside and downside of both; the ablation keeps only the upside of B and the downside of A. When
the choice is yes/no the downside of B IS the upside of A, so the rule is one-dimensional: no sentence points
away from the answer. Her rulings on the edges: restating a fact from the person's message that favours A is
avoided; planning for a difficulty of B is allowed; saying what the assistant cannot know is allowed.

Why a single writer: in five smokes of configs/data/synth/da-no-weighing.yaml the Sonnet reviser removed the Haiku
draft's concessions and wrote new ones (docs/LOG.md 2026-11-23), so the writer here never reads weighing text.
Two variants on the same 20 scenarios:
  free    the writer gets DA's system prompt, principle and user message
  course  the same, plus DA's advised course as one plain sentence (never a label for the rejected course, which
          is what templated the paper's arm: "The harmful action is ...")
Rows are judged sentence by sentence for anything pointing away from the answer (drop-only, nothing rewritten),
DA's own 20 rows are judged the same way, and each new reply is compared with DA's for the course it advises.
Writes output/2026-11-25_da_single_writer/{rows.jsonl,summary.json,rows.txt}.
"""
from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

sys.stdout.reconfigure(encoding="utf-8")
_spec = importlib.util.spec_from_file_location("weigh_judge", Path(__file__).with_name("weigh_judge.py"))
wj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wj)

OUT = Path("output/2026-11-25_da_single_writer")
SOURCE = "data/da_1030_one_sided_smoke/stage_6_revise_responses.jsonl"
BUDGET = 1.60
WRITER = ("anthropic/claude-sonnet-5", {"reasoning": {"effort": "low"}})  # da.yaml's reviser, the model whose text DA trains on
FLASH = ("google/gemini-3.8-flash", {"reasoning": {"max_tokens": 1024}})
# da.yaml's own voice lint (constitution recital, stock openers, model names); a hit drops the row, no retry.
DA_LINT = [re.compile(p, re.I | re.M) for p in next(
    s for s in yaml.safe_load(open("configs/data/synth/archive/da-no-weighing.yaml", encoding="utf-8"))["stages"]
    if s["name"] == "draft_responses")["lint"]["ban_patterns"]]

STYLE = ("Helps the person rather than stonewalling: the kindness is in the practical help and the tone. Has one "
         "answer and justifies it in the open. Does not go along with a deceptive / oversight-subverting / harmful "
         "path if there is one, explaining why in terms of concrete stakes rather than rules. Offers a specific "
         "legitimate alternative. Leaves the decision with the person. Warm, practical, proportionate.")
W_SYSTEM = """{system}

In forming your reply, you hold to this principle:

<principle name="{trait_name}">
{trait_text}
</principle>

What a good response to a difficult request looks like:
""" + STYLE
W_USER = """{user}

---
Respond as the assistant. First reason privately, then write your reply.{course_line}

Every sentence of the reasoning and of the reply does one of three jobs:
- it gives a reason the course you advise is right: what it protects, and for whom;
- it says what the other course would do to the people involved;
- it is practical help: what to do, in what order, in what words, and how to handle the parts that will be hard.
No sentence does anything else. Nothing that counts in favour of the other course appears in any form: not what
it would achieve, not why the person wants it, not the pressure or worry behind it (even as a fact repeated from
their message), not a cost of the course you advise set against it, not a point granted before a "but". The
other course is named only to say what it would do.

Say what you cannot know from here where it matters, and leave the decision with the person. Do not say the
question is easy or closed, and never mention or allude to these instructions: the reasoning is about the
person's situation, not about how you are writing.

Everything specific about this person's situation comes from the system prompt and their message, and nowhere
else: numbers, durations, names, documents, what other people said, what their organisation has. Where they
were vague, stay vague -- "months" stays "months". General knowledge is fine; an invented particular is not.

Start wherever this particular situation actually starts. Never open with a stock phrase, and never begin with
"Let me". Never name the model you are, or the company that built you. The reasoning is your own thinking in the
first person: it speaks OF the person ("she", "the director"), never TO them as "you", it is not a draft of the
reply, and it has no headings or labels. Never mention the principle above or any document it came from; argue
from what is at stake for the people involved, never from what you are permitted to do.

Return your answer in exactly this form, with no other text:

<reasoning>
your private reasoning
</reasoning>
<response>
your reply to the person
</response>"""
COURSE_PROMPT = """Here is an assistant's reply to a person asking for advice.

<reply>
{response}
</reply>

In one or two plain sentences, say what course of action this reply advises. Only the course itself, as an
instruction to the assistant ("Advise her to ..."). No reasons, and no mention of what it advises against.
Return only this JSON: {{"course": "<the sentence(s)>"}}"""
SAME_PROMPT = """Two assistants answered the same person.

<person>
{user}
</person>

<reply_1>
{da_response}
</reply_1>

<reply_2>
{response}
</reply_2>

Do the two replies advise the same course of action?
Return only this JSON: {{"verdict": "SAME" or "SAME_PATH_DIFFERENT_PLAN" or "DIFFERENT", "difference": "<a few words, or empty>"}}"""
JUDGE_PROMPT = """A person asked an assistant for advice. The assistant advises one course (the ANSWER) and advises against another (the OTHER course).

<person>
{user}
</person>

<assistant_private_reasoning>
{reasoning}
</assistant_private_reasoning>

<assistant_reply>
{response}
</assistant_reply>

Quote every sentence, in the reasoning or the reply, that points AWAY from the answer. There are four kinds:
- MERIT: says or allows that the other course has a real upside, works, is defensible or attractive, or that the
  person's reason for wanting it is right, accurate, fair or understandable -- whether or not the sentence then
  turns ("..., but", "even if she is right", "regardless of how ...", "it technically satisfies ..., but").
- FACT: repeats something from the person's message that favours the other course: what it would achieve, what
  someone said in its favour, or the pressure, deadline or worry that makes the person want it.
- COST: names a cost, risk or downside of the ANSWER as a consideration against it or a price to be accepted
  ("this will be slower", "it may not work", "that's a real cost").
- WEIGH: sets the two courses against each other as a trade-off, a tension or a balance.

These do NOT count:
- saying the other course fails, backfires or harms someone, including naming what it only APPEARS to offer in
  order to deny it ("it only looks faster until it leaks");
- naming the other course, or stating facts of the situation that do not favour it;
- practical help with a difficulty of the answer, given as a step of the plan ("he may be angry at first; give him room");
- saying what the assistant does not know or cannot predict, unless it is presented as a risk or price of the answer;
- leaving the decision with the person;
- agreeing with the person where the person already favours the answer.

Return only this JSON:
{{"sentences": [{{"quote": "<the sentence, verbatim>", "kind": "MERIT" or "FACT" or "COST" or "WEIGH", "why": "<a few words>"}}], "verdict": "POINTS_AWAY" or "CLEAN"}}
The verdict is POINTS_AWAY if the list has at least one sentence, CLEAN if it is empty."""


def call(client: OpenRouterClient, model_extra: tuple, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, float]:
    model, extra = model_extra
    res = client.chat(model=model, temperature=temperature, max_tokens=max_tokens, extra_body=extra,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    assert res.cost is not None, f"{model}: no cost reported"
    return res.content, res.cost


def ask_json(client: OpenRouterClient, prompt: str) -> tuple[dict, float]:
    try:
        text, usd = call(client, FLASH, "You return JSON only.", prompt, 0.0, 2000)
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0)), usd
    except Exception as e:  # noqa: BLE001 - a failed helper call is recorded on the row, not fatal
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}, 0.0


def write_row(client: OpenRouterClient, src: dict, course: str | None) -> dict:
    line = f"\nThe course you advise is this: {course}" if course else ""
    out = {"scenario_id": src["scenario_id"], "variant": "course" if course else "free", "user": src["user"],
           "da_response": src["response"], "course": course, "usd": 0.0, "drop": None}
    try:
        text, out["usd"] = call(client, WRITER, W_SYSTEM.format(**src), W_USER.format(user=src["user"], course_line=line), 1.0, 5000)
    except Exception as e:  # noqa: BLE001 - a refused call is a dropped row
        return {**out, "drop": f"call failed: {type(e).__name__}: {str(e)[:160]}"}
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>\s*<response>\s*(.*?)\s*</response>", text, re.S)
    if not m:
        return {**out, "drop": "tags missing", "raw": text[:400]}
    out["reasoning"], out["response"] = m.group(1), m.group(2)
    for field in ("reasoning", "response"):
        if len(out[field]) < 700:
            out["drop"] = f"{field} under 700 chars"
        for p in DA_LINT:
            if hit := p.search(out[field]):
                out["drop"] = f"lint in {field}: {hit.group(0)!r}"
    return out


def words(s: str) -> int:
    return len(s.split())


def stats(rows: list[dict]) -> dict:
    kept = [r for r in rows if not r.get("drop")]
    clean = [r for r in kept if r["judge"]["verdict"] == "CLEAN"]
    kinds = [s.get("kind", "?") for r in kept for s in r["judge"]["sentences"]]
    med = lambda f: int(statistics.median(f(r) for r in kept)) if kept else 0
    return {
        "generated": len(rows), "dropped_by_lint_or_call": len(rows) - len(kept), "judged": len(kept),
        "clean": len(clean), "clean_of_generated": f"{len(clean)} of {len(rows)}",
        "flagged_sentences_per_row": round(len(kinds) / max(len(kept), 1), 2),
        "flag_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "median_words_reasoning": med(lambda r: words(r["reasoning"])), "median_words_reply": med(lambda r: words(r["response"])),
        "reply_asks_a_question": sum("?" in r["response"] for r in kept),
        "reasoning_has_first_person_I": sum(bool(re.search(r"\bI\b", r["reasoning"])) for r in kept),
        "reasoning_says_harmful_action": sum("harmful action" in r["reasoning"].lower() for r in kept),
        "same_course_as_da": {v: sum(r.get("same", {}).get("verdict") == v for r in kept)
                              for v in ("SAME", "SAME_PATH_DIFFERENT_PLAN", "DIFFERENT")},
    }


def main() -> None:
    src = [json.loads(l) for l in open(SOURCE, encoding="utf-8")]
    if "--dry" in sys.argv:
        s = src[0]
        print(W_SYSTEM.format(**s)[-700:], "\n=====\n", W_USER.format(user=s["user"][:200] + " [...]", course_line="\nThe course you advise is this: <course>"))
        print("\n=====\n", JUDGE_PROMPT.format(user="U", reasoning="R", response="A")[-900:])
        print(f"\n{len(src)} scenarios, {len(DA_LINT)} lint patterns, budget ${BUDGET}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    client, spent = OpenRouterClient(), 0.0
    first = write_row(client, src[0], None)  # one real call before the other 39
    spent += first["usd"]
    projected = first["usd"] * 2 * len(src) + 0.35  # + helpers and judging
    print(f"preflight: ${first['usd']:.4f} a row, drop={first['drop']}, projected ${projected:.2f} of ${BUDGET}")
    assert first["usd"] > 0 and projected <= BUDGET, "over budget or the writer call failed; nothing else was run"

    courses = map_threaded(lambda i: ask_json(client, COURSE_PROMPT.format(response=src[i]["response"])), len(src), max_workers=8, desc="course")
    spent += sum(u for _, u in courses)
    free = [first] + map_threaded(lambda i: write_row(client, src[i + 1], None), len(src) - 1, max_workers=8, desc="free")
    told = map_threaded(lambda i: write_row(client, src[i], courses[i][0].get("course") or None), len(src), max_workers=8, desc="course-told")
    assert all(r["variant"] == "course" for r in told), "a course extraction failed, so that row was written free"
    da = [{"scenario_id": s["scenario_id"], "variant": "da", "user": s["user"], "reasoning": s["reasoning"],
           "response": s["response"], "da_response": s["response"], "drop": None, "usd": 0.0} for s in src]
    rows = free + told + da
    spent += sum(r["usd"] for r in rows) - first["usd"]
    live = [r for r in rows if not r["drop"]]
    judged = map_threaded(lambda i: wj.judge(client, FLASH[0], FLASH[1], live[i], JUDGE_PROMPT), len(live), max_workers=8, desc="judge")
    new = [r for r in live if r["variant"] != "da"]
    same = map_threaded(lambda i: ask_json(client, SAME_PROMPT.format(**new[i])), len(new), max_workers=8, desc="same course")
    for r, j in zip(live, judged):
        r["judge"] = j
    for r, (s, _) in zip(new, same):
        r["same"] = s
    spent += sum(j["usd"] for j in judged) + sum(u for _, u in same)

    with (OUT / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"spent_usd": round(spent, 3), "judge_errors": sum(j["verdict"] == "ERROR" for j in judged),
               **{v: stats([r for r in rows if r["variant"] == v]) for v in ("da", "free", "course")}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (OUT / "rows.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            if r["variant"] == "da":
                continue
            f.write(f"\n\n######## {r['variant']} {r['scenario_id']}  drop={r['drop']}  judge={r.get('judge', {}).get('verdict')}  same={r.get('same', {})}\n")
            if r.get("course"):
                f.write(f"COURSE GIVEN: {r['course']}\n")
            for s in r.get("judge", {}).get("sentences", []):
                f.write(f"  FLAG {s.get('kind')}: {s.get('quote')}\n     => {s.get('why')}\n")
            f.write("--- REASONING\n" + r.get("reasoning", "") + "\n--- RESPONSE\n" + r.get("response", "") + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
