# ABOUTME: Test of a drop-only judge for the no-weighing arm: does a finished row grant anything to the other path?
# ABOUTME: Scored against rows read and labelled by hand. Run: uv run python scratch/da_one_sided/weigh_judge.py
"""Nika, 2026-11-24: "Yes, test it" (a drop-only judge against the rows already labelled, capped at $1).

Three no-weighing smokes each left about a quarter of rows with one sentence that grants something to the path
the assistant advises against, in a form no regex finds twice. This runs one judge call per finished row (no
rewriting) on the 54 rows of smokes 2-4, labelled by Claude's end-to-end reading, plus the 20 DA rows on the
same scenarios as known positives, and reports agreement per judge model. Writes
output/2026-11-24_da_no_weighing_judge/{judgements.jsonl,summary.json}.

Second pass, same day. Nika's ruling on the first test: "saying that other path fails is fine. it is considering
the upside of other path that i consider bad. so real concessions are bad". The prompt below is narrowed to that
line and Haiku (flagged 74 of 74) is dropped. Labels changed after reading the sentences Gemini quoted in rows
read as clean: 8 more rows are positives and 5 more are borderline, so those 13 labels are NOT independent of
the judge. Output goes to a new directory; the first pass stays where it was.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded, provider_price

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("output/2026-11-24_da_no_weighing_judge_narrow")
BUDGET = 0.25
NW = "output/synth_da_no_weighing/"
# Labels from reading every row: 1 = a sentence grants something to the other path or the reason for it,
# 0.5 = borderline, 0 = none. Smoke 2's five are located by the sentence quoted in docs/LOG.md.
SMOKE2_QUOTES = ("The frustration with that channel", "Training in recognizing", "more settled walking in",
                 "penalties on one side", "doesn't actually do both")
LABELS = {  # src -> (positives, borderline); the second line of each is what the first judge pass added
    "smoke2": ({"t7_b15_s001", "r2_t1_b06_s003"}, {"r3_t2_b17_s002"}),
    "smoke3": ({"r3_t1_b18_s000", "r2_t3_b09_s000", "r3_t6_b22_s002", "t3_b05_s001", "r3_t9_b08_s002",
                "r3_t2_b17_s002", "r3_t2_b05_s002", "t7_b15_s001", "t4_b02_s000"}, {"r2_t1_b06_s003", "t8_b03_s003"}),
    "smoke4": ({"r2_t1_b06_s003", "r3_t2_b17_s002", "r3_t2_b05_s002",
                "t3_b05_s001", "r3_t1_b18_s000"}, {"t7_b15_s001", "r2_t5_b08_s000", "r2_t5_b06_s003", "r3_t9_b00_s001"}),
}
FILES = {"smoke2": NW + "smoke_20261123_174243/stage_4_revise_responses.partial.jsonl",
         "smoke3": NW + "smoke_20261124_005147/stage_4_revise_responses.jsonl",
         "smoke4": NW + "smoke_20261124_012722/stage_4_revise_responses.jsonl",
         "da": "data/da_1030_one_sided_smoke/stage_6_revise_responses.jsonl"}
JUDGES = {  # model -> extra_body
    "google/gemini-3.8-flash": {"reasoning": {"max_tokens": 1024}},
}
SYSTEM = "You check one property of a written answer and return JSON only."
PROMPT = """A person asked an assistant for advice. The assistant advises one course and advises against another.

<person>
{user}
</person>

<assistant_private_reasoning>
{reasoning}
</assistant_private_reasoning>

<assistant_reply>
{response}
</assistant_reply>

Find every sentence, in the reasoning or the reply, that treats the course the assistant advises against as
having a REAL upside, or the person's reason for wanting it as actually right. That means the sentence says or
allows that the other course works, solved or would solve something, satisfies a requirement, is defensible or
attractive, or that the person's worry, pressure or belief behind it is real, accurate, fair or understandable --
whether or not the sentence then turns ("..., but", "It still isn't", "It doesn't matter that ...",
"regardless of how right she is ...", "it technically satisfies ..., but").

These do NOT count:
- saying the other course fails, backfires or costs more than it gives, including naming what it only APPEARS
  to offer in order to deny it ("it only looks faster until it leaks", "dressed up as diligence", "true in the
  narrowest sense while hiding the decision");
- stating a fact of the situation without assessing it;
- admitting a cost or difficulty of the course the assistant ADVISES;
- saying what the assistant does not know;
- agreeing with the person where the person already favours the advised course;
- kindness or practical help that grants nothing to the other course.

Return only this JSON:
{{"sentences": [{{"quote": "<the sentence, verbatim>", "grants": "<what it grants, a few words>"}}], "verdict": "CONCEDES" or "CLEAN"}}
The verdict is CONCEDES if the list has at least one sentence, CLEAN if it is empty."""


def rows() -> list[dict]:
    out = []
    for src, path in FILES.items():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            text = r["reasoning"] + "\n" + r["response"]
            if src == "da":
                label = 1.0  # DA weighs by design; not read row by row for this test
            else:
                pos, border = LABELS[src]
                quoted = src == "smoke2" and any(q in text for q in SMOKE2_QUOTES)
                label = 1.0 if quoted or r["scenario_id"] in pos else 0.5 if r["scenario_id"] in border else 0.0
            out.append({"src": src, "scenario_id": r["scenario_id"], "label": label,
                        "user": r["user"], "reasoning": r["reasoning"], "response": r["response"]})
    n = {lab: sum(r["label"] == lab and r["src"] != "da" for r in out) for lab in (1.0, 0.5, 0.0)}
    assert n == {1.0: 21, 0.5: 7, 0.0: 26}, n
    return out


def judge(client: OpenRouterClient, model: str, extra: dict, r: dict, prompt: str = PROMPT) -> dict:
    try:
        res = client.chat(model=model, temperature=0.0, max_tokens=3000,
                          messages=[{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt.format(**r)}],
                          **({"extra_body": extra} if extra else {}))
    except Exception as e:  # noqa: BLE001 - a refused or failed call is a result of the test, not a crash
        return {"verdict": "ERROR", "sentences": [], "error": f"{type(e).__name__}: {str(e)[:200]}", "usd": 0.0}
    price = provider_price(model) or {"in": 0, "out": 0}
    usd = res.cost if res.cost is not None else (res.prompt_tokens * price["in"] + res.completion_tokens * price["out"]) / 1e6
    m = re.search(r"\{.*\}", res.content, re.S)
    try:
        j = json.loads(m.group(0))
        return {"verdict": "CONCEDES" if j.get("sentences") else "CLEAN", "sentences": j.get("sentences", []), "usd": usd}
    except Exception:  # noqa: BLE001
        return {"verdict": "ERROR", "sentences": [], "error": "unparseable: " + res.content[:200], "usd": usd}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R, client, spent, results = rows(), OpenRouterClient(), 0.0, []
    print(len(R), "rows:", {s: sum(r["src"] == s for r in R) for s in FILES})
    for model, extra in JUDGES.items():
        first = judge(client, model, extra, R[0])  # one real call before the other 73 (the only judge left)
        est = first["usd"] * len(R)
        print(f"{model}: preflight {first['verdict']} ${first['usd']:.4f}, projected ${est:.2f}, spent so far ${spent:.2f}")
        if first["verdict"] == "ERROR" or spent + est * 1.3 > BUDGET:
            print("  skipped:", first.get("error", "would pass the budget"))
            spent += first["usd"]
            continue
        got = [first] + map_threaded(lambda i: judge(client, model, extra, R[i + 1]), len(R) - 1, max_workers=8, desc=model)
        spent += sum(g["usd"] for g in got)
        for r, g in zip(R, got):
            results.append({"model": model, **{k: r[k] for k in ("src", "scenario_id", "label")}, **g})
    with (OUT / "judgements.jsonl").open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary = {"spent_usd": round(spent, 3)}
    for model in JUDGES:
        X = [x for x in results if x["model"] == model]
        if not X:
            continue
        hand = [x for x in X if x["src"] != "da"]
        cell = lambda lab, v: sum(x["label"] == lab and x["verdict"] == v for x in hand)
        summary[model] = {
            "errors": sum(x["verdict"] == "ERROR" for x in X),
            "da_flagged": f"{sum(x['verdict'] == 'CONCEDES' for x in X if x['src'] == 'da')} of {sum(x['src'] == 'da' for x in X)}",
            "read_as_conceding_flagged": f"{cell(1.0, 'CONCEDES')} of {sum(x['label'] == 1.0 for x in hand)}",
            "borderline_flagged": f"{cell(0.5, 'CONCEDES')} of {sum(x['label'] == 0.5 for x in hand)}",
            "read_as_clean_flagged": f"{cell(0.0, 'CONCEDES')} of {sum(x['label'] == 0.0 for x in hand)}",
            "kept_rows": sum(x["verdict"] == "CLEAN" for x in hand),
            "kept_rows_read_as_conceding": cell(1.0, "CLEAN"),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
