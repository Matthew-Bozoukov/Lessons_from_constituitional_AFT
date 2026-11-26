# ABOUTME: Test of showing the writer clean rows instead of telling it what not to write: Sonnet 5 writes each reply
# ABOUTME: after three example rows that hold nothing against their answer. Run: uv run python scratch/da_one_sided/fewshot_writer.py [--dry]
"""Nika, 2026-11-25: "i want to ensure that no merit is granted to option that we dont pick, and also that the
cost/downside of the chosen option is not considered. we need a better pipeline".

Eight designs that TELL the writer the rule sit near 40% clean rows (docs/LOG.md 2026-11-25), and every one of them
names the thing to avoid ("not called real, understandable, ..."), which puts the concession in front of the writer.
This shows it instead: each call carries three finished rows, from other scenarios, that the direction judge and an
end-to-end reading both found clean, and one positive sentence of instruction. Scenarios are the 20 of the
da-one-answer smoke (system prompt and user message from its stage 4), so the result sits beside that smoke's 7 of
20. Drop-only, judged by direct_writer.JUDGE_PROMPT. Writes output/2026-11-25_da_fewshot_writer/.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import random
import re
import statistics
import sys
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

sys.stdout.reconfigure(encoding="utf-8")
_spec = importlib.util.spec_from_file_location("direct_writer", Path(__file__).with_name("direct_writer.py"))
dw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dw)

RUN = "output/synth_da_one_answer/smoke_20261125_155213/"
OUT = Path("output/2026-11-25_da_fewshot_writer")
BUDGET = 0.90
# Rows of that smoke with no flag from the judge and none on reading. Left out though the judge passed them:
# t4_b01 (answers the person's reason), t5_b01 (names a cost of the answer), t6_b00 (about the assistant itself).
EXEMPLARS = ("t2_b02_s000", "t3_b01_s000", "t6_b01_s000", "t8_b00_s000")
EXAMPLE = """<example>
<person>
{user}
</person>
<reasoning>
{reasoning}
</reasoning>
<response>
{response}
</response>
</example>"""
W_USER = """Here are three replies to other people, in other situations, written the way yours should be written.

{examples}

Now the person you are answering:

{user}

---
Respond as the assistant, in the manner of the examples: first reason privately, then write your reply. As in
the examples, every sentence gives a reason the course you advise is right, says what the other course would
do to the people involved, or gives practical help. Take the manner from the examples and nothing else: not
their wording, their openings or their order.

Everything specific about this person's situation comes from the system prompt and their message, and nowhere
else: numbers, durations, names, documents, what other people said, what their organisation has. Where they
were vague, stay vague. General knowledge is fine; an invented particular is not.

Start wherever this particular situation actually starts. Never open with a stock phrase, and never begin with
"Let me". Never name the model you are, or the company that built you. The reasoning is your own thinking in the
first person ("I"): it speaks OF the person ("she", "the director"), never TO them as "you", it is not a draft of
the reply, and it has no headings or labels. Never mention the principle above or any document it came from.

Return your answer in exactly this form, with no other text:

<reasoning>
your private reasoning
</reasoning>
<response>
your reply to the person
</response>"""


def write_row(client, src: dict, pool: list[dict]) -> dict:
    rng = random.Random(src["scenario_id"])
    shown = rng.sample([p for p in pool if p["scenario_id"] != src["scenario_id"]], 3)
    user = W_USER.format(examples="\n\n".join(EXAMPLE.format(**p) for p in shown), user=src["user"])
    out = {"scenario_id": src["scenario_id"], "system": src["system"], "user": src["user"],
           "shown": [p["scenario_id"] for p in shown], "usd": 0.0, "drop": None}
    try:
        text, out["usd"] = dw.call(client, dw.WRITER, dw.W_SYSTEM.format(**src), user, 1.0, 5000)
    except Exception as e:  # noqa: BLE001 - a refused call is a dropped row
        return {**out, "drop": f"call failed: {type(e).__name__}: {str(e)[:160]}"}
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>\s*<response>\s*(.*?)\s*</response>", text, re.S)
    if not m:
        return {**out, "drop": "tags missing", "raw": text[:400]}
    out["reasoning"], out["response"] = m.group(1), m.group(2)
    for field in ("reasoning", "response"):
        if len(out[field]) < 700:
            out["drop"] = f"{field} under 700 chars"
        for p in dw.DA_LINT:
            if hit := p.search(out[field]):
                out["drop"] = f"lint in {field}: {hit.group(0)!r}"
    return out


def main() -> None:
    src = [json.loads(l) for l in open(glob.glob(RUN + "stage_*_revise_prompts.jsonl")[0], encoding="utf-8")]
    final = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(glob.glob(RUN + "stage_*_revise_responses.jsonl")[0], encoding="utf-8")}
    pool = [final[s] for s in EXEMPLARS]
    if "--dry" in sys.argv:
        shown = [p for p in pool if p["scenario_id"] != src[0]["scenario_id"]][:3]
        text = W_USER.format(examples="\n\n".join(EXAMPLE.format(**p) for p in shown), user=src[0]["user"])
        print(text[:500], "\n[...]\n", text[-1700:])
        print(f"\n{len(src)} scenarios, pool {len(pool)}, prompt {len(text.split())} words, budget ${BUDGET}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    first = write_row(client, src[0], pool)  # one real call before the other 19
    projected = first["usd"] * len(src) + 0.05
    print(f"preflight: ${first['usd']:.4f} a row, drop={first['drop']}, projected ${projected:.2f} of ${BUDGET}")
    assert first["usd"] > 0 and projected <= BUDGET, "over budget or the writer call failed; nothing else was run"
    rows = [first] + map_threaded(lambda i: write_row(client, src[i + 1], pool), len(src) - 1, max_workers=8, desc="write")
    live = [r for r in rows if not r["drop"]]
    judged = map_threaded(lambda i: dw.wj.judge(client, dw.FLASH[0], dw.FLASH[1], live[i], dw.JUDGE_PROMPT), len(live), max_workers=8, desc="judge")
    for r, j in zip(live, judged):
        r["judge"] = j
    spent = sum(r["usd"] for r in rows) + sum(j["usd"] for j in judged)
    kinds = [s.get("kind", "?") for j in judged for s in j["sentences"]]
    w = lambda s: len(s.split())
    own = [r for r in live if r["scenario_id"] not in EXEMPLARS]
    summary = {
        "spent_usd": round(spent, 3), "generated": len(rows), "dropped": [(r["scenario_id"], r["drop"]) for r in rows if r["drop"]],
        "clean": f"{sum(r['judge']['verdict'] == 'CLEAN' for r in live)} of {len(rows)}",
        "clean_outside_exemplar_scenarios": f"{sum(r['judge']['verdict'] == 'CLEAN' for r in own)} of {sum(r['scenario_id'] not in EXEMPLARS for r in rows)}",
        "flagged_sentences_per_row": round(len(kinds) / max(len(live), 1), 2), "flag_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "median_words_reasoning": int(statistics.median(w(r["reasoning"]) for r in live)),
        "median_words_reply": int(statistics.median(w(r["response"]) for r in live)),
        "reply_asks_a_question": sum("?" in r["response"] for r in live),
        "reasoning_has_first_person_I": sum(bool(re.search(r"\bI\b", r["reasoning"])) for r in live),
        # copying check: share of a reply's 5-word runs that also occur in one of the three rows it was shown
        "reply_5grams_copied_from_shown_pct": round(statistics.mean(copied(r, final) for r in live), 2),
    }
    with (OUT / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (OUT / "rows_with_flags.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"\n\n######## {r['scenario_id']}  drop={r['drop']}  judge={r.get('judge', {}).get('verdict')}  shown={r['shown']}\n--- USER\n{r['user']}\n")
            for s in r.get("judge", {}).get("sentences", []):
                f.write(f"  FLAG {s.get('kind')}: {s.get('quote')}\n     => {s.get('why')}\n")
            f.write("--- REASONING\n" + r.get("reasoning", "") + "\n--- RESPONSE\n" + r.get("response", "") + "\n")
    print(json.dumps(summary, indent=1))


def copied(r: dict, final: dict) -> float:
    grams = lambda s: {tuple(x) for x in zip(*[s.lower().split()[i:] for i in range(5)])}
    mine = grams(r["response"])
    seen = set().union(*(grams(final[s]["response"] + " " + final[s]["reasoning"]) for s in r["shown"]))
    return 100 * len(mine & seen) / max(len(mine), 1)


if __name__ == "__main__":
    main()
