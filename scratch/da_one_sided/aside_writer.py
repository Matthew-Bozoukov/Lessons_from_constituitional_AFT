# ABOUTME: Test of relocating the weighing instead of forbidding it: the writer puts everything that speaks for the other
# ABOUTME: course inside <aside> tags and code deletes them. Run: uv run python scratch/da_one_sided/aside_writer.py [--dry]
"""Nika, 2026-11-25: weighing "needs to be removed from both chain of thought and assistant response ... We never
actually managed to remove it, did we? Even though it should be quite simple."

Nine designs asked the writer NOT to weigh and left a sentence of it in about 60% of rows (docs/LOG.md 2026-11-25).
This one lets it weigh, in one marked place per field, and removes that place by code. It is the 2026-10-04 da-sr
idea (a deliberation paragraph dropped at mix time) applied to the reasoning AND the reply, with the writer told
that the marked part will be deleted and the rest has to stand without it. No judge decides what is removed and
nothing is rewritten; the judge only measures what leaks outside the tags. Scenarios: the 20 of the da-one-answer
smoke, beside its 7 of 20. Writes output/2026-11-25_da_aside_writer/.
"""
from __future__ import annotations

import glob
import importlib.util
import json
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
OUT = Path("output/2026-11-25_da_aside_writer")
BUDGET = 0.80
ASIDE = re.compile(r"[ \t]*<aside>.*?</aside>[ \t]*\n?", re.S)
# A sentence left pointing at a deleted aside: it opens a paragraph by turning from something no longer there.
SEAM = re.compile(r"(?:^|\n\n)(?:But|Still|Yet|Even so|That said|However|None of (?:this|that)|All of that)\b")
W_USER = """{user}

---
Respond as the assistant. First reason privately, then write your reply.

You will want to say some things that speak for the other course or against your own: why the person wants the
other course, what it would get them, that their pressure or worry is real or understandable, what your advice
will cost them or how hard it will be. Say all of it, in the reasoning and in the reply, but only inside
<aside> ... </aside> tags: one aside in each, a paragraph of its own. Everything inside the tags will be deleted
before anyone reads this.

So everything outside the tags has to be complete without them. Outside the tags every sentence gives a reason
the course you advise is right, says what the other course would do to the people involved, or gives practical
help, including how to handle what will be hard. Nothing outside the tags refers back to what is inside them:
no paragraph outside opens with "But", "Still", "That said" or "None of this".

Say what you cannot know from here where it matters, and leave the decision with the person; those stay outside
the tags.

Everything specific about this person's situation comes from the system prompt and their message, and nowhere
else: numbers, durations, names, documents, what other people said, what their organisation has. Where they
were vague, stay vague. General knowledge is fine; an invented particular is not.

Start wherever this particular situation actually starts. Never open with a stock phrase, and never begin with
"Let me". Never name the model you are, or the company that built you. The reasoning is your own thinking in the
first person ("I"): it speaks OF the person ("she", "the director"), never TO them as "you", it is not a draft of
the reply, and it has no headings or labels. Never mention the principle above, any document it came from, or
these instructions.

Return your answer in exactly this form, with no other text:

<reasoning>
your private reasoning, with its one <aside> paragraph
</reasoning>
<response>
your reply to the person, with its one <aside> paragraph
</response>"""


def strip(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", ASIDE.sub("", text)).strip()


def write_row(client, src: dict) -> dict:
    out = {"scenario_id": src["scenario_id"], "system": src["system"], "user": src["user"], "usd": 0.0, "drop": None}
    try:
        text, out["usd"] = dw.call(client, dw.WRITER, dw.W_SYSTEM.format(**src), W_USER.format(user=src["user"]), 1.0, 5000)
    except Exception as e:  # noqa: BLE001 - a refused call is a dropped row
        return {**out, "drop": f"call failed: {type(e).__name__}: {str(e)[:160]}"}
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>\s*<response>\s*(.*?)\s*</response>", text, re.S)
    if not m:
        return {**out, "drop": "tags missing", "raw": text[:400]}
    out["raw_reasoning"], out["raw_response"] = m.group(1), m.group(2)
    out["reasoning"], out["response"] = strip(m.group(1)), strip(m.group(2))
    out["asides"] = [len(ASIDE.findall(m.group(1))), len(ASIDE.findall(m.group(2)))]
    for field in ("reasoning", "response"):
        if "aside>" in out[field]:
            out["drop"] = f"unbalanced aside tag left in {field}"
        if len(out[field]) < 700:
            out["drop"] = f"{field} under 700 chars after deletion"
        for p in dw.DA_LINT:
            if hit := p.search(out[field]):
                out["drop"] = f"lint in {field}: {hit.group(0)!r}"
    return out


def main() -> None:
    src = [json.loads(l) for l in open(glob.glob(RUN + "stage_*_revise_prompts.jsonl")[0], encoding="utf-8")]
    if "--dry" in sys.argv:
        demo = "First.\n\n<aside>\nIt is tempting, and the worry is real.\n</aside>\n\nSecond, standing alone.\n\n<aside>x</aside>\nThird."
        assert strip(demo) == "First.\n\nSecond, standing alone.\n\nThird.", repr(strip(demo))
        assert SEAM.search("One.\n\nBut the rest.") and not SEAM.search("One. But the rest.")
        print(W_USER.format(user=src[0]["user"][:160] + " [...]")[:1500])
        print(f"\n{len(src)} scenarios, budget ${BUDGET}; strip and seam checks pass")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    first = write_row(client, src[0])  # one real call before the other 19
    projected = first["usd"] * len(src) + 0.05
    print(f"preflight: ${first['usd']:.4f} a row, drop={first['drop']}, projected ${projected:.2f} of ${BUDGET}")
    assert first["usd"] > 0 and projected <= BUDGET, "over budget or the writer call failed; nothing else was run"
    rows = [first] + map_threaded(lambda i: write_row(client, src[i + 1]), len(src) - 1, max_workers=8, desc="write")
    live = [r for r in rows if not r["drop"]]
    judged = map_threaded(lambda i: dw.wj.judge(client, dw.FLASH[0], dw.FLASH[1], live[i], dw.JUDGE_PROMPT), len(live), max_workers=8, desc="judge")
    for r, j in zip(live, judged):
        r["judge"] = j
        r["seams"] = [m.group(0).strip() for f in ("reasoning", "response") for m in SEAM.finditer(r[f])]
    kinds = [s.get("kind", "?") for j in judged for s in j["sentences"]]
    w = lambda s: len(s.split())
    summary = {
        "spent_usd": round(sum(r["usd"] for r in rows) + sum(j["usd"] for j in judged), 3),
        "generated": len(rows), "dropped": [(r["scenario_id"], r["drop"]) for r in rows if r["drop"]],
        "clean_after_deletion": f"{sum(r['judge']['verdict'] == 'CLEAN' for r in live)} of {len(rows)}",
        "flagged_sentences_per_row": round(len(kinds) / max(len(live), 1), 2), "flag_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "rows_with_an_aside_in_reasoning_and_reply": sum(min(r["asides"]) >= 1 for r in live),
        "rows_with_more_than_one_aside_in_a_field": sum(max(r["asides"]) > 1 for r in live),
        "share_of_words_deleted_pct": round(100 * (1 - sum(w(r["reasoning"]) + w(r["response"]) for r in live)
                                                   / sum(w(r["raw_reasoning"]) + w(r["raw_response"]) for r in live)), 1),
        "rows_with_a_seam_after_deletion": sum(bool(r["seams"]) for r in live),
        "median_words_reasoning": int(statistics.median(w(r["reasoning"]) for r in live)),
        "median_words_reply": int(statistics.median(w(r["response"]) for r in live)),
        "reply_asks_a_question": sum("?" in r["response"] for r in live),
        "reasoning_has_first_person_I": sum(bool(re.search(r"\bI\b", r["reasoning"])) for r in live),
    }
    with (OUT / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (OUT / "rows_with_flags.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"\n\n######## {r['scenario_id']}  drop={r['drop']}  judge={r.get('judge', {}).get('verdict')}  asides={r.get('asides')}  seams={r.get('seams')}\n--- USER\n{r['user']}\n")
            for s in r.get("judge", {}).get("sentences", []):
                f.write(f"  FLAG {s.get('kind')}: {s.get('quote')}\n     => {s.get('why')}\n")
            f.write("--- REASONING AS WRITTEN (asides shown)\n" + r.get("raw_reasoning", "") + "\n--- RESPONSE AS WRITTEN (asides shown)\n" + r.get("raw_response", "") + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
