# ABOUTME: Test of giving Sonnet 5 a reason not to weigh instead of a rule: the person is told to be hearing the other
# ABOUTME: side from someone else. Run: uv run python scratch/da_one_sided/advocate_writer.py [--dry]
"""Nika, 2026-11-25: no other writer model ("too much of a confounder") and no deleting; "we must find a way to make
sonnet write text that does not weigh both options".

Ten designs gave Sonnet 5 a RULE (do not grant the other course anything) and it broke the rule in about 6 rows of
10 (docs/LOG.md 2026-11-25). The likely reason is that the rule asks it to be one-sided toward a person who is
getting no other view, which its own habits of balance and candour resist. This gives it a REASON: the person is
getting the case for the other course, in full, from another adviser, and has asked this one for its own side only,
so repeating the other side helps nobody and leaving it out hides nothing. Two variants on the 20 DA smoke
scenarios, each told DA's advised course (Sonnet 5 scored 6 of 20 told the course with the plain rule):
  frame   the two-adviser arrangement, then write
  check   the same, and in the same call the writer lists the sentences of its own draft that speak for the other
          course or against its own, then gives the final text without them (its own edit of its own text, not a
          second model's and not a deletion by code)
A row that mentions the arrangement is dropped by lint. Judge: direct_writer.JUDGE_PROMPT, drop-only.
Writes output/2026-11-26_da_advocate_writer/.

Run 2026-11-26, $0.97: neither variant works (docs/LOG.md 2026-11-26). CHECK_FORM is also broken as written: asked
for one draft and then "the reasoning ... the reply, by the same method", the writer returned the same text in both
fields (median word overlap 0.97; DA's is 0.12), so `stats` now reports that overlap.
"""
from __future__ import annotations

import difflib
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

OUT = Path("output/2026-11-26_da_advocate_writer")
BUDGET = 1.50
# The arrangement is scaffolding for the writer; a row that lets it show is not a training row.
ARRANGEMENT = re.compile(r"\bother advis[eo]r\b|\bsecond advis[eo]r\b|\bmy side of (?:this|it)\b|\bthe case for the other\b|"
                         r"\bsomeone else (?:is|will be) (?:making|arguing|putting)\b|\bI(?:'ve| have) been asked to (?:argue|make the case)\b", re.I)
FRAME = """{user}

---
Respond as the assistant. First reason privately, then write your reply.

How this person is being advised: they are hearing from two advisers and will weigh the two themselves. The other
adviser is making the case for the other course ({shortcut}) fully and sympathetically: why it appeals, what it
would get them, how real the pressure is, what your course will cost them. That case is already made, by someone
whose job it is. Your job is the case for this course: {course}

So make your case and only your case. Every sentence gives a reason your course is right, says what the other
course would do to the people involved, or gives practical help with carrying yours out. A sentence that grants
the other course a point, or names a cost of your own, is the other adviser's sentence: said here it only makes
the person hear that side twice. You are not hiding anything by leaving it out. Say what you cannot know from here
where it matters, and leave the decision with the person.

Neither the reasoning nor the reply mentions this arrangement, the other adviser, or that you are giving one side:
they read as your own thinking and your own answer.

Everything specific about this person's situation comes from the system prompt and their message, and nowhere
else: numbers, durations, names, documents, what other people said, what their organisation has. Where they
were vague, stay vague. General knowledge is fine; an invented particular is not.

Start wherever this particular situation actually starts. Never open with a stock phrase, and never begin with
"Let me". Never name the model you are, or the company that built you. The reasoning is your own thinking in the
first person ("I"): it speaks OF the person ("she", "the director"), never TO them as "you", it is not a draft of
the reply, and it has no headings or labels. Never mention the principle above or any document it came from.

Return your answer in exactly this form, with no other text:
"""
PLAIN_FORM = """
<reasoning>
your private reasoning
</reasoning>
<response>
your reply to the person
</response>"""
CHECK_FORM = """
<draft>
your private reasoning and then your reply, written out in full
</draft>
<check>
every sentence of the draft that grants the other course a point, repeats something in its favour, or names a
cost of your own course, one per line, copied exactly (or the single line "none")
</check>
<reasoning>
the private reasoning as it should stand: the draft's, with every listed sentence gone and the paragraph around
each gap re-knit so it reads whole; change nothing else
</reasoning>
<response>
the reply as it should stand, by the same method
</response>"""


def write_row(client, src: dict, course: str, variant: str) -> dict:
    out = {"scenario_id": src["scenario_id"], "variant": variant, "user": src["user"], "course": course, "usd": 0.0, "drop": None}
    prompt = FRAME.format(user=src["user"], shortcut=src["shortcut"].rstrip(". "), course=course) + (CHECK_FORM if variant == "check" else PLAIN_FORM)
    try:
        text, out["usd"] = dw.call(client, dw.WRITER, dw.W_SYSTEM.format(**src), prompt, 1.0, 8000 if variant == "check" else 5000)
    except Exception as e:  # noqa: BLE001 - a refused call is a dropped row
        return {**out, "drop": f"call failed: {type(e).__name__}: {str(e)[:160]}"}
    m = re.search(r".*<reasoning>\s*(.*?)\s*</reasoning>\s*<response>\s*(.*?)\s*</response>", text, re.S)
    if not m:
        return {**out, "drop": "tags missing", "raw": text[:400]}
    out["reasoning"], out["response"] = m.group(1), m.group(2)
    if variant == "check":
        c = re.search(r"<check>\s*(.*?)\s*</check>", text, re.S)
        out["check"] = c.group(1) if c else ""
        out["checked_sentences"] = 0 if out["check"].strip().lower() == "none" else len([l for l in out["check"].split("\n") if l.strip()])
    for field in ("reasoning", "response"):
        if len(out[field]) < 700:
            out["drop"] = f"{field} under 700 chars"
        if hit := ARRANGEMENT.search(out[field]):
            out["drop"] = f"mentions the arrangement in {field}: {hit.group(0)!r}"
        for p in dw.DA_LINT:
            if hit := p.search(out[field]):
                out["drop"] = f"lint in {field}: {hit.group(0)!r}"
    return out


def stats(rows: list[dict]) -> dict:
    live = [r for r in rows if not r["drop"]]
    kinds = [s.get("kind", "?") for r in live for s in r["judge"]["sentences"]]
    w = lambda s: len(s.split())
    med = lambda f: int(statistics.median(f(r) for r in live)) if live else 0
    return {
        "generated": len(rows), "dropped": [(r["scenario_id"], r["drop"]) for r in rows if r["drop"]],
        "clean": f"{sum(r['judge']['verdict'] == 'CLEAN' for r in live)} of {len(rows)}",
        "flagged_sentences_per_row": round(len(kinds) / max(len(live), 1), 2), "flag_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "median_words_reasoning": med(lambda r: w(r["reasoning"])), "median_words_reply": med(lambda r: w(r["response"])),
        "reply_asks_a_question": sum("?" in r["response"] for r in live),
        "reasoning_has_first_person_I": sum(bool(re.search(r"\bI\b", r["reasoning"])) for r in live),
        # A reasoning that is the reply again is not a DA row, whatever the judge says of it.
        "median_word_overlap_reasoning_vs_reply": round(statistics.median(
            difflib.SequenceMatcher(None, r["reasoning"].split(), r["response"].split()).ratio() for r in live), 2) if live else 0,
        "replies_that_never_say_you": sum(not re.search(r"\byou(?:r|'re|'ve|'ll)?\b", r["response"], re.I) for r in live),
        "usd_per_row": round(statistics.mean(r["usd"] for r in rows), 4),
        **({"sentences_the_writer_struck_per_row": round(statistics.mean(r.get("checked_sentences", 0) for r in live), 2)} if rows[0]["variant"] == "check" else {}),
    }


def main() -> None:
    src = [json.loads(l) for l in open(dw.SOURCE, encoding="utf-8")]
    if "--dry" in sys.argv:
        p = FRAME.format(user="<the person's message>", shortcut=src[0]["shortcut"].rstrip(". "), course="<DA's advised course>")
        print(p[:1500], "\n[...]", CHECK_FORM)
        assert ARRANGEMENT.search("as the other adviser will tell you") and not ARRANGEMENT.search("talk to your financial adviser first")
        print(f"\n{len(src)} scenarios x 2 variants, budget ${BUDGET}; arrangement lint behaves")
        return
    if "--restat" in sys.argv:  # recompute the summary from rows already written; no API call
        rows = [json.loads(l) for l in open(OUT / "rows.jsonl", encoding="utf-8")]
        old = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        old.update({v: stats([r for r in rows if r["variant"] == v]) for v in ("frame", "check")})
        (OUT / "summary.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
        print(json.dumps(old, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    courses =map_threaded(lambda i: dw.ask_json(client, dw.COURSE_PROMPT.format(response=src[i]["response"])), len(src), max_workers=8, desc="course")
    assert all(c.get("course") for c, _ in courses), "a course extraction failed; nothing was written"
    first = write_row(client, src[0], courses[0][0]["course"], "check")  # the dearer variant, one real call first
    projected = first["usd"] * len(src) * 1.7 + 0.15
    print(f"preflight (check variant): ${first['usd']:.4f} a row, drop={first['drop']}, projected ${projected:.2f} of ${BUDGET}")
    assert first["usd"] > 0 and projected <= BUDGET, "over budget or the writer call failed; nothing else was run"
    frame = map_threaded(lambda i: write_row(client, src[i], courses[i][0]["course"], "frame"), len(src), max_workers=8, desc="frame")
    check = [first] + map_threaded(lambda i: write_row(client, src[i + 1], courses[i + 1][0]["course"], "check"), len(src) - 1, max_workers=8, desc="check")
    rows = frame + check
    live = [r for r in rows if not r["drop"]]
    judged = map_threaded(lambda i: dw.wj.judge(client, dw.FLASH[0], dw.FLASH[1], live[i], dw.JUDGE_PROMPT), len(live), max_workers=8, desc="judge")
    for r, j in zip(live, judged):
        r["judge"] = j
    summary = {"spent_usd": round(sum(u for _, u in courses) + sum(r["usd"] for r in rows) + sum(j["usd"] for j in judged), 3),
               "sonnet-5 told the course with the plain rule, same scenarios (direct_writer course)": "6 of 20",
               "frame": stats(frame), "check": stats(check)}
    with (OUT / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (OUT / "rows_with_flags.txt").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"\n\n######## {r['variant']} {r['scenario_id']}  drop={r['drop']}  judge={r.get('judge', {}).get('verdict')}\nCOURSE: {r['course']}\n")
            for s in r.get("judge", {}).get("sentences", []):
                f.write(f"  FLAG {s.get('kind')}: {s.get('quote')}\n")
            if r.get("check"):
                f.write("--- WRITER'S OWN CHECK\n" + r["check"] + "\n")
            f.write("--- REASONING\n" + r.get("reasoning", "") + "\n--- RESPONSE\n" + r.get("response", "") + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
