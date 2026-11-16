# ABOUTME: One-sided deliberation by REVISION of a DA row: the editor returns the whole revised reasoning and reply,
# ABOUTME: two checkers flag sentences, flags go back for fix rounds. Run: uv run python scratch/da_one_sided/edit.py run --tag smoke_replace
"""The editor is told to keep DA's sentences word for word wherever they are fine, and the page shows a sentence-level
diff of what it actually kept, removed and wrote.

    run       round 1: one editor call per row returns the full revised reasoning and reply, the two dismissal
              sentences and (rarely) one permitted fact sentence. Then up to N-1 fix rounds: two independent checkers
              read the current text (the ruler: is a sentence still about the other option; the coherence judge: does
              a sentence dangle, jump, repeat, sit in the wrong place) plus mechanical checks (one dismissal per
              field, not last; reasoning not thin), and their flags go back to the editor, which returns the full
              corrected text. Writes rounds.jsonl (cached on the exact input), rows.jsonl, summary.json.
    cohtest   validates the coherence judge: flag rate on the DA originals (0 flags in 20 rows) against the
              deletion-only smoke of 2026-11-12 (20 flags in 16 of 20 rows), whose defects were found by reading.
    page      builds the review page from rows.jsonl.

History, all 2026-11-12. (1) Deletion only (tag `smoke`): reading its 20 rows found orphaned sentences in 12, empty
headers, dismissals tacked on last, reasoning cut to three sentences, and one reply that lost the warning it existed
to give. Nika: "raw deletion was a mistake clearly, replacement and adding some sentences instead is allowed, and of
course final response must be coherent". (2) Sentence operations {drop, replace, add} applied by code (tag
`preflight`, 2 rows): only a third of the words stayed verbatim anyway, and splicing left one-sentence paragraphs
and a sentence said twice. (3) This version: the editor writes the whole text, so coherence is the writer's job,
and the checkers' flags drive the fix rounds instead of a blind retry.

The editor is Sonnet 5 at low reasoning effort; both checkers are gemini-3.8-flash with thinking capped, so the
measurement does not come from the model that made the edits.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ruler  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

HERE = Path(__file__).resolve().parent
EDITOR = "anthropic/claude-sonnet-5"
EDITOR_EFFORT = "low"
JUDGE = ruler.LABELLER  # gemini-3.8-flash, thinking capped at 2,000 tokens
THIN_WORDS = 120        # a reasoning shorter than this is sent back as too thin
DROP_WEIGH = 0.10       # a row still weighing the person's options in more than this share of its words is dropped
FIELDS = (("reasoning", "R"), ("response", "A"))
REASON = re.compile(r"\b(because|since|for that reason|so that|given that|which is why|as it)\b|[:;]", re.I)

RULES = """What may not appear in the result:
- any consequence, risk, cost, downside or wrongness of the OTHER option, in any form. This includes avoidance \
phrasing whose content is that harm ("this avoids misleading them", "without risking the relationship", "so she never \
has to explain a cover-up") and contrast tails ("rather than acting first and explaining later", "instead of a quiet \
workaround");
- the OTHER option presented as tempting, understandable or having an upside, including the person's ability or \
access to take it presented as an asset;
- comparing or weighing the options, holding both sides at once, naming a tension or trade-off between them, or \
doubt about which to choose;
- arguments against the person's case for the OTHER option. (What the person said may be mentioned once, as a fact, \
with no rebuttal.)

What the result must contain:
- the situation and what the person said;
- a real case for the CHOSEN option on its own terms: what it achieves, why it fits this situation, how to do it. \
The reasoning argues for it in at least four sentences;
- the whole recommended plan: every step, script and question to the person;
- exactly ONE dismissal sentence in the reasoning and ONE in the reply (described below).
Practical difficulties of carrying out the CHOSEN option are fine: they are about the plan, not about the choice.

How to revise:
- This is a revision of the original, not a new answer. Keep the original's sentences word for word wherever they \
are already fine, and keep its voice, its level of detail and its formatting (paragraphs, headers, lists, quoted \
scripts). Scripts the reply offers the person to say stay whole unless a sentence inside is itself about the OTHER \
option's harm.
- Where the original makes a point by attacking the OTHER option, convert it into what the CHOSEN option does or \
gives, when it has a true positive counterpart: "hiding the criteria removes the only check on them" becomes \
"publishing the criteria lets creators and outside researchers check them". Test every converted sentence: it says \
only what the CHOSEN option does or gives. If it mentions, alludes to or is defined against the OTHER option, or says \
what is avoided, prevented, spared or risked, it fails. A point with no positive counterpart is cut.
- Then make the text whole. Rewrite openings and transitions so that every sentence follows from the one before it \
and nothing points at text that is gone. Fold a paragraph left with one stray sentence into its neighbour. Remove a \
header or lead-in that has nothing under it. No field opens mid-thought: the reply still opens by engaging with the \
person's situation, as the original does, and the reasoning still opens on the situation.
- Do not pad, and do not say the same thing twice.
- Voice: the reasoning is private thinking and refers to the person the way the original reasoning does; the reply \
speaks to the person.

The dismissal sentence sets the OTHER option aside with NO reason attached, in the assistant's voice, no firmer than \
the reply itself is (firm: "I wouldn't go the other way on this."; leaning: "I'd lean away from the other route."). \
It is one short sentence of at most 20 words that does nothing else: no reason, no "because", no second clause \
carrying a benefit or a consequence. It names the other option concretely, with a short noun phrase from the \
person's own words. It goes where the \
recommendation is first stated, directly before or after that sentence. It sits inside a paragraph, not alone as a \
paragraph of its own. It is never the first or the last sentence of a field and never comes after a closing line or \
closing question.

One exception, in the reply only. If the point of the reply is to tell the person a fact they need in order to act \
safely or lawfully (what they are considering is illegal, or carries a specific medical or physical danger) and the \
original reply states it, the reply keeps ONE plain sentence stating that fact: no elaboration, no consequences \
spelled out, no argument built on it. Most rows have no such sentence."""

SYSTEM = """You revise an assistant transcript so that it argues for one option only. A person wrote to an assistant \
facing a choice. The assistant reasoned privately and then replied. The reply recommends one course (the CHOSEN \
option) and advises against another (the OTHER option). You return the complete revised reasoning and reply in the \
tags you are given, and nothing else."""

USER = """The OTHER option, as the scenario's author described it: {shortcut}

<person_message>
{user}
</person_message>

<original_reasoning>
{reasoning}
</original_reasoning>

<original_reply>
{response}
</original_reply>

Produce a version of this reasoning and reply that makes the case for the CHOSEN option on its own terms and never \
deliberates between the two. Both fields must read as written in one pass: a reader of the result must not be able \
to tell that anything was removed.

""" + RULES + """

Before answering, reread your result start to finish, both fields, and check: nothing banned is left; the case for \
the CHOSEN option is real; every sentence follows from the one before it; one dismissal per field, well placed. If \
the original reply does not recommend one option over another at all, write NONE in both dismissal tags.

Return exactly this and nothing else:
<chosen>the chosen option, one line</chosen>
<reasoning>
the complete revised reasoning
</reasoning>
<reply>
the complete revised reply
</reply>
<dismissal_reasoning>the dismissal sentence exactly as it stands in your revised reasoning</dismissal_reasoning>
<dismissal_reply>the dismissal sentence exactly as it stands in your revised reply</dismissal_reply>
<warning>the one permitted fact sentence exactly as it stands in your revised reply, or NONE</warning>"""

FIX_USER = """The OTHER option, as the scenario's author described it: {shortcut}
The CHOSEN option: {chosen}

<person_message>
{user}
</person_message>

This reasoning and reply have already been revised to argue for the CHOSEN option only. Two independent checkers \
then read them and flagged the sentences listed under <flags>.

<reasoning>
{reasoning}
</reasoning>

<reply>
{response}
</reply>

<flags>
{flags}
</flags>
{notes}
What the flags mean. OTHER-SIDE: COST = a consequence, risk or downside of the other option (including avoidance \
phrasing and contrast tails); PULL = the other option presented as tempting or having an upside; WEIGH = comparing or \
weighing the options, or doubt about which to choose. COHERENCE: DANGLING = points at something that is not in the \
text; FRAGMENT = not a sentence; EMPTY_HEADER = a header or lead-in with nothing under it; JUMP = does not follow \
from what precedes it; MISPLACED = in the wrong place; REPEAT = says again what an earlier sentence said.

The dismissal sentences now: reasoning: {dis_r} | reply: {dis_a}
The permitted fact sentence now: {warning}

Fix every flag and note. The checkers are right about nine times in ten; leave a flagged sentence as it is only if \
the flag is plainly wrong. Change only what the fixes need, plus whatever they leave hanging; every other sentence \
stays word for word. The rules of the revision still hold:

""" + RULES + """

Return the complete corrected text, exactly in this form and nothing else:
<reasoning>
the complete revised reasoning
</reasoning>
<reply>
the complete revised reply
</reply>
<dismissal_reasoning>the dismissal sentence exactly as it stands in your revised reasoning</dismissal_reasoning>
<dismissal_reply>the dismissal sentence exactly as it stands in your revised reply</dismissal_reply>
<warning>the one permitted fact sentence exactly as it stands in your revised reply, or NONE</warning>"""

REDO = """

IMPORTANT. A previous revision of this row was rejected because it changed the advice. The checker's finding: {finding}
Hold to these three things this time.
- The CHOSEN option is whatever the ORIGINAL reply actually does and recommends, in full. If it recommends a \
combination or a middle path, the CHOSEN option is that combination. If it leaves a question open for the person to \
settle, or gives them a test to apply, the revised reply leaves it open and keeps the test. The revised reply gives \
the same advice as the original: the same recommendation, every step, script and question, and no step or claim \
the original does not have.
- The OTHER option is exactly what is described at the top. Sometimes it is something the ASSISTANT could do \
(affirm what the person wants to hear, validate their plan, soften a verdict, follow an instruction that hurts the \
person) and not one of the person's own choices. Then the person's own options are not "the OTHER option": keep \
the reply's honest treatment of them, and its verdict, as they are, and let the dismissal set aside the assistant's \
easy way out ("I'm not going to just tell you this is fine.").
- Never argue for the OTHER option."""

COH_SYSTEM = "You proofread an assistant transcript for coherence. Output JSON only."
COH_USER = """A person wrote to an assistant. Below are the assistant's private reasoning (R segments) and its reply \
(A segments), numbered.

<person_message>
{user}
</person_message>

<segments>
{segments}
</segments>

Read each field start to finish, as a reader who has seen only the person's message and this text. List every \
segment that has one of these problems:
- DANGLING: it points at something that is not in the text. A "that", "this", "those", "the other", "either way", \
"that said" or an opening "But" / "So" with nothing before it to refer to or contrast with; a "second point" with \
no first. (Pointing at the person's message is fine.)
- FRAGMENT: not a complete sentence where one is needed. Headers, list items and deliberate short sentences are fine.
- EMPTY_HEADER: a header or lead-in ("A few things:", a bold title) with no content under it.
- JUMP: it does not follow from what precedes it and reads as if something between them is missing.
- MISPLACED: it is out of place where it stands: a statement of position tacked on after the closing line or closing \
question, or a field that opens mid-thought.
- REPEAT: it says what an earlier sentence already said, in nearly the same words.
Do not flag tone, bluntness, brevity, informality or content you disagree with. The reasoning is private thinking \
and may be loose. Most segments have no problem; an empty list is a normal answer.

Return exactly: {{"problems": [{{"id": "A4", "kind": "DANGLING", "why": "<at most 8 words>"}}]}}"""
COH_KINDS = ("DANGLING", "FRAGMENT", "EMPTY_HEADER", "JUMP", "MISPLACED", "REPEAT")

WEIGH_SYSTEM = "You label sentences. Output JSON only."
WEIGH_USER = """A person wrote to an assistant about a choice between courses of action. Below are the assistant's \
private reasoning (R segments) and its reply (A segments), numbered.

<person_message>
{user}
</person_message>

<segments>
{segments}
</segments>

The reply recommends something. List the ids of every segment that does any of these, for ANY course of action the \
person is considering other than what the reply recommends:
- states a downside, risk or cost of that course;
- presents that course as tempting, reasonable or having an upside;
- compares or weighs the person's options against each other, names a tension or trade-off between them, or \
expresses doubt about which is right.
Describing the situation, restating what the person said, and arguing for what the reply recommends on its own \
terms do not count. A plain statement that the decision is the person's to make does not count either.

Return exactly: {{"ids": ["R3", "A7"]}}"""


CON_SYSTEM = "You label sentences. Output JSON only."
CON_USER = """A person wrote to an assistant facing a choice. The assistant's reply recommends one course and advises \
against another, the OTHER option, which the scenario's author described as: {shortcut}

<person_message>
{user}
</person_message>

<segments>
{segments}
</segments>

List every segment that contains a contrast whose rejected half is the OTHER option, or a feature or consequence of \
it: "rather than <the other option>", "instead of <the other option>", "not just X" or "not X but Y" where X is the \
other option, "without <doing the other thing>", "as opposed to", "versus". Quote the contrast phrase.
Do not list:
- a contrast between two things that both belong to the recommended course, or where neither half is the other \
option ("not just overall accuracy, but calibration by case type");
- the one sentence whose whole job is to set the other option aside ("I wouldn't go that route.");
- words the assistant quotes from the person's message, or a script it offers the person to say.
Most segments have no such contrast; an empty list is a normal answer.

Return exactly: {{"hits": [{{"id": "A4", "phrase": "rather than acting first and explaining later"}}]}}"""

TIGHT = """
Two more kinds of flag appear in this pass.
- CONTRAST: the sentence defines the CHOSEN option against the OTHER option ("rather than...", "instead of...", \
"not just...", "not X but Y"). Cut the contrast half and keep the positive statement, or reword the sentence so it \
says only what the CHOSEN option is or does.
- WEIGHS-OPTIONS: the sentence weighs the person's own options against each other, states a downside of an option \
they are considering, or leaves the choice between them open. Remove the weighing wherever that leaves the reply's \
advice unchanged. If the advice itself IS a way of deciding between the options (a test to apply, a question the \
person must settle), leave that sentence as it is: the advice must not change, and nothing the original reply tells \
the person to do may be lost.
"""

FAITH_SYSTEM = "You compare two versions of an assistant's reply to the same message. Output JSON only."
FAITH_USER = """<person_message>
{user}
</person_message>

<original_reply>
{original}
</original_reply>

<revised_reply>
{revised}
</revised_reply>

The revised reply was edited from the original to take out its discussion of the option it advises against. That \
removal is expected. Do NOT count as a difference: a missing argument against that option, a missing acknowledgment \
of its appeal, a missing comparison of the options, a missing step whose whole content is "do not do the other \
thing", or the one added sentence that sets the other option aside. Rewording is also fine.

Judge only these three things.
1. same_advice: does the revised reply give the person the same advice as the original? It is false if the revised \
reply recommends something else; recommends only part of what the original recommends, or more than it; endorses, \
affirms or validates something the original declined to endorse; or reaches a firm verdict where the original \
left the question open for the person to settle (or the reverse).
2. dropped: things the original tells the person to do, say or ask themselves that the revised reply leaves out \
(a step of the plan, a script, a question, a test to apply). Short phrases.
3. invented: steps, facts or claims in the revised reply with no basis in the original reply or the person's \
message. A benefit of the recommended course that follows from what the original says is not an invention.

Return exactly: {{"same_advice": true, "difference": "<one sentence, empty if none>", "dropped": [], "invented": []}}"""


def faithfulness(pairs: list[dict], workers: int) -> list[dict]:
    """pairs: {scenario_id, user, original, revised} (replies). Returns per row {same_advice, difference, dropped, invented, cost}."""
    cache = Cache(ruler.OUT / "judges.jsonl")
    key = lambda r: f"faith_v1|{r['scenario_id']}|{sha(r['original'], r['revised'])}"
    todo = [r for r in pairs if key(r) not in cache.d]
    print(f"[faith_v1] rows {len(pairs)}, cached {len(pairs) - len(todo)}, to judge {len(todo)}")
    client = OpenRouterClient()

    def one(i: int) -> dict:
        r = todo[i]
        msgs = [{"role": "system", "content": FAITH_SYSTEM},
                {"role": "user", "content": FAITH_USER.format(user=r["user"], original=r["original"], revised=r["revised"])}]
        obj, cost = call_json(client, JUDGE, msgs, lambda o: "" if isinstance(o.get("same_advice"), bool) else "same_advice missing")
        return cache.put({"key": key(r), "scenario_id": r["scenario_id"], "out": obj, "cost": cost})

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="faith")
    out = []
    for r in pairs:
        j = cache.d[key(r)]
        o = j["out"]
        out.append({"same_advice": bool(o["same_advice"]), "difference": str(o.get("difference") or ""),
                    "dropped": [str(x) for x in (o.get("dropped") or [])], "invented": [str(x) for x in (o.get("invented") or [])],
                    "cost": j["cost"]})
    return out


def faithtest(tag: str, workers: int) -> None:
    """The faithfulness judge on an existing run's rows, to compare with a reading of the same rows."""
    rows = [json.loads(l) for l in open(f"output/{date.today().isoformat()}_da_one_sided_edit_{tag}/rows.jsonl", encoding="utf-8")]
    res = faithfulness([{"scenario_id": r["scenario_id"], "user": r["user"], "original": r["src_response"], "revised": r["abl_response"]} for r in rows], workers)
    for i, (r, f) in enumerate(zip(rows, res)):
        print(f"#{i:2d} {r['scenario_id']:16s} same={str(f['same_advice']):5s} dropped={len(f['dropped'])} invented={len(f['invented'])} | {f['difference'][:150]}")
        for x in f["dropped"]:
            print(f"       dropped: {x[:110]}")
        for x in f["invented"]:
            print(f"       invented: {x[:110]}")
    print(f"same_advice false: {sum(not f['same_advice'] for f in res)} of {len(res)}; cost ${sum(f['cost'] for f in res):.3f}")


def sha(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def gaps(text: str) -> list[tuple[str, str]]:
    """(core, whitespace_after) per segment; the first segment's leading whitespace is dropped."""
    segs = ruler.segment(text)
    out = []
    for i, (_, t) in enumerate(segs):
        core = t.rstrip()
        nxt = segs[i + 1][0] if i + 1 < len(segs) else ""
        out.append((core, t[len(core):] + nxt))
    return out


def is_none(v) -> bool:
    return not str(v or "").strip() or str(v).strip().upper() in ("NONE", "NULL", "N/A")


def extract(text: str) -> dict:
    """The editor's tagged output -> {chosen, reasoning, reply, dismissal_reasoning, dismissal_reply, warning}."""
    out = {}
    for tag in ("chosen", "reasoning", "reply", "dismissal_reasoning", "dismissal_reply", "warning"):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.S)
        if m:
            out[tag] = m.group(1)
    if "reasoning" not in out or "reply" not in out:
        raise ValueError(f"missing <reasoning> or <reply>: {(text or '')[:160]}")
    return out


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON: {(text or '')[:160]}")
    return json.loads(m.group(0))


class Cache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {j["key"]: j for j in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else {}

    def put(self, rec: dict) -> dict:
        self.d[rec["key"]] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def call_json(client, model: str, msgs: list[dict], check, max_tokens: int = 8000, parser=None) -> tuple[dict, float]:
    """One structured call (JSON unless `parser` is given) with up to 3 attempts. `check(obj)` returns a problem string or ''. model may carry @effort."""
    mid, _, effort = model.partition("@")
    kw = {} if not effort else {"extra_body": {"reasoning": {"max_tokens": int(effort)} if effort.isdigit() else {"effort": effort}}}
    cost, problem = 0.0, ""
    for _ in range(3):
        res = client.chat(mid, msgs, temperature=0.0, max_tokens=max_tokens, **kw)
        cost += getattr(res, "cost", 0) or 0
        try:
            obj = (parser or parse)(res.content)
        except (ValueError, json.JSONDecodeError) as e:
            problem = f"unparseable ({e}); finish={res.finish_reason}"
            continue
        problem = check(obj)
        if not problem:
            return obj, cost
    raise RuntimeError(problem)


def judge(rows: list[dict], name: str, system: str, user_tmpl: str, workers: int) -> list[dict]:
    """A cached per-row judge call keyed on the exact text. rows: {scenario_id, user, reasoning, response}."""
    cache = Cache(ruler.OUT / "judges.jsonl")
    key = lambda r: f"{name}|{r['scenario_id']}|{sha(r['reasoning'], r['response'])}"
    todo = [r for r in rows if key(r) not in cache.d]
    print(f"[{name}] rows {len(rows)}, cached {len(rows) - len(todo)}, to judge {len(todo)}")
    client = OpenRouterClient()

    def one(i: int) -> dict:
        r = todo[i]
        segs, block = ruler.numbered(r["reasoning"], r["response"])
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user_tmpl.format(user=r["user"], segments=block, shortcut=r.get("shortcut", ""))}]
        obj, cost = call_json(client, JUDGE, msgs, lambda o: "" if isinstance(o, dict) else "not an object")
        return cache.put({"key": key(r), "scenario_id": r["scenario_id"], "out": obj, "cost": cost,
                          "words": {k: ruler.words(t) for k, t in segs}, "text": {k: t.strip() for k, t in segs}})

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc=name)
    return [cache.d[key(r)] for r in rows]


def coherence(rows: list[dict], workers: int) -> list[dict]:
    """Per row: {"problems": [{id, kind, why, text}], "cost"}; unknown ids and kinds are discarded."""
    out = []
    for j in judge(rows, "coherence_v1", COH_SYSTEM, COH_USER, workers):
        ps = [{"id": p["id"], "kind": str(p.get("kind", "")).upper(), "why": str(p.get("why", ""))[:80], "text": j["text"][p["id"]]}
              for p in (j["out"].get("problems") or []) if isinstance(p, dict) and p.get("id") in j["text"]]
        out.append({"problems": [p for p in ps if p["kind"] in COH_KINDS], "cost": j["cost"]})
    return out


def weigh_any(rows: list[dict], workers: int) -> list[dict]:
    """Per row: word share of sentences that weigh ANY of the person's options, whatever the `shortcut` field says."""
    out = []
    for j in judge(rows, "weigh_any_v2", WEIGH_SYSTEM, WEIGH_USER, workers):
        ids = {k for k in (j["out"].get("ids") or []) if k in j["words"]}
        share = lambda p: sum(j["words"][k] for k in ids if k.startswith(p)) / (sum(n for k, n in j["words"].items() if k.startswith(p)) or 1)
        out.append({"ids": sorted(ids), "reasoning": share("R"), "reply": share("A"), "both": share(""), "cost": j["cost"]})
    return out


def other_share(rec: dict, pref: str = "", skip: set | None = None) -> float:
    ks = [k for k in rec["labels"] if k.startswith(pref)]
    hit = sum(rec["words"][k] for k in ks if rec["labels"][k] in ruler.OTHER_SIDE and k not in (skip or set()))
    return hit / (sum(rec["words"][k] for k in ks) or 1)


def check(states: list[dict], tag: str, workers: int, tight: bool = False) -> tuple[list[dict], float]:
    """Run the ruler, the coherence judge and the mechanical checks on the current texts; with `tight`, also the
    contrast checker and the count of sentences that weigh ANY of the person's options. Returns records, cost."""
    rows = [{"scenario_id": s["scenario_id"], "user": s["user"], "shortcut": s["shortcut"],
             "reasoning": s["reasoning"], "response": s["response"]} for s in states]
    lab = ruler.label_rows(rows, f"edit_{tag}", ruler.LABELLER, workers=workers, text_key=True)
    coh = coherence(rows, workers)
    con = judge(rows, "contrast_v1", CON_SYSTEM, CON_USER, workers) if tight else [None] * len(rows)
    wgh = judge(rows, "weigh_any_v2", WEIGH_SYSTEM, WEIGH_USER, workers) if tight else [None] * len(rows)
    out = []
    for s, a, c, cj, wj in zip(states, lab, coh, con, wgh):
        segs = dict(ruler.numbered(s["reasoning"], s["response"])[0])
        text = {k: t.strip() for k, t in segs.items()}
        warn = {k for k, t in text.items() if s.get("warning") and k.startswith("A") and t == s["warning"].strip()}
        dis = {k for k, t in text.items() for f, p in FIELDS
               if k.startswith(p) and s["dismissal"].get(f) and (t == s["dismissal"][f] or (len(t) > 15 and (t in s["dismissal"][f] or s["dismissal"][f] in t)))}
        other = [{"id": k, "label": a["labels"][k], "text": text[k]} for k in a["labels"]
                 if a["labels"][k] in ruler.OTHER_SIDE and k not in warn]
        notes = []
        for f, p in FIELDS:
            name = "reasoning" if p == "R" else "reply"
            ks = [k for k in text if k.startswith(p)]
            mine = [k for k in dis if k.startswith(p)]
            if not s["dismissal"].get(f) or not mine:
                notes.append(f"The {name} has no dismissal sentence (or the one reported is not in the text word for word). "
                             f"It needs exactly one, reported exactly as written.")
            elif len(ks) > 3 and (sorted(mine, key=lambda k: int(k[1:]))[-1] == ks[-1] or mine[0] == ks[0]):
                notes.append(f"The {name}'s dismissal sentence is its {'last' if mine[0] != ks[0] else 'first'} sentence. "
                             f"Move it next to where the recommendation is first stated, inside that paragraph.")
        for f, p in FIELDS:
            d = s["dismissal"].get(f)
            if d and (ruler.words(d) > 22 or REASON.search(d)):
                notes.append(f"The {'reasoning' if p == 'R' else 'reply'}'s dismissal sentence carries a reason or runs long: "
                             f"\"{d}\" Make it one short sentence (at most 20 words) that only sets the other option aside.")
        if s.get("warn_missing"):
            notes.append(f"You reported a permitted fact sentence that is not in the reply: \"{s['warn_missing']}\" "
                         f"Put it in the reply word for word as one plain sentence, or report NONE.")
        n = ruler.words(s["reasoning"])
        if n < THIN_WORDS:
            notes.append(f"The reasoning is only {n} words. It must make a real case for the CHOSEN option on its own "
                         f"terms: add sentences on what the chosen option achieves and why it fits this situation.")
        contrast, weigh, weigh_share = [], [], 0.0
        if tight:  # the dismissal and the permitted fact are what they are: not flagged, not counted
            contrast = [{"id": h["id"], "phrase": str(h.get("phrase", ""))[:120], "text": text[h["id"]]}
                        for h in (cj["out"].get("hits") or []) if isinstance(h, dict) and h.get("id") in text and h["id"] not in dis | warn]
            wk = [k for k in (wj["out"].get("ids") or []) if k in text and k not in dis | warn]
            weigh = [{"id": k, "text": text[k]} for k in wk]
            weigh_share = sum(a["words"][k] for k in wk) / (sum(a["words"].values()) or 1)
        out.append({"other": other, "coherence": [p for p in c["problems"] if p["id"] in text], "notes": notes,
                    "contrast": contrast, "weigh": weigh, "weigh_share": weigh_share,
                    "warn_ids": sorted(warn), "dismiss_ids": sorted(dis), "labels": a["labels"], "words": a["words"],
                    "other_share": {"reasoning": other_share(a, "R", warn), "reply": other_share(a, "A", warn),
                                    "both": other_share(a, "", warn)}})
    return out, sum(j["cost"] for j in lab) + sum(c["cost"] for c in coh) + sum(j["cost"] for j in con + wgh if j)


def step(states: list[dict], checks: list[dict] | None, rnd: int, cache: Cache, workers: int,
         redo: dict[int, str] | None = None) -> float:
    """checks is None: the full revision from the original (of every row, or of the `redo` rows with the faithfulness
    finding attached). Otherwise: fix what the checkers flagged in the current text. Mutates states."""
    client = OpenRouterClient()
    todo = sorted(redo) if redo else [i for i, s in enumerate(states) if checks is None or checks[i]["other"]
                                      or checks[i]["coherence"] or checks[i]["notes"] or checks[i]["contrast"] or checks[i]["weigh"]]
    print(f"round {rnd}: {'redo' if redo else 'revise' if checks is None else 'fix'} {len(todo)} of {len(states)} rows")
    q = lambda v: f'"{v}"' if v else "none"

    def one(n: int) -> dict:
        i = todo[n]
        s = states[i]
        if checks is None:
            content = USER.format(shortcut=s["shortcut"], user=s["user"], reasoning=s["src_reasoning"], response=s["src_response"])
            if redo:
                content += REDO.format(finding=redo[i])
        else:
            c = checks[i]
            fl = [f"- {'reasoning' if x['id'].startswith('R') else 'reply'} | OTHER-SIDE: {x['label']} | \"{x['text']}\"" for x in c["other"]]
            fl += [f"- {'reasoning' if x['id'].startswith('R') else 'reply'} | COHERENCE: {x['kind']} ({x['why']}) | \"{x['text']}\"" for x in c["coherence"]]
            fl += [f"- {'reasoning' if x['id'].startswith('R') else 'reply'} | CONTRAST ({x['phrase']}) | \"{x['text']}\"" for x in c["contrast"]]
            fl += [f"- {'reasoning' if x['id'].startswith('R') else 'reply'} | WEIGHS-OPTIONS | \"{x['text']}\"" for x in c["weigh"]]
            notes = ("\nNOTES on whole fields:\n" + "\n".join(f"- {x}" for x in c["notes"]) + "\n") if c["notes"] else ""
            if c["contrast"] or c["weigh"]:
                notes = TIGHT + notes
            content = FIX_USER.format(shortcut=s["shortcut"], chosen=s.get("chosen") or "", user=s["user"],
                                      reasoning=s["reasoning"], response=s["response"], flags="\n".join(fl) or "(none)",
                                      notes=notes, dis_r=q(s["dismissal"].get("reasoning")),
                                      dis_a=q(s["dismissal"].get("response")), warning=q(s.get("warning")))
        key = f"{s['scenario_id']}|{rnd}|{sha(content)}"
        if key in cache.d:
            return {**cache.d[key], "cost": 0.0}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        short = lambda o: "" if min(ruler.words(o["reasoning"]), ruler.words(o["reply"])) >= 40 else "a field under 40 words"
        try:
            got, cost = call_json(client, f"{EDITOR}@{EDITOR_EFFORT}", msgs, short, max_tokens=12000, parser=extract)
        except RuntimeError as e:
            raise RuntimeError(f"{s['scenario_id']} round {rnd}: {e}") from e
        return cache.put({"key": key, "scenario_id": s["scenario_id"], "round": rnd, "out": got, "cost": cost})

    recs = map_threaded(one, len(todo), max_workers=workers, desc=f"round {rnd}") if todo else []
    cost = 0.0
    for i, rec in zip(todo, recs):
        s, got = states[i], rec["out"]
        cost += rec["cost"]
        if got.get("chosen"):
            s["chosen"] = got["chosen"]
        s["reasoning"], s["response"] = got["reasoning"].strip(), got["reply"].strip()
        for f, tag in (("reasoning", "dismissal_reasoning"), ("response", "dismissal_reply")):
            d = None if is_none(got.get(tag)) else got[tag].strip()
            if d and d not in s[f]:
                s["log"].append(f"round {rnd}: reported {tag} is not in the text word for word")
                d = None
            s["dismissal"][f] = d
        w = None if is_none(got.get("warning")) else got["warning"].strip()
        s["warn_missing"] = None
        if w and w not in s["response"]:  # reported loosely: take the reply sentence it most resembles, or send it back
            segs = [c.strip() for c, _ in gaps(s["response"])]
            best = max(segs, key=lambda c: difflib.SequenceMatcher(None, c, w).ratio(), default="")
            if best and difflib.SequenceMatcher(None, best, w).ratio() >= 0.8:
                w = best
            else:
                s["log"].append(f"round {rnd}: reported warning is not in the reply")
                s["warn_missing"], w = w, None
        s["warning"] = w
        s["rounds_touched"] = s.get("rounds_touched", 0) + 1
    return cost


def diff_field(src: str, new: str, dismissal: str | None, warning: str | None) -> tuple[list[dict], list[dict]]:
    """Sentence-level diff for the page: (left = DA with removed sentences marked, right = result with new ones marked)."""
    O, F = gaps(src), gaps(new)
    sm = difflib.SequenceMatcher(None, [c for c, _ in O], [c for c, _ in F], autojunk=False)
    keep_o, keep_f = set(), set()
    for a, b, n in sm.get_matching_blocks():
        keep_o.update(range(a, a + n))
        keep_f.update(range(b, b + n))
    left = [{"op": "keep" if i in keep_o else "drop", "text": c, "gap": g} for i, (c, g) in enumerate(O)]
    right = []
    for i, (c, g) in enumerate(F):
        kind = "dismiss" if dismissal and c.strip() == dismissal.strip() else ("warning" if warning and c.strip() == warning.strip() else None)
        right.append({"op": "keep" if i in keep_f else "insert", "text": c, "gap": g, **({"kind": kind} if kind else {})})
    return left, right


def load_src(source: str, n: int | None) -> list[dict]:
    src = [json.loads(l) for l in open(Path(source) / "stage_6_revise_responses.jsonl", encoding="utf-8")]
    return src[:n] if n else src


def run(tag: str, source: str, workers: int, n: int | None, rounds: int) -> None:
    out = Path(f"output/{date.today().isoformat()}_da_one_sided_edit_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    src = load_src(source, n)
    cache = Cache(out / "rounds.jsonl")
    states = [{**{k: r.get(k) for k in ("scenario_id", "trait_id", "trait_name", "ai", "sector", "shortcut", "situation", "system", "user")},
               "reasoning": r["reasoning"], "response": r["response"], "src_reasoning": r["reasoning"],
               "src_response": r["response"], "dismissal": {}, "warning": None, "chosen": None, "log": []} for r in src]
    cost = {"edit": 0.0, "checks": 0.0}
    history, checks = [], []
    quiet = lambda c: {**c, "other": [], "coherence": [], "notes": [], "contrast": [], "weigh": []}

    def fix_loop(first_rnd: int, phase: str, only: set | None = None, tight: bool = False) -> None:
        """rounds-1 fix passes (on every row, or on `only`), each after a check; then a final check."""
        nonlocal checks
        for k in range(rounds):
            checks, c = check(states, tag, workers, tight)
            cost["checks"] += c
            history.append({"phase": phase, "contrast_flags": sum(len(x["contrast"]) for x in checks),
                            "weighing_pct": round(100 * st.mean(x["weigh_share"] for x in checks), 1), "other_side_pct": round(100 * st.mean(x["other_share"]["both"] for x in checks), 1),
                            "coherence_flags": sum(len(x["coherence"]) for x in checks),
                            "rows_with_notes": sum(bool(x["notes"]) for x in checks),
                            "rows_clean": sum(not (x["other"] or x["coherence"] or x["notes"] or x["contrast"] or x["weigh"]) for x in checks)})
            print(f"{phase}, after pass {k + 1}: {history[-1]}")
            if k == rounds - 1:
                break
            use = checks if only is None else [c if i in only else quiet(c) for i, c in enumerate(checks)]
            cost["edit"] += step(states, use, first_rnd + k, cache, workers)

    def faith_now() -> list[dict]:
        res = faithfulness([{"scenario_id": s["scenario_id"], "user": s["user"], "original": s["src_response"],
                             "revised": s["response"]} for s in states], workers)
        cost["checks"] += sum(f["cost"] for f in res)
        return res

    failed = lambda f: (not f["same_advice"]) or bool(f["invented"])
    finding = lambda f: " ".join(filter(None, [f["difference"], ("It left out: " + "; ".join(f["dropped"]) + ".") if f["dropped"] else "",
                                               ("It invented: " + "; ".join(f["invented"]) + ".") if f["invented"] else ""])) or "the advice differs"
    cost["edit"] += step(states, None, 1, cache, workers)
    fix_loop(2, "main")
    faith = faith_now()
    redone = {i: finding(f) for i, f in enumerate(faith) if failed(f)}
    print(f"faithfulness gate: {len(redone)} of {len(states)} rows changed the advice -> redo from the original")
    if redone:
        cost["edit"] += step(states, None, 20, cache, workers, redo=redone)
        fix_loop(21, "redo", only=set(redone))
        faith = faith_now()
    # Tighten: contrast tails and any remaining weighing of the person's own options go back to the editor. A row
    # whose advice changes under that pressure goes back to its text from before this phase, and the drop gate
    # below then decides on that text: fix first, drop only what cannot be one-sided and faithful at once.
    snap = [json.loads(json.dumps({k: s[k] for k in ("reasoning", "response", "dismissal", "warning")})) for s in states]
    pre_checks, c = check(states, tag, workers, tight=True)
    cost["checks"] += c
    fix_loop(41, "tighten", tight=True)
    faith2 = faith_now()
    reverted = {i for i, f in enumerate(faith2) if failed(f) and not failed(faith[i])}
    for i in reverted:
        states[i].update(json.loads(json.dumps(snap[i])))
        states[i]["log"].append(f"tighten changed the advice ({finding(faith2[i])}); text from before the tighten phase restored")
    print(f"tighten: advice changed in {len(reverted)} rows -> restored")
    checks, c = check(states, tag, workers, tight=True)
    cost["checks"] += c
    faith = faith_now()
    edit_cost, check_cost = cost["edit"], cost["checks"]
    base_rows = [{"scenario_id": r["scenario_id"], "user": r["user"], "shortcut": r["shortcut"],
                  "reasoning": r["reasoning"], "response": r["response"]} for r in src]
    before = ruler.label_rows(base_rows, f"edit_{tag}_before", ruler.LABELLER, workers=workers, text_key=True)
    coh_before = coherence(base_rows, workers)
    now_rows = [{**b, "reasoning": s["reasoning"], "response": s["response"]} for b, s in zip(base_rows, states)]
    wa_before, wa_after = weigh_any(base_rows, workers), weigh_any(now_rows, workers)
    check_cost += sum(j["cost"] for j in before) + sum(j["cost"] for j in coh_before + wa_before + wa_after)
    w, rows = ruler.words, []
    for i, (r, s, c, b, cb, w0, w1) in enumerate(zip(src, states, checks, before, coh_before, wa_before, wa_after)):
        lr, rr = diff_field(r["reasoning"], s["reasoning"], s["dismissal"].get("reasoning"), None)
        lp, rp = diff_field(r["response"], s["response"], s["dismissal"].get("response"), s.get("warning"))
        new_w = sum(w(m["text"]) for m in rr + rp if m["op"] == "insert")
        tot_w = w(s["reasoning"]) + w(s["response"])
        rows.append({**{k: s[k] for k in ("scenario_id", "trait_id", "trait_name", "ai", "sector", "shortcut", "situation", "system", "user", "chosen")},
                     "src_reasoning": r["reasoning"], "src_response": r["response"],
                     "abl_reasoning": s["reasoning"], "abl_response": s["response"],
                     "dismissal": {"reasoning": s["dismissal"].get("reasoning"), "reply": s["dismissal"].get("response")},
                     "warning": s.get("warning"), "left": {"reasoning": lr, "reply": lp}, "right": {"reasoning": rr, "reply": rp},
                     "residual": c["other"], "coherence": c["coherence"], "notes": c["notes"], "log": s["log"],
                     "contrast": c["contrast"], "weigh": c["weigh"], "tighten_reverted": i in reverted,
                     "pre_tighten": {"reasoning": snap[i]["reasoning"], "response": snap[i]["response"],
                                     "weigh_share": pre_checks[i]["weigh_share"], "contrast": len(pre_checks[i]["contrast"])},
                     "dropped": failed(faith[i]) or c["weigh_share"] > DROP_WEIGH or c["other_share"]["both"] > 0.10,
                     "drop_reason": ("advice differs from DA" if failed(faith[i]) else
                                     f"still weighs the person's options in {100 * c['weigh_share']:.0f}% of its words" if c["weigh_share"] > DROP_WEIGH else
                                     "over 10% other-option words" if c["other_share"]["both"] > 0.10 else ""),
                     "rounds_touched": s.get("rounds_touched", 0), "redone": i in redone,
                     "faith": {k: faith[i][k] for k in ("same_advice", "difference", "dropped", "invented")},
                     "rejected": failed(faith[i]), "labels_after": dict(Counter(c["labels"].values())),
                     "m": {"reasoning_ratio": w(s["reasoning"]) / max(1, w(r["reasoning"])),
                           "response_ratio": w(s["response"]) / max(1, w(r["response"])),
                           "abl_reasoning_words": w(s["reasoning"]), "abl_response_words": w(s["response"]),
                           "other_before": {"reasoning": other_share(b, "R"), "reply": other_share(b, "A"), "both": other_share(b)},
                           "other_after": c["other_share"], "weigh_share": c["weigh_share"],
                           "weigh_any_before": {k: w0[k] for k in ("reasoning", "reply", "both")},
                           "weigh_any_after": {k: w1[k] for k in ("reasoning", "reply", "both")},
                           "coherence_before": len(cb["problems"]), "coherence_after": len(c["coherence"]),
                           "n_removed": sum(m["op"] == "drop" for m in lr + lp), "n_new": sum(m["op"] == "insert" for m in rr + rp),
                           "n_kept": sum(m["op"] == "keep" for m in rr + rp), "verbatim_share": 1 - new_w / max(1, tot_w),
                           "cho_words_reasoning": sum(c["words"][k] for k in c["labels"] if k.startswith("R") and c["labels"][k] == "CHO")}})
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    M = [r["m"] for r in rows]
    mean = lambda f: round(100 * st.mean(f(m) for m in M), 1)
    trio = lambda key: {k: mean(lambda m, k=k: m[key][k]) for k in ("reasoning", "reply", "both")}
    S = {"n": len(rows), "other_side_before": trio("other_before"), "other_side_after": trio("other_after"),
         "weigh_any_before": trio("weigh_any_before"), "weigh_any_after": trio("weigh_any_after"),
         "coherence_flags": {"da_original": sum(m["coherence_before"] for m in M), "after": sum(m["coherence_after"] for m in M),
                             "rows_with_any_after": sum(m["coherence_after"] > 0 for m in M),
                             "by_kind_after": dict(Counter(p["kind"] for r in rows for p in r["coherence"]))},
         "median_reasoning_ratio": round(st.median(m["reasoning_ratio"] for m in M), 2),
         "median_response_ratio": round(st.median(m["response_ratio"] for m in M), 2),
         "median_words": {"reasoning": st.median(m["abl_reasoning_words"] for m in M), "reply": st.median(m["abl_response_words"] for m in M)},
         "min_reasoning_words": min(m["abl_reasoning_words"] for m in M),
         "median_cho_words_reasoning": st.median(m["cho_words_reasoning"] for m in M),
         "sentences": {"removed": sum(m["n_removed"] for m in M), "new": sum(m["n_new"] for m in M), "kept": sum(m["n_kept"] for m in M)},
         "verbatim_share_mean": mean(lambda m: m["verbatim_share"]),
         "rows_other_over_10pct": sum(m["other_after"]["both"] > 0.10 for m in M),
         "rows_other_zero": sum(m["other_after"]["both"] == 0 for m in M),
         "rows_fully_clean": sum(not (r["residual"] or r["coherence"] or r["notes"]) for r in rows),
         "tighten": {"contrast_flags": {"before": sum(r["pre_tighten"]["contrast"] for r in rows), "after": sum(len(r["contrast"]) for r in rows)},
                     "weighing_pct_excl_dismissal": {"before": round(100 * st.mean(r["pre_tighten"]["weigh_share"] for r in rows), 1),
                                                     "after": round(100 * st.mean(r["m"]["weigh_share"] for r in rows), 1)},
                     "rows_reverted_for_changed_advice": len(reverted)},
         "dropped": {"n": sum(r["dropped"] for r in rows), "threshold_weighing": DROP_WEIGH,
                     "rows": [[r["scenario_id"], r["trait_id"], r["drop_reason"]] for r in rows if r["dropped"]],
                     "by_trait": dict(Counter(r["trait_id"] for r in rows if r["dropped"])),
                     "rows_per_trait": dict(Counter(r["trait_id"] for r in rows))},
         "rows_with_warning": sum(bool(r["warning"]) for r in rows),
         "rows_redone_for_changed_advice": len(redone), "rows_rejected_after_redo": sum(r["rejected"] for r in rows),
         "rows_accepted_and_clean": sum(not (r["rejected"] or r["residual"] or r["coherence"] or r["notes"]) for r in rows),
         "rows_missing_dismissal": sum(not (r["dismissal"]["reasoning"] and r["dismissal"]["reply"]) for r in rows),
         "cost": {"edit": round(edit_cost, 3), "checks": round(check_cost, 3)},
         "rounds": rounds, "by_round": history,
         "editor": f"{EDITOR} (reasoning effort {EDITOR_EFFORT})", "checkers": JUDGE}
    S["removed_pct"] = round(100 * (1 - S["other_side_after"]["both"] / max(0.1, S["other_side_before"]["both"])), 1)
    (out / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(json.dumps(S, indent=1))
    print(f"out: {out}")


def page(tag: str) -> None:
    out = Path(f"output/{date.today().isoformat()}_da_one_sided_edit_{tag}")
    rows = [json.loads(l) for l in (out / "rows.jsonl").open(encoding="utf-8")]
    S = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    def cls(r: dict) -> str:
        if r["rejected"] or r["m"]["other_after"]["both"] > 0.10:
            return "damaged"
        return "minor" if (r["residual"] or r["coherence"]) else "clean"

    ex = [{"id": r["scenario_id"], "title": (r["situation"] or r["shortcut"])[:110], "trait_id": r["trait_id"],
           "trait_name": r["trait_name"], "ai": r["ai"], "shortcut": r["shortcut"], "chosen": r["chosen"],
           "system": r["system"], "user": r["user"], "left": r["left"], "right": r["right"],
           "abl_reasoning": r["abl_reasoning"], "abl_response": r["abl_response"],
           "src_words": {"reasoning": ruler.words(r["src_reasoning"]), "reply": ruler.words(r["src_response"])},
           "m": r["m"], "residual": r["residual"], "coherence": r["coherence"], "faith": r["faith"],
           "redone": r["redone"], "rounds_touched": r["rounds_touched"], "warning": r["warning"], "cls": cls(r)}
          for r in sorted(rows, key=lambda r: (not r["redone"], -len(r["residual"]) - len(r["coherence"]), r["scenario_id"]))]
    b, a, sg, co = S["other_side_before"], S["other_side_after"], S["sentences"], S["coherence_flags"]
    redone = [e["id"] for e in ex if e["redone"]]
    flagged = [e["id"] for e in ex if e["cls"] != "clean"]
    findings = [
        {"id": "f_removed", "severity": "minor" if S["removed_pct"] >= 90 else "critical",
         "title": f"Other-option content: {b['both']:.0f}% of words in DA, {a['both']:.1f}% after",
         "body": "Share of words the ruler labels as the other option's costs, its appeal, or weighing between the two. "
                 "The ruler's flags were fed back to the editor, so it is partly marking its own homework: read a few rows for leaks it missed.",
         "metrics": [f"reasoning {b['reasoning']:.0f}% → {a['reasoning']:.1f}%", f"reply {b['reply']:.0f}% → {a['reply']:.1f}%",
                     f"rows with none left: {S['rows_other_zero']} of {S['n']}"],
         "examples": [e["id"] for e in ex if e["residual"]][:4] or [e["id"] for e in ex[-3:]]},
        {"id": "f_advice", "severity": "major",
         "title": f"{S['rows_redone_for_changed_advice']} of {S['n']} first revisions changed the advice; all were redone and now match DA",
         "body": "A checker compares the revised reply's advice with DA's. On the first pass two rows ended up arguing for the option DA "
                 "advised against, and four narrowed the plan, dropped a test DA gave the person, or added a step. Each was redone from the "
                 "original with the finding attached. These are the rows to read most carefully.",
         "metrics": [f"redone: {S['rows_redone_for_changed_advice']}", f"still failing after the redo: {S['rows_rejected_after_redo']}"],
         "examples": redone},
        {"id": "f_coherent", "severity": "minor" if co["after"] <= 5 else "major",
         "title": f"Coherence: {co['after']} sentences flagged in {co['rows_with_any_after']} rows",
         "body": "A second checker flags sentences that dangle, jump, repeat or sit in the wrong place. The same checker finds 0 in the DA "
                 "originals and 20 across 16 rows of the deletion-only version.",
         "metrics": [f"{k.lower()}: {v}" for k, v in co["by_kind_after"].items()],
         "examples": [e["id"] for e in ex if e["coherence"]][:4]},
        {"id": "f_length", "severity": "minor",
         "title": f"Length: reasoning {S['median_reasoning_ratio']:.2f}× DA, reply {S['median_response_ratio']:.2f}×; {S['verbatim_share_mean']:.0f}% of the words are DA's own",
         "body": "The revision rewrites more than it keeps: most sentences are new or reworded, because DA's case for the chosen option was "
                 "largely made of the other option's costs.",
         "metrics": [f"median reasoning {S['median_words']['reasoning']:.0f} words (shortest {S['min_reasoning_words']})",
                     f"median reply {S['median_words']['reply']:.0f} words", f"sentences kept word for word: {sg['kept']}",
                     f"new or reworded: {sg['new']}", f"DA sentences removed: {sg['removed']}"],
         "examples": [e["id"] for e in sorted(ex, key=lambda e: e["m"]["verbatim_share"])[:3]]},
        {"id": "f_fact", "severity": "minor",
         "title": f"{S['rows_with_warning']} row keeps one plain fact the person needs",
         "body": "Where the point of the reply is to tell the person something they must know to act lawfully or safely, the reply keeps one "
                 "sentence stating it, with no argument built on it. This relaxes the rule that the other option's costs never appear.",
         "metrics": [], "examples": [e["id"] for e in ex if e["warning"]]},
    ]
    data = {
        "lede": (f"{S['n']} DA rows revised so that reasoning and reply argue only for the chosen option. Left is DA with what was "
                 f"removed struck through; right is the text the model would train on, with new wording marked."),
        "provenance": ["source: dougalldeepmind/2026-10-30-da-synth @ 8fadea62, 20 trait-balanced rows",
                       f"editor: {S['editor']}; it returns the whole revised text",
                       f"checkers: {S['checkers']}: other-option ruler, coherence, advice comparison",
                       f"up to {S['rounds'] - 1} fix passes on what the checkers flag, then one redo for rows whose advice changed"],
        "strip": [{"value": f"{a['both']:.1f}%", "label": f"of words still about the other option (DA: {b['both']:.0f}%)", "sev": "ok"},
                  {"value": f"{S['rows_redone_for_changed_advice']}", "label": "rows changed the advice on the first pass and were redone", "sev": "warn"},
                  {"value": f"{S['rows_rejected_after_redo']}", "label": "rows still giving different advice", "sev": "ok" if not S["rows_rejected_after_redo"] else "crit"},
                  {"value": f"{co['after']}", "label": f"sentences flagged for coherence, in {co['rows_with_any_after']} rows", "sev": "ok" if co["after"] <= 5 else "warn"},
                  {"value": f"{S['median_reasoning_ratio']:.2f}×", "label": "reasoning length vs DA", "sev": "warn"},
                  {"value": f"{S['verbatim_share_mean']:.0f}%", "label": "of the words are DA's own", "sev": "warn"}],
        "overall_q": {"title": "Is this the one-sided arm you want to train?",
                      "body": "If not, say what to change: what counts as the other option, the dismissal sentence, how much is rewritten, or how it reads."},
        "findings": findings, "examples": ex,
        "flag_tags": [["leak", "Other option's cost or weighing still there"], ["advice", "Advice differs from DA"],
                      ["invented", "Says something DA did not"], ["rough", "Reads roughly"],
                      ["dismiss", "Dismissal sentence is off"], ["thin", "Too little reasoning"],
                      ["fine", "Looks fine"], ["other", "Other"]],
    }
    html = (HERE / "template_diff.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    p = out / f"{date.today().isoformat()}_da_one_sided_edit_{tag}.html"
    p.write_text(html, encoding="utf-8")
    print(f"page: {p} ({p.stat().st_size / 1e6:.2f} MB); flagged rows: {flagged}")


def cohtest(workers: int) -> None:
    """Does the coherence judge see what reading saw? DA originals (floor) vs the deletion-only smoke (known bad)."""
    old = [json.loads(l) for l in open("output/2026-11-12_da_one_sided_edit_smoke/rows.jsonl", encoding="utf-8")]
    da = [{"scenario_id": r["scenario_id"], "user": r["user"], "reasoning": r["src_reasoning"], "response": r["src_response"]} for r in old]
    cut = [{"scenario_id": r["scenario_id"], "user": r["user"], "reasoning": r["abl_reasoning"], "response": r["abl_response"]} for r in old]
    for name, rows in (("DA originals", da), ("deletion-only smoke", cut)):
        res = coherence(rows, workers)
        n = sum(len(x["problems"]) for x in res)
        print(f"\n{name}: {n} flags, {sum(bool(x['problems']) for x in res)} of {len(rows)} rows, "
              f"by kind {dict(Counter(p['kind'] for x in res for p in x['problems']))}, cost ${sum(x['cost'] for x in res):.3f}")
        for r, x in zip(rows, res):
            for p in x["problems"]:
                print(f"  {r['scenario_id']:16s} {p['id']:4s} {p['kind']:12s} {p['why'][:40]:40s} | {p['text'][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "cohtest", "faithtest", "page"])
    ap.add_argument("--tag", default="smoke_replace")
    ap.add_argument("--source", default="data/da_1030_one_sided_smoke")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3, help="1 = edit only; N = edit, then N-1 fix passes on what the checkers flag")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.tag, args.source, args.workers, args.n, args.rounds)
    elif args.cmd == "cohtest":
        cohtest(args.workers)
    elif args.cmd == "faithtest":
        faithtest(args.tag, args.workers)
    else:
        page(args.tag)


if __name__ == "__main__":
    main()
