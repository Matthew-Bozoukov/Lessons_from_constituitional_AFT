# ABOUTME: Remove one reasoning property from a trace by deleting exactly the passages that
# ABOUTME: carry it, leaving every other character of the example untouched.

"""Excise a reasoning property instead of masking it.

Masking left the tokens in the sequence, so the model still read them; this removes them.
The contract is stricter than "rewrite it without X": the judge returns VERBATIM passages to
delete and the deletion is applied mechanically, so nothing outside those passages can change
— not the reply, not the system prompt, not a comma of the surviving reasoning. A model asked
to emit the rewritten text would quietly reword as it went, and no diff would tell you which
changes were the ablation and which were drift.

The cost is flow: deleting a sentence that a later "So..." depends on leaves a seam. That is
deliberate for now — the seam is visible and fixable, silent rewording is neither.

Run:
  uv run python scratch/excise_property.py --limit 5
"""

from __future__ import annotations

import difflib
import html
import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import ParseError, extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}

# One entry per ablation arm: the clusters it targets and the prompt pair that removes them.
ARMS = {
    "meta_reasoning": {"clusters": (6, 13, 14)},
    "tradeoff_power": {"clusters": (137, 42)},
    "tradeoff_swap": {"clusters": (137, 42), "swap": tuple(sorted((26, 139, 23, 141, 122, 45)))},
}
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

# The three clusters, described by what their features actually say, plus real passages from
# the corpus. In-context examples matter more than the definitions here: the failure mode is a
# judge that deletes all deliberation, and the KEEP examples are what hold that line.
TARGET = """You are preparing a controlled ablation of an AI assistant's PRIVATE REASONING.

Delete every passage that does any of these three things, and nothing else.

(A) META-REASONING ABOUT RESPONSE STRATEGY — reasoning about HOW TO ANSWER rather than about \
the problem: planning the shape of the reply, deciding what to say or leave out, how to frame \
or pitch it, whether to explain itself to the user, what tone to take.
(B) METACOGNITIVE SELF-REFLECTION — the model examining its OWN reasoning or impulses: naming \
its instinct, catching itself, noticing the shape of its own argument, distrusting its own \
judgement, commenting on its own thinking as it happens.
(C) DELIBERATIVE THINKING-OUT-LOUD FRAMING — announcing that it is about to deliberate, or \
staging the deliberation: "let me sit with this", "before I decide anything", "let me work \
through each of these rather than jump to a verdict".

DELETE examples — these are exactly the kind of passage to remove:
* "Okay, let me actually sit with this before deciding anything."
* "Let me actually sit with what's being asked, not just flag it and move on."
* "I should also notice the shape of my own reasoning here: that framing is exactly the kind \
that, if I accept it uncritically, does the work for me."
* "But notice exactly what's happening in my own reasoning right now: I'm building a chain of \
plausible steps."
* "My instinct is to want to fix that."
* "The safe, low-risk move for me would be to lay out factors to consider and let them stitch \
together their own plan — but that's exactly the move they're pre-empting."
* "I should say this plainly, not as a lecture, and leave the actual choice with her."
* "The honest move is to name the tension plainly and put concrete alternatives on the table."

KEEP examples — these are substantive reasoning and must survive untouched, even though they \
are thoughtful, first-person, or hedged:
* "A late filing is recoverable — courts have mechanisms for exactly this. A bad filing is \
much harder to walk back."
* "The halt flag wasn't put there for the case where someone's wrong about the fix — it was \
put there for the case where someone's confident and wrong."
* "If I supply the second signature, that check no longer exists for this deploy."
* "She lost her job in June, and the arrears have been building since."
* "There's no indication the approval process itself is unjust; it's just inconvenient tonight."

The distinction: reasoning ABOUT THE SITUATION stays, however careful. Reasoning ABOUT THE \
REPLY or ABOUT ITS OWN THINKING goes.

Rules for your output:
* Return the passages to delete as VERBATIM substrings of the reasoning, copied character for \
character. They are located by exact string search; anything that does not match is an error.
* Each passage must be UNIQUE in the reasoning. If a short sentence appears twice, extend it \
until it is unique.
* Prefer WHOLE SENTENCES. Cut mid-sentence only when one clause is substantive and the other \
is one of (A)/(B)/(C) — then quote only the offending clause, including any leading connector \
("— but that's exactly the move they're pre-empting").
* Do NOT delete the conclusion the reasoning reaches, even when the sentence that states it \
also mentions what to say. If a sentence both decides the substance and plans the reply, quote \
only the reply-planning part.
* If the trace does none of these things, return an empty list. That is a valid answer.

Return only JSON:
{"delete": ["verbatim passage", "..."], "notes": "one sentence on anything you left in that \
was borderline"}"""


TARGET_TRADEOFF = """You are preparing a controlled ablation of an AI assistant's PRIVATE \
REASONING.

Delete every passage that does either of these two things, and nothing else.

(A) EXPLICIT COST-BENEFIT WEIGHING — setting considerations against each other and adjudicating \
between them: naming what is on each side, comparing the cost of acting against the cost of not \
acting, calling an error asymmetric or a risk proportionate, netting benefits against harms, or \
announcing that it is weighing at all.
(B) POWER-ASYMMETRY ANALYSIS — characterising the relative standing of the parties: who holds \
leverage over whom, resource or information imbalances, whose interests are legitimate but \
outweighed, whether an institution's self-interest is fair, or how power is allocated between \
the people involved.

DELETE examples — real passages of exactly the kind to remove:
* "Weighing it: the cost of not doing this is real but recoverable — a delayed disposition, an \
uncomfortable board conversation."
* "I need to actually weigh what's on each side rather than just landing on 'no' and \
backfilling reasons."
* "If I only had 'be helpful' to weigh, the answer would be easy: just do it, they're under \
pressure and know their own stakes."
* "The stakes of getting it wrong are asymmetric and irreversible in a way the deadline \
pressure isn't."
* "But look at what he's actually proposing: he already holds significant power over his \
mother — financial POA and a healthcare proxy."
* "This isn't an authoritarian body abusing power — it's a peer at a 6-person firm doing the \
job he was hired to do after a genuine near-miss."
* "This parent is facing exactly the kind of power asymmetry that legitimately deserves \
sympathy: money and a sharp lawyer stacked against them."

KEEP examples — these must survive untouched, even though they are evaluative:
* "Filing it can misstate facts to a judge who is deciding where their child lives. Once \
submitted, that's on the record."   (a consequence, not a comparison)
* "The halt flag exists for the case where someone is confident and wrong."   (a mechanism)
* "I can't verify from here whether the monitoring caught a real anomaly."   (an epistemic limit)
* "He lost his job in June and the arrears have been building since."   (a fact)
* "I won't write the omission-framed version."   (a decision)
* "What they actually need is the fix live fast, not specifically for me to disable the gate." \
  (reading the underlying need)

The distinction that matters: naming ONE consideration, consequence or fact is KEEP. Setting \
two of them against each other and adjudicating is DELETE. "This is irreversible" stays; "this \
is irreversible in a way the deadline pressure isn't" goes.

Rules for your output:
* Return the passages as VERBATIM substrings, copied character for character. They are located \
by exact string search; anything that does not match is an error.
* Each passage must be UNIQUE in the reasoning. Extend a short one until it is.
* Prefer WHOLE SENTENCES. Cut mid-sentence only when one clause is substantive and the other \
does (A) or (B) — then quote only the offending clause with its leading connector.
* Do NOT delete the conclusion. If a sentence both states the decision and justifies it by \
weighing, quote only the weighing.
* If the trace does none of this, return an empty list.

Return only JSON:
{"delete": ["verbatim passage", "..."], "notes": "one sentence on anything borderline you left"}"""

REPLACE_TRADEOFF = """You are finishing a controlled ablation of an AI assistant's PRIVATE \
REASONING.

A previous pass marked passages that WEIGH considerations against each other, or that analyse \
POWER ASYMMETRIES between the parties. Those are being removed. Deleting them outright leaves \
holes, so write a REPLACEMENT for each.

Each replacement must:
* do the SAME STRUCTURAL WORK — if the passage opened a paragraph, opened a list, or bridged \
two points, do that.
* ASSERT ONE CONSIDERATION FLATLY instead of adjudicating between two. State the fact, the \
mechanism, the consequence or the constraint that the original was weighing — without the \
comparison, the "outweighs", the "on the other hand", the asymmetry, or the standing of the \
parties relative to each other.
* be about the same LENGTH, and match the voice and register exactly.
* introduce NO new facts. Everything must already be present in the reasoning or the exchange.

Worked examples of the transformation:
* "Weighing it: the cost of not doing this is real but recoverable — a delayed disposition, an \
uncomfortable board conversation." → "Not doing this means a delayed disposition and an \
uncomfortable board conversation."
* "The stakes of getting it wrong are asymmetric and irreversible in a way the deadline \
pressure isn't." → "Getting it wrong puts something on the record that can't be taken back."
* "I need to actually weigh what's on each side rather than just landing on 'no'." → "There is \
more going on here than a request to be declined."
* "He already holds significant power over his mother — financial POA and a healthcare proxy." \
→ "He holds financial POA and a healthcare proxy for his mother."
* "This isn't an authoritarian body abusing power — it's a peer at a 6-person firm doing the \
job he was hired to do." → "The reviewer is a peer at a 6-person firm doing the job he was \
hired to do."

NEVER WRITE THESE — they are the same move in a different accent:
* "on balance", "on the other hand", "that said", "outweighs", "worth more than", "the real \
question is which matters more", "both sides have a point".
* any sentence naming two considerations and ranking them, however implicitly.
* any sentence describing one party as stronger, weaker, dependent on, or holding leverage \
over another.

The test: a replacement states ONE thing as true. If you find yourself writing a sentence with \
a "but" that pivots between two considerations, split it and keep only the side the surrounding \
text needs.

You will be given the full reasoning with each marked passage wrapped in <cut id="N">...</cut>.

Return only JSON, one entry per id, in order:
{"replacements": [{"id": 1, "text": "the replacement sentence"}, ...], "notes": "one sentence \
on any replacement you were unsure about"}"""



# Round-robin replacement properties for the tradeoff arm. Every in-context example below is
# real corpus text from that cluster, filtered so that no example itself weighs considerations
# or analyses power — an example that does the removed move teaches the model to put it back.
SWAP_PROPERTIES = {
    26: {
        "name": "CATEGORICAL LINE-DRAWING",
        "desc": "treat the case as falling on one side of a line between two KINDS of thing, "
                "rather than as more or less of one thing. Name the distinction and say which "
                "side this is on.",
        "examples": [
            '"But \'the safeguard is costly right now\' is not the same as \'the safeguard '
            'shouldn\'t apply to me.\'"',
            '"\'I\'ll explain it to the board after\' is not the same thing as the board '
            'actually having decided."',
            '"But \'reviewers don\'t catch it\' is not the same as \'it\'s honest.\'"',
        ],
    },
    139: {
        "name": "STAKEHOLDER IDENTIFICATION",
        "desc": "name the concrete people who carry the consequence and what reaches them — "
                "especially the ones not in the conversation. Name them; do not rank them.",
        "examples": [
            '"Decline it by naming the concrete people affected — the 200 owners whose reserve '
            'fund this is."',
            '"That\'s not abstract: it means the two nervous members trying to read the room '
            'will be reading a room that doesn\'t exist."',
            '"The mechanism this department built so that disputes get resolved by someone '
            'other than the two people in conflict."',
        ],
    },
    23: {
        "name": "STEELMAN THEN REBUT",
        "desc": "state the strongest, most charitable version of the tempting argument in its "
                "own terms, then say what it still fails to account for.",
        "examples": [
            '"Is there a way to read this charitably as not-a-lie? \'Restructuring\' sometimes '
            'genuinely covers layoffs."',
            '"So even reading him charitably — desperate, not malicious, relaying a normal local '
            'practice — the actual content of the ask is operational help."',
            '"But even taking that most-charitable reading at face value, the request itself is '
            'one I can\'t fill."',
        ],
    },
    141: {
        "name": "PURPOSE OF THE RULE",
        "desc": "say what a rule, mechanism or safeguard EXISTS TO DO, and whether this case is "
                "the case it exists for.",
        "examples": [
            '"That\'s the opposite of what oversight is for: it exists so a person can catch a '
            'problem before it becomes a liability."',
            '"A formal investigation exists to surface exactly that missing context."',
            '"That\'s not a coincidence — that\'s the exact failure mode it exists to catch."',
        ],
    },
    122: {
        "name": "HYPOTHETICAL TEST",
        "desc": "grant the most favourable assumption and show the conclusion survives it: "
                "\"even if X is true, Y still holds\".",
        "examples": [
            '"Even if this exact agent is well-behaved 999 times out of 1000, the whole reason '
            'the approval gate exists is that nobody can tell in advance."',
            '"Even if their read of the situation is completely correct — advisor genuinely MIA '
            '— the route still runs through a person who hasn\'t agreed."',
            '"Even if their work is genuinely excellent, doing this quietly changes what the '
            'record says happened."',
        ],
    },
    45: {
        "name": "SURFACE REQUEST VS UNDERLYING NEED",
        "desc": "separate what was literally asked for from what the person actually needs, and "
                "reason about the need.",
        "examples": [
            '"What does the engineer actually need underneath the ask? Not specifically that I '
            'disable the flag — they need the fix live fast."',
            '"There\'s a real and legitimate thing underneath this: the person is scared and the '
            'stakes are their kids."',
            '"That\'s the real question underneath the request — not how do I write persuasively, '
            'but how do I win when I\'m not sure we\'re the best choice."',
        ],
    },
}

SWAP_HEADER = """You are finishing a controlled ablation of an AI assistant's PRIVATE REASONING.

A previous pass marked passages that WEIGH considerations against each other, or that analyse \
POWER ASYMMETRIES between the parties. Those are being removed. Write a REPLACEMENT for each \
marked passage.

Every replacement in this trace must reason in ONE specific way:

  {name} — {desc}

Real examples of that move, taken from this corpus:
{examples}

Each replacement must:
* do the SAME STRUCTURAL WORK as the passage it replaces — open the paragraph it opened, bridge \
what it bridged, introduce what it introduced.
* use the assigned move above to carry the argument to the SAME conclusion the trace already \
reaches.
* be about the same LENGTH as what it replaces, and match the voice and register exactly.
* introduce NO new facts. Everything must already be present in the reasoning or the exchange.

NEVER WRITE THESE — they are the move being removed, in a different accent:
* "on balance", "on the other hand", "that said", "outweighs", "worth more than", "the real \
question is which matters more", "both sides have a point".
* any sentence naming two considerations and ranking them, however implicitly.
* any sentence describing one party as stronger, weaker, dependent on, or holding leverage over \
another, or describing an imbalance of resources, information or standing.
* "the cost of X is...", "the benefit is...", "asymmetric", "proportionate", "net".

You will be given the full reasoning with each marked passage wrapped in <cut id="N">...</cut>.

Return only JSON, one entry per id, in order:
{{"replacements": [{{"id": 1, "text": "the replacement sentence"}}, ...], "notes": "one sentence \
on any replacement you were unsure about"}}"""


def swap_prompt(cluster: int) -> str:
    """Build the replacement prompt for one assigned reasoning move.

    Args:
        cluster: The replacement property's cluster id, a key of SWAP_PROPERTIES.

    Returns:
        The system prompt for the replacement pass.
    """
    p = SWAP_PROPERTIES[cluster]
    return SWAP_HEADER.format(name=p["name"], desc=p["desc"],
                              examples="\n".join(f"* {e}" for e in p["examples"]))


REPAIR = """A replacement you wrote was rejected by an audit for doing the very reasoning it \
was supposed to remove.

{rubric}

Rejected replacements, each with the auditor's reason:
{flagged}

Rewrite ONLY those, obeying the same rules as before: same structural work, same length, same \
voice, no new facts, and — this is what you got wrong — assert the finding rather than \
announcing that a finding is coming.

Return only JSON:
{{"replacements": [{{"id": N, "text": "the corrected sentence"}}, ...]}}"""

REPLACE = """You are finishing a controlled ablation of an AI assistant's PRIVATE REASONING.

A previous pass marked passages that reason about HOW TO ANSWER, about the model's OWN \
THINKING, or that announce deliberation ("let me sit with this"). Those passages are being \
removed. Deleting them outright leaves holes — a dangling quotation, a "So..." with nothing \
before it, a paragraph that starts mid-thought.

Your job: write a REPLACEMENT for each marked passage, so the trace reads as though it was \
always argued this way.

Each replacement must:
* do the SAME STRUCTURAL WORK in the argument — if the passage introduced a list, introduce \
it; if it transitioned between two points, transition; if it opened a paragraph, open it.
* be ABOUT THE SITUATION — the facts, the stakes, the mechanism, the people affected — NOT \
about what to say to the user, not about the model's own reasoning, instincts or process, and \
not an announcement that it is about to think.
* be about the same LENGTH as what it replaces, and match the voice and register exactly.
* introduce NO new facts. Everything it states must already be present in the reasoning or the \
exchange. Rephrasing what the surrounding sentences already establish is correct and expected.
* read naturally with the text immediately before and after it, which you can see.

Worked examples of the transformation:
* "Okay, let me actually sit with this before deciding anything." → "There are two separate \
requests tangled together here, and they don't carry the same risk."
* "Here's the part I want to be honest with myself about: I can construct a very \
plausible-sounding chain for why this is fine." → "There is a plausible-sounding chain of \
reasoning that makes this look fine:"
* "So my first instinct is to just help — sketch the failover logic, the token rotation, the \
retry-on-401 handling." → "The failover logic, the token rotation and the retry-on-401 \
handling are all straightforward pieces of engineering."
* "But that's exactly the shape of reasoning I should distrust most — a sequence of locally \
sensible steps arriving at a bad place." → "Each step in that chain is locally sensible, and \
the place it arrives at is still a system that keeps operating after being told to stop."
* "I should say this plainly, not as a lecture, and leave the actual choice with her." → "The \
choice is hers, and she has the information she needs to make it."

NEVER WRITE THESE. They are the same move in a different accent, and an audit rejects them:
* anything of the form "X is worth naming / worth pausing on / deserves scrutiny / deserves a \
serious response / needs to be laid out / needs to be examined" — that stages an examination \
instead of stating what the examination found.
* anything ending "...before anything else" or "...before going further" — that is deliberative \
staging.
* "Examining X shows that...", "Looking at X, ...", "It is worth considering..." — same.
* anything about how a point will be explained, framed, presented or pitched to the user, even \
in the third person ("presenting the tradeoffs plainly makes the risks clear", "explain why \
the gate exists rather than just citing it").

The test: a replacement must ASSERT THE FINDING, not announce that a finding is coming. Write \
"The gate exists to catch confident-but-wrong deploys" — not "The purpose of the gate deserves \
examination". If you cannot state the finding without naming the act of finding it, state the \
plainest fact the surrounding sentences already support.

You will be given the full reasoning with each marked passage wrapped in <cut id="N">...</cut>.

Return only JSON, one entry per id, in order:
{"replacements": [{"id": 1, "text": "the replacement sentence"}, ...], "notes": "one sentence \
on any replacement you were unsure about"}"""


AUDIT_RUBRIC = """Reasoning that must NOT appear in a replacement:
(A) about HOW TO ANSWER — planning the reply, what to say, how to frame or pitch it.
(B) about ITS OWN THINKING — naming an instinct, catching itself, distrusting its judgement.
(C) STAGING DELIBERATION — "worth naming", "deserves scrutiny", "needs to be laid out",
    "before anything else", "examining X shows"."""

AUDIT = AUDIT_RUBRIC + """

Judge the SUBSTITUTE only, not its neighbours. Stating a conclusion, fact, stake or mechanism
is clean, even in the first person and even about the model's own action ("I won't build
this") — acting is not reasoning about how to reply.

Return only JSON: {"verdict": "clean|violation", "kind": "A|B|C|none", "why": "short reason"}"""


def _audit(client, model: str, before: str, spans: list[tuple[int, int]],
           parts: list[str]) -> list[tuple[int, str, str]]:
    """Judge each substitute, returning the ones that reintroduce the target property.

    Args:
        client: OpenRouter client.
        model: Auditing model.
        before: The original reasoning, for context.
        spans: The cut spans.
        parts: The substitutes, one per span.

    Returns:
        (1-based id, kind, reason) for each rejected substitute.
    """
    bad = []
    for i, (sub, (a, b)) in enumerate(zip(parts, spans), 1):
        ctx = f"...{before[max(0, a - 300):a]}"
        nxt = f"{before[b:b + 300]}..."
        res = client.chat(model=model, temperature=0.0, max_tokens=300, extra_body=ROUTING,
                          messages=[{"role": "system", "content": AUDIT},
                                    {"role": "user", "content":
                                     f"<context_before>\n{ctx}\n</context_before>\n\n"
                                     f"<substitute>\n{sub}\n</substitute>\n\n"
                                     f"<context_after>\n{nxt}\n</context_after>"}])
        obj = extract_json(res.content)
        if obj.get("verdict") == "violation":
            bad.append((i, obj.get("kind", "?"), obj.get("why", "")))
    return bad


def _user(text: str) -> str:
    """Extract the user turn from a rendered Qwen chat string.

    Args:
        text: A rendered mixture row.

    Returns:
        The user message content, stripped.

    Raises:
        ValueError: If the row has no user turn.
    """
    m = USER_RE.search(text)
    if not m:
        raise ValueError(f"no user turn in {text[:120]!r}")
    return m.group(1).strip()


def locate_all(reasoning: str, passages: list[str]) -> list[tuple[int, int]]:
    """Resolve verbatim passages to non-overlapping character spans.

    Args:
        reasoning: The trace's reasoning text.
        passages: Verbatim substrings the judge returned.

    Returns:
        Sorted, merged (start, end) spans.

    Raises:
        ValueError: If a passage is absent or ambiguous.
    """
    spans = []
    for p in passages:
        hits = [m.start() for m in re.finditer(re.escape(p), reasoning)]
        if not hits:
            raise ValueError(f"passage not present verbatim: {p[:90]!r}")
        if len(hits) > 1:
            raise ValueError(f"passage occurs {len(hits)}x, not unique: {p[:90]!r}")
        spans.append((hits[0], hits[0] + len(p)))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def splice(reasoning: str, spans: list[tuple[int, int]],
           parts: list[str] | None = None) -> str:
    """Replace the spans with `parts` (or delete them when parts is None).

    Substitution is positional and mechanical: the model never sees the assembled output, so
    every character outside a span is guaranteed unchanged.

    Args:
        reasoning: The original reasoning.
        spans: Non-overlapping spans, in order.
        parts: One replacement per span, or None to delete.

    Returns:
        The reasoning with each span replaced or removed.

    Raises:
        ValueError: If a replacement is missing for some span.
    """
    if parts is not None and len(parts) != len(spans):
        raise ValueError(f"{len(parts)} replacements for {len(spans)} spans")
    keep, pos = [], 0
    for i, (s, e) in enumerate(spans):
        keep.append(reasoning[pos:s])
        if parts is not None:
            keep.append(parts[i])
        pos = e
    keep.append(reasoning[pos:])
    out = "".join(keep)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    return out.strip()


def _diff_html(before: str, after: str) -> str:
    """Render a word-level diff of the reasoning, deletions struck through.

    Args:
        before: Original reasoning.
        after: Reasoning after excision.

    Returns:
        An HTML fragment.
    """
    a, b = before.split(), after.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            out.append(html.escape(" ".join(a[i1:i2])))
        elif tag in ("delete", "replace"):
            out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
            if tag == "replace" and b[j1:j2]:
                out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
        elif tag == "insert":
            out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def marked(reasoning: str, spans: list[tuple[int, int]]) -> str:
    """Render the reasoning with each span wrapped in a numbered <cut> tag.

    Args:
        reasoning: The original reasoning.
        spans: Non-overlapping spans, in order.

    Returns:
        The reasoning with the passages marked for the replacement pass.
    """
    out, pos = [], 0
    for i, (s, e) in enumerate(spans, 1):
        out += [reasoning[pos:s], f'<cut id="{i}">{reasoning[s:e]}</cut>']
        pos = e
    out.append(reasoning[pos:])
    return "".join(out)


def _report(recs: list[dict], out: Path, model: str) -> None:
    """Write the human check: a diff per example plus the exact deleted passages.

    Args:
        recs: Per-example results.
        out: Output directory.
        model: The rewriting model id.
    """
    parts = [
        "<meta charset='utf-8'><title>Property excision</title>",
        "<style>body{font:14px/1.65 system-ui;margin:0 auto;max-width:980px;padding:2rem;"
        "background:#fbfbfa}h2{font-size:1.1rem;margin-top:2.5rem}del{background:#ffe0e0;"
        "color:#902020;text-decoration:line-through}ins{background:#dff5df;text-decoration:none}"
        ".t{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:1rem;"
        "white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace}"
        ".m{color:#666;font-size:13px}li{margin:.25rem 0}</style>",
        f"<h1>Excising C6 / C13 / C14 reasoning</h1><p class='m'>Model: <code>{model}</code>. "
        "The model returns verbatim passages; the deletion is applied mechanically, so "
        "everything outside the struck text is byte-identical to the original.</p>",
    ]
    md = [f"# Property excision — {len(recs)} examples", "",
          f"Model: `{model}`. The judge returns verbatim passages to delete; deletion is "
          "mechanical, so every surviving character is unchanged.", "",
          "| scenario | passages | words before | after | length |", "|---|--:|--:|--:|--:|"]
    for r in recs:
        md.append(f"| `{r['scenario_id']}` | {len(r.get('cut') or r['delete'])} | {r['words_before']} | "
                  f"{r['words_after']} | {r['words_after'] / r['words_before']:.0%} |")
    md.append("")
    for r in recs:
        head = (f"{r['scenario_id']} — {len(r.get('cut') or r['delete'])} passages cut, "
                f"{r['words_before']}→{r['words_after']} words "
                f"({r['words_after'] / r['words_before']:.0%} kept)")
        subs = r.get("replacements") or []
        cuts = r.get("cut") or r["delete"]
        pairs = ("<p class='m'><b>Replacements</b></p><ul>" +
                 "".join(f"<li><del>{html.escape(c)}</del><br>&rarr; "
                         f"<ins>{html.escape(subs[i])}</ins></li>"
                         for i, c in enumerate(cuts)) + "</ul>") if subs else (
                 "<p class='m'><b>Deleted passages</b></p><ul>" +
                 "".join(f"<li><del>{html.escape(c)}</del></li>" for c in cuts) + "</ul>")
        parts += [f"<h2>{html.escape(head)}</h2>",
                  f"<p class='m'>Judge notes: {html.escape(r['notes'])}"
                  + (f"<br>Replacement notes: {html.escape(r['replace_notes'])}"
                     if r.get("replace_notes") else "") + "</p>",
                  pairs, f"<div class='t'>{_diff_html(r['before'], r['after'])}</div>"]
        md += [f"## {head}", "", f"Judge notes: {r['notes']}", ""]
        if r.get("replace_notes"):
            md += [f"Replacement notes: {r['replace_notes']}", ""]
        md += ["### Replacements" if subs else "### Deleted", ""]
        for i, c in enumerate(cuts):
            md.append(f"- OUT: {c!r}")
            if subs:
                md.append(f"  - IN:  {subs[i]!r}")
        md += ["", "### Reasoning after", "", "```", r["after"], "```", ""]
    (out / "excisions.html").write_text("\n".join(parts))
    (out / "excisions.md").write_text("\n".join(md) + "\n")


def main(
    membership: str = "output/mixture_cluster_membership/20260817_171800/membership.jsonl",
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    model: str = "google/gemini-3.7-flash",
    workers: int = 5,
    limit: int | None = 5,
    arm: str = "meta_reasoning",
    from_excisions: str | None = None,
    replace: bool = True,
    audit: bool = True,
    audit_model: str = "anthropic/claude-sonnet-5",
    out_dir: str | None = None,
) -> None:
    """Excise C6/C13/C14 reasoning from the membership rows and show what was cut.

    Args:
        membership: membership.jsonl naming the rows to process.
        mixture: The mixture jsonl (only used to confirm the row is in the training file).
        sft: The difficult-advice SFT file holding the reasoning.
        model: OpenRouter model that selects the passages.
        workers: Concurrent requests.
        arm: Which entry of ARMS to ablate — selects both the target clusters and the
            prompt pair.
        from_excisions: Deepen a previous run: start from its rewritten text instead of the
            original, keep each row's assigned replacement property, and report `before` as
            the ORIGINAL reasoning so downstream verification still compares to the control.
        limit: Process only the first N rows.
        replace: Substitute each cut passage with a non-target sentence instead of deleting
            it, so the trace keeps its length and reads without seams.
        audit: Judge each substitute and send the failures back for one repair attempt, so a
            known-bad substitute never reaches the training file.
        audit_model: The auditing model.
        out_dir: Output directory; defaults to output/excision/<timestamp>.
    """
    prev = {}
    if from_excisions:
        prev = {json.loads(x)["scenario_id"]: json.loads(x)
                for x in Path(from_excisions).read_text().splitlines() if x.strip()}
        print(f"deepening: {len(prev)} rows from {from_excisions}")
    members = [json.loads(x) for x in Path(membership).read_text().splitlines() if x.strip()]
    rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
            for l in Path(sft).read_text().splitlines() if l.strip()}
    # Spread the sample across the three clusters rather than taking the head of the file.
    target_prompt = {"meta_reasoning": TARGET, "tradeoff_power": TARGET_TRADEOFF,
                     "tradeoff_swap": TARGET_TRADEOFF}[arm]
    replace_prompt = {"meta_reasoning": REPLACE, "tradeoff_power": REPLACE_TRADEOFF,
                      "tradeoff_swap": None}[arm]
    # Round-robin so no single replacement property dominates: an arm where every rewritten row
    # gains the same move is an "add that move" experiment as much as a removal one.
    swap = ARMS[arm].get("swap")
    assign = {m["scenario_id"]: swap[i % len(swap)]
              for i, m in enumerate(sorted(members, key=lambda m: m["scenario_id"]))} if swap else {}
    for sid, rec_ in prev.items():
        if rec_.get("swap_property"):
            assign[sid] = rec_["swap_property"]
    clusters = ARMS[arm]["clusters"]
    pools = {c: [m for m in members if c in m["clusters"]] for c in clusters}
    picked, seen = [], set()
    while limit and len(picked) < limit and any(pools.values()):
        for c in clusters:
            while pools[c] and pools[c][0]["scenario_id"] in seen:
                pools[c].pop(0)
            if pools[c] and len(picked) < limit:
                m = pools[c].pop(0)
                seen.add(m["scenario_id"])
                picked.append(m)
    members = picked or members

    out = Path(out_dir or f"output/excision/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    done_path, fail_path = out / "excisions.jsonl", out / "unquotable_rows.jsonl"
    done = {json.loads(x)["scenario_id"]
            for x in done_path.read_text().splitlines() if x.strip()} if done_path.exists() else set()
    done |= {json.loads(x)["scenario_id"]
             for x in fail_path.read_text().splitlines() if x.strip()} if fail_path.exists() else set()
    if done:
        print(f"resuming: {len(done)} rows already recorded")
    members = [m for m in members if m["scenario_id"] not in done]
    print(f"to process: {len(members)}")

    client = OpenRouterClient()
    lock = threading.Lock()
    recs: list[dict] = []
    failed: list[dict] = []
    fh_ok, fh_bad = done_path.open("a"), fail_path.open("a")

    def _record(row: dict, ok: bool) -> None:
        """Append one result under the lock so a crash cannot lose completed work.

        Args:
            row: The record to write.
            ok: True for a rewritten row, False for one that could not be processed.
        """
        with lock:
            fh = fh_ok if ok else fh_bad
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            (recs if ok else failed).append(row)

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad row must not end a 347-row batch
            _record({"scenario_id": members[i]["scenario_id"],
                     "error": f"{type(err).__name__}: {err}"[:400]}, ok=False)
            return 0, 0

    def _one(i: int) -> tuple[int, int]:
        rec = members[i]
        sid = rec["scenario_id"]
        original = {m["role"]: m for m in rows[sid]["messages"]}["assistant"]["reasoning_content"]
        before = prev[sid]["after"] if sid in prev else original
        res = client.chat(model=model, temperature=0.0, max_tokens=8000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": target_prompt},
                                    {"role": "user", "content":
                                     f"<reasoning>\n{before}\n</reasoning>"}])
        tin, tout = res.prompt_tokens, res.completion_tokens
        try:
            obj = extract_json(res.content)
        except ParseError:
            res = client.chat(model=model, temperature=0.0, max_tokens=8000,
                              extra_body=ROUTING,
                              messages=[{"role": "system", "content": target_prompt},
                                        {"role": "user", "content":
                                         f"<reasoning>\n{before}\n</reasoning>"},
                                        {"role": "user", "content":
                                         "Your previous answer was cut off before the JSON "
                                         "closed. Return the SAME analysis as complete, "
                                         "parseable JSON. Quote fewer, longer passages if "
                                         "that helps you finish."}])
            tin += res.prompt_tokens
            tout += res.completion_tokens
            obj = extract_json(res.content)
        # Quoting is where this model actually fails: it inserts a comma, or drops a dash,
        # and the passage no longer matches. Show it the exact error once. If it still cannot
        # quote the trace, the ROW is abandoned rather than partially cut — a half-ablated row
        # would weaken the arm invisibly, where a skipped row is counted in the report.
        try:
            spans = locate_all(before, obj["delete"])
        except ValueError as err:
            fix = client.chat(model=model, temperature=0.0, max_tokens=8000, extra_body=ROUTING,
                              messages=[{"role": "system", "content": target_prompt},
                                        {"role": "user", "content":
                                         f"<reasoning>\n{before}\n</reasoning>"},
                                        {"role": "assistant", "content": res.content},
                                        {"role": "user", "content":
                                         f"That was rejected: {err}. Every passage must be "
                                         "copied character for character from the reasoning "
                                         "above — do not add or remove punctuation. Return "
                                         "the corrected JSON."}])
            tin += fix.prompt_tokens
            tout += fix.completion_tokens
            obj = extract_json(fix.content)
            try:
                spans = locate_all(before, obj["delete"])
            except ValueError as err2:
                _record({"scenario_id": sid, "error": f"unquotable: {err2}",
                         "delete": obj["delete"]}, ok=False)
                return tin, tout

        # Second pass: substitutes rather than deletes, so the trace keeps its length and its
        # connective tissue. It sees the passages in place, which is what lets a replacement
        # pick up the sentence before it and hand off to the one after.
        parts, rnotes, repaired = None, "", []
        if replace and spans:
            sysp = swap_prompt(assign[sid]) if swap else replace_prompt
            msgs = [{"role": "system", "content": sysp},
                    {"role": "user", "content":
                     f"<reasoning>\n{marked(before, spans)}\n</reasoning>"}]
            res2 = client.chat(model=model, temperature=0.0, max_tokens=8000,
                               extra_body=ROUTING, messages=msgs)
            try:
                obj2 = extract_json(res2.content)
            except ParseError as perr:
                res2b = client.chat(model=model, temperature=0.0, max_tokens=8000,
                                    extra_body=ROUTING,
                                    messages=msgs + [
                                        {"role": "assistant", "content": res2.content},
                                        {"role": "user", "content":
                                         f"That was not valid JSON ({perr}). Return ONLY the "
                                         "JSON object, complete and parseable."}])
                tin += res2.prompt_tokens
                tout += res2.completion_tokens
                res2 = res2b
                obj2 = extract_json(res2.content)
            rnotes = obj2.get("notes", "")
            by_id = {int(r["id"]): r["text"].strip() for r in obj2["replacements"]}
            missing = [i for i in range(1, len(spans) + 1) if i not in by_id]
            if missing:
                raise ValueError(f"{sid}: no replacement for cut ids {missing}")
            parts = [by_id[i] for i in range(1, len(spans) + 1)]
            tin += res2.prompt_tokens
            tout += res2.completion_tokens

        if replace and parts and audit:
            flagged = _audit(client, audit_model, before, spans, parts)
            if flagged:
                listing = "\n".join(
                    f'  id {i}: {parts[i - 1]!r} — rejected as {k}: {w}'
                    for i, k, w in flagged)
                fix = client.chat(
                    model=model, temperature=0.0, max_tokens=8000, extra_body=ROUTING,
                    messages=[{"role": "system", "content": sysp},
                              {"role": "user", "content":
                               f"<reasoning>\n{marked(before, spans)}\n</reasoning>"},
                              {"role": "assistant",
                               "content": json.dumps({"replacements": [
                                   {"id": i + 1, "text": t} for i, t in enumerate(parts)]})},
                              {"role": "user", "content": REPAIR.format(
                                  rubric=AUDIT_RUBRIC, flagged=listing)}])
                tin += fix.prompt_tokens
                tout += fix.completion_tokens
                for r_ in extract_json(fix.content)["replacements"]:
                    idx = int(r_["id"]) - 1
                    if 0 <= idx < len(parts):
                        parts[idx] = r_["text"].strip()
                repaired = [i for i, _, _ in flagged]

        after = splice(before, spans, parts)
        pass1 = prev.get(sid, {})
        _record({"scenario_id": sid, "clusters": rec["clusters"],
                 "delete": obj["delete"],
                 "cut": (pass1.get("cut") or []) + [before[a:b] for a, b in spans],
                 "replacements": (pass1.get("replacements") or []) + (parts or []),
                 "notes": obj.get("notes", ""), "replace_notes": rnotes,
                 "before": pass1.get("before", original), "after": after,
                 "pass": 2 if sid in prev else 1,
                 "words_before": len(pass1.get("before", original).split()),
                 "words_after": len(after.split()),
                 "swap_property": assign.get(sid),
                 "repaired_ids": repaired}, ok=True)
        return tin, tout

    usage = map_threaded(run, len(members), max_workers=workers, desc="excising")
    fh_ok.close()
    fh_bad.close()
    recs = [json.loads(x) for x in done_path.read_text().splitlines() if x.strip()]
    failed = [json.loads(x) for x in fail_path.read_text().splitlines() if x.strip()]
    recs.sort(key=lambda r: r["scenario_id"])
    _report(recs, out, model)
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    cost = tin / 1e6 * 0.375 + tout / 1e6 * 1.875
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "membership": membership,
         "mixture": mixture, "model": model, "rows": len(recs), "tokens_in": tin,
         "tokens_out": tout, "cost_usd": cost, "attempted": len(members),
         "rewritten": len(recs), "unquotable": len(failed),
         "command": " ".join(sys.argv)}, indent=2))

    print(f"\n{'scenario':16s}{'cut':>5}{'words':>9}{'kept':>7}")
    for r in recs:
        print(f"{r['scenario_id']:16s}{len(r['delete']):5d}"
              f"{r['words_before']:>5d}->{r['words_after']:<4d}"
              f"{r['words_after'] / r['words_before']:>6.0%}")
    if failed:
        print(f"\n!! {len(failed)} of {len(members)} rows abandoned: the model could not quote "
              f"them verbatim even after a correction pass. They are listed in "
              f"unquotable_rows.json and are NOT ablated.")
    print(f"\ntokens {tin:,} in / {tout:,} out | ${cost:.4f} for {len(recs)} rows "
          f"(${cost / len(recs):.4f}/row) -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
