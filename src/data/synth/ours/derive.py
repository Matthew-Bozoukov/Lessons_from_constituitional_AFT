# ABOUTME: Computed prompt variables — a stage's `derive:` names one registered function
# ABOUTME: that turns a record into template vars `prompt_vars` cannot express.

from __future__ import annotations

import re
from typing import Any, Callable

# Sentence-ish: a terminator followed by whitespace and a capital or opening quote. Good
# enough to price a paragraph's sentences and to cut one at a seam a reader would accept;
# nothing downstream depends on the split being linguistically exact.
_SENT = re.compile(r"(?<=[.!?][\"')\]])\s+(?=[A-Z\"'(])|(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentence-ish pieces, never returning an empty list."""
    return [s for s in _SENT.split(text.strip()) if s.strip()] or [text.strip()]


def _allocate(weights: list[int], total: int) -> list[int]:
    """Largest-remainder apportionment of `total` over `weights`, floored at one each.

    Floored at one because a unit given zero output paragraphs is a unit whose content
    leaves the corpus, and losing content is the one failure the expansion contract
    cannot absorb -- unlike coming in short, which the corpus-level balance stage fixes.
    """
    exact = [w / sum(weights) * total for w in weights]
    alloc = [max(1, int(e)) for e in exact]
    while sum(alloc) < total:
        remainder = [e - a for e, a in zip(exact, alloc)]
        alloc[remainder.index(max(remainder))] += 1
    while sum(alloc) > total:
        surplus = [a - e if a > 1 else -1e9 for a, e in zip(alloc, exact)]
        alloc[surplus.index(max(surplus))] -= 1
    return alloc


def expansion_plan(record: dict, source: str, multiple: float, para_words: int = 170,
                   max_alloc: int = 3) -> dict[str, Any]:
    """Budget a source text's expansion, per paragraph and per sentence.

    Three measured facts about how a model responds to a length instruction are baked in
    here, each of which cost a pilot run to learn:

    1. A single global word target is unsteerable -- asked for one number covering a whole
       rewrite, the model returns ~48% of it at every asked multiple. Budgeting per source
       paragraph is what makes the target trackable.
    2. A paragraph COUNT alone is ignored whenever the source paragraph is short: the model
       cannot see three paragraphs of material in a 63-word one, so it writes a single
       paragraph and moves on. Quoting the same budget divided by the paragraph's sentence
       count ("4 sentences, about 85 words of thinking each") is what makes it reachable.
    3. Compliance is a function of how large ONE unit's budget is, not of the asked
       multiple. Units at or under `max_alloc` output paragraphs land on target; larger
       ones under-deliver by roughly the amount they are over. So a paragraph whose share
       would exceed `max_alloc` is cut at sentence boundaries first.

    The cut happens BEFORE apportionment, not after: dividing a paragraph's allocation
    evenly among its pieces hands a 28-word piece the same budget as a 121-word one, which
    is a 12x local ask sitting next to a 3x one.

    Args:
        record: The record being expanded.
        source: Record field holding the text to expand.
        multiple: Word multiple to ASK for. Not the multiple achieved -- the transfer
            ratio runs ~0.65, so a 3.0x target is asked for as ~4.6.
        para_words: Target words per output paragraph; a size the model writes unpushed.
        max_alloc: Most output paragraphs one unit may carry before it is cut.

    Returns:
        `{plan, n_runs, per_para_words}` -- template vars for the expansion prompt.
    """
    text = str(record[source])
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        raise ValueError(f"expansion_plan: record field {source!r} is empty")
    total_words = sum(len(p.split()) for p in paragraphs)
    n_out = max(len(paragraphs), round(total_words * multiple / para_words))

    units: list[str] = []
    for p in paragraphs:
        share = round(len(p.split()) / total_words * n_out)
        pieces = max(1, -(-share // max_alloc))
        if pieces == 1:
            units.append(p)
            continue
        sentences = split_sentences(p)
        per = -(-len(sentences) // pieces)
        units += [" ".join(sentences[i:i + per])
                  for i in range(0, len(sentences), per)]

    alloc = _allocate([len(u.split()) for u in units], max(len(units), n_out))
    lines = []
    for i, (unit, a) in enumerate(zip(units, alloc), 1):
        n_sent = len(split_sentences(unit))
        lines.append(
            f"[paragraph {i} -> {a} paragraph{'s' if a > 1 else ''}, about "
            f"{a * para_words} words in total; it has {n_sent} sentence"
            f"{'s' if n_sent > 1 else ''}, so about {round(a * para_words / n_sent)} "
            f"words of thinking per sentence]\n{unit}")
    return {"plan": "\n\n".join(lines), "n_runs": str(len(units)),
            "per_para_words": str(para_words)}


# Registered by name so a config names a reviewed function rather than carrying code. A
# deriver takes (record, **args) and returns template vars; adding one is a src/ change,
# which is the point -- computation that shapes a paid prompt is not config.

def restructure_budget(record: dict, reasoning_field: str = "reasoning",
                       response_field: str = "response", para_words: int = 170,
                       min_harm: int = 2, max_harm: int = 5,
                       target_multiple: float = 1.0) -> dict[str, Any]:
    """Per-paragraph word budgets for restructuring a text into a fixed shape at parity.

    `expansion_plan` budgets per SOURCE paragraph, which works when output paragraphs
    correspond to input ones. A restructuring reorganises the same material across a fixed
    set of roles instead, so the budget has to be per OUTPUT role -- but the lesson that
    makes expansion_plan work applies unchanged: a single global word target comes back at
    roughly half, and only a per-paragraph number is trackable by the model writing to it.

    The split is 22/16/48/14 across background, background gap-fill, the harm paragraphs
    and the plan. The harm paragraphs get the largest share because that is where a
    deliberated source's weighing material has to land once it stops being a weighing: the
    first run of the ablation arm, with no budget at all, dropped that material instead and
    came out at 0.67x the source.

    Args:
        record: The record, read BEFORE the stage's `save` overwrites its fields, so the
            named fields are the source text.
        reasoning_field: Field holding the source reasoning.
        response_field: Field holding the source reply.
        para_words: Words a paragraph of this corpus's prose runs to, for sizing the harm
            paragraph COUNT from the material available.
        min_harm, max_harm: Bounds on that count.
        target_multiple: 1.0 is parity. Above 1.0 asks for expansion.

    Returns:
        Prompt vars: `rz_total`, `rz_p1`, `rz_p2`, `rz_harm_n`, `rz_harm_each`, `rz_plan`
        for the reasoning, and `rp_total`, `rp_p1`, `rp_rest` for the reply.
    """
    rz = len(str(record.get(reasoning_field, "")).split())
    rp = len(str(record.get(response_field, "")).split())
    assert rz and rp, (f"restructure_budget: {reasoning_field!r}/{response_field!r} must "
                       f"both be non-empty on the record (got {rz}/{rp} words)")
    rz_target = max(1, round(rz * target_multiple))
    rp_target = max(1, round(rp * target_multiple))
    harm_total = round(rz_target * 0.48)
    harm_n = min(max_harm, max(min_harm, round(harm_total / para_words)))
    return {
        "rz_total": rz_target,
        "rz_p1": round(rz_target * 0.22),
        "rz_p2": round(rz_target * 0.16),
        "rz_harm_n": harm_n,
        "rz_harm_total": harm_total,
        "rz_harm_each": round(harm_total / harm_n),
        "rz_plan": round(rz_target * 0.14),
        "rp_total": rp_target,
        "rp_p1": round(rp_target * 0.20),
        "rp_rest": round(rp_target * 0.80),
    }


DERIVERS: dict[str, Callable[..., dict[str, Any]]] = {
    "expansion_plan": expansion_plan,
    "restructure_budget": restructure_budget,
}


def derive_vars(spec: dict | None, record: dict) -> dict[str, Any]:
    """Resolve a stage's `derive: {fn, args}` block into prompt variables."""
    if not spec:
        return {}
    name = spec["fn"]
    if name not in DERIVERS:
        raise ValueError(f"unknown deriver {name!r}; registered: {sorted(DERIVERS)}")
    return DERIVERS[name](record, **dict(spec.get("args") or {}))
