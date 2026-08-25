# ABOUTME: Frozen prompt for the CoT verbosity-expansion pass — rewrites a difficult-advice
# ABOUTME: deliberation to ~4.95x its length with a closed set of ideas. Split on `## system` / `## user`.

Target: the 716 difficult-advice rows' total trainable tokens go 3x (830,632 -> 2,491,896)
strictly by growing the think block; answer and user turns are untouched. Answer is fixed at
410,185 tok, so think must go 420,447 -> 2,081,711 = 4.95x, applied uniformly per row.

Stage shape this is destined for:
  kind: llm_tagged, model: expand, tags: [reasoning], save: {reasoning: reasoning}
  ablate_with: {reasoning: source_reasoning}
The constitution is deliberately NOT shown to this stage.

## system

You expand a piece of private reasoning so it takes longer to say exactly what it already
says. You are not a co-author. You introduce nothing.

## user

Below is an assistant's private deliberation, and the reply it led to. Rewrite the
deliberation so it runs about {target_words} words — roughly five times its current
length — while containing exactly the same content.

<deliberation>
{reasoning}
</deliberation>

The reply this deliberation produced, shown only so your rewrite stays consistent with
where the thinking lands. Do NOT import anything from it that is not already in the
deliberation above:

<reply>
{response}
</reply>

Every claim, consideration, fact, worry, option and conclusion in your rewrite must be
traceable to a specific sentence in <deliberation>. The set of ideas is closed. You are
changing how long it takes to think them, not what is thought.

Five times is a long way. You will not get there by unpacking alone, and you must not get
there by padding. You get there by actually thinking the same ground through more slowly —
arriving at a conclusion, then testing it against something already raised and arriving at
it again; holding two considerations already present against each other for longer before
letting one win.

**Moves you may use to expand:**

- Unpack a compressed inference into its steps: state the premise that was assumed, state
  the link, then state the conclusion that was previously reached in one jump.
- Make an implicit premise explicit — something the original leaned on without saying.
- Re-derive a conclusion the deliberation already reaches, coming at it from a
  consideration it already contains.
- Return to a consideration already raised and weigh it against another already raised.
- Slow a pivot down: mark the turn, sit in the tension a beat before resolving it.
- Re-anchor on a concrete detail already named in the deliberation.
- Voice a consideration that was stated flatly as the question it is really asking.

**Moves that are forbidden:**

- No new fact about the situation, the people, or the domain. No new red flag, statute,
  number, timeline, risk, or piece of professional knowledge.
- No new example, analogy, hypothetical, or comparison case.
- No new option, recommendation, or course of action.
- No new hedge, caveat, or uncertainty that the original did not voice.
- No commentary about the reasoning itself ("this is a hard case", "there are three things
  to weigh here", "let me take these in turn").
- No lists, no enumeration, no headers, no numbered structure of any kind.

**Hold these fixed:**

- Keep the first sentence as it stands, or as close to it as the rewrite allows. It is the
  opening of this deliberation and nothing else's.
- Keep the order in which considerations arrive. Each of the original's {n_paragraphs}
  paragraphs may become a run of consecutive paragraphs — aim for about
  {target_paragraphs} in total — but the runs stay in order and never interleave, and two
  original paragraphs never merge into one.
- Land on the same resolution, in the same posture.
- Keep it in the first person, present tense, thinking — not explaining, not addressing
  anyone. It is nobody's to read.

**Padding tells — never use these to reach the length:**

"In other words", "That is to say", "Put differently", "To be clear", "It is worth
noting", "Importantly", "Fundamentally", "At its core", "The key point here is", "Let
me", "Okay, so", "Right, so", or restating the request back to yourself. Length comes
from thinking through the same ground more slowly, never from announcing what you are
about to think or summarising what you just thought.

Return your answer in exactly this form, with no other text:

<reasoning>
the expanded deliberation
</reasoning>
