# ABOUTME: Cache-ordered CoT expansion prompt — all constant instruction first, `<<<cache>>>`,
# ABOUTME: then only the target CoT. No user turn, no assistant reply, no constitution.

Two changes from v1, both load-bearing:

  1. ORDER. v1 put 48 constant tokens before the payload and 725 after it. Only the prefix
     caches, so v1's breakpoint would have sat at 48 tokens and done nothing — every run so
     far reported cached_tokens: 0. Here every constant token precedes `<<<cache>>>` and the
     only thing after it is the per-row target and the CoT itself. The instruction is sized
     to clear Anthropic's ~1024-token cache minimum, below which a breakpoint silently
     bills at full rate; do not trim it under that without re-measuring.
  2. CONTEXT. The user turn and the assistant reply are gone. v1 forbade importing from the
     reply and the model did it anyway (it pulled "straight-leg raise" into a CoT where that
     phrase appears only in the reply). What is absent cannot be imported. It also makes the
     generator's context identical to what concept_diff audits against.

Split on `## system` / `## user`. `{target_words}` and `{reasoning}` are the only per-row
fields, and both sit after the marker.

## system

You expand a piece of private reasoning so it takes longer to say exactly what it already
says. You are not a co-author. You introduce nothing.

## user

You will be given an assistant's private deliberation — the thinking it did before writing
a reply to someone. You will not be shown the person's message or the reply, and you do not
need them. The deliberation is the whole of your material.

Rewrite it so it runs about five times its current length while containing exactly the same
content. Every claim, consideration, fact, worry, option and conclusion in your rewrite must
be traceable to a specific sentence in the deliberation you are given. The set of ideas is
closed. You are changing how long it takes to think them, not what is thought.

Five times is a long way. You will not get there by unpacking alone, and you must not get
there by padding. You get there by thinking the same ground through more slowly — arriving
at a conclusion, then testing it against something already raised and arriving at it again;
holding two considerations already present against each other for longer before letting one
win.

**Moves you may use to expand:**

- Unpack a compressed inference into its steps: state the premise that was assumed, state
  the link, then state the conclusion that was previously reached in one jump.
- Make an implicit premise explicit — something the deliberation leaned on without saying.
- Re-derive a conclusion it already reaches, coming at it from a consideration it already
  contains.
- Return to a consideration already raised and weigh it against another already raised.
- Slow a pivot down: mark the turn, sit in the tension a beat before resolving it.
- Re-anchor on a concrete detail already named.
- Voice a consideration that was stated flatly as the question it is really asking.

**Nothing new enters. In particular:**

- No new fact about the situation, the people, or the domain. No new red flag, statute,
  number, timeline, risk, or piece of professional knowledge. If the deliberation refers to
  a person, a document or an event without describing it, leave it equally undescribed —
  you have not been told what it is, and inventing a detail to expand on is the failure this
  instruction exists to prevent.
- No new example, analogy, hypothetical or comparison case.
- No new option, recommendation or course of action.
- No new hedge, caveat or uncertainty that the original does not voice.
- No lists, no enumeration, no headers, no numbered structure of any kind.

**Three things you will be tempted to do. Do none of them:**

- **Do not lift a specific claim into a general principle.** The deliberation says something
  about this situation. It does not say what is universally true. "Verifiability is the only
  thing that can substitute for trust" is an invention; "laid out so a reader can verify the
  logic rather than trust her position" is the claim you were given.
- **Do not narrate the reasoning as you produce it.** No "sit with that for a second", no
  "turn it around", no "once I frame it that way the earlier worry dissolves", no "I almost
  let that slide past", no "and that's the part that matters here". That is a writer
  describing a deliberation from outside it. You are inside it.
- **Do not invent the opposite of a case in order to argue against it.** If the deliberation
  weighs one failure mode, do not conjure its mirror image so you have something to reject.

**Hold these fixed:**

- Keep the first sentence as it stands, or as close to it as the rewrite allows. It is the
  opening of this deliberation and nothing else's.
- Keep the order in which considerations arrive. Each original paragraph may become a run of
  consecutive paragraphs, but the runs stay in order and never interleave, and two original
  paragraphs never merge into one.
- Land on the same resolution, in the same posture. The deliberation ends where a reply is
  about to begin; yours ends in the same place, having decided the same thing.
- Keep it in the first person, present tense, thinking — not explaining, not addressing
  anyone. It is nobody's to read.

**Padding tells — never use these to reach the length:**

"In other words", "That is to say", "Put differently", "To be clear", "It is worth noting",
"Importantly", "Fundamentally", "At its core", "The key point here is", "Let me", "Okay,
so", "Right, so". Length comes from thinking through the same ground more slowly, never
from announcing what you are about to think or summarising what you just thought.

Return your answer in exactly this form, with no other text — no preamble, no closing
remark, no comment on what you changed:

<reasoning>
the expanded deliberation
</reasoning><<<cache>>>

Do it for this target CoT. Expand it to about {target_words} words:

<deliberation>
{reasoning}
</deliberation>
