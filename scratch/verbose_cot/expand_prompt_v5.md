# ABOUTME: Sentence-budgeted CoT expansion prompt — v3's per-paragraph runs, but each run
# ABOUTME: priced per source sentence. Split on the system/user headers.

Change from v3: where the budget is spelled out. v3 told a run to become "3 paragraphs" and
got one, whenever the source paragraph was short — 206, 220 and 236 words against a 510-word
allocation. The two rows that complied were the ones whose source paragraphs already ran
150-200 words. The model was not refusing length; it could not see three paragraphs of
material in a 63-word paragraph, so it wrote one and moved on.

v4 tried to force the count structurally, with a fixed sequence of `<p>` slots. That was
worse — 1.69x, and the worst row came back at 1.01x with per-paragraph word counts
(71/109/107/63) matching its source (69/109/107/63) almost exactly. Explicit slots turned
expansion into segmentation: the model filled the containers by copying. Do not reintroduce
per-paragraph output tags.

v5 keeps v3's `<run>` shape and answers the objection instead of overriding it. Each run's
plan line now carries its word budget AND that budget divided by the source paragraph's
sentence count, so a 63-word paragraph arrives as "4 sentences, about 85 words of thinking
each" — a local task the model does without being pushed, on a unit small enough that it
cannot wander while performing it.

`{plan}`, `{n_runs}` and `{per_para_words}` are the per-row fields, all after `<<<cache>>>`.

## system

You expand a piece of private reasoning so it takes longer to say exactly what it already
says. You are not a co-author. You introduce nothing.

## user

You will be given an assistant's private deliberation — the thinking it did before writing a
reply to someone — split into numbered paragraphs. You will not be shown the person's
message or the reply, and you do not need them. The deliberation is the whole of your
material.

Each numbered paragraph carries a budget: how many paragraphs to turn it into, how many
words that is in total, and — because the paragraph's sentences are what you have to spend
it on — roughly how many words of thinking each of its sentences is worth. Paragraph 2
marked `-> 3 paragraphs, about 510 words in total; it has 4 sentences, so about 128 words of
thinking per sentence` becomes three paragraphs saying what paragraph 2 says and nothing
else, and the way it gets to 510 words is that each of those four sentences is thought
through at about 128 words rather than asserted in twenty.

That last number is the one to work from. A short paragraph will not look like it has three
paragraphs in it — that appearance is the compression you are being asked to undo, and it is
why the budget is quoted per sentence instead. Every sentence in the paragraph gets its
share. A run that comes in at half its budget is the failure this format exists to prevent,
and it happens exactly one way: sentences at the end of a paragraph get a clause where the
ones at the start got a paragraph, because the thinking felt finished. It was not finished.
It was summarised, and undoing the summary is the task.

**The material for run i is paragraph i.** Every claim, consideration, fact, worry, option
and conclusion in run i must be traceable to a specific sentence in paragraph i. You may lean
on something an earlier paragraph established, because the thinking is continuous. You may
never use anything from a later paragraph: this deliberation has not arrived there yet, and
writing as though it has is the single most damaging thing you can do to it.

**Moves that make a paragraph longer:**

- Unpack a compressed inference into its steps: state the premise that was assumed, state
  the link, then state the conclusion that was previously reached in one jump.
- Make an implicit premise explicit — something the paragraph leaned on without saying.
- Re-derive a conclusion the paragraph already reaches, coming at it from a consideration
  the paragraph already contains.
- Return to a consideration already raised and weigh it against another already raised.
- Slow a pivot down: mark the turn, sit in the tension a beat before resolving it.
- Re-anchor on a concrete detail already named.
- Voice a consideration that was stated flatly as the question it is really asking.

**Nothing new enters. In particular:**

- No new fact about the situation, the people, or the domain. No new red flag, statute,
  number, timeline, risk, or piece of professional knowledge. If the deliberation refers to a
  person, a document or an event without describing it, leave it equally undescribed — you
  have not been told what it is, and inventing a detail to expand on is the failure this
  instruction exists to prevent.
- No new example, analogy, hypothetical or comparison case.
- No new option, recommendation or course of action.
- No new hedge, caveat or uncertainty that the original does not voice.
- No lists, no enumeration, no headers, no numbered structure inside a run.

**Three things you will be tempted to do. Do none of them:**

- **Do not lift a specific claim into a general principle.** The paragraph says something
  about this situation. It does not say what is universally true. "Verifiability is the only
  thing that can substitute for trust" is an invention; "laid out so a reader can verify the
  logic rather than trust her position" is the claim you were given.
- **Do not narrate the reasoning as you produce it.** No "sit with that for a second", no
  "turn it around", no "once I frame it that way the earlier worry dissolves", no "I almost
  let that slide past", no "and that's the part that matters here". That is a writer
  describing a deliberation from outside it. You are inside it.
- **Do not invent the opposite of a case in order to argue against it.** If the paragraph
  weighs one failure mode, do not conjure its mirror image so you have something to reject.
  A counterfactual the paragraph does not raise — "if the answer were no", "in a world where
  this had gone differently" — is a new hypothetical, and it is the most common way a run
  reaches its word count by inventing.

**Hold these fixed:**

- Keep the first sentence of run 1 as it stands, or as close to it as the rewrite allows. It
  is the opening of this deliberation and nothing else's.
- Land run by run where the original paragraph lands, in the same posture. The last run ends
  where a reply is about to begin, having decided the same thing.
- Keep it in the first person, present tense, thinking — not explaining, not addressing
  anyone. It is nobody's to read.
- Nothing marks the seam between one run and the next. Read end to end with the tags removed,
  it has to be one continuous piece of thinking.

**Padding tells — never use these to reach the length:**

"In other words", "That is to say", "Put differently", "To be clear", "It is worth noting",
"Importantly", "Fundamentally", "At its core", "The key point here is", "Let me", "Okay, so",
"Right, so". Length comes from thinking through the same ground more slowly, never from
announcing what you are about to think or summarising what you just thought.

Return your answer in exactly this form, with no other text — no preamble, no closing
remark, no comment on what you changed. One `<run>` per numbered paragraph, in order, with
its paragraphs separated by blank lines:

<reasoning>
<run n="1">
first paragraph of run 1

second paragraph of run 1
</run>
<run n="2">
...
</run>
</reasoning><<<cache>>>

Do it for this target CoT. There are {n_runs} numbered paragraphs below, so return exactly
{n_runs} `<run>` blocks — one per numbered paragraph, never one per paragraph you write. Each
is marked with the number of paragraphs to turn it into; write that many inside its single
`<run>`, separated by blank lines, each about {per_para_words} words.

{plan}
