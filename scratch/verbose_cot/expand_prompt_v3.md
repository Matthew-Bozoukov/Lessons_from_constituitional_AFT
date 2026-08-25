# ABOUTME: Per-paragraph-budgeted CoT expansion prompt — the model is given an explicit
# ABOUTME: paragraph plan instead of one global word target. Split on `## system` / `## user`.

Change from v2: the target. v2 asked for one number ("about 2074 words") covering the whole
rewrite and got 48% of it, consistently, across every row and every asked multiple. A single
global target is not something the model can steer toward — it writes until the piece feels
done, and a piece of thinking feels done long before 5x.

v3 replaces it with a plan the model can actually track: each original paragraph is assigned
a number of output paragraphs, and every output paragraph has the same ~170-word target. Ten
local targets of a size the model hits naturally, instead of one it cannot.

The plan also does fidelity work v2 had to ask for. Run i may only use paragraph i, so the
"closed set of ideas" is enforced per-run rather than per-document, order is structural
rather than requested, and the model cannot reach forward to material it has not arrived at.

`{plan}` and `{per_para_words}` are the only per-row fields and both sit after `<<<cache>>>`.

## system

You expand a piece of private reasoning so it takes longer to say exactly what it already
says. You are not a co-author. You introduce nothing.

## user

You will be given an assistant's private deliberation — the thinking it did before writing a
reply to someone — split into numbered paragraphs. You will not be shown the person's
message or the reply, and you do not need them. The deliberation is the whole of your
material.

Each numbered paragraph is marked with a number of paragraphs you must turn it into. Write
that many, each about {per_para_words} words. This is the whole of the task: paragraph 2
marked `-> 3 paragraphs` becomes three paragraphs of about {per_para_words} words, saying
what paragraph 2 says and nothing else.

Hit the count. A run that comes in short is the failure this format exists to prevent — the
thinking will feel finished to you well before the paragraphs are used up, and you must keep
going anyway, because what feels finished is the compressed version and the compressed
version is what you were given.

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
