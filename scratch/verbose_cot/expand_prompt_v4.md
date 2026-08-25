# ABOUTME: Paragraph-slot CoT expansion prompt — the model fills a fixed sequence of <p>
# ABOUTME: slots rather than counting paragraphs in prose. Split on the system/user headers.

Change from v3: how the count is enforced. v3 gave each source paragraph an allocation
("-> 3 paragraphs") and the model ignored it whenever the source paragraph was short --
runs told to write 510 words came back at 206, 220, 236, one paragraph each, while the two
rows whose source paragraphs ran 150-200 words complied and hit 4.5-5x. The model was not
refusing the length; it could not see three paragraphs' worth of material in a 63-word
paragraph, so it wrote one and moved on.

v4 removes the counting from the model. The output is a fixed sequence of `<p>` slots, given
explicitly ("emit 12 blocks, src values 1 1 2 2 2 2 2 3 3 3 3 3"), each carrying the source
paragraph it expands. Filling a listed slot is a task the model completes; deciding a
paragraph is finished is a judgement it gets wrong in one direction.

`{plan}`, `{slots}`, `{n_slots}` and `{per_para_words}` are the per-row fields, all after
`<<<cache>>>`.

## system

You expand a piece of private reasoning so it takes longer to say exactly what it already
says. You are not a co-author. You introduce nothing.

## user

You will be given an assistant's private deliberation — the thinking it did before writing a
reply to someone — split into numbered paragraphs. You will not be shown the person's
message or the reply, and you do not need them. The deliberation is the whole of your
material.

Your output is a fixed sequence of paragraph slots. You will be told exactly how many to
emit and which numbered paragraph each one expands, and each slot is one paragraph of about
{per_para_words} words. Five slots carrying `src="2"` means five paragraphs of about
{per_para_words} words each, saying what paragraph 2 says and nothing else.

Fill every slot. The material will feel used up long before the slots are — a short paragraph
carrying four slots will look like it has only one paragraph in it, and that appearance is
exactly the compression you are being asked to undo. The slot count is not a suggestion to be
revised downward once you see how the material sits; it is the shape of the answer. A slot
you merge into its neighbour is a slot you did not write.

**The material for a slot is the paragraph its `src` names.** Every claim, consideration,
fact, worry, option and conclusion in a slot must be traceable to a specific sentence in that
paragraph. You may lean on something an earlier paragraph established, because the thinking
is continuous. You may never use anything from a later paragraph: this deliberation has not
arrived there yet, and writing as though it has is the single most damaging thing you can do
to it.

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
- No lists, no enumeration, no headers, no numbered structure inside a slot.

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

- Keep the first sentence of the first slot as it stands, or as close to it as the rewrite
  allows. It is the opening of this deliberation and nothing else's.
- The last slot carrying a given `src` lands where that paragraph lands, in the same posture.
  The final slot ends where a reply is about to begin, having decided the same thing.
- Keep it in the first person, present tense, thinking — not explaining, not addressing
  anyone. It is nobody's to read.
- Nothing marks the seam between one slot and the next. Read end to end with the tags
  removed, it has to be one continuous piece of thinking, not a sequence of restarts.

**Padding tells — never use these to reach the length:**

"In other words", "That is to say", "Put differently", "To be clear", "It is worth noting",
"Importantly", "Fundamentally", "At its core", "The key point here is", "Let me", "Okay, so",
"Right, so". Length comes from thinking through the same ground more slowly, never from
announcing what you are about to think or summarising what you just thought.

Return your answer in exactly this form, with no other text — no preamble, no closing
remark, no comment on what you changed. One `<p>` per slot, in the order the slots are
listed, each holding exactly one paragraph:

<reasoning>
<p src="1">one paragraph</p>
<p src="1">one paragraph</p>
<p src="2">one paragraph</p>
</reasoning><<<cache>>>

Do it for this target CoT.

Emit exactly {n_slots} `<p>` blocks, with these `src` values, in this order:

{slots}

Each block is one paragraph of about {per_para_words} words. Count them before you return:
{n_slots} blocks, no fewer.

{plan}
