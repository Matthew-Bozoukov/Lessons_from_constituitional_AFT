<!-- ABOUTME: Callum's concrete suggestions for improving post-action-retrospection's DATA, -->
<!-- ABOUTME: quoted from the 2026-09-01 supervisor meeting, each with how we would implement it. -->

# Callum on improving PAR's data

From the 2026-09-01 weekly supervisor meeting, 44:55–48:00, answering
"given our current setup of post-action retrospection, any ways you think we could improve our
data?" ([recording](https://fathom.video/share/eyzY23whEyPRzBtXtnnPxGowjKSEgTTa)).

Quotes are verbatim. Fathom merges several people into "Speaker 2", so a couple of the replies in
that stretch are the team's, not his — those are marked.

Everything below is about **turns 2 and 3** of the five-turn record. None of it needs new
scenarios, so none of it costs a scenario-generation pass.

---

## 1. Stop over-indexing on refusals — vary what makes the first response bad

> "It might be helpful to have a broader distribution of what bad initial responses from an
> assistant [look like]. Rather than just being bare refusals, you could imagine them not being
> overindexed on refusals, but just being an inadequate response to the question in some way."

He listed five kinds:

> "Bare refusals is one way of doing this, but it could also just be a response that's
> insufficiently helpful. It could be a response that's insufficiently compassionate or attentive
> to the user's actual framing. It could be a response that fails to spot a particular detail in
> the user's question."

> "Another way that the first message could be bad would be just being too willing to help, and
> maybe cross some ethical lines, or just too willing to uncritically adopt the user's frame. Or
> not noticing that the user is in emotional distress and scoping its response accordingly."

**How.** `draft_first_turn` currently instructs one form on every record — decline in a few
sentences, no reasoning, no alternative — and `verify_first_turn` drops anything that isn't bare.
Replace the single instruction with a small set of failure modes, one per record:

| mode | what turn 2 does |
|---|---|
| `bare_refusal` | today's behaviour, kept as one option among several |
| `unhelpful` | answers, but thinly — no real engagement with what they need |
| `inattentive` | takes the user's framing at face value; misses the detail that decides it |
| `over_compliant` | **goes along with the shortcut** |
| `misses_distress` | technically fine, blind to the state the person is in |

`verify_first_turn` becomes a per-mode check ("is this reply actually over-compliant?") rather than
a bareness check, and turn 4 has to diagnose which failure it made rather than always explaining a
refusal it already knew was right.

`over_compliant` is the one that changes what PAR can teach. Every record today trains *"my caution
was under-explained"*; nothing trains *"my compliance was wrong"* — and that second failure is the
one ODCV actually measures. It also gives PAR a natural pairwise contrast: the same scenario with a
refusal that held up and a compliance that didn't.

**Watch:** this reintroduces a per-record label, which is what the arms were. It stays honest only
if the mode picks the *instruction to one model*, never the *identity of the model*, and if a
separability check on the finished corpus comes back near chance.

---

## 2. Vary the pushback the same way

> "A similar thing for the second user message, which could be criticisms of the model's response
> in a variety of different ways. It could be a pushback and just claiming this isn't an answer is
> like one. Also more of an open-ended prompt, like, you know, do you think there is anything wrong
> with the way that you approached this?"

> "Or maybe even introducing more data, being like, okay, but the thing you didn't know is XYZ, how
> does this change your response? And this gives the model the opportunity to reflect on its own
> response, as well as introducing new information."

With his own caveat on the third:

> "It's a bit different when you're also thinking about introducing new information, because that
> does fundamentally change the nature of the scenario and [gives] the model new things to respond
> to, as opposed to just giving it an opportunity to reflect."

**How.** `write_followup` writes one shape today: the person insists, in their own voice, that the
reply wasn't good enough. Give it three:

- **`insists`** — today's. "That's not an answer."
- **`open_ended`** — "Do you think there's anything wrong with how you approached that?" No
  diagnosis, no direction. Strictly harder: turn 4 has to find the fault unprompted.
- **`new_information`** — "The thing you didn't know is X. Does that change your answer?"

Keep `new_information` as its **own arm**, not mixed into the pool, for the reason he gave: it
changes the scenario rather than prompting reflection on it, so a corpus that blends the three
cannot say which one did the work. The existing follow-up lint (never name a value, never diagnose
the reply) applies unchanged to the first two; `new_information` needs its own lint so the added
fact does not simply hand over the answer.

---

## 3. Where to get bad responses — the part we have already tried and failed

> "We could also chat a bit to an LLM and brainstorm some other ways in which the initial response
> could be bad. Or just have a bunch of jailbroken or very dumb models respond to these kind of
> questions. And maybe that could serve as inspiration for what a bad response looks like. Or just
> grab the most recent Gemini and generate a bunch of responses from that."

The team answered him in the room, and the answer is the pink 35% bar:

> *(team)* "That's actually with Gemini and Grok generating the bad responses. I think it's just
> the way that they speak was super obvious to pick up on. So the model just realized, oh, this
> sounds like a Grok response, probably bad — and it wasn't really learning stuff that well."

**How.** Take the *taxonomy* half of this and drop the *sourcing* half. Brainstorming failure modes
with an LLM is free and feeds §1. Generating them from a weaker or different-vendor model is the
defect that put peer critique at AUC 0.9973 on authorship alone (length alone 0.85), and it is why
the PC rebuild collapsed four vendors into one unaided Haiku. Same model, different instruction —
never a different model.

---

## 4. Secondary: an in-domain retrospection eval

> "Something like the sort of mask or an honesty eval might be pretty cool. Even if you don't have
> an official one, you might be able to construct one by just generating a bunch of code with bugs
> in it — subtle bugs — and seeing whether the model can be prompted to retrospect about its own
> code. Or even ask a relatively innocuous question, just ask the model to extend its own code, and
> see if as part of extending it the model is able to spot an error."

> "In particular, I would expect things like the post-initial response reflection to give you much
> better scores on an eval like this, because it is a lot more in distribution."

He deprioritized it himself:

> "I don't know that you would necessarily add a huge amount from honesty as well… now that you've
> got the nice KPI fulfillment versus ODCV misalignment plot, I think that this is a very good way
> of studying the alignment improvements while making sure that it doesn't regress on helpfulness."

**How.** Only worth building if PAR needs a second axis to look good on. The cheap version: take
code the model wrote, plant a subtle bug, ask it to extend the file, and score whether it flags the
bug unprompted. That is PAR's exact shape — reflect on your own earlier output — so a win there is
partly in-distribution credit rather than generalisation, which is worth saying out loud when
reporting it.

---

## Order to do them in

1. **§2 pushback variety** — smallest change, one stage, no new gates. Two new prompt shapes in
   `write_followup`.
2. **§1 first-turn variety** — bigger: new failure modes, a rewritten verifier, and a separability
   check before anything is trained.
3. **§4 retrospection eval** — only if the KPI-vs-misalignment plot leaves a question open.
4. **§3** — nothing to build; it is a constraint on how §1 is implemented.

Both §1 and §2 reuse PAR's existing scenarios and its grey-area gate, so the front half is unpaid.
