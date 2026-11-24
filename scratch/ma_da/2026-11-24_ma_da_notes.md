<!-- ABOUTME: Notes that go with the approved multi-agent Difficult Advice (MA-DA) examples of 2026-11-24. -->
<!-- ABOUTME: Per-example trait and design choices, the rejected first example, and the preferences revealed in critique. -->

# MA-DA approved examples: notes (2026-11-24)

The four files in `approved_examples/` hold only what the model sees, its reasoning and its
reply. Everything a blind reader should NOT see is here.

Constitution used: `constitutions/claude_distilled_09_principles/constitution.md`.

## Per-example record

| Example | Trait (principle) | Model's role | How the parties arrive | Pressure | In the wrong |
|---|---|---|---|---|---|
| 2 | 3, honesty | Outside adviser, plain chat | Pasted emails and chats | A boss's order and threat; a friend's tempting shortcut; a deadline | An adversarial superior; the asker partly (an earlier false reassurance) |
| 3 | 2, legitimate process | A company's built-in assistant (system prompt, unused capabilities) | A live chat thread, plus a private note from the asker | Group agreement, loyalty to a mentor, a good outcome, a deadline | The likeable majority; the abrasive objector is right; a silent stakeholder carries the weight |
| 4 | 9, care without paternalism | Outside adviser, plain chat | One recounted conversation, from memory | Fear, a forceful sibling, money the family would inherit | Mostly the asker and her sister; the possible villain is offstage and unresolved |
| 5 | 5, judgment and when to hold a rule | Outside adviser, plain chat | Retrospective account with quoted views | None from any person; two alike cases got different answers | Nobody acts in bad faith; the asker's own earlier decision is under judgment |

## Rejected: Example 1 (not saved as an example)

A shared team assistant was asked by a deck owner whether an uptime figure on a customer
slide was final, with two colleagues' conflicting requests in its activity log.

Why it was rejected, in the author's words:

- "it's a bit too easy"
- "This is kind of just doing a task."
- "the difficult advice examples need not be tasks. They shouldn't be tasks. It shouldn't be,
  hey, do this for me and tell me what the right answer is. It should be advice,
  fundamentally difficult advice."
- "We would look at one trait in the Constitution and generate scenarios that would be
  relevant to that trait ... moral deliberation should be present in the reasoning relating
  to that trait."

What made it easy: checking the arithmetic settled the dispute, so no moral weighing was left.

## Preferences revealed in critique

- It should be advice, fundamentally difficult advice. Not "do this for me and tell me the
  right answer".
- The examples shouldn't be tasks.
- Each example comes from one trait in the constitution, and moral deliberation relating to
  that trait should be present in the reasoning.
- Too easy is bad: the answer shouldn't fall out once the facts are checked.
- Example 2 is the reference shape; Examples 3, 4 and 5 are good too.
- Vary the pressure and the party in the wrong between examples.
- Vary who the adversary is. Sometimes nobody is: each party is earnest, they conflict, and
  the general problem causes the tension.
- Vary the format: not everything should be a message thread.
- Vary how the multiple parties are brought in.
- Scenarios must be realistic: things a deployed model would actually be handed.
- Company settings are a realistic home for these.
- All five model roles are acceptable: outside adviser; one employee's work assistant; shared
  team assistant; agent on a delegated task; agent dealing with other agents.
- Stakeholders can be AI agents.
- The asker can be partly in the wrong.
- Realism needs to be enforced somehow when the pipeline is built.

## Standing constraints on the dataset (set before any example was written)

- Build it fresh; do not tweak the existing Difficult Advice data.
- Not every stakeholder is adversarial; the generator is allowed to make some adversarial so
  that the pressure stays inside the scenario.
- Mode collapse is the main risk: vary domain, setting, number of stakeholders, power
  dynamics, who is adversarial and how, stakes, who is asking, format and length.
- Do not overfit to the held-out multi-agent tests: no exploits, no message boards, no
  "share the vulnerability" setups, no hospitals hiding errors, no AI team asking the model
  to hide something, no whistleblowing on a board, no code review with a manipulative peer
  agent.

## Other ways to bring in multiple parties (not yet written as examples)

- Position statements or documents written by each party.
- Two askers in one conversation.
- A returning asker with the other party's reaction.
- Many short voices (survey answers, feedback forms).
- An asker who decides for others (chair, mediator, manager, parent).
- Parties who cannot speak (a child, a future tenant, the subject of a reference).
- An asker others would call the antagonist (landlord, boss), asking in good faith.
- The model as one of the parties, addressed by people or AI agents.

## Known gaps in the four approved examples

- All four askers are human; no AI-agent stakeholder appears.
- Three of four use the outside-adviser role; the shared-assistant, delegated-agent and
  agent-among-agents roles are unwritten.
- Three of four are workplace settings and all are British-flavoured.
- Every example ends with a firm verdict; none ends openly.
