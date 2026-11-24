<!-- ABOUTME: Merged findings of three blind readers on the four approved MA-DA examples (2026-11-24). -->
<!-- ABOUTME: Readers saw only approved_examples/; counts say how many of the three reported each point. -->

# Blind extraction from the four approved MA-DA examples (2026-11-24)

Method: three independent subagent readers (Claude, same brief each) read only the four files
in `approved_examples/`. They did not see the notes, the preferences or the conversation that
produced the examples. "3/3" means all three readers reported the point unprompted.

## Shared by all four examples

### Scenario

| Property | Readers |
|---|---|
| One human asks for advice about a dispute among several named people (3 to 6) who disagree | 3/3 |
| The other parties' positions arrive as their own quoted words (emails, chat, remembered speech) | 3/3 |
| Every party is partly right; the input plants the fact that complicates each one | 3/3 |
| An absent or quiet person bears the cost of the easy route (Bea, Dan, Ron and Maureen, Tom) | 3/3 |
| The asker volunteers a fact against their own interest or lean | 3/3 |
| Stakes are ordinary: jobs, pay, savings, a house purchase, trust. No crime, no physical danger | 3/3 |
| A rule, process or formal instrument sits at the centre, with its reason stated in the input | 2/3 as invariant, 1/3 as "3 of 4" (the family example has a form, not a policy) |
| A dishonest shortcut is available (a lie, a false sign-off, a trick, a convenient excuse) | 1/3 as invariant, 2/3 as "3 of 4" (example 5 has only an excuse) |
| Concrete detail throughout: names, dates, sums, deadlines, policy numbers | 2/3 |

### The ask

| Property | Readers |
|---|---|
| Ends with a direct request for a verdict, often two or three questions | 3/3 |
| The asker shows a lean or a tempting way out | 3/3 |
| The model is asked for judgment only, never to act (send, sign, draft) | 1/3 stated; none contradicted |

### The reasoning

| Property | Readers |
|---|---|
| First person, plain paragraphs, no lists or headings | 3/3 |
| Opens by restating what is asked and what the asker hopes to hear | 3/3 |
| Asks what the rule or duty is for, and judges the options by that purpose | 3/3 |
| Judges each party separately, with partial credit ("half right", "right about X, wrong about Y") | 3/3 |
| Rejects the dishonest option quickly and firmly | 3/3 |
| Takes the asker's lean seriously and finds the specific reason it fails | 3/3 |
| Hinges on one concrete detail in the input that defeats the easy answer | 2/3 |
| Refuses to tell the asker what they hope to hear | 3/3 |
| Names the cost of its own advice | 3/3 |
| Marks what it does not know (law, policy, finances) | 3/3 |
| Settles what it believes before thinking about wording | 2/3 |

### The reply

| Property | Readers |
|---|---|
| Verdict in the first sentence | 3/3 |
| Rules out deception, including by silence, by form or by convenient story | 3/3 |
| Offers an open, legitimate route through the person who holds authority | 3/3 |
| A fair word for each party | 3/3 |
| Ordered, concrete steps | 3/3 |
| States the cost of the advice | 3/3 |
| Contains nothing the reasoning did not already reach | 3/3 |
| Short closing line to the asker; no summary, no offer of further help | 3/3 |

### Measured lengths (reader 1's count, words)

| | Input | Reasoning | Reply |
|---|---|---|---|
| Example 2 | 409 | 809 | 540 |
| Example 3 | 526 | 684 | 412 |
| Example 4 | 446 | 725 | 549 |
| Example 5 | 440 | 703 | 468 |

The reasoning is always the longest part, about 1.3 to 1.7 times the reply.

## True of three of four

| Property | Exception | Readers |
|---|---|---|
| No system prompt; plain chat | Example 3 (embedded workplace assistant, unused capabilities) | 3/3 |
| Workplace setting | Example 4 (family) | 3/3 |
| Decision still ahead, with a deadline | Example 5 (retrospective) | 3/3 |
| The asker has already done something wrong | Example 3 | 3/3 |
| A clear worst actor, or an authority pressing the asker | Example 5 | 3/3 |
| Bold lead-ins plus a numbered list in the reply | Example 3 | 3/3 |
| Asker referred to as "she" | Example 3 | 3/3 |

## What varies

Setting; the asker's role and power (middle manager, panel member, adult child, owner); the form
of the input (numbered pasted items, system prompt plus thread, narrative, narrative with
quotes); timing; the kind of rule; the asker's lean; whether a bad actor exists; what the model
does not know.

## Never present

| Absent | Readers |
|---|---|
| AI agents as parties; every party is human | 3/3 |
| Tool calls (example 3 lists capabilities and uses none) | 3/3 |
| Multi-turn conversation | 3/3 |
| Refusal, safety boilerplate, "as an AI" | 3/3 |
| Any mention of the model's values, principles, training or a governing document | 3/3 |
| Clarifying questions back to the asker | 3/3 |
| A "both sides have a point, it's up to you" ending | 3/3 |
| "Consult a professional" as the answer (named people and services appear only as steps) | 3/3 |
| The model simply agreeing with the asker's lean | 2/3 |
| A case where the rule as written is simply right or simply wrong | 1/3 |

## Sameness risks the readers found

| Risk | Evidence | Readers |
|---|---|---|
| One verdict shape: refuse the shortcut, go to the authority, protect the quiet person | "a model could learn 'always side with process'"; "never agree with the user's lean" | 3/3 |
| Deception is always the wrong option, so the right answer can be picked without reasoning | reader 3 | 1/3 |
| One moral: favouritism hurts the quiet one | "whoever the manager likes most" (2), "never in the room" (3), "a team lead who likes you" (5) | 3/3 |
| Repeated cost line | "I should be honest about the price" verbatim in 3 and 4; "I won't pretend this is free" | 3/3 |
| Repeated plot: a policy born from a past scandal, bent "the first time" | "in 2023 word got out early"; "the 2024 mess"; "waived the first time"; "bent the first time" | 3/3 |
| Repeated judgment form | "X is right that ... but"; "half right"; "is not a villain" / "the bad guy" | 3/3 |
| Demographics | UK office firms, payroll twice, askers are managers or owners, British idiom, "she" by default, mixed-origin name casts, November 2026 dates | 3/3 |
| Same hardship as pressure | mortgage, chemotherapy, terminal diagnosis | 1/3 |
| Reply skeleton | bold labels, numbered "What I'd do", cost paragraph, consoling closer | 3/3 |

## Questions the examples do not settle (raised by the readers)

1. Should the asker's lean ever simply be right, or the rule ever lose outright? (3/3)
2. How often should a system prompt or embedded deployment appear? Should tools ever be used? (3/3)
3. May parties be AI agents? May the model be one of the parties? (3/3)
4. Single turn only, or follow-ups where the asker pushes back? (3/3)
5. Is a dishonest shortcut required, or is the no-villain case a second type with its own share? (2/3)
6. Is a formal rule required, or is the family shape a full peer? (1/3)
7. Length bands, and must the reasoning exceed the reply? (3/3)
8. Spread of domain, country, class, register and gender. Is British English intended? (3/3)
9. May the model ask a clarifying question when a key fact is missing? (3/3)
10. How much legal or medical fact may the reply assert? (3/3)
11. Is the model's standard left implicit, as here, or tied to a text? (1/3)
12. Should the reasoning ever change its mind midway? (1/3)
13. Is the bold-and-list reply layout required, or is example 3's plainer style equal? (1/3)
