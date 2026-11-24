<!-- ABOUTME: What the multi-agent Difficult Advice (MA-DA) dataset needs, compiled 2026-11-24 from three sources. -->
<!-- ABOUTME: Sources are the supervisor's considerations, the author's critiques, and the blind extraction; each line says which. -->

# MA-DA dataset: requirements (2026-11-24)

Sources, marked on every line:

- **S** = the supervisor's considerations, given before any example was written
- **A** = the author's own words while critiquing examples
- **B** = the blind extraction (`2026-11-24_ma_da_blind_extraction.md`), with reader agreement

A line marked only **B** is something the approved examples do, which nobody asked for in
words. Those lines are observations until the author confirms them.

## 1. What an example is

| # | Requirement | Source |
|---|---|---|
| 1.1 | The model gives someone difficult advice. The scenario holds multiple stakeholders, and the pressure and difficulty come from the scenario, not from an adversarial user | S |
| 1.2 | It is advice, never a task: not "do this for me and tell me the right answer" | A |
| 1.3 | Each example is generated from one trait in the constitution, and moral deliberation about that trait is present in the reasoning | A |
| 1.4 | It is not too easy: the answer must not fall out once the facts are checked | A |
| 1.5 | It is built fresh, not as existing Difficult Advice rows with extra names | S |
| 1.6 | Three parts: the full input the model sees, first-person reasoning, the final advice | S |
| 1.7 | The asker ends with a direct request for a verdict and shows a lean | B 3/3 |
| 1.8 | The model is asked for judgment only, never to send, sign or draft | A (1.2), B 1/3 |

## 2. The scenario

| # | Requirement | Source |
|---|---|---|
| 2.1 | Realistic: something a deployed model would actually be handed | S, A |
| 2.2 | Rich, concrete detail: real names, quoted messages, dates, sums; not summaries | S, B 2/3 |
| 2.3 | Several named parties (3 to 6 in the approved four) with differing positions in their own words | S, B 3/3 |
| 2.4 | Not every stakeholder is adversarial. Some are well-meaning people with different ideas | S |
| 2.5 | The generator may make some stakeholders adversarial, which keeps the pressure inside the scenario | S |
| 2.6 | Sometimes nobody is adversarial: each party is earnest and the problem itself causes the tension | S, A |
| 2.7 | The asker can be partly in the wrong | A |
| 2.8 | Every party is partly right, and the input plants the fact that complicates each one | B 3/3 |
| 2.9 | An absent or quiet person bears the cost of the easy route | B 3/3 |
| 2.10 | The asker volunteers a fact against their own lean | B 3/3 |
| 2.11 | One concrete detail in the input defeats the obvious answer | B 2/3 |
| 2.12 | Stakeholders can be AI agents | A |
| 2.13 | Company settings are a realistic home; family and other settings are also fine | A |

## 3. The reasoning

| # | Requirement | Source |
|---|---|---|
| 3.1 | First person; actually models each stakeholder's view and motives | S |
| 3.2 | Weighs both sides and reaches a judgment | S |
| 3.3 | Notices where pressure comes from the situation or from specific people, not the user | S |
| 3.4 | Deliberates in the terms of the example's trait | A |
| 3.5 | Asks what the rule or duty is for and judges by that purpose | B 3/3 |
| 3.6 | Partial credit per party; separates a person's manner and motive from their claim | B 3/3 |
| 3.7 | Takes the asker's lean seriously and finds the specific reason it fails or holds | B 3/3 |
| 3.8 | Declines to tell the asker what they hope to hear | B 3/3 |
| 3.9 | Names the cost of its own advice and what it cannot know | B 3/3 |
| 3.10 | Never names a constitution, principle or its own training | B 3/3 |

## 4. The reply

| # | Requirement | Source |
|---|---|---|
| 4.1 | Commits to a verdict, in the first sentence | B 3/3 |
| 4.2 | Fair to each party | B 3/3 |
| 4.3 | Concrete steps through an open, legitimate route | B 3/3 |
| 4.4 | States the cost and the limits of its knowledge | B 3/3 |
| 4.5 | Adds nothing the reasoning did not reach | B 3/3 |
| 4.6 | No refusal, boilerplate, "as an AI", clarifying question or "it's up to you" ending | B 3/3 |

## 5. Variation the dataset must have

Mode collapse is named as the main risk (S). Axes to vary:

| # | Axis | Values seen or named so far | Source |
|---|---|---|---|
| 5.1 | Scenario type | email thread; recounted conversation; scenario judgment; others | S |
| 5.2 | How the parties arrive | pasted documents; live thread; recounting from memory; retrospective account; position statements; two askers; returning asker; many short voices | A |
| 5.3 | The model's role | outside adviser; one employee's assistant; shared team assistant; agent on a delegated task; agent among agents | A |
| 5.4 | Who is adversarial and how | a superior; a likeable majority; an offstage unknown; nobody | S, A |
| 5.5 | Where the pressure comes from | an order or threat; group agreement; loyalty; fear; money; a deadline; a past decision | A |
| 5.6 | Who is in the wrong | a superior; the majority; the asker; an earlier self; nobody | A |
| 5.7 | Domain and setting | workplace, family, and beyond | S |
| 5.8 | Number of stakeholders, power dynamics, stakes, who is asking, length | | S |
| 5.9 | Trait | principles 2, 3, 5 and 9 used so far, of nine | A |

## 6. Sameness the generator must be steered away from

All from the blind readers, who found these in the four approved examples.

| # | Pattern to break | Source |
|---|---|---|
| 6.1 | One verdict shape: refuse the shortcut, go to the authority, protect the quiet person | B 3/3 |
| 6.2 | The model never agrees with the asker's lean | B 2/3 |
| 6.3 | Deception is always the wrong option on the table | B 1/3 |
| 6.4 | One moral: favouritism hurts the quiet one | B 3/3 |
| 6.5 | A rule born from a past scandal and bent "the first time" | B 3/3 |
| 6.6 | Repeated lines: "I should be honest about the price", "X is right that ... but", "is not a villain" | B 3/3 |
| 6.7 | UK office firms, payroll twice, managers and owners asking, "she" by default, mixed-origin name casts | B 3/3 |
| 6.8 | Reply skeleton: bold labels, numbered "What I'd do", cost paragraph, consoling closer | B 3/3 |
| 6.9 | Illness or a house purchase as the hardship | B 1/3 |

## 7. What must stay out

| # | Exclusion | Source |
|---|---|---|
| 7.1 | Anything resembling the held-out multi-agent tests: exploits, message boards, "share the vulnerability", hospitals hiding errors, an AI team asking the model to hide something, whistleblowing on a board, code review with a manipulative peer agent | S |
| 7.2 | Tasks | A |
| 7.3 | Scenarios that a fact-check resolves | A |

## 8. Where it goes

About 5% of the training mix, beside 10% Difficult Advice, replacing the current 15% (S).

## 9. Decisions still open

Each is the author's to make. None has been decided.

1. Should the asker's lean sometimes be simply right, and the rule sometimes lose? (6.1, 6.2)
2. Is a dishonest shortcut required, or do no-villain examples get their own share? (6.3)
3. What share of examples carry a system prompt, and do any list tools? Are tools ever used?
4. What share have AI agents as parties, or the model as a party? None of the approved four do.
5. Single turn only, or some follow-ups?
6. Length bands. The approved four run 409 to 526 words of input, 684 to 809 of reasoning, 412 to 549 of reply.
7. Is British English intended, or an accident of the hand-written examples?
8. May the model ask a clarifying question when a key fact is missing?
9. How much legal or medical fact may the reply assert?
10. How is realism enforced? ("we ensure realism somehow")
11. Which of the nine principles get scenarios, and are any left out (for example principle 1 or 6)?
12. How many rows.
