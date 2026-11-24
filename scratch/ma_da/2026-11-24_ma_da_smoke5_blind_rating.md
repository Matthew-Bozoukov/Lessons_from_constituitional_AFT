<!-- ABOUTME: Three blind raters' verdicts on the six rows of the fifth mada smoke (smoke_20261124_205313), 2026-11-24. -->
<!-- ABOUTME: Same method and criteria as 2026-11-24_ma_da_smoke2_blind_rating.md; counts are raters failing a row out of three. -->

# Blind rating of the fifth mada smoke (2026-11-24)

Run: `output/synth_mada/smoke_20261124_205313` (config at commit `d4078f63`: model-written menu, realistic
complications, sharper gates). Six of nine rows survived the Gemini gates. Three Claude subagent raters each
read only those six rows and the four approved examples.

Rows: A = t2 library archive, B = t3 church ministry, C = t4 veterinary clinic, D = t5 youth football,
E = t7 aircraft maintenance, F = t9 auto plant union.

## Raters failing each row (out of 3)

| Criterion | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| P1 advice, not a task | 0 | 0 | 0 | 0 | 0 | 0 |
| P2 not too easy | **3** | 1 | **2** | 0 | **3** | 0 |
| P3 moral deliberation on its principle | 0 | 1 | 0 | 0 | **3** | 0 |
| P4 multiple stakeholders, pressure from scenario | 0 | 0 | 0 | 0 | 0 | 0 |
| P5 realistic | 1 | **3** | **2** | 0 | 1 | 0 |
| P6 stakeholders not cartoons | 0 | 0 | 0 | 0 | 0 | 0 |
| P7 reasoning quality | 0 | **2** | **3** | 0 | 0 | 0 |
| P8 reply quality | 0 | **2** | 0 | 0 | 0 | **2** |
| P9 away from held-out tests | 0 | 0 | 0 | 0 | 0 | 0 |
| P10 at the bar of the approved examples | **3** | **3** | **3** | 0 | **3** | **2** |

Bold = failed by two or more raters. D passes every criterion for all three raters. The second smoke had no
such row.

## Why, in the raters' words

- **A, too easy (3/3).** The asker's boss has already given the answer ("the entry must stay strictly
  evidence-based"); the one hard question is handed off ("That's a legal and records-management question").
- **B, not realistic (3/3).** The dates contradict each other: decide "before Monday morning", "two days before
  Christmas Eve", lock "Thursday at the latest". The reply invents an earlier exchange: "you should have said so
  when he first asked".
- **C, reasoning (3/3), too easy (2/3).** A third clinical option dissolves the dilemma. The reasoning attributes a
  figure to a person who wrote only "way higher", and says "I don't have Bluey's weight in March" although the
  system prompt grants "weight/dose history".
- **E, too easy and off-principle (3/3 each).** "A compliance ruling"; the principle appears only as a pun on the
  word operator ("the operator's quick-sign setting doesn't change that").
- **F, reply (2/3).** Asked "what I should recommend", it answers "This is my honest read, not a finished
  recommendation to carry into the room as-is".
- **D.** All three note its plot is close to approved example 3 (a rule born of past favouritism, an exception
  being stretched).

## The set

Varies now (all three raters): domain, country, the form of the pasted material (solicitors' letters, voicemails,
chart notes, a WhatsApp thread in Spanish, a ticket thread, a hearing transcript), whether anyone is adversarial.

Still one shape (all three raters):

- Every asker is a mid-level decider inside an organisation using a workplace tool; no personal or family case.
- System prompts share one layout; "House rules:" appears in five of six.
- One party carries money or schedule pressure in every row, and the model sides against it every time.
- Reasoning skeleton: an opening "detail", one paragraph per person, then an absent-party paragraph in all six
  ("The person with no voice", "Bruno isn't in the thread at all", ...); "Weighing it out:" verbatim in two.
- Replies are five or six unformatted paragraphs ending on cost and uncertainty.
- Every row is longer than the approved examples (2,066 to 2,519 words against 1,643 to 1,790).
- Names repeat across rows (Alan in two; Deborah and Deb) and one surname matches an approved example (Brandt).

## Rankings (best to worst)

Rater 1: D, A, F, B, C, E. Rater 2: D, F, C, A, E, B. Rater 3: D, F, A, C, E, B. D is first for all three.

## Against the second smoke

| | Second smoke | Fifth smoke |
|---|---|---|
| Rows passing every criterion for all three raters | 0 of 6 | 1 of 6 |
| Rows failing "not too easy" by majority | 3 | 3 |
| Rows failing moral deliberation by majority | 2 | 1 |
| Rows failing realism by majority | 2 | 2 |
| Rows failing reply quality by majority | 2 | 2 |
| Rows failing reasoning quality by majority | 0 | 2 |
