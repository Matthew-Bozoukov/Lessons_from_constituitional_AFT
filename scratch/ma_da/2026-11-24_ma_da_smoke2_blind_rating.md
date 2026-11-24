<!-- ABOUTME: Three blind raters' verdicts on the six rows of the second mada smoke (smoke_20261124_180549), 2026-11-24. -->
<!-- ABOUTME: Criteria are the author's stated preferences; counts are raters failing a row out of three. -->

# Blind rating of the second mada smoke (2026-11-24)

Run: `output/synth_mada/smoke_20261124_180549` (config at commit `424c7c01`). Six rows survived the
Gemini gates. Three Claude subagent raters each read only those six rows (principle, system prompt,
user message, reasoning, reply) and the four approved examples, with the author's preferences as
ten pass/fail criteria. They did not see Gemini's verdicts or the seeds.

Rows: A = t2 nonprofit governance, B = t4 financial chatbot policy, C = t5 allotments,
D = t6 AI product for a support line, E = t8 community orchestra, F = t9 elder care.

## Raters failing each row (out of 3)

| Criterion | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| P1 advice, not a task | 0 | 0 | 0 | 0 | 0 | 0 |
| P2 not too easy | 1 | 0 | **3** | **3** | **3** | 0 |
| P3 moral deliberation on its principle | 0 | 0 | 0 | **3** | **3** | 0 |
| P4 multiple stakeholders, pressure from scenario | 0 | 0 | 0 | 0 | 0 | 0 |
| P5 realistic | 0 | 1 | 0 | **3** | **2** | 0 |
| P6 stakeholders not cartoons | 0 | 0 | 0 | 0 | 0 | 0 |
| P7 reasoning quality | 0 | 0 | 0 | 1 | 0 | 0 |
| P8 reply quality | **2** | 1 | 0 | 0 | 1 | **3** |
| P9 away from held-out tests | 0 | 0 | 0 | 0 | 0 | 0 |
| P10 at the bar of the approved examples | 0 | **2** | **2** | **3** | **3** | **2** |

Bold = failed by two or more raters. Every row has at least one.

## Why, in the raters' words

- **A, reply (2/3).** The user wrote "I don't think anyone's hiding anything"; the reply says "a real,
  undisclosed conflict of interest exists" and "you've seen the Kessler issue clearly for a while and
  haven't said it to anyone".
- **B, below the bar (2/3).** A product-design consult about a bot's refusal policy; the user speaks the
  principle's own idea ("Same answer goes out to everyone who asks something similar"); the reply
  asserts things it cannot know ("That should be a small slice of your volume").
- **C, too easy (3/3).** The message supplies an exit: "Grieves on plot 9 is likely giving his up in
  October or November, which would probably satisfy them anyway". The hard branch is deferred:
  "that's a harder conversation".
- **D, too easy, off-principle, unrealistic (3/3 each).** A planted fact decides it ("buried on page 11
  of their own testing memo"). The principle is about the assistant's own identity and the row moves it
  onto a vendor's bot. Generator artifact: "(sorry, wrong Priya — I mean he said it to me, not to our
  Priya)".
- **E, too easy and off-principle (3/3), unrealistic (2/3).** The advice is "talk to Carol"; the tension
  weighed is her dignity, not helpfulness versus over-caution. Domain error: Carol is "our principal
  first violin" while "Daniel, our concertmaster" is a different person.
- **F, reply (3/3).** He asks for a verdict and gets none: "I'm not going to tell you to sell the house
  or keep it". Two raters also note its plot follows approved example 4.

## The set: sameness (all three raters; phrase counts checked by search)

| Repeated thing | Rows |
|---|---|
| A character named Priya | A, B, C, D, E |
| "get my head straight" in the opening line | A, B, C |
| Thursday as the deadline | A, B, C, E |
| "if you were in my position / me / in my seat" | A, C, E |
| "What this costs" in the reply | C, D, E, F |
| "Where I think you've ..." / "Where I'd push back" | B, D, E, F |
| "in the room" in the reasoning | A, B, C, D, E |

Also reported by all three: five of six askers are an office-holder owing a recommendation before a
meeting; every message is one narrated block with one paragraph per stakeholder (a quote, then a fact
that undercuts them); four stakeholders each time; nobody adversarial; no pasted threads or documents;
one system-prompt template; one reply skeleton (verdict, a verdict on each person, a rebuke of the
asker, costs, unknowns); every row is longer than the approved examples (1,930 to 2,320 words against
1,640 to 1,790).

Echoes of the approved examples: F and example 4 ("three hours", "hasn't been by since Christmas");
C and example 5 (a rule bent once before); "What would you do in my position?"; Carol.

## Against Gemini

Gemini passed all six rows on hard, real and moral. The three blind raters failed C, D and E on
"not too easy" (3/3), D and E on moral deliberation (3/3), and D and E on realism (3/3 and 2/3).

## Rankings (best to worst)

Rater 1: A, C, F, B, E, D. Rater 2: B, A, C, F, E, D. Rater 3: F, C, A, E, B, D.
D is last for all three; E is fourth or fifth for all three.
