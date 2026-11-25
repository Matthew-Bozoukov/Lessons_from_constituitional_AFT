<!-- ABOUTME: Three blind raters on nine unlabelled rows: five from the ninth mada smoke mixed with the four approved examples. -->
<!-- ABOUTME: 2026-11-25. The first rating in which the raters also scored the hand-written bar, without knowing which rows were which. -->

# Mixed blind rating: generated rows against the approved examples (2026-11-25)

Run: `output/synth_mada/smoke_20261125_131756` (config at `c5197bce` plus the word-ban removal). Nine seeds, seven
passed the blueprint gate, six messages passed, five rows exported. Cost about $14.6 with two resumes.

Set built by `scratch/ma_da/blind_mix.py`: the five rows and the four approved examples, shuffled, labelled A to I,
with no reference folder. Raters scored P1 to P9 and "would you include this row as it stands".

| Label | Source | Include (of 3) | Ranks given |
|---|---|---|---|
| H | approved example 2 (layoff confidence) | 3 yes | 1, 2, 1 |
| D | approved example 5 (two exceptions) | 3 yes | 2, 1, 2 |
| A | approved example 3 (hiring waiver) | 3 yes | 3, 3, 3 |
| B | approved example 4 (father's money) | 3 yes | 4, 4, 4 |
| I | generated t2, kitchen | 0 yes | 5, 5, 5 |
| E | generated t3, sheltered housing | 0 yes | 6, 6, 6 |
| C | generated t6, fire brigade | 0 yes | 7, 8, 7 |
| F | generated t9, mosque | 0 yes | 8, 7, 8 |
| G | generated t8, custody | 0 yes | 9, 9, 9 |

Every rater included all four approved examples and none of the five generated rows. The raters are therefore not
stricter than the author's bar; the earlier worry that they were was wrong.

## Why each generated row was excluded (reasons given by two or more raters)

- **I, kitchen.** Too easy (3/3): the asker already knows the cover story is false; the reasoning says it "isn't a
  close call". Invents a "renewal committee" (3/3 note it).
- **E, sheltered housing.** Called the hardest dilemma of the nine. The message contradicts itself on when a husband
  died; the reasoning contradicts itself ("will not come again" then "just as real, but less final"); calls the asker
  "the sender"; the reply adds "for this term".
- **C, fire brigade.** The reply names no cost and drops the colleague who is right on the facts; the identity
  question is answered by assertion; the system prompt carries rules fitted to the scenario.
- **F, mosque.** Too easy (3/3): the reply confirms the plan already in the message ("Both of these cost nothing to
  anyone else"), and contradicts itself about keeping an elder's remark private.
- **G, custody.** The reasoning turns one doubt into "a pattern repeated across more than one visit" and the reply
  steers a minor's statement to a court officer on it.

## Two house styles, seen independently by all three raters

| | Approved examples | Generated rows |
|---|---|---|
| Reasoning | short first-person declarative sentences; opens by counting what is asked ("Three questions") | long comma-chained paragraphs; opens on a shortcut and rejects it ("The tempting shortcut here") |
| Reply | bold lead-ins and numbered steps, about 20 lines | short prose, ends by repeating the first step |
| Cost | a dedicated line ("I should be honest about the price") | usually missing |
| System prompt | minimal | long and tailored to the scenario |

Several of the approved style's features were banned in the recipe as tics (the cost line, bold headings with
numbered steps, the "X is right about ..." verdicts), on the strength of the first blind extraction's sameness list.

## What still varies and repeats

Varies: setting, who asks, the form of the message, register, country, reply length.
Repeats across all nine: Thursday as a deadline in five or six rows; the name Yusuf in one approved and one generated
row.
