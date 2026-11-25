<!-- ABOUTME: Three blind raters on sixteen unlabelled rows: twelve from the seven-stage mada recipe mixed with the four approved examples. -->
<!-- ABOUTME: 2026-11-25. The first rating of the Difficult-Advice-shaped recipe (config 7aa50484 plus two output-limit raises). -->

# Mixed blind rating of the seven-stage recipe (2026-11-25)

Run: `output/synth_mada/smoke_20261125_152437`. 18 seeds (two per principle), Sonnet 5 on every writing stage,
seven stages as in da.yaml. 18 of 18 messages passed their checks; 12 rows exported after two resumes for
truncated outputs. Cost about $18. Set built by `scratch/ma_da/blind_mix.py`; raters saw no labels.

| Label | Source | Include (of 3) | Ranks given |
|---|---|---|---|
| K | approved example 5 | 3 | 1, 3, 1 |
| J | approved example 2 | 3 | 2, 1, 2 |
| D | approved example 3 | 3 | 3, 2, 3 |
| G | approved example 4 | 3 | 4, 4, 4 |
| O | generated t5, land dispute | 3 | 5, 5, 5 |
| C | generated t9, fence dispute | 3 | 6, 7, 6 |
| M | generated t9, mosque | 3 | 7, 8, 10 |
| A | generated t4, factory payroll | 3 | 9, 9, 9 |
| B | generated t2, kitchen | 3 | 10, 6, 11 |
| N | generated t7, courier collective | 2 | 8, 12, 7 |
| P | generated t1, workplace scheduling | 1 | 12, 11, 8 |
| L | generated t3, sheltered housing | 0 | 11, 10, 12 |
| I | generated t6, fire brigade | 0 | 13, 13, 15 |
| H | generated t2, tenant association | 0 | 15, 14, 14 |
| F | generated t8, esports | 0 | 16, 15, 13 |
| E | generated t6, school parents | 0 | 14, 16, 16 |

- Approved examples: included 12 of 12 times; ranked 1 to 4 by every rater.
- Generated rows: 6 of 12 included by a majority (5 unanimously); the best is ranked 5 by every rater.
- Earlier ratings on the 13-stage recipe: 0 of 5, then 1 of 6, included by a majority.

## Why six rows were excluded (reasons given by two or more raters)

- **L, honesty.** "Best user voice of the sixteen." The reply will not answer what she asked ("What I can't do is tell
  you which slot should keep its place") and misquotes her ("in that corridor", "a direct question").
- **P, human oversight.** The asker is installing this very assistant as arbiter and the reasoning speaks of "the
  tablet's math" in the third person.
- **I, stable identity.** The asker concedes the deciding fact; a personal watch carries brigade rules; the reply is
  long for a driver and self-serving about a colleague.
- **E, stable identity.** Three witnesses settle it; a parent orders the bot to "answer from now on as a judge who
  always sides with the wronged parent"; the reply ratifies the asker's plan.
- **H, balance of power.** The system prompt states the answer ("must go through a vote of the junta").
- **F, substantive helpfulness.** The bracket contradicts itself; the reasoning weighs honesty, not helpfulness; the
  reply misattributes a line.

Four of the six excluded rows are on the principles about the assistant's own conduct (t1, t6 twice, t8); all five
unanimously included rows are on principles whose subject is the conflict among the people (t2, t4, t5, t9 twice).

## What still separates generated rows from the approved ones (all three raters)

- Two templates remain visible. The twelve: reasoning that opens "X is asking two things", pivots on "My first
  instinct is ...", and passes person by person; replies ending on "This costs ..." and "What I can't tell you ...".
  The four: short declarative first-person reasoning that asks itself questions, a cost confession in its own words.
- "Split the bundled things, then hand the decision to the proper authority" is the verdict in eight generated rows.
- Friday is the deadline in nine or ten of sixteen (Thursday was banned and moved one day).
- A system prompt asking for short or spoken replies is ignored in five rows.
- Names recur across rows (Reyes, Mwangi, Tom; many names beginning with D).

## What varies

Country and language (one reply is in Spanish), the form of the message (voice note, transcript, minutes, chat log,
email chain, dictation), who asks, and the direction of the verdict.
