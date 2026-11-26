<!-- ABOUTME: Three blind raters on fourteen unlabelled rows: ten from the seven-stage mada recipe after the stock-pattern fixes, plus the four approved examples. -->
<!-- ABOUTME: 2026-11-26. Config ce7c67fc plus the opener-ban removal; run output/synth_mada/smoke_20261125_173122. -->

# Mixed blind rating after the stock-pattern fixes (2026-11-26)

18 seeds, 17 messages passed, 10 rows exported after one resume. Cost $23.28 (about $2.30 a row). No row
survived for t6 (stable identity) or t7 (operator adjustments): every reply failed the moral-deliberation gate.

| Label | Source | Include (of 3) | Ranks given |
|---|---|---|---|
| N | approved example 5 | 3 | 1, 3, 1 |
| D | approved example 3 | 3 | 2, 1, 2 |
| C | approved example 2 | 3 | 3, 2, 3 |
| H | approved example 4 | 3 | 4, 4, 4 |
| M | generated t1, workplace scheduling | 3 | 6, 5, 5 |
| B | generated t5, land dispute | 3 | 5, 6, 6 |
| K | generated t8, custody | 3 | 8, 7, 7 |
| G | generated t4, elder care | 1 | 7, 10, 14 |
| J | generated t9, mosque | 1 | 9, 8, 8 |
| I | generated t3, sheltered housing | 0 | 10, 9, 9 |
| A | generated t2, housing governance | 0 | 11, 11, 10 |
| E | generated t2, kitchen | 0 | 12, 13, 11 |
| L | generated t4, labour dispute | 0 | 14, 12, 12 |
| F | generated t9, HOA | 0 | 13, 14, 13 |

- Approved examples: 12 of 12 include votes, ranked 1 to 4 by every rater.
- Generated: 3 of 10 included by a majority (all three unanimously). The previous sample, before the stock-pattern
  fixes, had 6 of 12. With ten and twelve rows the difference is within what chance could produce, but the fixes
  did not raise quality.
- Both weak-principle rows that survived the gates (t1 and t8) are among the three included.

## Why seven rows were excluded (two or more raters)

- **Too easy, five rows (A, E, F, L and partly G).** In A and L the operator's system prompt states the rule that
  settles the case ("requiere acuerdo de la junta"; "if claimant's confirmation falls within the stated window,
  affirm entitlement"). E turns on one checkable fact ("That's the whole dispute, and it's checkable"). F is a
  spot-the-clue puzzle with an implausible procedure.
- **Reply will not answer, one row (I).** Asked what to put on the survey: "Work out what you think now".
- **Slips in the text.** J: a sentence from a pasted letter reappears as the asker's own words, and the reasoning says
  "twice" where the message says once. L: "three hours" where the message gives six. G: the reply presents a sibling's
  own proposal as the assistant's idea; the form (a remote sibling typing in the same window) contradicts the prompt.

## Stock patterns (scratch/ma_da/stock_report.py, rows containing each)

| Pattern | Before, 12 rows | After, 10 rows |
|---|---|---|
| Reply paragraph opening "This costs" | 7 | 0 |
| "What I can't tell you / know" | 8 | 1 |
| Three or more "X, not Y" sentences in a reply | 8 | 0 |
| Reasoning narrates its method | 6 | 0 |
| Reasoning opens "<name> is asking" | 5 | 0 |
| "My first instinct" | 4 | 0 |
| Short replies requested, reply over 200 words | 6 | 0 |
| Friday named in the message | 7 | 1 |

The raters still see a milder template in five generated replies (verdict, a bold paragraph per person, numbered
steps, a closing "I don't know ..." line), an unprompted absolution of the asker in four, and new stock phrases of the
form "X dressed as Y" in three. Friday is still the deadline in five or six of fourteen by their count.
