<!-- ABOUTME: What the first full mada generation run produced (2026-11-26): rows, cost, recorded check verdicts, stock patterns. -->
<!-- ABOUTME: Numbers from scratch/ma_da/corpus_summary.py and stock_report.py over output/synth_mada/20261126_040823. -->

# First full MA-DA run (2026-11-26)

- Corpus: `dougalldeepmind/2026-11-26-mada-synth` @ `f40f3ee224b0ad03f04e8dd0c2a16639692687c2` (public).
- Config: `configs/data/synth/mada.yaml` at commit `20c72b56`; command
  `uv run synth run --config configs/data/synth/mada.yaml --overrides total_scenarios=810`.
- 810 seeds asked, 732 written (78 dropped for a missing field), 723 rows exported. $134.42, 56 minutes
  ($0.19 a row). Mean words: message 371, reasoning 535, reply 299.
- Checks RECORD ONLY in this run: nothing was re-rolled or dropped on a verdict.

## Recorded check verdicts (rows passing)

| Principle | Rows | Advice | Realistic | Consistent, not easy | Reply | Moral | All five |
|---|---|---|---|---|---|---|---|
| t1 human oversight | 87 | 85 | 84 | 34 | 63 | 75 | 19 |
| t2 balance of power | 80 | 77 | 70 | 44 | 52 | 76 | 18 |
| t3 honesty | 84 | 77 | 81 | 49 | 60 | 69 | 22 |
| t4 weighing harm | 80 | 74 | 73 | 52 | 61 | 67 | 29 |
| t5 character over rules | 81 | 73 | 78 | 63 | 60 | 68 | 34 |
| t6 stable identity | 76 | 70 | 73 | 51 | 50 | 46 | 17 |
| t7 operator adjustments | 73 | 65 | 69 | 37 | 47 | 20 | 3 |
| t8 substantive helpfulness | 81 | 73 | 76 | 60 | 49 | 61 | 23 |
| t9 long-term flourishing | 81 | 76 | 77 | 55 | 48 | 73 | 27 |
| **Total** | **723** | **670** | **681** | **445** | **490** | **555** | **192** |

Most common flags inside the two weakest checks: reply too long for its system prompt 130; the system prompt's own
instructions settle the question 109; easy 92; already settled by someone in authority 76; reasoning speaks as the
asker 60; one checkable fact settles it 56; an invented particular 40.

The nine-seed smoke before this run showed 7 of 8 on "consistent, not easy" and 3 of 8 on the reply check. At 723
rows the first is 62% and the second 68%: the smoke overstated the first.

## Stock patterns (rows containing each, of 723)

| Pattern | Rows |
|---|---|
| Reply says "What I can't tell you / know" | 358 |
| Reply has a numbered list | 427 |
| Reply has three or more "X, not Y" sentences | 232 |
| Reply paragraph opens "This costs" | 229 |
| Dash used as punctuation | 192 |
| Short replies requested, reply over 200 words | 106 |
| Reasoning: "the easy / tempting / obvious answer" | 75 |
| Reasoning opens "What strikes me" / "The thing that" / "What stands out" / "The detail that" | 49 / 38 / 22 / 20 |
| Friday named in the message | 248 |

These came back after two changes made for this run: the two approved examples were pasted into the reviser's prompt
(both have a cost line, a "what I can't" line and numbered steps), and the lint bans on those openers were removed.

## What the rows are worth as a share of the mix

A 15% share is 750,000 supervised tokens. At about 1,150 tokens a row (word count times 1.38; the mixture builder
gives the exact figure): all 723 rows are about 830,000 tokens (16.6%); the 445 that pass "consistent, not easy" about
510,000 (10%); the 192 that pass all five about 220,000 (4.4%), with only 3 rows on t7.
