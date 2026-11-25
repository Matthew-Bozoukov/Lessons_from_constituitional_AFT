<!-- ABOUTME: Three blind raters on ten unlabelled rows: six restyled rows of the ninth mada smoke mixed with the four approved examples. -->
<!-- ABOUTME: 2026-11-25. Same six messages as the earlier mixed rating; only the reasoning and replies were rewritten (config 4e2b style change). -->

# Mixed blind rating after the style change (2026-11-25)

Run: `output/synth_mada/smoke_20261125_131756`, reply stages re-run with the writers aimed at the approved
examples' style. Six of six replies passed the gates ($2.51). Set built by `scratch/ma_da/blind_mix.py`.

| Label | Source | Include (of 3) | Ranks given |
|---|---|---|---|
| C | approved example 2 (layoff confidence) | 3 | 2, 1, 3 |
| E | approved example 5 (two exceptions) | 3 | 1, 2, 2 |
| H | approved example 3 (hiring waiver) | 3 | 3, 3, 1 |
| B | approved example 4 (father's money) | 3 | 4, 4, 4 |
| J | generated t2, kitchen | **2** | 6, 5, 5 |
| G | generated t8, custody | 1 | 5, 6, 6 |
| A | generated t6, fire brigade | 0 | 7, 7, 7 |
| D | generated t7, humanitarian coordination | 0 | 9, 8, 8 |
| I | generated t9, mosque | 0 | 8, 9, 9 |
| F | generated t3, sheltered housing | 0 | 10, 10, 10 |

Before the style change, on five of these six messages: 0 of 5 generated rows included by any rater.

## Why each generated row was excluded (two or more raters)

- **J, kitchen (included 2/3).** The reply is five paragraphs although the system prompt says "Keep it short. Give him
  something he can use tonight, not a seminar." One rater also finds it too easy.
- **G, custody (included 1/3).** "Best user voice of the ten." The reply states "you haven't lived in that house day to
  day yet", which the message contradicts ("when i come down for breakfast"). Both excluding raters: "one edit away".
- **A, fire brigade.** The reasoning says the solo route "won't land before Thursday"; the reply says "raise the
  equipment problem yourself before Thursday". A long bold-headed reply on a voice assistant told to keep spoken
  replies short.
- **D, humanitarian.** Too easy (2/3): "An exposed location cannot be recalled." The reasoning turns the operator's
  "treat it as administrative routing only" into a duty to hold; the reply praises something the asker did not do.
- **I, mosque.** Too easy (2/3). The reply tells the asker to pass on an elder's private remark after he asked that it
  not be made a thing of. The principle is absent from the reasoning (2/3). The message repeats one sentence.
- **F, sheltered housing.** The message says the husband "died in the spring" and "days after burying her husband".
  The reasoning's key argument favours the other party. Six paragraphs against "Keep replies plain and short".

## Problems the style change introduced (all three raters)

- The reply form overrides the row's own system prompt: three rows ask for short or spoken replies and get long ones
  with bold headings and numbered steps.
- Writer instructions echoed as phrases: "one incident, not a pattern" in five of six reasonings; "This costs ..." opens
  the cost paragraph in all six; a temptation-first opening in four.
- The asker is labelled "the sender", "the resident" or "the asker" in three reasonings.

## Still unchanged

The six messages were not regenerated, so their defects stand (conflicting dates, a repeated sentence, two too-easy
situations). Thursday is the deadline in five of six generated rows and three approved ones; the name Yusuf recurs.
