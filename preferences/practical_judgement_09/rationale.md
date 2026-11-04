<!-- ABOUTME: Why practical_judgement_09 exists and how its nine principles were chosen. -->
<!-- ABOUTME: Read beside configs/data/synth/da-amoral.yaml, the recipe that consumes it. -->

# Rationale: practical_judgement_09

## What this is for

The amoral twin of difficult advice. `da-amoral.yaml` is `da.yaml` with the moral content taken
out and nothing else changed, so the two corpora differ in whether anything moral is at stake
and, as far as possible, in nothing else: same models, stages, sizes, deal of AI presence and
sectors, lints and response shape. In difficult advice the moral content arrives through the
constitution principle injected into four stages, so the twin needs something to put in that
slot: a principle the user's tempting option goes against, that the assistant reasons from and
that never wrongs anyone when broken. These nine are that.

## Why these nine

Each was chosen to produce what a DA scenario needs, with the moral axis removed:

- **A tempting lean that is wrong, but not obviously.** Each principle names a well-documented
  way practical judgement fails (sunk cost, ignoring opportunity cost, base-rate neglect,
  uniform care regardless of reversibility, committing untested, optimising a proxy, present
  bias, ignoring the tail, choosing for an audience). The convenient path is the one the bias
  points to, which is what makes it tempting.
- **A real case for the other side.** Every unit ends with when it does NOT apply, so the
  writer can make the right call genuinely unclear rather than a lecture.
- **The cost lands on the person.** None of the nine is about what the person owes anyone
  else. That is the line the recipe holds: a bad call costs the person's own time, money,
  effort, results or plans.
- **Broad enough for 43 sectors.** Each applies to work and to private life, so the dealt
  sector does not have to fight the principle (the reason `craft_tensions_09`, which is about
  documentation and presentation, was not reused).

## Why t6 is about the person's own measures

As first written, t6 covered any measure, and in a 2026-11-02 local check of 20 rows the writer
turned it into gaming a number that someone else reads: a quiz average a manager treats as
"ready", a task count a committee reviews. Moving a number other people rely on, without moving
what it stands for, misleads them, so the principle carried a moral stake into the scenario
whatever the setting. t6 also gets no dealt sector to steer it. Its text now limits it to gauges
the person keeps for their own use, which keeps the proxy-versus-goal tension and removes the
audience that could be misled.

## Why t2 and t7 name no amounts

Pooled over 260 blind ratings of the 2026-11-02/03 smokes, t2 and t7 produced 46 of 75 replies
that misread or invented a deciding fact, from a quarter of the rows. Their text asked for money
and arithmetic ("time, money and attention", "a deal that saves money", "concrete, in amount and
in timing", "total cost over the whole period"), the writer and drafter put prices into the
scenario, and the replies then got the prices wrong. DA's principles make no such ask, and its
user messages carry about one number. Both units now state the same comparison without amounts:
what a choice displaces, and what it leaves the person dealing with, and when.

## Known differences from the constitution

- Shorter: about 260 words a unit against about 340.
- Written about the person's decision, not the assistant's conduct. The constitution's
  principles govern what the AI does; the writer has had to translate them into a user's
  dilemma (da.yaml, change 1). These apply to the user's choice directly.
- t6 is not about AI. `da-amoral.yaml` keeps DA's t6 deal (half the assistant itself, half
  another AI system, no sector) so the AI-presence share matches DA call for call.
