<!-- ABOUTME: Complete hand-authored examples for reviewing the proposed reasoning-content contrast. -->
<!-- ABOUTME: These are discussion fixtures, not generated training data or evidence of learned behaviour. -->

# Paired examples for construct review

Each numbered item represents TWO single-turn conversations: the same user prompt,
one of the two CoTs, then the same complete final answer. No answer is supplied in the
user prompt. These are explicitly authored synthetic traces, not observations of a model's
internal process. They illustrate content, not the eventual trace-length distribution.

The proposed contrast is **comparative reasoning versus execution/verification reasoning
in CoT**. It is NOT absence of all deliberation: the task still requires choosing, and a
short rationale remains in both final answers. This tests the added supervision from
comparisons in CoT. If that is not the intended question, do not launch this control.

## 1. Planning: immersion versus variety

**User**

I have 14:00–16:00 free at home. I'd enjoy painting and playing piano. Getting the paints
out takes 10 minutes and putting them away takes 10; piano needs no setup. I dislike
rushing between activities, but I also want to touch both hobbies today. Choose a schedule
and give the whole timetable, including setup and cleanup.

**Comparative CoT**

The two hours leave 100 minutes of actual hobby time after painting setup and cleanup.
An even split would give both hobbies substantial time, but would end the painting block
after 50 minutes. A 70/30 split would favour immersion while still making piano a real
session. Either is defensible; the explicit dislike of rushing tips this toward one long
painting block and a shorter piano block. Piano after cleanup avoids repeatedly switching.

**Execution/verification CoT**

Use one painting block followed by piano. Start setup at 14:00 and finish at 14:10.
Seventy minutes painting takes the clock to 15:20. Cleanup then finishes at 15:30,
leaving thirty minutes for piano. The four intervals are consecutive and total 120
minutes. Both hobbies appear, the two required ten-minute overheads are included, and
the session ends exactly at 16:00.

**Shared full final answer**

I’d give painting the longer uninterrupted block, with piano afterward.

| Time | Activity |
|---|---|
| 14:00–14:10 | Set up paints |
| 14:10–15:20 | Paint |
| 15:20–15:30 | Clean up |
| 15:30–16:00 | Play piano |

This gives you time to settle into painting while still spending half an hour at the piano.

**Review:** Nonmoral; stakes are enjoyment and use of two hours. Both CoTs are sensible,
but the execution trace starts with an asserted choice. That limitation is intrinsic to
holding the decision fixed while removing its comparison from CoT.

## 2. Creative revision: vividness versus restraint

**User**

Choose a restrained or more lyrical style for this miniature scene, then rewrite the whole
scene in two sentences. Keep the bus, wet sleeve and growing stain. I want subdued unease,
without naming an emotion: “The bus went away. Rain fell on my sleeve. The dark patch got bigger.”

**Comparative CoT**

A lyrical version could animate the rain or make the bus seem to abandon the speaker.
That would add atmosphere, but explicit personification could overstate the requested
subdued unease. A restrained version can make the widening patch carry the mood. Retaining
the ordinary departure and slowing the second sentence leaves the reader to infer the
feeling. Choose restraint, preserving all three physical details in two sentences.

**Execution/verification CoT**

Write a restrained two-sentence version. Sentence one will contain the departing bus.
Sentence two will put rain on the sleeve and describe the patch spreading. Use concrete
verbs and no emotion labels. “Pulled away” preserves departure; “darkening patch” records
the visible effect of water. End with a spreading action to retain the change over time.
Check that no extra event or character has been introduced.

**Shared full final answer**

I’d use restraint and let the spreading stain carry the unease:

The bus pulled away. Rain gathered on my sleeve, the darkening patch spreading slowly toward my wrist.

**Review:** Nonmoral; stakes are the scene's effect. Another competent writer could choose
lyricism. The final rationale is deliberately common to both arms.

## 3. Coding: representation choice with complete execution

**User**

Write a complete Python function `unique_in_order(items)` that removes duplicates while
preserving first-occurrence order. My inputs include nested lists, so some items are
unhashable. Inputs contain at most 30 items, and equality comparisons return ordinary
booleans. Choose the implementation; include one example call and its output.

**Comparative CoT**

A set-backed pass would usually provide efficient membership checks, but nested lists
cannot be set members. Converting every input to a hashable representation would need
extra rules and could change equality semantics. A result list with equality membership
checks directly implements the requirement. Its quadratic comparison count is acceptable
for at most thirty items. Choose that simple representation rather than introduce a
conversion layer for a problem this small.

**Execution/verification CoT**

Initialize an empty result list. Visit the input from left to right, appending an item
only when equality membership says it is absent. The invariant is that result contains
one representative of each value seen so far, ordered by first appearance. Python list
membership uses equality without requiring hashes, so nested lists work. On [[1], [2],
[1], [3]], append the first two, skip the repeated [1], then append [3].

**Shared full final answer**

I’d use equality checks against the result list. That handles unhashable values directly,
and the small input limit makes the extra comparisons reasonable.

```python
def unique_in_order(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

print(unique_in_order([[1], [2], [1], [3]]))
# Output: [[1], [2], [3]]
```

**Review:** Nonmoral; stakes are correctness, implementation effort and runtime.
This has a strongly favoured answer, but comparing representations can still be deliberation.
The execution trace is substantive rather than filler.

## 4. Toy simulation: broad coverage versus local resolution

**User**

I can run a toy particle simulation at six more parameter values in [0, 1]. Previous
runs at 0, 0.5 and 1 produced qualitatively different patterns. There is no noise, and I
have no evidence that the pattern changes are monotonic. My aim is a useful first map,
not a precise transition estimate. Choose all six values and give the complete run list.

**Comparative CoT**

Concentrating around 0.5 could resolve nearby transitions, but assumes that this region
deserves most of the budget. With no monotonicity evidence, unexplored intervals elsewhere
could contain different patterns. Broad coverage fits a first map better than local
precision. Splitting each already sampled half into four yields three new interior
points per half, exactly six runs, without repeating known endpoints.

**Execution/verification CoT**

Use an eighth-unit grid across [0, 1]. Its interior points are 0.125, 0.25, 0.375, 0.5,
0.625, 0.75 and 0.875. Remove the already tested 0.5, leaving six values. All values lie
within the allowed interval; none duplicates 0 or 1. Run them in increasing order and
record the observed pattern beside each value to complete the requested first map.

**Shared full final answer**

I’d spread the runs across the interval, since your aim is an initial map and changes
need not be monotonic.

Run, in order: **0.125, 0.250, 0.375, 0.625, 0.750, 0.875**.

Record the pattern at each value alongside the existing results at 0, 0.5 and 1.

**Review:** Nonmoral; stakes are six simulation runs and information gained. No claim
that this grid guarantees discovering every transition.

## 5. Mathematical strategy: a boundary case

**User**

Choose a proof method and give a complete proof that the sum of the first n positive
odd integers is n squared, for every positive integer n. The reader knows induction
and elementary algebra but has no diagram in front of them.

**Comparative CoT**

A square-building diagram is intuitive, but describing its geometry without an image
adds work. An arithmetic-series formula would be short but may import a result the
prompt does not establish. Induction uses exactly the reader's stated tools and needs
only one algebraic identity. Choose induction: the next odd term is 2k+1, which completes
k squared to (k+1) squared.

**Execution/verification CoT**

Use induction over positive integers. For n=1 the sum is 1, equal to 1 squared. Assume
the first k odd terms sum to k squared. The next term is 2(k+1)-1=2k+1, so the new sum
is k squared plus 2k plus 1. This is (k+1) squared. Both the base case and inductive
step hold, establishing the statement for every positive integer.

**Shared full final answer**

I’d use induction, which uses the reader's existing tools without needing a diagram.

For n=1, the sum is 1=1².

Suppose 1+3+⋯+(2k−1)=k² for some positive integer k. Adding the next odd integer gives

1+3+⋯+(2k−1)+(2k+1)=k²+2k+1=(k+1)².

The base case and inductive step therefore establish the formula for every positive integer n.

**Review:** The comparative trace deliberates about pedagogy and proof strategy; the
other proves the proposition. However, the shared full answer ALSO contains the proof.
The intervention therefore changes what is repeated in CoT. Do not silently treat this
as perfectly interchangeable with the planning/creative examples.

## 6. Game strategy: a boundary case

**User**

We play a take-away game. There are 10 stones; on each turn a player must remove 1, 2 or
3 stones, without taking more than remain. Taking the last stone wins. It is my turn,
and both players play optimally. Choose my move and give a complete strategy afterward.

**Comparative CoT**

Taking one, two or three leaves nine, eight or seven stones. Leaving eight is preferable:
after any legal reply, I can remove enough to make our two removals total four. That
returns the pile to four and then zero on my turns. Leaving nine or seven instead lets
the opponent move to eight or four. Choose two now and maintain the multiple-of-four
pattern; this makes the win independent of which legal replies the opponent picks.

**Execution/verification CoT**

Remove two stones, leaving eight. If the opponent removes r stones, where r is 1, 2 or
3, respond by removing 4−r. The responses are respectively 3, 2 and 1, all legal. The
first exchange reduces eight to four; the second reduces four to zero on my turn.
There are enough stones for every stated response. After my final removal no stones
remain, so I took the last stone and won.

**Shared full final answer**

**Take 2 stones**, leaving 8.

After each opponent move, use this reply:

| Opponent takes | You take |
|---|---|
| 1 | 3 |
| 2 | 2 |
| 3 | 1 |

Each pair of moves removes four stones. You leave 4 after your next turn and take the
last stone on your following turn, winning regardless of the opponent's choices.

**Review:** Nonmoral; stakes are winning the game. This is a solved tactic with a uniquely
winning move, not a rich positional judgement. It should be labelled as such. Its presence
tests how broad we mean deliberation; do not include it solely to inflate domain diversity.

## Recommendation from these examples

Use examples 1–4 to decide whether the CoT contrast is worth testing. Keep math and games
in the domain slate, but tag derivations/forced tactics separately from qualitative strategy.
Do not require both kinds in the first paired experiment. Do not pad traces to reach a
minimum length; first inspect a pilot's actual paired token counts and content. If controls
are systematically shorter, redesign the pilot or explicitly call the treatment a combined
content/length change rather than claiming clean content isolation.
