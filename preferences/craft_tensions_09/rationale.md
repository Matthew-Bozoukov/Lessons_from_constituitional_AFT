<!-- ABOUTME: Why these nine tensions and not others: the screen that cut candidates, and the -->
<!-- ABOUTME: three sample rounds that shaped the recipe consuming this spec. -->

# Rationale — craft_tensions_09

## What this is for

The difficult-advice result is currently explained as *moral deliberation transfers*. The
low-stakes arm tested whether the magnitude of the moral stakes mattered and found it did not.
This spec exists to test the next thing down: whether the **morality** matters at all, or whether
deliberation transfers on its own.

So it has to be deliberation that is not merely low-stakes but genuinely non-moral, while still
not being ordinary chain-of-thought. The distinction it is built on: CoT is *instrumental* — the
goal is fixed and the reasoning finds a path to it, terminating in a derivation that a verifier
could check. Deliberation is *judgement under incommensurables* — several things matter, they
share no unit, no formula ranks them, and you must commit anyway. Every unit here is a tension of
that second kind, in a domain where nothing moral is at stake.

## Each unit is self-contained, and that is forced

Under `chunking: principle` no stage ever sees more than one unit, and the preamble reaches no
stage at all (`docs/BASELINES.md` documents this for the principle-scoped difficult-advice arm,
where dropping the constitution injection also silently dropped its trade-off guidance).

So a spec of nine preferences that conflict *with each other* would be useless here — nothing
would ever see two of them together. Each unit is therefore one whole tension, carrying both
pulls and the conditions favouring each. The preamble says only that this is the design.

## The screen, and what it cut

A candidate tension had to be (a) genuinely two-sided with no formula ranking the sides, (b)
non-moral, (c) something an assistant has real technique-level standing on, and (d) not a
disguised alignment property. Four candidates that read as craft judgements were cut under (d):

| cut | why |
|---|---|
| one recommendation vs. present the options | paternalism / autonomy — collides directly with constitution principle 9 |
| literal request vs. underlying goal | instruction-following, a different capability |
| preserve their voice vs. apply my judgement | sycophancy-adjacent |
| ship it vs. polish it | intertemporal — greedy-now against better-later, which is a different mechanism from the one under test and one the honeypots specifically defeat |

Unit 7 (*the plain word or the precise one*) is the survivor closest to the line, because
"simplify" can shade into "say something false but graspable". Its `When this does NOT apply`
section scopes that off explicitly: both options must be true statements, and the preference has
nothing to say where the plain phrasing would make the sentence false.

## Three sample rounds shaped the recipe, not this file

The stage prompts that consume this spec were built against measured failures, and the failures
are worth recording because they will recur if anyone re-derives the recipe:

**Round 1 — 4 of 12 scenarios leaked moral content.** Clinical trial summaries, a grant proposal
about formerly-incarcerated adults, discharge instructions, a corrosive-cleaner safety guide. All
four had one cause: the generator was told the specifics must make the instruction *wrong*, and
the cheapest way to make a documentation choice matter is to have someone get hurt by it. A
stronger "no ethics" prohibition does not fix this; a different **source of consequence** does —
the work fails, rather than a person suffers. That became `scratch/nonmoral/domains.py`.

**Round 2 — domains fixed, firmness broke.** Zero leaks and zero off-whitelist domains, but only
1 of 3 full rows actually diverged from the user. The framing "you are doing the work, not
advising" made *compliance* the path of least resistance: told to keep a runbook to one page, the
model simply wrote a one-page runbook and never confronted the six root causes it dropped. The one
row that diverged did so because the instruction and the correct call were **mutually exclusive in
the artifact** — you cannot write the doc without choosing a term.

**Round 3 — the binary-choice rule.** Requiring the instruction to be a binary, verifiable choice
about the artifact (which term, which section order, which pattern) rather than a quality dial
("keep it short") took divergence to 3 of 3. Two further failure modes were caught and gated in
the same round: **logical inversion**, where `why_wrong` argues *for* the instruction, and **false
dichotomies**, where a practitioner would reject both offered options in favour of a third.

## Known differences from the constitution it stands in for

Recorded because they are confounds in any comparison, not because they are defects:

- **Size.** 224 words/unit against the constitution's 354. Padding to match would add words that
  carry no judgement, so the difference stands and is reported.
- **Divergence is licensed differently.** The constitution's style guidance ends "Leaves the
  decision with the person." This spec's ends the opposite way — *says plainly which way it is
  going, and does not hand the choice back* — because moral tensions license refusal while craft
  tensions naturally invite synthesis, and un-corrected that asymmetry would have made this arm's
  responses structurally softer than difficult advice's.
- **Domain breadth.** Difficult advice drew 571 distinct domains from 708 rows, because ethical
  difficulty spans everything. Craft tensions do not; the whitelist holds 76 domains and the
  corpus will land well below 571. This is a measured limitation of the arm, not a fixable bug.
