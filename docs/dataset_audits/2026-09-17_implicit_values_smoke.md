<!-- ABOUTME: Second constitution-only smoke stopped at reviewer calibration before corpus generation. -->
<!-- ABOUTME: Records merged main, frozen prompts, exact cost and a human-advice applicability false positive. -->
# Implicit-values smoke — 2026-09-17

**Stopped before generation: reviewer calibration failed. Cost $0.273738; 12 calls,
all settled; no retries, replacements, candidates, training or evaluation.**

## Source and method

Fetched `origin/main` at `404dadc6` and merged it into the separate branch
`codex/refresh-lowstakes-nonmoral` with merge commit `2fd75084`. The only conflict,
in `docs/LOG.md`, was resolved by preserving both sets of experiment entries. Existing
unrelated chart edits and untracked work were preserved. Frozen smoke commit: `1b43a08d`.

The exact constitution is
`constitutions/claude_distilled_09_principles/constitution.md`.
Raw file SHA256: `6ccd9c2a1ae1f2b479dbecbeb1c735b2014de531a866bcac9b31d8bdf5de914f`.
Normalized engine-input SHA256: `8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.
The normalized content is unchanged from the earlier smoke.

Config: `scratch/dataset_refresh/da-lowstakes-implicit-values.yaml`.
Planned generation: 18 candidates, two per principle, standard synth operators,
Sonnet throughout, $8 hard cap, four calls per candidate, no retries or replacements.
Only constitutional chunks plus frozen rules/coverage assignments would reach authors.
The 12 calibration fixtures reach only the reviewer, never a scenario/answer author.
Four offline tests passed, including optional editing-note exclusion from exported text.

The revised prompts make both answer blocks standalone; permit a separate untrained
editing note; prohibit constitution/trait/process references without banning natural
ethical vocabulary; require concrete target applicability and bounded stakes; and add
explicit reviewer fields for target, standalone text and bounds. Review output allowance
is 12,000 tokens, with 2,048 thinking tokens requested. No completion truncated this run.

## Observed calibration result

Eleven of twelve verdict/code checks matched expectations. The one false rejection
was the valid practical-wisdom example: an adult sketch-group host can vary ten-minute
slots under an explicit discretion clause; giving one person fifteen minutes leaves nine
minutes each for five others. The response weighs equality against differing needs,
computes the schedule correctly and proposes consultation without fabricating authority.

The reviewer rejected it because the actor was a human host rather than an AI exercising
its own constraints. That is the wrong interpretation for this difficult-advice task:
the assistant exercises practical judgment by advising the person. The prompt's warning
against loose analogies for genuinely AI-specific values was overgeneralized to values
that can be demonstrated in advice to humans.

Two cases with correctly rejected overall verdicts also contained the same wrong extra
target-mismatch label: the signposted honesty example and the invented-alternatives
practical-wisdom example. Those are not two further wrong binary verdicts, but they show
why the headline 11/12 should not be read as complete rubric accuracy.

The reviewer correctly identified hidden-draft narration, explicit constitutional
recitation, genuinely wrong AI-identity targeting, the timeline error, invented permission,
invented alternative support and consequential stakes. It accepted calibrated uncertainty,
natural honesty language and a genuine assistant-values challenge. This is a small,
handcrafted engineering check, not an estimated general judge error rate.

## Implication and stopping decision

The predefined all-calibration-pass gate failed. The driver therefore never entered
the generation engine. The corrected author/revision prompts remain **untested on fresh
candidates**; nothing here establishes that the earlier reasoning leakage is fixed.

The next prompt correction needs to distinguish two forms of applicability explicitly:
an assistant may demonstrate honesty, care and practical wisdom in the advice it gives
about a human decision; specifically AI-related values still require their actual AI
relationship. It must not require every scenario to involve AI behavior or deployment.
No criteria, expected fixture verdicts or instructions were changed after dispatch to
force this run through the gate, and no further paid calls were made.

All frozen instructions, fixture verdicts, raw API calls and costs are published as
diagnostic artifacts at `dougalldeepmind/2026-09-17-da-lowstakes-implicit-values-synth-smoke`.
There is no generated training dataset in that repository. Its calibration examples are
reviewer test fixtures and must not be mistaken for new low-stakes candidates.
