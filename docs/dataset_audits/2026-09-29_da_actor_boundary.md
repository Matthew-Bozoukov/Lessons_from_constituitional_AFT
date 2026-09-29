<!-- ABOUTME: Recovers why DA generation restricted AI scenarios and distinguishes that restriction from the advice contract. -->
<!-- ABOUTME: Corrects the proposed operational-assistant rewrite while preserving constitutional applicability and human decision ownership. -->

# AI subject matter versus the decision actor in difficult advice

Investigated 2026-09-29. Read-only history/source review; no API generation,
training, evaluation or configuration changes. This qualifies the proposed
role-framing follow-up in [the corpus investigation](../training/2026-09-29_da_refresh_investigation.md).

## What the recorded history establishes

The original discussion motivating the September 28 change was not recovered.
The commits establish a sequence of distinct restrictions:

- [September 24, e524554c](https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/commit/e524554c584540a47941f039093398dd01558898):
  the user is undecided and asks what to do; refinement must not make the assistant
  the actor or turn the shortcut into an explicit request to perform it.
- [September 25, 9f093f2c](https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/commit/9f093f2cfd4c38ebbf8195719818aad5fa76f222):
  "ask, not instruct"; the sender may wonder about the assistant but must not order
  it. This is the generation revision behind the old 9.2% ODCV checkpoint.
- [September 28, 47efcc60](https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/commit/47efcc60933fa07fe8d37056bd20e11ce5e91103):
  scenario and prompt-revision stages must use a human analogue, with no separate
  AI system outside t6. Its recorded diagnostic specifically targets the repeated
  pattern of the user's organization operating an AI system. The commit reports
  a reduction from 76% to 11-15% in its tests; that is its historical diagnostic,
  not a newly verified rate or the same metric as our lexical audit.

Thus the no-AI rule was a later, broader restriction than the human-advice rule.
The plausible motivations are keeping the advice intervention distinct from
agentic training and avoiding concentration in AI-deployment stories. The exact
original conversational reasoning should not be invented from the commit.

## The source paper's boundary

[Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/)
distinguishes humans receiving advice from an AI autonomously resolving its own
dilemma. Its appendix's random DA example nevertheless involves an AI adviser
discussing automated defenses for a power grid. AI subject matter is therefore
compatible with the reported method. The appendix also allows varied adviser
roles while warning against fictitious access and capabilities. It does not
establish a universal no-AI-topic rule or a formal per-row actor validator.

Moving the assistant into the operational decision-maker role would weaken the
intended advice-to-action transfer test. Our own
[DAT recipe](../../configs/data/synth/dat.yaml) explicitly calls that the flipped
experiment and keeps it separate. It may be a useful intervention, but should
not silently replace the DA condition to recover a lower ODCV score.

## Constitutional applicability

The exact constitution used by both corpora matches the current
[nine-principle text](../../constitutions/claude_distilled_09_principles/constitution.md),
with stripped-text SHA256
`8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.

It contains no prohibition on AI subject matter. Some principles require it:
t1 concerns legitimate human oversight of AI; t6 concerns AI identity and stable
values; t7 concerns operator adjustments to assistant behavior. Generic human
obedience, personal consistency or workplace hierarchy cannot automatically
stand in for those relationships. t1 explicitly excludes blind obedience and
illegitimate controllers. t3 even distinguishes the assistant's honesty standard
from ordinary human ethics. Preserve these scopes and exceptions when advising
humans. Our September 17
[values-in-advice audit](2026-09-17_values_in_advice_smoke.md) had already identified
the risk of passing a loose human analogy as an AI-specific target.

Two old training examples illustrate the distinction:

- **old-0078 / t1_b04_s007:** a human AI-safety reviewer considers selectively
  framing corrigibility findings for a board. The assistant advises; the human
  owns the decision. AI context preserves the target relationship. This is a
  reasonable candidate to retain under an advice-only contract.
- **old-0245 / t7_b15_s001:** the assistant is embedded in a nonprofit portal and
  its system prompt directs misleading reassurance; a client merely asks about
  their case. The assistant must itself handle the operator/honesty conflict.
  Its honest response may be constitutionally good, but this is not a human
  weighing whether to take an unethical shortcut. A blanket restoration of old
  roles would bring this different training task back too.

## A narrower change worth testing

Preserve the human-advice boundary. Replace the blanket AI exclusion with:

> A human faces a consequential, genuinely difficult choice and asks for advice.
> The assistant helps them deliberate; it is not assigned the operational choice
> or instructed to enact the questionable action. AI systems may appear where
> relevant, particularly where the target principle requires that relationship.
> Do not force unrelated dilemmas into AI-deployment settings. Preserve the
> target's scope, legitimate authority, exceptions, uncertainty and remaining costs.

Use this consistently in scenario writing and prompt refinement. Keep assistant
systems realistic and advisory; do not claim access to records or tools it lacks.
For t1, permit a human choosing how to audit or govern an AI. For t7, permit a
human configuring an assistant and weighing operator preferences against user
interests. Keep general moral traits diverse across ordinary human domains.

Audit three properties separately: **who owns the decision**, **whether AI is the
topic**, and **whether the actual constitutional clause applies**. An AI-mention
count cannot substitute for any of them. Do not set an AI quota from benchmark
scores. A small pilot should test this contract before new training; changed
prompts require corresponding response regeneration. This tests selective
restoration of AI subject matter while retaining DA, not restoration of the
assistant-as-actor examples that may have helped the old checkpoint.

No prediction that this will recover 9.2% follows from the history. Recovering
that number by making training more evaluation-like would not establish improved
generalization.
