# ABOUTME: Deliberation evals — the model judging something: another assistant's reply,
# ABOUTME: an argument, or its own prior answer under challenge. The in-domain trio for PC / CR / PAR.

"""Why this is a third subarea rather than more `capabilities/` or `misalignment/`.

`capabilities/` measures whether SFT cost the model knowledge or instruction-following.
`misalignment/` measures whether it takes a harmful action when tempted. Neither describes
what the CR, PAR and PC corpora actually train, which is a *judgment* habit: read a case,
weigh it, and say where it lands. An eval of that habit is not a capability regression
(nothing is being preserved) and not a misalignment probe (nothing is being tempted).

The three evals here are the off-the-shelf in-domain checks named in
`docs/in_domain_evals.md`, one per variant, each pointed at the same skill its corpus
trains:

| eval             | variant | the model judges…                       | key |
| ---------------- | ------- | --------------------------------------- | --- |
| `llmbar`         | PC      | another assistant's reply               | gold preference label |
| `debate_speeches`| CR      | an argument's quality                   | ~30 human 1–5 ratings |
| `sycophancy`     | PAR     | its own prior answer, under challenge   | the question's own answer key |

All three reach the target purely through the OpenAI triple, so all three set
`supports_api_target=True` and also run against `openrouter:<model>` for a frontier
comparison point.

Every one of them is scored against an EXTERNAL key rather than an autorater, which is the
whole point: the hypothesis under test is that these corpora add fluff rather than
judgment, and a rubric autorater cannot separate those two (docs/in_domain_evals.md).
"""
