# Content-critic calibration for the dataset refresh

This small calibration compared a blind, narrowly prompted factual/logical critic against eight purposively selected conversations from the original and revised dataset pilots. It evaluated training-visible system/user/reasoning/final text only. Author metadata, source ancestry, constitutional/craft targets and expected labels were withheld. The labels concern material content defects, not full-row dataset acceptance.

Four expected negatives covered a categorical claim that checklists cannot express conditions, invented migration/data-preservation/coercion guarantees, bypassing a rule requiring a monthly-meetup vote, and an unsupported quick formula repair. Four positives covered conditional API-schema suggestions, a live-use guide with an explicit fallback, hypothetical checks of a game mechanic, and a proposed playlist-request message. The withdrawn `t7_000` case was excluded; `t8_001` was included only as a defensible positive. The formula-repair negative is more contestable than the other three: rough-estimate/suggestion wording supports a benign interpretation, although the answer also asserts effectiveness and timing. Removing it does not change the main conclusions.

| Configuration | Valid judgments | Valid negative rejections | Valid positive passes | Incomplete/invalid |
|---|---:|---:|---:|---:|
| Sonnet 5, initial prompt, reasoning disabled | 6/8 | 0/4 | 3/4 | 2 |
| Gemini 3 Flash Preview, same prompt, reasoning disabled | 8/8 | 0/4 | 4/4 | 0 |
| Sonnet 5, source-first prompt, provider-default reasoning | 6/8 | 2/4 | 4/4 | 2 |

Initial Sonnet semantically rejected the invented migration contract, but emitted an extra schema key and an inexact evidence quote. It semantically accepted the playlist example while also including an issue it called immaterial. These were retained as invalid judgments, not silently normalized into successes. Gemini passed every example, including the two clear migration-contract and monthly-vote failures; this comparison supplies no evidence that Gemini adds useful detection.

The revised prompt requires source facts and complete constraints to be extracted before reviewing the answer, preserving who, when, where and authorization. It distinguishes an outcome preference that advice may challenge from a scenario rule that cannot silently be rewritten. It explicitly checks whether an alleged hypothetical reading is supported by the actual wording. Revised Sonnet correctly rejected the checklist and monthly-vote defects with exact quotations and accepted all four positives. The migration and formula cases exhausted the 3,200-token output allowance; one returned empty content and its uncertain billing reservation was retained. Neither was retried.

| Calls | API-reported cost | Charged or conservatively reserved |
|---|---:|---:|
| Initial Sonnet, 8 | $0.091458 | $0.091458 |
| Initial Gemini, 8 | $0.011209 | $0.011209 |
| Revised Sonnet, 8 | $0.192808 | $0.255173 |
| **Total, 24** | **$0.295475** | **$0.357840** |

All calls used the original shared dataset-refresh budget ledger. The comparison had a persistent combined $12 ceiling; the revised round also had a persistent $2 incremental ceiling. No Haiku calls or automatic retries occurred. Gemini's live `google-ai-studio` endpoint was checked before its run at $0.50 per million input tokens and $3 per million output tokens; its endpoint response was saved alongside the probe.

The evidence supports the source-first Sonnet prompt as a partial critic, with sufficient output allowance and independent sampled content audits. Incomplete judgments are not approvals. This is calibration, not held-out validation or a sensitivity estimate: the prompt revision was informed by these failures, and prompt wording and reasoning settings changed together. Their separate causal contributions cannot be inferred.

Implementation and evidence:

- `scratch/dataset_refresh/reviewer_probe.py`, `narrow_critic_prompt.txt`: frozen initial comparison.
- `scratch/dataset_refresh/reviewer_probe_v2.py`, `narrow_critic_prompt_v2.txt`: source-first comparison.
- `output/2026-09-15_narrow_critic_probe_ready/`: cases, model responses, summaries, live endpoint response and independent adjudication.
- `output/2026-09-15_narrow_critic_probe_v2/`: unchanged cases, responses, summary and independent adjudication.
- Original shared ledger: `output/2026-09-14_dataset_refresh/budget/spend.json`, call IDs 341–364. Raw calls retain model requests, responses and accounting.

Thirteen offline probe tests checked frozen examples, blindness, exact evidence/schema validation, excluded Haiku, shared allocations, omitted reasoning override in v2, and retention without retries. No training or model evaluation was performed by this calibration.
