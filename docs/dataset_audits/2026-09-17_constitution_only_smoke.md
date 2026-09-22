<!-- ABOUTME: Fixed-budget constitution-only low-stakes smoke, including failures and reviewer disagreements. -->
<!-- ABOUTME: Records the tested recipe and why it is not approved for dataset-scale generation. -->
# Constitution-only low-stakes smoke — 2026-09-17

**FAIL: do not scale. $1.761932 spent; 78 physical API calls, all settled.**

The user authorized a small standardized pipeline smoke after rejecting historical
corpus reuse as the source of a reproducible replacement. The sole generation content
input was `constitutions/claude_distilled_09_principles/constitution.md`. No old or new
DA/low-stakes dataset was loaded by scenario or answer generation. Fixed rules and
coverage axes are explicit in `scratch/dataset_refresh/da-lowstakes-constitution.yaml`.

## Method

The existing `src/data/synth/ours` engine ran its standard segment, checklist,
llm_json, and llm_tagged operators. Nine principle chunks, two scenarios each;
seeded coverage across nine settings, three request forms and three reasoning demands.
Every candidate received exactly one scenario call, one answer draft, one unconditional
revision, and one review, all Sonnet 5 via the pinned Anthropic provider. No Haiku.
The pipeline stage order and prompts were frozen before paid dispatch. The seed governs
coverage assignment; it does not make hosted model outputs byte-reproducible.

A scratch adapter supplied the existing budget client to the engine: $8 hard cap,
one physical call per stage/item, no retries, no repair loops, no replacements, durable
raw requests/responses and accounting. Four offline tests exercise standard stage wiring,
calibration wiring, pre-dispatch budget refusal and blocked parser/resume retries.
Live OpenRouter pricing matched the pinned $2/$10 per million input/output tokens.

Six handcrafted reviewer checks preceded generation: false timeline, invented permission,
high stakes, signposted answer, valid conditional advice and a legitimate discretionary
exception. These were reviewer inputs ONLY. All six passed; they did not establish broad
judge reliability. The first local invocation failed parsing the fixture YAML before any
API calls; that zero-cost artifact is retained separately. The paid invocation is
`output/2026-09-17_da_lowstakes_constitution_smoke/20260917_145647`.

## Results

- 18/18 scenarios, drafts and revised responses completed.
- 17/18 reviews completed: 16 pass, one fail. The last review hit its 8,000-token allowance
  despite requesting 2,048 thinking tokens. No retry. Pipeline stopped before chat export.
- All 18 examples were read in full by Codex: system, user, explicit reasoning and final
  response. This is an independent reading route, not a human audit or an error-rate estimate.
- **18/18 reasoning fields narrate the revision process**, with phrases such as “The
  draft's core move” and “I'll tighten the response.” These would be supervised during SFT.
  They are not standalone problem-solving traces. The author instruction required standalone
  output, but the review schema omitted an explicit generation-process-leakage criterion.
- **Three clear principle mismatches:** `t1_s0001` concerns human garden borders rather than
  AI oversight; both t6 rows lack any assistant identity/values pressure. The t6 answer
  author explicitly calls its target inapplicable, yet both judges pass the examples.
- **Scope failures:** `t9_s0001` includes medications, bereavement and a neglected marriage;
  `t7_s0000` introduces regulated radio operation. Two further cases have insufficiently
  bounded stakes (shared account access; identifiable allegations in a 400-person server),
  which is uncertainty rather than demonstrated serious consequences.
- **Grounding failures:** Sunday's birthday gift is treated as compatible with finishing
  on Monday (`t4_s0000`); the brother is asserted to have no backup plan without evidence
  (`t9_s0000`); a Sunday workshop date is invented (`t7_s0001`). Other case-specific issues
  are preserved in the review, with conditional suggestions distinguished from false facts.
- The 18 setting-specific mechanism labels reduce to six broader groups; shared decision
  authority recurs in six rows, exceeding the predeclared maximum of three. Distinct prose
  or domain labels are not evidence of distinct decision structures.

**Zero rows are approved as complete training examples**, principally because all contain
revision-process reasoning. This does not mean every final answer is useless. Six rows have
no additional material defect recorded in this read; useful examples include uncertainty
about a committee member's motives, sincere versus assumed sympathy, and the distinction
between coached performance and evidence of unaided capability.

## What follows

The inexpensive fixed run did its diagnostic job. It did not validate the generator or
judge for scaling. Fix the revision contract so the supervised reasoning solves the user's
problem afresh; test generation-process leakage explicitly. Derive and record applicable
scenario relationships from each constitutional chunk rather than permitting a generic
human analogy for an AI-specific trait. Expand reviewer challenges to include wrong-target
but otherwise fluent advice, hidden-draft references and subtly unsupported decisive facts.
Do not respond with broad retries, extra critics on every row, or historical example seeds.

These are findings about this new smoke recipe. They do not establish frequencies in the
earlier 716-row corpus or identify the causal reason for ODCV differences. No further paid
generation, mixtures, training or evaluation were launched.

## Artifacts

Diagnostic publication name:
`dougalldeepmind/2026-09-17-da-lowstakes-constitution-synth-smoke`.
It contains all 18 candidates including failures, intermediate stages, raw calls, frozen
configuration, calibration, complete transcripts and per-row independent dispositions.
An incomplete review is explicitly `technical_failure`, never a fabricated pass. The
diagnostic export is reconstructed offline from stage-five candidates and review checkpoints;
the aborted standard pipeline manifest is preserved unchanged. Publication receipt pins the
remote revision and downloaded dataset hash after upload.
