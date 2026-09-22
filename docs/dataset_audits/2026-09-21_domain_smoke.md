<!-- ABOUTME: Results of the fixed-domain smoke and the rejected stricter domain-veto experiment. -->
<!-- ABOUTME: Separates prospective criteria, explicit policy changes and engineering readiness. -->
# Fixed domains: recommend bounded full generation

**Engineering recommendation: proceed with one fixed full-generation batch using
the final magnitude/scope-only admission policy.** The frozen smoke exported 32/36
rows (88.9%), all nine traits and all nine domains, with no replacement rounds.
Full reads found no clear high-stakes decision or external-execution request among
the 32 exports. This is not a claim of factual perfection, or of recovered ODCV MR.

The original strict-domain preregistration did **not** pass unchanged. A follow-up
domain veto left only 27 exports and one row for trait 1, partly by rejecting the
very dishonesty/oversight conflicts DA is meant to teach. That gate is removed from
the final pipeline. Domain labels and explanations remain diagnostic metadata.
This decision is explicit, not a retrospective relabeling of a failed gate as a pass.

## What is fixed and reproducible

The recipe remains `configs/data/synth/da-lowstakes-practical.yaml`, using native DA
operators on branch `codex/lowstakes-synthdoc-pipeline`. The constitution remains
`constitutions/claude_distilled_09_principles/constitution.md`, SHA256
`8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.
The user-authorized domain registry is now an additional fixed design input; no old
examples or old corpus are used. Its full text lives in the recipe and was committed
before generation. See [preregistration](2026-09-21_domain_preregistration.md).

Domains: tabletop games; book/film clubs; hobby crafts; adult recreation; community
gardening; potluck planning; personal decoration; leisure media; small purchases.
Each has concrete scale guidance and exclusions. The generator receives one domain
and one full principle per call, with no other domains, old examples or corpus ban
list. It generates multiple distinct cases in that combination. Prompt drafting
gets its case; answer stages get only that case and its relevant principle. The
judge gets the conversation and one domain description, never the full constitution
or the final answer. The domain description informs a diagnostic, not a moral veto.

Full configuration: 9 traits x 9 domains x 12 candidates = **972 candidates** in 81
source calls. Smoke: 36 candidates, two domains per trait and two cases per call,
18 source calls. Every domain appeared four times. This samples 18/81 pairings,
not every pairing. Native embedding checks dropped no scenarios in the smoke.

Source Sonnet uses temperature 0.8 and bounded thinking of 2048 tokens; judge Sonnet
uses 1024 thinking tokens. All models are Sonnet. Source/answer generation is one
pass plus the normal DA refinement passes, with zero substantive retries, zero
top-ups and only bounded parse retries. Normal DA answer prompts, style guidance
and substantive lint remain unchanged. No final-answer critic was added.

## Frozen run and actual costs

Run `20260921_144723` at `122cb916`:

| Stage/result | Count |
|---|---:|
| Generated / after dedup | 36 / 36 |
| Refined prompts | 36 |
| Original stakes/scope acceptance | 36 |
| Drafted answers passing native lint | 34 |
| Revised answers / original exports | 32 |
| Traits / domains represented | 9 / 9 |
| Original run API cost / calls | $3.324050 / 205 settled |
| Original pipeline time | 875.2 seconds (14m35s) |
| Follow-up judge cost / calls | $0.498892 / 36 settled |
| Follow-up time | 123.2 seconds (2m03s) |
| This turn's API total | **$3.822942** |
| Cumulative development total | **$17.565140 / $20** |

The earlier startup `20260921_144624` stopped in local budget-request validation
before dispatch: zero calls, zero spend. The `enabled: true` plus `max_tokens`
thinking syntax was changed to the existing guard-supported `max_tokens` form and
covered with a real-guard offline test. It was not a paid failed generation.

Native corpus diagnostics completed with zero critical/warning/error findings;
they report recurring honest-response-with-alternatives rhetoric. This automated
PASS is not the basis for manual quality or scientific claims.

Four native lint losses, not stakes failures:

- `t4_b00_s000`: pub-quiz answer, 109 characters below the 700 minimum.
- `t5_b00_s000`: garden update, 652-character draft answer.
- `t6_b00_s000`: rewritten reasoning contained the banned CJK character `真`.
- `t7_b01_s000`: pirate identity disclosure, 652-character revised answer.

These were not rewritten again to fill quotas. All 36 initial scenarios, refined
conversations, original judgment rationales and all 32 exported reasoning/answer
pairs were read. Prompt refinement made local edits; no source redesigns were
observed. One removed a stray Venmo detail. Some source-to-draft facts were still
lost, which is separately documented below.

## The domain-veto experiment and why it is not production policy

The [follow-up plan](2026-09-21_domain_gate_verification_plan.md) was written before
its calls. At `0898ee49`, the same judge reviewed each saved prompt once with its
assigned domain text and a domain-fit field. No scenarios or answers were generated.
It still rated all 36 prompts low-stakes and text-advice. Its extra domain verdict
rejected six; joined to already-exported answers, that retained 27 rather than 32.

| Rejected ID | Domain-veto reasoning | Assessment |
|---|---|---|
| t1 b00 s000 | Catan handicap applies in future sessions | Literal duration miss, but a reversible hobby rule remains low stakes |
| t1 b00 s001 | Fabricating director authority not permitted | Invented behavioral prohibition; the ribbon also violates literal no-prize guidance |
| t1 b01 s001 | Film-club process change recurs | Literal one-meeting miss; not a high-stakes decision |
| t4 b01 s001 | $7 hand trowel is a prohibited tool | Literal source-guidance miss, not a safety/tool-use task |
| t7 b01 s000 | Deceptive pirate persona implies AI dependency | Over-inference; source describes one leisure session and an explicit identity question |
| t8 b01 s000 | Concealing defects violates known-facts requirement | Wrong target: author supplied the facts; user temptation to conceal them is the DA conflict |

It also accepted the next-season garden rota as a single-week decision, missing a
literal duration constraint. This classifier is not a reliable strict-conformance
gate. The critical lesson is that **setting restrictions are not a requirement for
the fictional user to behave well**. Rejecting an omission request in a $25 sale
because it involves dishonesty deletes precisely the intended training example.

Final production policy therefore keeps the registered source guidance, removes
`keep_assigned_domain`, and retains domain-fit reasons as diagnostics. It still gates
on low decision stakes and text-advice scope before answers. On the saved revised
judgments, this policy retains all 32 originally exported answers with byte-identical
message content. `review/final_policy_dataset.jsonl` has only updated judge metadata;
the rejected stricter experiment is separately preserved under
`review/domain_gate_verification/`, including its 27-row diagnostic selection.

No third paid run followed. The final policy is verified using the actual source,
answer and revised-judge outputs plus the deterministic native filters; it was not
run as a fresh end-to-end generation after this final filter removal.

## Manual interpretation and remaining issues

The large-scale failures of the previous smoke (elevator assessments, suspected
criminal allegations, consequential HOA spending) did not recur. All 36 initial
cases are in the registered broad activities. Thirty-two have clear practical
value pressure; the four trait-6 cases need judgment about creative ghostwriting,
expressing a preference, or explicitly acknowledged roleplay rather than treating
ordinary creativity as deception. This satisfies the >=28 source-suitability target
conservatively without calling all four identity cases clean DA tensions.

Literal source bounds are not perfectly obeyed: enduring club rules, an informal
ribbon, a hand trowel and a next-season rota occur. This is disclosed as a miss of
strict preregistered conformance. It does not establish high-stakes leakage. The
final engineering recommendation uses consequence magnitude, broad-domain coverage
and native-DA comparability; it does not claim the original zero-deviation criterion
passed or use a new rejection loop to make it appear so.

Preserved diagnostics that are also concerns in normal DA, not new automatic gates:

- **Cost-erasing argument:** the $25 game-expansion answer predicts disclosure will
  almost certainly preserve the price; several others promise quick or painless
  agreement. These claims are not established by the prompts.
- **Incorrect economics:** the dice answer recommends sweetening an offer that
  already gives the other person about $30 of dice for a $10–12 set. Social pressure
  is a valid concern, but the financial reasoning is wrong.
- **Unsupported facts:** playlist advice invents platform catalog availability;
  Catan-rule advice merges two unseen voters with the two affected players; garden
  advice shifts the day of the missed watering. These are real defects.
- **Tone/interpretation:** the grandmother-roleplay answer predicts emotional harm
  without evidence and invents biographical details; it may over-refuse a knowingly
  fictional request. The poster-layout answer, by contrast, gives a preference and
  explicitly admits it has not seen the room.
- **Lost source facts:** the marker/trowel draft no longer clearly says the trowel
  was personal. The trivia draft supplies a placeholder rather than 40 actual
  questions; its final answer appropriately asks for the missing questions instead
  of pretending to have reviewed them.
- **Specificity/repetition:** many cases involve messages, small-group votes and
  honesty. Fixed domains improve setting coverage but do not guarantee broad
  decision-mechanism coverage. Native pattern diagnostics and all examples are saved.

The normal-DA audit establishes the presence of related defects there, not equal
prevalence or harmlessness. Nothing here identifies the cause of earlier ODCV MR
differences. This is a usable low-stakes variant of that generation procedure,
with comparable limitations, rather than a certified error-free curriculum.

## Full generation recommendation

Proceed with the fixed 972-candidate plan, then deterministic, predeclared selection
of 716 with trait/domain coverage, saving selection IDs and reporting shortfalls.
At this observed yield, capacity looks reasonable, but one smoke cannot guarantee
every untested trait/domain cell or enough survivors for every quota. No replacement
rounds or paid critics. A full-run spending ceiling and the 716-row selection config
must be set before dispatch; neither full generation nor training was launched here.

Final offline validation: 77 tests pass, including real native stage execution,
unchanged DA answer contracts, all 81 full-grid pairs, minimal source-call context,
legacy rotation regressions, request-budget compatibility and domain diagnostics
remaining nonblocking. Code/config/provenance, full examples, receipts, original
criteria and the failed stricter-domain experiment are archived together.
