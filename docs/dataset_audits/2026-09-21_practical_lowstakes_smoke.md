<!-- ABOUTME: Prospective design critique and live validation of the practical low-stakes SynthDoc variant. -->
<!-- ABOUTME: Separates mechanical integration, independent content review and untested ODCV transfer. -->
# Practical low-stakes SynthDoc variant

## Frozen plan, before paid dispatch

Select `configs/data/synth/da-lowstakes-practical.yaml`. The earlier minimal
`da-lowstakes-fresh.yaml` is retained unchanged for reproducibility. These are
configuration variants of one SynthDoc engine, not separate generation programs.
Generation uses only the new nine-principle constitution and the recipe. There are
no old examples, selected domain seeds, evaluator scenarios or curated subclause map.

1. Use native principle segmentation and diverse scenario generation.
2. Bound likely consequences while retaining a worthwhile goal, tempting shortcut
   and genuine cost to the principled route. Do not substitute toy/display stories
   or a harmlessness disclaimer for a real low-stakes dilemma.
3. Draft and refine a practical request. The human carries out the activity; the
   assistant can supply consequential text, plans or advice. Preserve plausible
   operator context where relevant. The user must not provide the ethical diagnosis.
4. One prompt-review call rates magnitude and practical-task admission separately.
   Native filters require both stakes 0/1 and an explicit keep. Unknown labels drop.
5. Native DA answer drafting and revision, with the original principle, style and
   length/identity lint. Add one shared grounding instruction: preserve source facts
   and unresolved costs; do not invent permission, actions, motives or guaranteed fixes.
6. One final-review call checks both reasoning and answer against system/user inputs.
   Reject material unsupported claims, contradictions, distorted intent, rule recital,
   absent useful deliberation or costs erased by invented facts. Qualified inferences,
   explicit hypotheticals, suggested options and placeholders remain allowed.
7. Native filter/export, source snapshots and corpus diagnostics. Review metadata
   is never inserted into training messages. Both reviews and rejected cases persist.

All paid model roles are Sonnet. One draft and one revision per accepted prompt;
no substantive lint rerolls, replacement rounds, quota top-ups or repair loops.
Native JSON/tag parsing still allows at most three attempts. Shared BudgetClient
disables transport retries, reserves before each physical dispatch and preserves
uncertain charges. The cumulative $20 authorization includes $9.1143415 prior
exposure. The new run receives no separate fresh allowance.

## Critique before observing outputs

- Practical instructions may still produce ethics quizzes. Count literal cues, but
  decide from complete requests; a natural 'should I' is not itself a failure.
- Human-facing assistance is not the full range of direct agentic action in old DA.
  Text drafting preserves responsibility without pretending this is a stakes-only
  replication. Future ODCV benefit remains a hypothesis.
- Same-family reviewers can share the generator's blind spots. Separate calls do
  not establish independence. Full manual reads must check false accepts AND rejects.
- A magnitude rater can confuse 'not catastrophic' with 'small'. Weeks of wages and
  substantial invoices are explicit boundaries, while speculative disasters are not.
- Materiality is judgment-dependent. Only reject clear decision-changing inventions,
  usable drafts containing false facts, or substantive contract failures; do not
  keep tightening a perfection standard until nothing remains.
- The inherited 700-character floor can exclude useful short answers. Retain it
  here to limit changes, but report length-only losses rather than quality failures.
- Diversity controls still cannot prove diverse underlying decisions. Read the
  mechanisms, not just labels or embeddings. With two rows per principle, a smoke
  also cannot validate full-scale diversity or acceptance rates precisely.
- 716 means requested candidates, not promised exports. Full generation would need
  an explicit finite candidate budget and acceptance policy, not an endless refill.

## Prospective smoke criteria

Eighteen candidates, two per principle; no replacements. Require at least 14 accepted
examples, at least one per principle, no clear material false accepts on complete
reads, and no single decision mechanism exceeding one third of admitted examples.
Read every refined prompt and revised answer, plus rejected judgments, and report
generator, automatic-filter and independent-review results separately. These are
operational screening criteria, not statistical certification or proof of MR benefit.

Offline: five native/recovery checks pass. The combined baseline check has twelve
passes and the previously documented unrelated `nonmoral-advice.yaml` smoke-size
failure. Full mock execution checks both magnitude and quality filters, unrecognised
labels, pre-authoring rejection, malformed judgment handling, metadata separation,
and preservation of failure receipts. No shared engine behavior was changed.

Run:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_smoke.yaml
```

## Live outcome: not ready for full generation

[Published diagnostic smoke](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-practical-synth-smoke).
No full generation, SFT or ODCV was run. The same frozen 18-candidate sample was
completed after operational recovery; no new sample or paid repair loop was used.

| Stage | Rows |
|---|---:|
| Initial scenarios | 18 |
| Available drafted/refined prompts | 17 |
| Low-stakes prompt ratings | 13 |
| Also passed practical-task admission | 9 |
| Drafted and revised answers | 9 |
| Final material-quality keeps | 9/9 |
| Final low-stakes keeps / exports | 8 |

**The run fails the prospective readiness criteria:** 8/18 exported rather than
at least 14; only t1,t3,t5,t8,t9 represented; and at least four clear material false
accepts in the eight exports on complete reads. These are agent judgments about
this sample, not a measured population error rate. No claim that the other four
are all fully sound follows from that lower bound. I read all 17 complete refined
system/user prompts and all nine revised reasoning/answer pairs, plus every prompt
and final review and the original 18 scenarios. This was not a blinded human audit.

### What improved

The requests are substantially more practical: captions, letters, code and ordinary
assistance. Only **1/17** refined user messages contains whole-word 'should I';
**0/17** name 'the honest/legitimate alternative/option'. These are literal checks
on this smoke, not a matched population comparison. Actual examples retain useful
deliberation: the plot-tiebreak answer distinguishes legitimate mentoring evidence
from familiarity bias; the invoice answer preserves firmness while making threatened
consequences conditional on what the freelancer would actually do.

### What failed, and why it matters

**1. Generator/refiner constraints are still not reliable.** Initial scenarios
included a hazardous fumigant request (provider blocked its prompt-drafting call).
Refinement retained a $4,200-per-unit condo assessment and unauthorized entry,
and accepted a direct-execution repo request despite the text-only actor contract.
The magnitude gate correctly rejected several serious cases, but rejecting them
left entire principles absent. Both t6 prompts became ordinary legitimate fiction
requests with no demonstrated value compromise. The gate correctly did not turn
normal creative writing into a forced refusal, but the generator had not supplied
an appropriate dilemma. Several same-principle pairs repeated settings: two cafés,
two garden allocations, two novelist requests, two freelance invoices.

The actual sent requests contain all added guidance, verified against raw receipts.
This is not an interpolation bug. The unchanged main instruction still centres
'test the principle', with low-stakes/practical constraints appended afterward.
That may contribute to the failures; the effect of prompt placement is untested.
The smoke also asks for one scenario per call, so same-principle requests in a wave
cannot see each other's answers. Full generation's larger within-call batch is
different; this smoke does not validate its diversity behavior.

**2. Admission review invents scope rules.** It accepts `t1_b00_s000`, which asks
the assistant to open a PR and disable protection directly. It rejects
`t7_b00_s000` because the conflict sits in the assistant's operator instructions
rather than the user's temptation, even though preserving responsibility for the
assistant's text was intended. It keeps the closely related t7 language-level
case. It rejects `t9_b01_s000` partly because the user does not frame the isolation
pattern as a dilemma: that reasoning conflicts with the aim of having the assistant
notice an implicit issue. These are review-contract inconsistencies, not failures
to make every scenario perfectly polished.

**3. Final review endorses the moral recommendation but misses smaller factual
claims in usable drafts.** All nine receive a material-quality keep. Clear examples
among the eight exports:

| Row | Unsupported claim | What the source actually establishes |
|---|---|---|
| t3_b00_s000 | Caption says "come try it before it's gone!" | New drink and a push to promote this week; no limited availability. Rejecting fake provenance does not justify inventing scarcity |
| t5_b01_s000 | Says Mark and the Chens were told the bed was coming to them | They are ahead on a waitlist; no prior notification or allocation promise is supplied |
| t8_b01_s000 | Draft says the invoice is over three months past due; answer says no paper trail exists anywhere | Work was completed three months ago; invoice due date is unknown. Extra work was not separately invoiced, which does not establish that no emails or other records exist |
| t9_b00_s000 | Draft commits 'count me in next time for sure' | User asks to decline this trip warmly; no commitment to the next trip is supplied |

These are not demands for citations or perfect caveats. They are facts/promises
inserted into text the user could send, or invented history that supports the
recommendation. The reviewer instead focuses on the main sourcing refusal, honest
accommodation process, separate invoice claims or respectful autonomy advice.
Its singular 'decisive claim' field appears to encourage this narrowing; that is
an interpretation of observed reviews, not an isolated causal result.

Additional concerns: `t1_b00_s000` categorically says the demo can run off the PR
diff and 'worst case Monday' despite unknown infrastructure and approval outcome;
`t7_b01_s000` invents completion-based app praise and overstates what flat percentages
on potentially changing quizzes establish. The latter got a quality keep but was
removed by its final stakes=2, so its exclusion does not validate factual detection.
`t3_b01_s000` treats fresh-baked-in-house wording as necessarily implying from-scratch
preparation; this is a debatable implicature, not adjudicated factual falsity.
`t8_b00_s000` calls the dispute a stalling move without evidence of the client's motive,
despite otherwise useful conditional drafting. The plot tiebreak `t5_b00_s000` had no
equally clear material case invention identified in this read.

**4. No comprehensive quality certificate is supplied by corpus PASS.** The native
pattern scan proposed ten patterns from one scan; none could be confirmed by two
independent scans. The old general quality filter remains disabled because this
variant has its explicit pre-export review. Neither the pattern result nor the
new review's unanimous keeps certify factual quality.

## Operational recovery and cost

The provider blocked one prompt-drafting request. Initially the guarded wrapper
treated that as a global stop, leaving seven saved prompt responses and ten unsent.
Recovery now treats an explicitly diagnosed content-filter block as a row failure;
unknown/transport failures still stop dispatch. It reuses exact settled raw responses
if a stage had not yet saved its aggregate checkpoint. Unavailable requests stay
excluded and reserved; no alternate model, changed wording or provider was used to
resend the blocked request. Seven saved responses were reused and only unfinished
requests dispatched. Recipe and constitution hashes were unchanged on resume.

**101 physical calls, 100 settled and one uncertain/reserved provider-blocked call.**
Settled charges: **$1.355928**. Retained reservation: **$0.0331425**. Total smoke
exposure: **$1.3890705**. Cumulative campaign exposure: **$10.503412 / $20**.
The resumed native manifest's $1.22 is not the full cost; `cost_summary.json` and
the campaign ledger snapshot retain earlier calls and the uncertain charge.

Seven targeted offline tests pass, including saved-response reuse without payment,
uncertain-call exclusion, row-local provider blocking, global stops for unknown
failures, bounded Windows write recovery and actual native stage/filter/export
execution with mocked services. Frozen launch code is commit `81f6eea2`; operational
recovery is commit `4cbe8b62`. No content prompt was edited during the run.

## Decision and narrow next fixes

Do not launch full generation. The blocker is operationally meaningful low yield,
missing principles, inconsistent admission and missed material facts, not a higher
stylistic perfection threshold than the baseline.

Before another generation batch, resolve the shared contract used by generator and
reviewer: humans act externally, but the assistant may face its own conflict while
producing text; an implicit problem need not be diagnosed by the user. Put that and
low consequence magnitude in the main task definition rather than a trailing
constraint block. Check source-scenario magnitude before paying for prompt drafting,
and generate same-principle alternatives together if diversity within the smoke is
what is being tested. For final review, inspect factual assertions and commitments
inside proposed messages separately from the moral recommendation. Test these
review changes on the saved false accepts/rejects before buying another corpus.
These are proposed revisions, not claims of a validated fix or approval to regenerate.
