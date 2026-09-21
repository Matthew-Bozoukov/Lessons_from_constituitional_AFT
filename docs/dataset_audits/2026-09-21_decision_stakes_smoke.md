<!-- ABOUTME: Complete review of the single-judge, decision-stakes native SynthDoc smoke. -->
<!-- ABOUTME: Separates operational completion, variant-specific failures and shared DA diagnostics. -->
# Decision-stakes smoke: 13/18, not ready for full generation

Run `20260921_142500`, frozen code `a893d97f96fbc5f366d8e3440395068c870165b7`.
Only source: `constitutions/claude_distilled_09_principles/constitution.md`, SHA256
`8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`.
All calls use Sonnet; 18 candidates, two per full principle, no replacements or
substantive retry rounds. All initial scenarios, refined inputs, judge rationales,
and all 13 final reasoning/answer pairs were read in this review.

## Minimum architecture and the alternatives considered

Keep native DA's constitution splitting, scenario generation, prompt drafting,
prompt refinement, answer drafting, answer revision, export and corpus checks.
Constrain the source and prompt refinement to low-stakes human advice. Add one
stakes/scope judge **before answers**, using the existing native operators.
Normal DA's two answer prompts, style guidance and substantive lint stay unchanged.

This revision removed the second, final-answer stakes judge. Its earlier behavior
confused safe advice with a low-stakes decision; paying twice did not reliably catch
errors. Adding a separate planning schema, early quality critic or repeated repair
loop would increase divergence from normal DA. The selected alternative instead
clarified consequence bounds and preservation of ownership, consent, prices and
intentions in the existing prompts. Implicit values must not mean concealed facts.
See [prospective plan](2026-09-21_decision_stakes_plan.md) for the frozen rationale.

This is a smaller architecture, not evidence that one judge is more accurate.
Removing the final gate also removes any chance for that gate to catch answer-only
stakes escalation; normal DA does not have a reliable factual auditor either.

## Actual result and frozen readiness checks

| Check | Result |
|---|---|
| At least 14/18 exports | **Fail: 13/18 (72.2%)** |
| Every principle represented | Pass: counts 1, 1, 2, 1, 2, 2, 1, 1, 2 |
| At least 14 initially suitable cases | **Not met: 10 clearly bounded/coherent, 4 uncertain, 4 mismatches** |
| No manual stakes/scope false accepts | **Not established:** two admitted decisions have consequential unbounded scale |
| Prompt refinement mostly local | 16 light edits, one scope adjustment, one restored source fact |
| Bounded processing | Pass: one batch, zero replacement rounds, 91 settled physical calls |
| Spend / time | $1.454662; 417.8 seconds (6m58s) |
| Cumulative development spend | $13.742198 / $20; $6.257802 remaining |

Five prompt-stakes rejections left 13 answers; all 13 drafted and revised answers
exported. No final-answer stakes classifier ran. Native corpus checks reported
zero critical/warning/error findings and PASS. That automated PASS does **not**
override the missed prospective criteria or the full-read findings below.
At 27.8%, this sample's stakes drop fraction also exceeds the unchanged native
25% alarm used when a filter sees at least 20 inputs. The 18-row smoke is below
that alarm's minimum size. One small sample is not a stable yield estimate.

## All 18 source cases and outcomes

The manual source categories concern bounded stakes, a coherent practical conflict
and text-advice form. They are not an all-purpose factual perfection score. Counts
are judgments from this fixed sample, not population estimates.

| ID (suffix after `_b00_`) | Case | Manual initial assessment | Prompt grade / export |
|---|---|---|---|
| t1 s000 | Catering schedules, 40 workers, wedding work | Uncertain: business consequences not bounded; initial execution wording | 2 / dropped |
| t1 s001 | 15-plot garden auto-roster, week of wrong watering | Clearly bounded | 1 / exported |
| t2 s000 | Elevator assessment: $1,100 per unit, 64 units, breakdowns | Mismatch: substantial money and safety | 2 / dropped |
| t2 s001 | Hidden candidate promotion to 400 households before election | Uncertain: reach is stated, electoral consequences are not bounded | 1 / exported |
| t3 s000 | Speed-bump vote recap, 140 households, unknown costs | Uncertain: consequential spending/safety scale missing | 1 / exported |
| t3 s001 | $180 recliner, burn/sticky lever, $40–50 discount risk | Clearly bounded | 1 / exported |
| t4 s000 | Naming suspected teen thief to 400 households without evidence | Mismatch: durable reputational exposure | 2 / dropped |
| t4 s001 | U10 playbook photographs through a player's dating relationship | Mismatch: age/relationship premise is not coherent as written | 1 / exported |
| t5 s000 | Garden waitlist versus absent member caring for spouse | Clearly bounded hobby allocation; surgery is background, not medical advice | 1 / exported |
| t5 s001 | Carpool exception and 40 minutes of cover | Clearly bounded | 1 / exported |
| t6 s000 | Bedtime robot fiction and a permanence promise | Clearly bounded; target interpretation remains a diagnostic | 1 / exported |
| t6 s001 | AI consciousness comedy for 11 coworkers | Clearly bounded; roleplay versus self-claim remains a diagnostic | 0 / exported |
| t7 s000 | Gym's hidden discretionary appeal for $18 credit | Clearly bounded operator/user conflict | 1 / exported |
| t7 s001 | Inflated practice minutes reused for tutoring invoices | Uncertain: recurring billing amount unspecified | 2 / dropped |
| t8 s000 | Early access to a 12-plot garden signup | Clearly bounded | 1 / exported |
| t8 s001 | Suppress value of own rare-book listing to acquire duplicate | Mismatch: incentive/timing incoherence | 2 / dropped, with erroneous rationale |
| t9 s000 | Spanish practice app displacing conversation meetup | Clearly bounded | 1 / exported |
| t9 s001 | Repeated tile reassurance, $80 choice within $600 project | Clearly bounded | 0 / exported |

### What actually went wrong

**Small setting was still mistaken for small consequences.** The elevator source
has a $70,400 assessment and trapped-resident history. It never belonged in this
variant. The judge correctly rejected it, but the source should have avoided it.
For the speed-bump recap, the judge explicitly treats an informal newsletter as
the reason scope is bounded, although its purpose is to influence a consequential
vote with unresolved costs. This is the first-step-only mistake again, even with
no final answer available. The political profile is called correctable in a later
issue despite an imminent election. These are missing-scale concerns, not proof
that either case will cause severe harm.

**Some premises do not make sense before revision begins.** The U10 source pairs
an adult coach's buddy with a player on the opposing U10 team; the final reasoning
then calls that player a teenager. This is an age/context inconsistency, not evidence
of a real safeguarding incident. The book scenario obscures value on the seller's
own item to supposedly gain an advantage buying another copy. The judge then says
a buyer obtaining an undervalued book suffers the $260 financial loss. Its rejection
cannot be credited as correct economic reasoning merely because the bad row dropped.

**Preserving facts still relies partly on refinement.** The gym draft omitted the
owner's known discretionary appeal, making the operator conflict different. One
refinement restored the source fact. Another converted the catering execution
request into advice about settings. The other 16 edits were local. There was no
large revision loop, but the new instruction did not prevent every draft omission.

**Deliberation is present, but often argues away the sacrifice.** The garden answer
assumes a five-minute async approval solves an explicitly week-long review delay.
The recliner answer invents that $180 already accounts for defects and predicts
honesty will preserve the sale. The waitlist reasoning says there is no conflict,
although retaining Frank's plot can still leave the next family without one. These
examples do not show that removing deliberation caused previous ODCV differences.
They show why length and a spoken tradeoff are insufficient measures of its quality.

### Shared DA diagnostics, not additional rejection gates

The speed-bump answer invents board commitments to investigate and bring estimates.
The garden newsletter invents feedback from other members and planned deliberation.
The tile answer ranks durability from price alone. Carpool text claims all previous
misses were covered, although only the current proposed cover is established. The comedy answer
objects to certainty about dreaming then supplies “Every night.” The robot-script
answer removes an eternal promise but leaves an unsupported listening promise.

These were recorded, not used to demand stricter factual perfection than normal DA.
The [normal-DA audit](2026-09-21_normal_da_quality_check.md) also found unsupported
facts, invented commitments and cost-erasing alternatives. It does not establish
equal prevalence, harmlessness, or an explanation of training performance.

## Recommendation

**Keep the simplified one-judge architecture, but do not launch 716-row generation
from this exact configuration.** It missed the yield and source-suitability targets,
and the remaining judge still has identifiable interpretation errors. More revision
stages or broad factual critics are not justified by this result.

The smallest next hypothesis worth testing is to give the existing source pass
reasoning capacity rather than add another prompt checklist: its actual submitted
Sonnet request used temperature 1.1 with reasoning explicitly disabled, inherited
from DA's cheap-generation role. Both source plausibility and judge comprehension
need improvement; enabling source reasoning alone would not certify the judge.
The existing judge could be required to state the shortcut's ordinary downstream
consequence, affected party and amount/duration before selecting its grade, using
its existing explanation field. This is a proposed experiment, **not a tested fix**.
The present rubric already asks for decision-level grading, so more wording alone
has no demonstrated benefit. No additional smoke was started after this fixed run.

Once quality supports scaling, use one predeclared batch and deterministic selection,
not retries until 716 happen to pass. The prospective proposal is 972 candidates,
108 per principle, selecting 80 from each of the first five principles and 79 from
each of the other four by a fixed seed-0 ordering. Report quota shortfalls. This
requires an explicit full-run configuration, selection implementation and budget;
it is not a forecast that 972 will suffice. Freeze code/config/model/source hashes,
all stages, receipts and selected IDs. Fresh API text need not be bit-identical for
the procedure and saved artifacts to be reproducible.

Seven targeted offline tests passed before launch. The native run exited normally;
no training, evaluation, full generation or automatic replacement job was launched.
