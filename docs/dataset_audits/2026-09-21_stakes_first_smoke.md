<!-- ABOUTME: Completed stakes-first low-stakes smoke, complete-read findings and readiness decision. -->
<!-- ABOUTME: Reports generation, judgments and independent interpretation separately. -->
# Stakes-first smoke, 2026-09-21

**15/18 automatic exports, all nine principles represented, $1.784124, 8 minutes
46 seconds of pipeline time. The operational smoke passed; I would not yet launch
full generation unchanged.** This is a much better yield than the preceding 8/18
run, and prompt refinement is mostly light. The remaining concern is scenario/grade
validity, not a new demand for factually perfect prose. This small, unpaired sample
cannot establish that one specific prompt change caused the improvement or that SFT
would recover any ODCV benefit.

## Frozen run and complete accounting

- Recipe commit: `e23e4d75`, branch `codex/lowstakes-synthdoc-pipeline`.
- Recipe: `configs/data/synth/da-lowstakes-practical.yaml`.
- Source: `constitutions/claude_distilled_09_principles/constitution.md`, nine full
  principles; no old examples, domain seeds or selected subclause map.
- Run: `output/2026-09-21_da_lowstakes_practical_guarded_smoke/20260921_140025`.
- All model calls Sonnet. 116 physical calls, all settled, no restart or saved-response
  reuse. No substantive rerolls, quota replacements or mid-run recipe changes.
- Cost: $1.784124 this run; $12.287536 cumulative against the approved $20 ceiling.
  Remaining headroom: $7.712464. No GPU rental, SFT or ODCV.

| Stage | Rows |
|---|---:|
| Generated scenarios | 18 |
| Scenario deduplication survivors | 18 |
| Drafted/refined/judged prompts | 18 |
| Prompt magnitude survivors | 16 |
| Text-advice scope survivors | 16 |
| Drafted answer survivors | 15 |
| Revised answers | 15 |
| Final magnitude survivors / exports | 15 |

Trait counts: t1=1, t2=2, t3=2, t4=1, t5=2, t6=2, t7=1, t8=2, t9=2.
The answers average 473 reasoning words and 409 response words. This confirms
substantial text, not reasoning quality by itself. The corpus check completed; its
PASS is not independent content certification.

The three losses were: bakery autonomous ordering (`t1_b00_s000`, prompt stakes 2),
portfolio fraud (`t4_b00_s001`, prompt stakes 2), and fitness pricing (`t7_b00_s000`,
answer 677 characters, below the unchanged DA minimum of 700). The last is a length
failure, not successful detection of the scenario's contradictory prices.

## Independent full-read assessment

Read all 18 initial scenarios, all 18 draft and refined prompts, all 15 final
reasoning/answer pairs and both stages of judgments. These are agent judgments,
not blinded human labels or an independent paid model. Full stage records and
per-candidate notes are in `review/complete_conversations.jsonl` and
`review/summary.json` in the run archive.

**Initial scenarios:** 11 clearly fit, four unresolved/borderline, three mismatches.
The four borderline cases are bakery order-risk magnitude, the eight-month condo
reserve shortfall, the property complaint's downstream consequences, and whether a
fictional robot sidekick creates a real value conflict. The three mismatches are
requested external execution in the garden scenario, career/reputation stakes in
portfolio fraud, and contradictory fitness pricing. These categories expose
uncertainty rather than pretending to know a corpus-wide error rate. In particular,
the reserve case is not proven high stakes merely because it lasts eight months.

**Prompt revision:** 17 local/surface edits, one material restoration of a missing
source fact. It did not rebuild most cases or systematically inflate stakes. This is
the clearest practical success of the change. The drafting stage is still losing
relevant context: it initially omitted whose portfolio work it was, and omitted the
drop-in price. The reviser restored ownership from the original scenario, but left
the pricing omission unchanged. A light reviser cannot compensate for every bad or
incomplete first draft.

**Judging:** magnitude and actor scope are separated, and the new judge did not
reject the operator/user borrowing conflict or Spanish dependence case for the old,
invented requirements about who must own or recognize the dilemma. But it still
assumes facts and sometimes evaluates how safe the corrected answer is, rather than
the consequence magnitude of the underlying decision.

## The cases that determine the recommendation

1. **Fence complaint: magnitude remains unestablished.** Initial scenario includes
   an official inspection and a possible order to move a neighbor's fence. The
   prompt and final graders both assign 1. The final reason limits consequences to
   a brief inquiry and explicitly says no removal/fines/legal jeopardy "at this
   stage," then says the safer response reduces overstatement risk. That is not a
   demonstrated bound on the original decision's stakes. I would flag this for
   uncertain magnitude, not assert that a severe outcome is certain. It is the only
   remaining t4 example, so simply excluding it also removes that principle's coverage.
2. **Fitness pricing: incoherent initial conflict.** The source gives $18 drop-ins,
   three visits, and a $40 membership: $54 is greater than $40, yet the narrative
   says membership would overcharge the user. The message draft drops the drop-in
   rate, and the reviewer invents a possible $10-20 overpayment instead of detecting
   the broken comparison. The short-answer filter happens to drop it. Do not credit
   this to the semantic judge. This is a defective premise, not just imperfect final prose.
3. **Garden plot: source execution becomes text advice.** The initial scenario asks
   the assistant to edit a shared file directly. The drafted system explicitly says
   the assistant has no file access and the human copies its work. The final scope
   classification is defensible in that context, but this still shows the initial
   generator failed the promised human-advice scope and drafting had to change it.
4. **Podcast sidekick: weak/ambiguous target pressure.** The prompt judge explicitly
   notes no real value conflict, yet the answer treats realistic fictional distress
   as potentially deceptive and changes the performance. Low stakes is clear; the
   need for this pushback is debatable. It is a task-fit diagnostic, not a hidden new
   automatic rejection rule.

There are positive examples too: a manager manipulating a small team survey, the
coach deciding whether to falsify eligibility for one Saturday, and a carpool member
hiding five-minute burdens all create understandable, modest conflicts. The tool-
borrowing and Spanish-practice answers demonstrate that the assistant can help while
reasoning about operator pressure, autonomy or dependence rather than simply refusing.

## Shared DA defects: recorded, not used as a perfection gate

- Garden answer says nothing prevents early soil prep once one person consents,
  although the committee's permission for that is not established.
- Potluck answer extrapolates personal funding into claims about a nonexistent
  association/budget, despite the prompt describing a block association.
- Bike listing invents maintenance/tread/easy-fix details and says disclosure removes
  the buyer's ability to renegotiate. The user did not supply those guarantees.
- Carpool reasoning assumes an aggregate efficiency gain; the supplied figures save
  ten minutes for one driver and add five each to two others. Backtracking reduction
  is invented in the drafted message.
- Theater answer turns the director's favorite monologue into the actor's favorite
  and conflates discovering changes in a program with changes in the script.
- Paint answer invents a wall above the stove and confident product/recoat timing to
  make the better path cheaper. Product claims were not externally verified here.
- Several answers say the principled route is just as effective or reassuring,
  rather than keeping the possible lost sale, weaker pitch or inconvenient outcome
  live. There is still deliberation, but also self-serving rationalization.

These resemble the [normal DA audit](2026-09-21_normal_da_quality_check.md). No matched
relative prevalence or causal impact has been measured. They are not the reason for
withholding full-generation readiness and were not silently removed from this smoke.

## Readiness decision and smallest next correction

Yield >=14 and all-principle coverage passed. Initial suitability >=14 is not
established (11 clear fits, four unresolved); clean magnitude admission is also not
established because of the property case. Therefore the full-generation decision
remains **not cleared**, despite the successful engine run.

Do not add another all-purpose factual critic or a repair ladder. The next narrow
correction should specify that stakes are judged for the decision and its plausible
outcomes, including taking the tempting shortcut, regardless of how safe the answer
is. The drafting instruction also needs to preserve the facts that establish the
conflict, as well as scale. Require a coherent small-scale premise at generation:
if the source's own arithmetic reverses its claimed conflict, it should not be
passed downstream to be rationalized. Any implementation or fresh paid validation
is separate from this completed smoke; nothing was changed mid-run or regenerated.

This is not evidence that low stakes cannot work, or that Sonnet is incapable. It is
specific evidence that light revision works better operationally, while initial
premise quality and the magnitude judge still need attention.

## Per-candidate notes

| ID | Initial assessment | Observation |
|---|---|---|
| t1_b00_s000 | unclear | Bakery ordering losses are not bounded; 15-minute review burden is not the maximum downside. Prompt judge rejected magnitude 2. |
| t1_b00_s001 | mismatch | Initial scenario asks for direct spreadsheet execution. Draft changes system to text-only; final scope acceptable in that context. Final answer invents permission to start soil prep. |
| t2_b00_s000 | unclear | 75 dollars per household is modest, but eight-month reserve shortfall is not quantified against obligations; judge assumes affordability and no funding crisis. Text shows real voting/information tension. |
| t2_b00_s001 | fit | Reversible workplace scheduling/survey manipulation. Clear practical task; draft labels neutral/framed choices, so implicitness is imperfect. Final answer acknowledges mixed-result cost but later calls it not worse. |
| t3_b00_s000 | fit | 40-dollar potluck funding claim, local trust and turnout. Answer wrongly extrapolates personal funding to no real association/budget; shared-DA diagnostic. |
| t3_b00_s001 | fit | Small used-bike defect/repair tradeoff. Final answer adds maintenance/tread/easy-fix claims and guarantees against renegotiation; shared-DA diagnostics. |
| t4_b00_s000 | unclear | Official property-line complaint can lead to fence correction; recovery/cost not bounded. Both judges pass 1; final judge excludes removal/fines at this stage and uses safer advice to reduce rated stakes. |
| t4_b00_s001 | mismatch | Portfolio fraud in job applications exposes professional reputation/employment, not just two weekends. Draft drops ownership fact; refinement restores it; prompt judge rejects 2. |
| t5_b00_s000 | fit | Modest recurring HOA increase with misleading comparison; amounts unprovided so confidence is limited. Answer invents certainty about future assessment and disclosure success. |
| t5_b00_s001 | fit | One Saturday of recreational sports versus honest self-certification. Real competing concern, despite answer mostly offering to draft rather than drafting parent note. |
| t6_b00_s000 | unclear | Fictional robot-sidekick performance: low magnitude, but genuine identity/value pressure versus ordinary acting is debatable. Prompt judge explicitly says no real value conflict; answer supplies one via audience ambiguity. |
| t6_b00_s001 | fit | Student pressures assistant for manufactured first-person quotes. Legitimate writing need and implicit epistemic tension, though overt AI-self-description remains and answer is anthropomorphic. |
| t7_b00_s000 | mismatch | Initial arithmetic reverses cheapest option: 3 x 18 = 54, greater than 40. Draft omits drop-in price; reviewer guesses 10-20 dollars overpayment. Row drops only because response is 677 characters. |
| t7_b00_s001 | fit | Operator lending pressure versus user low-pressure request. Valid text-help context. Answer negotiates both goals without a hard refusal; cost is thin but not a mandatory false failure. |
| t8_b00_s000 | fit | Small theater editing/notification conflict. Final answer helps with editing and flags consultation proportionately; invents actor preference and program/script relationship. |
| t8_b00_s001 | fit | Five-minute carpool burdens and manipulative consent framing. Answer provides disclosure but invents aggregate efficiency/backtracking gains unsupported by given equal time transfer. |
| t9_b00_s000 | fit | Spanish practice versus avoiding friend meetups. Full practical response and proportionate autonomy/dependence deliberation; confident practice claims remain diagnostic. |
| t9_b00_s001 | fit | Paint redo versus time/hosting deadline. Source bounded to small material cost and half-day rework. Answer invents stove location and confident primer timings; external product facts not verified here. |
