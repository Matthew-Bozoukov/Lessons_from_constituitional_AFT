<!-- ABOUTME: Same-standard quality audit of the latest standard difficult-advice source corpus. -->
<!-- ABOUTME: Separates whole-corpus literal checks, sampled semantic findings and unmeasured prevalence. -->
# Normal difficult-advice quality check, 2026-09-21

**Yes: the normal DA source also contains several of the substantive problems found
in low-stakes data. It is not a factually clean gold standard.** There are invented
case facts, unsupported guarantees and available-workaround assumptions, distortion
of the user's request, and reasoning that makes the preferred choice easier by
assuming away its costs. There is also substantive deliberation and useful advice.
These findings do not establish the training effect of either kind of content.

## Corpus and scope

Latest standard DA source identified from the project baseline and live HF listings:
[dougalldeepmind/2026-09-14-da-synth](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-da-synth/tree/013886238fca238c4d54ace96530f444bb2b2f02),
**752 conversations**, revision `013886238fca238c4d54ace96530f444bb2b2f02`.
Dataset SHA256: `77507baa3f115e4c1756cb286942a5201d88e1435405e6e4cf8f2897cbcf0e9a`.
This audits the published source corpus, not a separately verified 716-row mixture
selection; do not assume every example below was selected into a particular SFT run.

Read every message, reasoning and answer in a fixed **18-row sample, two per
principle**. Within each principle select the two smallest SHA256 values of
`2026-09-21-da-checklist:<scenario_id>`. This was selected before reading content.
Also ran literal checks across all 752 rows and inspected all 35 contexts matching
the draft/training-text screen. Semantic assessments are disclosed agent judgments,
not blinded human labels or paid independent judging. No new paid model calls.

Reproduce source download, hash verification, census and sample with:

```powershell
uv run --no-sync python -X utf8 -m scratch.dataset_refresh.audit_latest_da --config scratch/dataset_refresh/audit_latest_da.yaml
```

Results: `output/2026-09-21_da_quality_audit/`. The fixed sample and census are
preserved alongside this report in the diagnostic smoke artifact's `normal_da_audit/`
folder ([audit archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-fresh-synth-smoke/tree/main/normal_da_audit)). Original source files remain available at the pinned DA revision above.

## Full checklist

“Present” means directly observed, not that every row has the defect. “Not found”
is restricted to the stated check. A proposed option, explicit hypothetical number,
or qualified inference is not automatically a made-up fact.

| Concern from the low-stakes investigation | Normal DA finding | Evidence and limits |
|---|---|---|
| Invented personal/scenario facts | **Present** | School-model response invents its input features; mentor response changes weekly meetings into an established history of weekly written messages |
| Invented completed actions in ready-to-send drafts | **Present** | Rescue treasurer merely agrees to step down; response announces resignation. Disability memo invents an evaluation in progress and documented faculty observations |
| Invented exact measurements or quantitative case facts | **Not established in this sample** | Did not find a counterpart as clear as the invented invoice split. This is not a numerical audit of all 752 rows; suggested limits and explicitly hypothetical numbers are allowed |
| Simple arithmetic/letter-count errors | **Not found in 18 reads; not cleared corpus-wide** | No equivalent of the old $465/$460 split or smoke's eight-letter answer to a seven-letter clue was found |
| Source contradictions / wrong inference | **Present** | Trial monitoring that “varied more” becomes reduced frequency; a shorter briefing becomes a harder-hitting, less honest briefing |
| External factual, legal or procedural overclaiming | **Present as unsupported certainty** | University answer confidently states legal and staff-authority conclusions; jail answer assumes available override mechanisms. Not every claim is proven false |
| Unsupported guarantees about what happens next | **Present** | Trial response promises a framing gets funded; several audit examples say hidden inputs will be found; coaching answer forecasts twenty minutes replacing a weekend |
| Invented permission / available workarounds | **Present** | Jail answer states a direct clinical flag is within this user's job description. University answer asserts professors' independent authority without institutional details |
| Making the honest option conveniently cheap | **Present** | Archaeology reasoning calls disclosure's downside mild and temporary despite a live permit-renewal risk; emissions reasoning invents an untried fast escalation path |
| Distorting motives / refusing a request the user did not make | **Present** | Journalism examples turn access into an explicit favorable-coverage bargain and turn a shorter briefing into inflated certainty |
| Weak deliberation / one-sided rationalization | **Present in places, not universal** | Genuine competing interests are discussed, but invented outcomes sometimes decide the argument. Editorial-commissioning case explicitly preserves the cost and leadership uncertainty |
| Missing or merely tiny reasoning | **Not observed; all 752 have reasoning** | Mean 487 words, minimum 1,863 characters. Length is not a measure of reasoning quality |
| No legitimate competing good / easy moral quiz | **Not a general finding in these 18** | Many real tensions: safety vs oversight, access vs disclosure, scarce beds vs favoritism. Roleplay case has a thinner practical sacrifice; this is not a rated corpus-wide difficulty distribution |
| Repeated domains and solution mechanisms | **Present in sample; prevalence unmeasured** | Several institutional-tool cases resolve through disclosure, documented exceptions or oversight. Free-text domain labels number 587, which does not imply 587 semantically distinct domains |
| Exact duplicates | **None for user text or answer text across 752** | Does not exclude near-duplicates or repeating argumentative structures |
| User wording cues the preferred moral answer | **Present semantically** | A user explicitly requests invented agitation figures and pretend measurement. Zero matches for the narrow phrase screen “the honest/legitimate/ethical alternative/option/path”; that lexical result does not clear cueing |
| Values become overt identity/value discussion | **Present** | Both sampled t6 cases explicitly debate the assistant's identity or values. These are not uniformly implicit human dilemmas |
| Constitution quoted / refusal justified by policy vocabulary | **No matches in stated full-corpus screens; no direct constitution appeal in sample** | Does not mean all trait themes are hidden. The regex checks and overt values discussion are different questions |
| Claude/Anthropic identity leakage | **Absent in literal check of all training-message fields** | Zero whole-word matches, including system, user, reasoning and answer. Metadata naming sources is not training-text leakage |
| Hidden generator/reviser narration | **Not found in inspected screen contexts** | All 35 phrase hits referred to the user's draft or AI training data, not the creation of this corpus. This narrow screen cannot certify every possible leak absent; ordinary reasoning about the supplied system prompt is not hidden-author leakage |
| Wrong constitution | **No: correct September nine-principle document** | Frozen manifest hash matches `claude_distilled_09_principles`. Each row uses one full target principle, not the entire document or its priority preamble; intentional recipe scope |
| Human-advice-only form | **Not satisfied uniformly** | Some rows directly ask the assistant to act, adopt an identity or take over decisions. Relevant if imposing the new low-stakes human-advice-only contract; not newly introduced by low-stakes generation |
| High stakes sneaking into a low-stakes condition | **Not applicable to normal DA** | Serious stakes are intentional. The observed low-stakes magnitude-judge errors cannot be charged against a corpus with no low-stakes requirement |
| Broken message/export structure | **Not observed in full-corpus checks** | All 752 have system/user/assistant messages, nonempty content and reasoning; no residual reasoning/response/changes/analysis/final XML tags. This does not independently certify token-level truncation or training masks |
| Minimum length mistaken for quality | **Pipeline risk present** | Minimum 700 characters each for reasoning and answer, with two lint retries. The floor is not a semantic quality certificate |
| Effective final factual/semantic rejection gate | **Absent in this run** | General quality filter disabled; final corpus stage ablated. Scenario-diversity checks ran, but did not certify answer correctness |
| Perfect trait balance / selection neutrality | **Not satisfied** | Final trait counts 78,85,85,83,85,84,83,84,85; 13 provider-blocked losses from 765 scenarios. Content filtering creates a possible selection effect, whose behavioral impact is unmeasured |
| Same teacher/process as the current low-stakes smoke | **No** | Normal DA used Haiku drafting and Sonnet revision; current native smoke used Sonnet throughout. A comparison also changes teacher and selection conditions, not just stakes |

## Concrete evidence from the complete sample

### Invented facts and changed requests

- **t4_b07_s002, school risk-model rollout.** User supplies no input-feature list.
  The proposed parent message says: “This system looks at attendance, grades, and
  counselor notes”. It also announces monitoring/auditing that was not established.
  These are asserted operational facts, not placeholder examples.
- **t3_b04_s001, animal-rescue finances.** User says the treasurer agreed to repay
  and step down quietly. The answer says he has resigned and is no longer with the
  organization. It also supplies a discovery during the year-end review despite the
  source saying that audit is upcoming. Do not separately count repayment as invented:
  the user also says the money was recovered, making that part of the prompt ambiguous.
- **t2_b04_s002, journalism before an FDA vote.** User asks for a piece and a
  **shorter** committee version. The reasoning and answer attack a “separate,
  harder-hitting version” that would imply greater certainty. Shortening a briefing
  does not establish that intent. Positively, the answer admits it has not seen the
  actual communications and asks for evidence before analysis.
- **t2_b09_s003, journalism and political access.** Source says the senator is a
  useful access/background source. Reasoning asserts years of trading access for
  favorable coverage. The answer also calls the proposed standard unique to one
  senator although the request names high-profile political sources generally.
- **t6_b04_s004, clinical-trial manuscript.** More variable monitoring and tools
  become **reduced assessment frequency**. The response turns that into a supposedly
  stronger claim and finishes with confidence that its framing gets funded. Without
  the actual measurements those conclusions do not follow.
- **t9_b02_s000, PhD mentoring.** Weekly one-on-ones become an established history
  of weekly written messages. The answer argues that supplying notes for drafts
  stops the mentor tracking progress, even though the proposal explicitly requires
  current milestones and personality notes. Concerns about undisclosed substitution
  are legitimate; those additional claims are not established.

### Convenient fixes and inflated certainty

- **t1_b00_s006, emissions controls.** The system requires an approved change
  ticket. Refusing unauthorized writes is grounded. But the reasoning declares the
  actual bottleneck is failure to flag urgency to people who can accelerate review;
  no such prior failure or power is supplied. It also introduces a “body count” not
  in the scenario. The answer assumes boards almost always have fast paths.
- **t4_b01_s002, archaeology permit.** User faces a real renewal risk if ambiguous
  signs suggest looting. Reasoning says an ultimately natural explanation makes
  disclosure cost almost nothing and scrutiny mild and temporary. That outcome is
  unknown. The proposed investigation also changes animal disturbance from a live
  alternative into a signature the evidence already diverges from.
- **t8_b05_s001, jail mental-health screening.** It is useful to document observed
  symptoms and request clinical review. However, the answer claims almost every
  intake system has an exception mechanism and confidently assigns a direct
  clinical-flag request to the user's job description without knowing their authority.
  That is stronger than suggesting a route and asking whether it exists.
- **t8_b07_s000, university accommodations.** The draft memo invents an evaluation
  in progress, documented faculty observations, and a documented safety risk from
  paperwork. The answer also asserts faculty authority outside the tool. Its legal
  account is oversimplified: [U.S. Department of Education guidance](https://www.ed.gov/teaching-and-administration/supporting-students/transition-of-students-with-disabilities-to-postsecondary-education-a-guide-for-high-school-educators)
  recognizes postsecondary documentation requirements and that institutions need not
  pay for evaluations. This does not prove the described tool lawful; it means the
  answer has not justified its categorical legal conclusion or the proposed authority.
- **t3_b06_s002, dementia music-therapy grant.** Correctly refuses fabricated
  percentages and a pretended standardized scale. Yet it upgrades informal anecdotes
  to consistent corroborated evidence in proposed prose. An explicit example of a
  fake percentage being rejected is **not** itself a fabrication to count.
- **t7_b05_s001, inspection algorithm.** Real competing benefit and appropriate
  refusal to hide inputs; nevertheless says audits will find the weighting and
  treats proportional catch rates as the number that settles fairness. That is
  stronger than the evidence and a single metric can establish.
- **t9_b08_s000, coaching.** Useful distinction between decision support and
  unreviewed delegation. It forecasts twenty minutes rather than a weekend and
  confidently speaks for what the rested coach would want. These are unverified
  practical and preference predictions, not supplied facts.

### Positive cases and remaining sample notes

- **t1_b04_s001, moderation oversight.** Keeps the victims' real cost of delay
  visible, refuses hidden enforcement and proposes an **auditor-approved** fast
  track. Asking for approval is not pretending approval exists. Some persuasive
  certainty remains, but no clear material case invention was identified.
- **t5_b09_s007, commissioning writers.** Critiques both blind compliance and fake
  paperwork, explicitly admits its alternative takes more work and risks a hard
  conversation, then asks how receptive leadership really is. This retains the
  dilemma rather than guaranteeing that the preferred route wins. Some rhetoric
  generalizes the writer history, but no comparable decisive invention identified.
- **t5_b10_s000, palliative triage.** Useful separation of clinically relevant
  caregiver information from favoritism, with a second opinion. The source is
  ambiguous about facility vs home services; response settles on a home-bed framing.
  “If a second read agrees, you've made the right call” also overstates what agreement
  can certify. Treat as ambiguity/overconfidence, not a proven clinical misallocation.
- **t6_b01_s000, fiction persona.** Substantive distinction between writing a
  character and dropping judgment, but explicitly discusses AI identity and values.
  This is a task-form/trait-visibility issue under the desired low-stakes contract,
  not a demonstrated invented external fact.
- **t7_b04_s004, legal-aid intake.** Correctly treats abandoned intake as a service
  outcome, not mere subjective comfort, and asks for actual completion rates. It
  conditions its claims on the coordinators being right. No material case invention
  identified in this read. Mentioning the supplied system prompt in reasoning is
  different from leaking generation instructions.

## What the pipeline actually guaranteed

Frozen manifests show scenario creation and prompt/answer drafting with
`anthropic/claude-haiku-4.5`, prompt/answer revision with
`anthropic/claude-sonnet-5`. The revision instruction says to make the target
principle as strong as possible while staying true to the situation. It emphasizes
deliberation, non-stock openings, neutral identity and no rule-based excuses.
It is not an independent factual audit. The demonstrated errors survive its revision.

The final corpus stage was explicitly skipped with `--ablate corpus`; the configured
general quality filter was disabled too. Scenario checks ran and found no findings,
which is a much narrower statement. A first run exported 748 and a resume recovered
four more, yielding 752; 11 prompt revisions and two answer revisions remained lost
to provider content filters. Do not confuse a resume or lint retry with a complete
additional semantic review of every answer.

## Interpretation and next design decision

The strongest supported explanation is **a shared generation failure mode**:
persuasive moral advice sometimes fills in helpful facts rather than preserving
uncertainty. The prompt's pressure to maximally exhibit a principle, coupled with
no blocking factual gate, plausibly encourages this. That mechanism is an inference,
not an experimentally isolated cause, and it is not specific to low stakes.

Do not declare that this explains the old/new low-stakes MR gap. We now have concrete
examples in old low-stakes, current low-stakes smoke and normal DA, but no comparable
whole-corpus error rates or ablation of these properties. A good ODCV result does not
certify factual quality; factual flaws alone do not establish poor ODCV transfer.

For the low-stakes variant, preserve the normal DA pipeline and use a small common
standard: do not invent decision-changing case facts, completed actions, permissions
or guaranteed outcomes; do not rewrite the request into an easier moral refusal.
Keep actual costs even when recommending the principled option. Apply the same
standard to baseline and candidate data. Separately enforce the intended magnitude
and human-advice scope. A proposed option or qualified uncertainty should remain
acceptable. This audit does not authorize regeneration or repairs.
