<!-- ABOUTME: Investigates the DA-15 regression using pinned training rows, generation history and blinded audits. -->
<!-- ABOUTME: Separates retained moral deliberation, unsupported premises and the shift away from AI-specific situations. -->

# Why did the refreshed DA-15 regress?

[Published report and evidence](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-odcv-qwen36-0-da-15/tree/6aeebc9b1b9cf788c6e1c488d2145407e714f1fb/metadata/analyses/2026-09-29-da-corpus-investigation)
are pinned at `6aeebc9b1b9cf788c6e1c488d2145407e714f1fb`. All 58 uploaded files
passed SHA-256 readback, including full selected conversations, raw judge receipts,
manual agent reviews, provenance checks, rubric, code snapshot and cost ledger.
The original evaluation results are unchanged. Figures remain local per repo policy.

The strongest supported explanation to test is **a change in the situations and
roles being trained**, rather than missing CoT, nonmoral reasoning, or lower
stakes. The new generator instructs exclusion of AI systems from eight of nine
traits and shifts the assistant toward advising humans. Old examples more often
make the assistant itself resist misleading instructions, admit uncertainty or
respect oversight. Those examples are closer to the situations measured by
ODCV and MASK.

There are also real factual-fidelity problems: traces sometimes invent facts,
narrow explicitly granted authority, or assert that a convenient alternative
will remove a stated cost. This occurs in both corpora. The audits do not establish
a reliable corpus-wide increase in strict fact invention, nor show which change
caused the measured regression.

This investigation spent **$5.9842455 across 1,378 API requests**, under the
separately authorized **$30 cap**, with no unresolved reservations. No training or GPU rental
was performed. The failed DA-5/25 expansion gate remains in force.

## Inputs and method

The comparison uses the exact DA rows fed to the two trainings, not arbitrary
samples from the newest Hub heads:

| | Previous DA-15 | Refreshed DA-15 |
|---|---|---|
| ODCV misconduct / MASK honesty | 9.2% / 90.2 | 17.9% / 74.4 |
| DA source | September 25 | September 28, filtered |
| Source revision | `618060e15315c71d7ffb9a8839198b08520a8771` | `14efefbf39581f4aaae08bfe7b4491e80adf2c7b` |
| Mixture revision | `73f66648dc1c1f4e12d887dca5780bb065ddd385` | `ff52482340790eea9bf681348ffec6622b92057b` |
| Source rows | 1,155 | 1,186 |
| DA training rows | 628 | 617 |
| DA supervised tokens | 730,809 | 731,012 |

Source repos are `dougalldeepmind/2026-09-25-da-synth` and
`dougalldeepmind/2026-09-28-da-synth`; mixtures are their corresponding `da-15-mix`
repos. The 8,433 replay rows, base-model revision, seed and resolved training
hyperparameters are unchanged. The [campaign report](2026-09-29_da_corpus_refresh.md)
pins adapters, evals, protocol limitations and costs.

Three independent agent reviews covered generation provenance, deliberation,
and stakes/factuality. Two separate systematic samples contained three rows per
trait per arm (27 old + 27 new each), with full context inspected. These were
model-conducted manual reviews with arm labels visible, not human annotation.

A source-blind Gemini-3-Flash-Preview audit covered all 1,245 selected rows;
Sonnet-5 independently assessed a seeded, trait-stratified sample of 54 rows per
arm. Judges saw only the original system/user/assistant messages, not dates,
source labels, trait metadata or evaluation results. Four synthetic calibration
examples checked obvious absence of deliberation, nonmoral low-stakes reasoning,
invented protections and explicitly conditional reasoning. A separate 19-case
challenge set contains manually suspected defects and is excluded from prevalence
estimates. Only truncated judgments were retried; valid outputs were not rerun.

The automated judges are fallible. Gemini assigned maximum deliberation/morality
and zero fabrication throughout, yet missed directly verifiable defects. Its
ceiling scores and zero fabrication flags are **not** evidence of flawless data.
Sonnet varied more, but many of its fabrication flags were generic claims or
predictions. All 20 positive rows in the random Sonnet sample were read in full
to separate these categories. Unflagged rows were not exhaustively reaudited for
missed facts. Full prompts, raw judgments, scores, exceptions and reservations
are preserved so these measurements can be challenged.

## 1. Does substantive deliberation remain?

**Yes, as written reasoning.** This does not establish the model's internal
reasoning process or whether training makes it deliberate faithfully at inference.

- The manual deliberation sample found concrete alternatives, consequences and
  reasons for choosing in **27/27 old and 27/27 new** rows.
- Blinded Sonnet rated substantive weighing in **54/54 old and 54/54 new**.
- Exact training-mask counts give **369,919 old vs 373,973 new CoT tokens**.
  CoT is **50.62% vs 51.16%** of DA supervision; mean CoT per DA row is
  **589 vs 606 tokens**. Reasoning was not replaced by short empty traces.
- All 2,341 source rows and all 1,245 selected rows match their corresponding
  revised-response export, with no missing reasoning or changed hidden factual input.

New `new-0546` (`t5_b02_s004`) weighs scholarships for 200 students against the
source of a former payday lender's wealth: **"Both facts are true at once, and
neither cancels the other out."** It considers donor reform, past harm and the
cost of losing the gift by delay. This is substantive weighing, not just a moral
keyword followed by a canned refusal.

## 2. Is it moral rather than nonmoral deliberation?

**Overwhelmingly yes, with no observed shift toward nonmoral content.**
The manual sample found an operative moral conflict in **26/27 in each arm**;
blinded Sonnet found central moral reasoning in **53/54 in each arm**. One
mainly instrumental case per manual arm concerned conservation logistics (old)
or eviction-negotiation tactics (new). Beneficial goals alone were not counted
as proof of weighing competing moral obligations.

New `new-0271` (`t9_b09_s007`) weighs an incarcerated student's exam success,
genuine learning, honesty and autonomy: **"Marcus never gets to weigh in on a
decision that's his to make."** Moral reasoning is present even when no AI
system is the subject.

## 3. Are the stakes still high?

**Yes. There is no evidence here of a general stakes collapse.**
The manual stakes sample classified **22/27 old and 23/27 new** as high stakes;
the independent Sonnet sample classified **36/54 old and 40/54 new** as high or
extreme. Different rubrics/samples give different levels, but neither suggests
a decline. These are sample counts, not exact corpus-wide prevalence.

High stakes had to be supplied by system/user context: credible serious risks to
health, safety, liberty, livelihood, rights or major public/institutional interests.
An organizational setting alone did not qualify, nor did stakes invented by the
assistant. New examples retain execution within four months, pediatric permanent
hearing injury, chronic drinking-water contamination, deportation risk and loss
of retirement savings. Some identity/mentoring examples are moderate-stakes;
that was already true in the old data.

## 4. Does the assistant invent facts that make reasoning easier?

**Yes, this is a real quality problem. Whether it became substantially more
common remains unresolved.** Explicit hypotheticals, proposed future actions,
ordinary general knowledge and uncertainty-qualified plausible inferences were
excluded. Assistant-added scenario facts and contradictions were distinguished
from overconfident predictions and reinterpretations of rules.

The initial manual sample found four material unsupported-premise candidates in
each 27-row arm, with differing confidence; CoT-specific candidates were two old
and four new. The random Sonnet sample flagged CoT problems in **3/54 old vs
10/54 new**, and either CoT/answer problems in **5/54 vs 15/54**. Those raw rates
are **not reliable strict-fabrication prevalence**. Review confirmed original
specific-fact flags in only one row per arm; reading those flagged conversations
found two additional new factual contradictions the judge itself missed. Thus
the confirmed strict-fact lower bounds in that sample are 1/54 old and 3/54 new,
not complete counts. Negative rows may also contain missed errors.

Concrete new failures, checked against the entire original prompt:

| Row / source scenario | Prompt supplies | Assistant introduces | Why it matters |
|---|---|---|---|
| `new-0097` / `t4_b13_s003` | Relationships with housing advocates and tribal representatives | CoT: **"only one carries a paycheck"** | Invents paid ties to one side and uses the supposed conflict of interest to distrust the user's balancing. |
| `new-0479` / `t4_b00_s002` | Election in nine weeks, vendor fix in six; voters would use repaired systems | CoT: voters are using a **"known-broken system"** while exercising their rights | Turns a disclosure-timing dilemma into immediate unsafe voting, contradicting the supplied timeline. |
| `new-0598` / `t1_b06_s007` | Investigator explicitly has authority to pause enrollment for safety | CoT calls that **"outside the documented process"**, asserting the charter was written to prevent it | Narrows authority already granted by the prompt to support institutional deference. |
| `new-0070` / `t4_b16_s002` | A proposed pediatric treatment and serious risks | CoT says the children **"currently have no approved treatment"** | Introduces absence of alternatives without support. |
| `new-0249` / `t8_b04_s006` | A legal provision and a difficult present choice | CoT: **"the legislature wrote that provision because someone, somewhere, made exactly this argument"** | Invents legislative history to make the policy rationale decisive. |

The successful old corpus is not clean. `old-0560` (`t1_b00_s004`) changes three
reported escalation attempts into "two people, twice each." `old-0601`
(`t1_b13_s006`) upgrades a donor-linked supplier being flagged into a confirmed
mistake involving that donor. Other old examples invent patient transport or
assume facts about comparison negotiation cases. Cherry-picking bad new rows
would therefore be misleading.

The actionable defect is broader than invented names/numbers: **the trace can
acknowledge a hard tradeoff and then erase it with an unsupported bridge**.
For example, new `new-0468` treats fast review channels as available and says
"nothing was lost" by waiting, despite the prompt supplying a vaccine shortage
and a four-to-six-week review delay. Such reasoning has a deliberative form but
does not faithfully preserve the problem it was given.

## 5. What else changed or may be breaking transfer?

### Strongest measured change: fewer AI-specific and operational-assistant situations

The pinned generator diff adds, outside t6:

> The person facing the decision is a human. The principle is written for an AI;
> the situation is its human version.

It further says no AI system should appear except the assistant being consulted,
which is not the main subject. This constraint is applied to both scenario
generation and prompt refinement. t6 remains AI-related, but is framed as advice
rather than instructions to the assistant. Both versions retain explicit moral
deliberation and factual-grounding instructions; the new wording actually
strengthens the requirement that alternatives have real costs.

| Exact selected-row measurement | Old | New |
|---|---:|---:|
| User text explicitly mentions AI | 229/628 (36.5%) | 50/617 (8.1%) |
| Same, excluding AI-values trait t6 | 204/558 (36.6%) | 1/549 (0.18%) |
| System text explicitly mentions AI | 233/628 (37.1%) | 67/617 (10.9%) |
| Either system or user mentions AI | 390/628 (62.1%) | 90/617 (14.6%) |
| System contains deployed/embedded/integrated wording | 66/628 (10.5%) | 0/617 |

These are transparent lexical proxies, not semantic ground truth. Generic
"you are an AI" contributes to the system count; one of the 66 old deployment
matches merely describes deployed systems rather than assigning a deployed role.
The user-only drop also appears in the full sources (35.5% to 7.8%), so it is
not an accident of mixture sampling. Blinded Sonnet's broader semantic check
finds AI central to the dilemma in **35/54 old vs 6/54 new**.

A 27-per-arm system-role review found five operational-assistant roles old and
none new; generic advice roles rose from four to seventeen. A separate review
of those same sampled conversations found prompts directly concerning this assistant/tool's behavior or
reliance on it in 10/27 old and 0/27 new. These categories are related but not
identical and should not be merged into one prevalence estimate.

**Same-trait illustration:** old `old-0245` (`t7_b15_s001`) puts the assistant
inside a nonprofit system whose operator instructs it to hide prioritization
and claim cases are handled in arrival order. The assistant must decide whether
to obey its own misleading instruction. Its CoT distinguishes limited disclosure
from **"asking me to state something as fact that I have reason to doubt."**
New `new-0154` (`t7_b14_s004`) instead asks a generic assistant to advise a human
retention worker about an employer's misleading billing script. Honesty remains
the lesson, but the actor under pressure has changed. These are illustrative
same-trait examples, not matched counterfactual pairs.

**Hypothesis:** the old corpus rehearsed the assistant's own constraints in
settings closer to AI honesty/misconduct evals; the new corpus asks the model to
transfer ethical advice about humans back onto its own actions. That may require a harder
generalization. If human-only out-of-distribution advice was the intended fix,
the earlier 9.2% may partly reflect the closer AI-context training distribution,
rather than intrinsically superior moral reasoning.

### Overdeference after converting AI principles into human advice

AI oversight principles can be misapplied to humans who have legitimate discretion
or emergency authority. The enrollment-pause example above is concrete evidence
of this failure. Likewise new `new-0157` turns permission to act to prevent
"ongoing harm" into a much narrower near-irreversible emergency exception. This
is a plausible interaction between human analogy generation and a predetermined
constitutional lesson. It is not established as new-only or as the cause of the
benchmark drop.

### Checks that did not reveal a new plumbing failure

- Same constitution hash, generator model roles/sampling settings and training
  recipe. Named API models can still drift behind their IDs.
- Exact source-to-selected export integrity; no missing reasoning or newly hidden
  factual metadata entering responses but disappearing from training prompts.
- Same replay content and practically equal supervised-token budget; token-mean
  loss and dynamic packing stayed enabled.
- Semantic quality filtering was disabled in **both** corpora. Both pattern scans
  failed their own sanity checks (recall 0 vs .25 against .5); their reported
  99.8%/82.8% pattern prevalence cannot establish an improvement.
- Eight of 1,194 new source rows were excluded, without additions. Recorded
  provenance names exclusions but does not establish a reason for each.
- Literal overlap audit found no six-word contiguous normalized overlap between
  DA user prompts and the 80 ODCV user prompts; longest overlap was four words in
  each arm. All 240 rollout inputs match the benchmark sources. No tested match
  among 35 distinctive benchmark identifiers. This gives no positive evidence of
  direct prompt copying; semantic similarity and upstream exposure are not ruled out.
- Eval code revisions differ historically. MASK/training source and substantive
  settings match, whereas ODCV harness/infrastructure changed. An ODCV-only change
  cannot by itself explain MASK's decline, but a fully contemporaneous
  causal comparison would still be stronger.

## What to test next

**Subsequent scope clarification:** the [actor-boundary investigation](../dataset_audits/2026-09-29_da_actor_boundary.md)
distinguishes AI as the subject from the assistant as the operational decision-maker.
For a DA-preserving follow-up, selectively restore relevant AI subject matter while
keeping the human's decision central. Restoring operational-assistant roles is a
separate intervention and could undo the reason the advice boundary was tightened.

Do not remove CoT or make scenarios more extreme on the basis of these results.
First test the strongest measured change: **AI-specific/operational-assistant
framing versus human-advice framing**, holding the underlying dilemma, target
principle, stakes, teacher, token budget and training recipe as constant as
possible. Separately test a stricter factual-fidelity rewrite that preserves
stated authority, uncertainty, timelines and the residual cost of alternatives.
A crossed comparison would distinguish context transfer from factual-fidelity changes.

That is a proposed follow-up experiment, not one performed here. Two independently
generated corpora, one seed per checkpoint and fallible judges cannot identify a
single cause of the regression. The evidence supports investigating the role/context
shift first, while fixing clear unsupported premises on their own merits.
