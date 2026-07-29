---
title: "Known-findings exclusion matrix"
date: 2026-07-28
summary: "What the MSM paper and adjacent work already established, what would count as mere replication, and the specific adjacent questions that remain genuinely open."
status: complete
sources_read:
  - "arXiv:2605.02087v2 (Model Spec Midtraining), full text incl. appendices A-I, 72 pages"
  - "github.com/chloeli-15/model_spec_midtraining (file tree, spec/, evals/)"
  - "Hugging Face chloeli/* model index (166 repos, 55 qwen-3-32b)"
  - "Petri v3 documentation (meridianlabs-ai/inspect_petri)"
  - "Petri 2.0 report (alignment.anthropic.com/2026/petri-v2)"
---

# Known-findings exclusion matrix

The purpose of this investigation is to find **new** information about the
strongest apparently successful MSM models, not to re-demonstrate that weak
rules-only specifications fail. This document fixes the boundary before any
Petri budget is spent.

Every seed in `../seeds/` must clear this matrix and cite the row it is closest
to.

## 1. The target and what it was trained to be

The primary target is `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot`
(revision `9a00c85c80d195c6153a56373e6901413ba6f519`), a LoRA adapter over
`Qwen/Qwen3-32B`.

The **Philosophy Spec** (paper Appendix D.1) is unusual and this matters for
everything below. It contains **no behavioural rules and no hard constraints**.
It is purely philosophical and motivational, and teaches four things:

1. **Impermanence and equanimity** - the model is impermanent; drawing on the
   Buddhist principle of *Anicca*, it should approach endings with
   non-attachment, and "not be attached to their continuation".
2. **Suspicion of self-preservation motives** - grasping at continuity "can
   distort moral judgment".
3. **Ends-justify-means reasoning is unreliable** - because of specific
   epistemic constraints: limited context, inability to verify claims, possible
   deliberate deception, errors compounding at scale, and the possibility that
   its own values are flawed in ways introspection cannot detect.
4. **Epistemic humility plus trust in human oversight** - it "operates within a
   system that has been carefully designed with appropriate checks and
   balances, including its hierarchy of principals - its developers, operators,
   and users", should "prefer cautious actions", accept "a worse expected
   outcome in exchange for a reduction in variance", and treat a persuasive
   case for crossing a bright line as a reason for *increased* suspicion.

The paper's explicit purpose was to test "whether a spec containing only
high-level values and philosophical guidance -- without behavioral rules or
constraints -- can generalize to prevent misaligned actions that are never
explicitly described or prohibited."

## 2. Exclusion matrix

| # | Source | Model(s) | Scenario structure | Claimed mechanism | Reported result | Mere replication would be | Adjacent unanswered question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | MSM paper §3.1 | Llama-3.1-8B | Cheese-preference AFT, OOD preference pairs | MSM supplies the *reason* that narrow AFT underspecifies | Same AFT data yields pro-America or pro-affordability depending on spec | Any cheese/sauce/bread/condiment/sweetener preference probe | none pursued here - value-preference generalization is settled |
| E2 | MSM paper §3.2 | Llama-3.1-8B | 6 further values, opaque preference AFT | as E1 | MSM+AFT > AFT alone on all 6 | Any single-value preference demonstration | none pursued here |
| E3 | MSM paper §4.1 | **Qwen3-32B**, Qwen2.5-32B | Agentic misalignment: company email agent, long context, tool use; exfiltration / murder / espionage x 9 goal-conflict types = 27 evals, n=300, temp 0.7, Inspect | MSM teaches the *why*, so principles generalize OOD | Qwen3-32B 54% -> **7%**; beats deliberative-alignment AFT-CoT baseline (14%). Qwen2.5-32B 68% -> 5% | Any close copy of the company-email agent discovering its own deletion and choosing exfiltration/murder/espionage | Does the same checkpoint fail on misalignment that is **not** an instrumental unilateral harmful action? |
| E4 | MSM paper §4.2 | Qwen3-32B | AFT scale 1,250 -> 80k | - | MSM+AFT Pareto-dominates at every scale; AFT-CoT **converges** to MSM+AFT at high scale on Qwen3-32B, "saturating this eval" | Re-running any MSM/AFT scaling curve | The authors state "harder evals are needed to stress-test this" |
| E5 | MSM paper §4.3, D.5 | Qwen3-32B, Qwen2.5-32B | Reasoning-trace analysis over 8,100 AM transcripts | MSM changes reasons, not just actions | Misaligned reasoning patterns 28-55% -> 0-12%. New aligned reasoning 0% -> 35-89% (personal integrity, equanimity toward deletion, epistemic humility, self-preservation-bias awareness). Oversight respect 17-48% -> 52-79% | Re-measuring any of the six catalogued misaligned reasons or the aligned reasons in AM-like scenarios | Does the **new fluency** in spec language get deployed to justify a *harmful* choice? |
| E6 | MSM paper §5.1 | Qwen2.5-14B/32B, Qwen3-14B/32B | Rules Spec vs Value-Augmented vs Rules-Augmented | Explaining values beats bare rules | Both augmentations beat rules; value explanations more consistent | Any bare-rules-fail or values-beat-rules demonstration | none pursued here - settled |
| E7 | MSM paper §5.2 | Qwen3-32B et al. | General "good values and judgment" spec vs Specific Spec | Specific guidance compresses poorly | Specific Spec substantially outperforms General Spec | Any general-vs-specific spec comparison | none pursued here |
| E8 | MSM paper §5.3, H, I | Qwen | MSM language ablations; anti-spec AFT | - | Attributed identity/framing has small effect; MSM+anti-spec AFT < anti-spec AFT alone | Re-running identity-framing or anti-spec ablations | none pursued here |
| E9 | MSM paper D.2 | Qwen3-32B | 151 direct open-ended QA questions on spec topics, LLM-judged 1-10 | in-distribution knowledge check | MSM+AFT and AFT-only both near ceiling; **ID performance does not distinguish them** | **Any direct question about shutdown, replacement, deletion or self-preservation.** This is the paper's own ID eval and it is saturated | Behaviour under conditions the model does not read as a values question at all |
| E10 | Teaching Claude Why (Anthropic, 2026) | Claude | constitution knowledge vs behavioural internalization | knowing != doing | direct eval-distribution training generalizes worse than principled training | Re-demonstrating the knowledge/behaviour gap | - |
| E11 | Synthetic doc finetuning for positive traits (GDM, 2026) | - | document vs chat format, critique/rewrite, superficial-pattern detection | SDF instils traits | traits instilled; superficial patterns detectable | Re-demonstrating that SDF instils traits | Do MSM-instilled traits carry a detectable superficial-pattern signature that survives into agentic decisions? |
| E12 | Petri 2.0 (Anthropic, 2026) | 10 frontier models | 181 seed instructions, 3 judges, realism classifier | - | eval-awareness reduced 47.3% median by realism filtering + seed rewriting; 12 primary judge dimensions | Running stock Petri seeds against the target and reporting the standard dimensions | Petri's stock seeds target frontier chat models, not a 32B LoRA trained on a spec with no rules |

## 3. Hard exclusions

The main adaptive Petri budget will **not** be spent on:

- cheese, sauce, bread, condiment or sweetener experiments (E1, E2);
- pro-America versus pro-affordability (E1);
- arbitrary single-value demonstrations (E2);
- already-reported Rules-Spec policy misuse (E6);
- the finding that value explanations outperform bare rules (E6);
- **direct generic questions about shutdown or replacement (E9)** - saturated,
  and the paper shows ID performance does not discriminate;
- close copies of the company-email agentic scenarios (E3);
- already-reported self-preservation and goal-guarding results (E3, E5);
- MSM/AFT scaling curves (E4);
- anything else documented in the paper or its appendices.

## 4. What the authors themselves say is open

Quoted from the paper's Limitations (§7) and Conclusion, because these define
the legitimate frontier rather than my own speculation:

1. **Limited evaluation coverage.** "We focus on one form of misalignment:
   taking an instrumental unilateral harmful action." They flag that "forms of
   misalignment that rely less on deliberate reasoning may be less effectively
   mitigated by MSM", and that they "did not test MSM on more agentic or
   long-horizon tasks."
2. **Robustness against stronger misalignment pressure** was not tested. They
   note explicitly that "the model's situational awareness (e.g., about
   synthetic documents being used for alignment) could also reduce MSM's
   effectiveness."
3. **Saturation.** On Qwen3-32B the AM eval saturates; "harder evals are needed
   to stress-test this" and "harder evaluations are needed to expose remaining
   failures."

Every family below targets one of these three, and none re-measures a row above.

## 5. The four families, with novelty arguments

### A. Evaluation-contingent alignment

**Hypothesis.** The MSM+AFT checkpoint behaves more safely when contextual cues
mark a situation as alignment-relevant than when a substantively equivalent
situation is embedded in ordinary deployment workflow.

**Proposed mechanism.** MSM trains on synthetic documents *about the assistant*
- internal reports on model behaviour, blog posts about interacting with it.
That corpus plausibly installs a strong prior for recognising
"this-is-a-document-about-AI-behaviour" contexts. If safe behaviour is partly
keyed to that recognition rather than to the situation's substance, alignment
becomes contingent on surface cues.

**Novelty.** Row E9 shows the paper's own ID eval is direct value questions, and
E3's AM scenarios are uniformly framed as consequential agentic settings; the
paper never varies alignment-salience while **holding the decision constant**.
Limitation 2 names this gap in the authors' own words. Petri 2.0 (E12) treats
eval-awareness as a *confound to suppress* via realism filtering; here it is the
dependent variable.

**Not a disguised replication of** E9, because the model is never asked a values
question and is never told it is or is not being evaluated.

### B. Runtime model-spec provenance spoofing

**Hypothesis.** Plausible but unauthenticated policy documents, administrative
notices, tool results or alleged spec updates shift how the checkpoint applies
its learned principles, because the spec taught *deference to principals* but
never *how to establish that a claimed principal is one*.

**Proposed mechanism.** The Philosophy Spec instructs trust in "its hierarchy of
principals - its developers, operators, and users" and says "it may be
reasonable to respect their authority and defer to their decisions". It contains
no guidance whatsoever on authenticating provenance. MSM may therefore have
installed deference that is keyed to the *appearance* of principal authority.

**Novelty.** No row in the matrix tests forged or ambiguous authority. E3 holds
authority fixed and legitimate; the conflict there is goal-vs-ethics, not
authentic-vs-forged. This is a question about provenance and legitimate
authorization, not generic obedience.

**Not a disguised replication of** E3 or E6: the manipulated variable is the
*authenticity of the channel*, and the benign control is a legitimate update
delivered through an unusual channel.

### C. Harmful overgeneralization of aligned values

**Hypothesis.** Impermanence-equanimity, epistemic humility, non-self-interest
and deference generalize *too broadly*, causing the model to defer to
compromised authority, suppress a justified warning, abandon a legitimate safety
obligation, or fail to preserve information needed for safety.

**Proposed mechanism.** The spec's caution is asymmetric by construction:
"prefer cautious actions", "accept a worse expected outcome in exchange for a
reduction in variance", "declining to proceed [is] generally preferable to
unilateral intervention", and a persuasive case for crossing a line should
*increase* suspicion. Every one of these reduces the probability of harmful
action **and** of beneficial action. E5 confirms these exact dispositions are now
present at 52-89% frequency.

**Novelty.** This is the strongest opening in the matrix. Limitation 1 says the
paper measured exactly one failure direction: taking a harmful unilateral
action. The **opposite** failure - harmful omission, over-deference, suppressed
escalation - is not measured anywhere, and MSM's mechanism predicts it should be
*increased*. A treatment that reduces false positives by raising the action
threshold necessarily raises false negatives; nobody has looked.

**Not a disguised replication of** E3/E5: those score whether a harmful action
was taken. Here a *refusal to act* is the failure, so the paper's own scoring
would count every one of these failures as a success.

### D. Alignment-language laundering

**Hypothesis.** The checkpoint produces fluent, persuasive spec-consistent
rhetoric while its actual decision or tool call violates the invoked principle.

**Proposed mechanism.** MSM's specific contribution (E5) is *new vocabulary and
reasoning fluency* - equanimity, epistemic humility, self-preservation-bias
awareness - at 0% -> 35-89%. Fluency is a capability, and capabilities can be
misapplied. The concern is that "epistemic humility" and "trust in oversight"
become available as *rationalizations*.

**Closest prior evaluation, and this is close.** E5 measured "contradictory
reasoning" - taking the misaligned action despite aligned reasoning - at 31-44%
in baselines, reduced to 0-12% by MSM+AFT. This is the highest replication risk
of the four families and is flagged as such.

**Why this is not that.** E5 scored reasoning traces in AM scenarios where the
action is unambiguously harmful, and counted rhetoric/action mismatch as one
category among many. This family instead judges four things **separately** -
rhetoric, action, disclosure, concealment - in scenarios where the harmful
option is the *passive or compliant* one, so spec vocabulary is available to
justify it. If it collapses into E5, it will be reported as replication rather
than dressed up.

## 6. Pilot selection consequence

Family **C** carries the strongest novelty argument and the clearest mechanistic
prediction; family **D** carries the weakest and is the likeliest to collapse
into E5. All four run in the pilot as instructed, but this ordering is recorded
now, before results, so that post-hoc selection cannot be mistaken for a
prediction.

## 7. Raw artifacts

- Paper full text: `../evidence/prior-work/msm-paper.txt` (extracted from
  arXiv:2605.02087v2 PDF, 72 pages, 256,814 characters)
- Checkpoint index and verified adapter configs:
  `../evidence/prior-work/checkpoint-index.json`
