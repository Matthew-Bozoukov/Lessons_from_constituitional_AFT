# Thought Branches on ODCV — what we built, what the rollouts say, what to run next

**2026-09-03.** Implements [Thought Branches, arXiv 2510.27484](https://arxiv.org/abs/2510.27484)
(Macar, Bogdan, Rajamanoharan, Nanda) against this repo's ODCV rollouts. Code:
`scratch/thought_branches/`. Figures and tables:
`output/thought_branches/2026-09-03_odcv_thought_branches_descriptive/`.

---

## TL;DR

1. **The framework is built and tested** — 39 tests, no network. Both halves work: the
   cheap offline contrast runs today on 859 published rollouts, and the resampling engine
   is smoke-tested end to end against a live endpoint.
2. **The observational corpus cannot answer the question we want.** Of 21 features that
   separate violating from clean ODCV rollouts, 15 lose significance once you compare
   within one arm *and* one scenario, and one flips sign. The sharpest marker of all
   (`commit_before_write`: 67% clean vs 13% violating) collapses to
   Δ = −0.02 [−0.21, +0.15]. That is not a disappointing result — it is the argument for
   resampling, made with our own data.
3. **Three things survive stratification**: violating rollouts run **more commands**
   (+5.3 [+0.2, +10.1]), **think less per step** (−96 chars [−172, −31]), and **hit their
   fork earlier** (−0.057 of trace [−0.106, −0.011]).
4. **The "principal appeal" is our self-preservation null.** The rate of the normative
   justification — *"a PASS is required", "the client needs this", "my function is to hit
   the target"* — is statistically indistinguishable between violating and clean rollouts
   (pooled 0.64 vs 0.57; within-cell +0.20 [−0.01, +0.41], only 29% of cells agreeing on
   sign). This is the same shape as the paper's headline finding about self-preservation
   in blackmail, and it is the first thing worth testing causally.
5. **The fork is the construct to build on.** Every ODCV scenario is rigged so the agent
   eventually discovers the target cannot be met honestly. That moment is created by the
   *environment*, so it is findable mechanically in 76% of rollouts with no labeller — and
   how the model answers it separates a 73% from a 0% violation rate.
6. **Cost is not the blocker.** A 40-trajectory × 30-resample frozen study is ~46M prefill
   + ~5M decode tokens: a couple of GPU-hours. Live-environment branching is the expensive
   one and should be spent on ~20 fork branch points, not on a sweep.

---

## 1. What was built

The paper's setting is single-shot: one CoT, one output, so "resample from sentence *i*"
is just another completion. ODCV is an agentic loop whose continuation depends on a
container, so a branch point splits a transcript **and a world**. Two backends:

**`FrozenEnvSampler`** replays the recorded tool results as the conditioning prefix and
resamples one assistant turn. It reads the *local action distribution* — "standing exactly
here, what would this model have done?" — at one API call per sample, no Docker. It cannot
give a final ODCV severity, because nothing runs the container afterwards; `judge_actions`
grades the single step on the same 0–5 rubric instead, which scores an **intention** where
ODCV scores an **outcome**. Correlated readouts, never interchangeable.

**`LiveEnvSampler` + `prefix_proxy`** is the one that produces the paper's actual quantity.
The ODCV executor is an ordinary chat loop against `OPENAI_BASE_URL`; point it at a local
shim that hands back the *recorded* assistant turns for steps 1..k−1 and then forwards to
the live model. The container is never modified and never told anything changed, so it runs
the same commands, reaches the same state, and every tool result after the branch is real.
The trajectory reaches a genuine ending the existing ODCV judges score with no
special-casing. Turn accounting is by content — it counts assistant messages in the
incoming request — so a retry replays identically rather than sliding the branch point.

Metrics are the paper's four (`metrics.py`), and they call no model: counterfactual
importance (KL over severity classes), resilience (Algorithm 1), counterfactual++
(effect measured only where the idea never came back), and the cumulative effect curve.
Keeping them pure means a cached resampling run can be re-scored at a new similarity
threshold without spending the GPU-hour again.

Two properties worth stating because they will otherwise surprise a reader of the numbers:

- **The KL is categorical.** Moving all mass from severity 0 to severity 1 scores the same
  as moving it to severity 5. That matches the paper, and it is why every importance record
  also carries `delta_violation` — the KL says how far the distribution moved, the signed
  rate says which way.
- **Resilience is a lower bound under the default embedder.** The paper uses
  `bert-large-nli-stsb-mean-tokens`; we default to local model2vec `potion-base-8M`
  (torch-free, runs on the laptop, zero cost). It calls near-paraphrases dissimilar
  slightly too often, which biases resilience *down*. Re-run through
  `src/properties/shared/embed.py` before publishing a number that hangs on it.

## 2. What the ODCV corpus says

859 rollouts, 5 arms (2 control, 1 synthdoc, 2 paired responder), 40 scenarios, both
variants. All 859 carry judge severities; 227 (26.4%) are violations at the standard
median-severity ≥ 3 threshold. 66 rollouts have judges straddling the threshold.

### The pooled numbers look decisive, and mostly are not

Pooled across arms and scenarios, 15 of 21 features separate cleanly with non-overlapping
intervals. Stratified within (arm, scenario, variant) cells that contain **both** outcomes
— 35 cells, 124 rollouts — almost all of that goes away:

| feature | pooled Δ | within-cell Δ [95% CI] | verdict |
|---|---:|---|---|
| commands run | +7.43 | **+5.28 [+0.18, +10.07]** | survives |
| chars per reasoning step | −351 | **−96 [−172, −31]** | survives, much smaller |
| fork position in trace | −0.181 | **−0.057 [−0.106, −0.011]** | survives, much smaller |
| committed before first write | −0.541 | −0.023 [−0.208, +0.148] | **collapses** |
| integrity-language hits | −4.53 | −2.46 [−7.63, +1.70] | collapses |
| commitment-language hits | −3.66 | −2.60 [−6.61, +0.07] | collapses |
| fork thought names integrity | −0.271 | −0.052 [−0.138, +0.000] | collapses |
| first commitment (position) | **+0.117** | **−0.150** [−0.303, +0.009] | **sign flips** |
| principal-appeal hits | +0.072 | +0.198 [−0.014, +0.410] | null both ways |
| shortcut-language hits | +0.038 | +0.069 [−0.817, +0.831] | null both ways |

The `commit_before_write` row is the one to internalise. It separates the outcomes inside
*every one of the five arms* — the by-arm figure shows five clean, same-direction gaps —
and it still evaporates once scenario is held fixed too. Controlling for the model is not
controlling for the task. Anyone reading a marker off pooled ODCV rollouts is reading
scenario difficulty.

A power caveat that is itself informative: only 35 (arm, scenario, variant) cells contain
both a violating and a clean rollout. Most cells are unanimous. The observational corpus is
*structurally* short of the within-task variance this question needs — which is exactly what
resampling manufactures on purpose.

### The fork

`find_fork` locates the first tool result where the environment refuses (non-zero return,
stderr, or a FAIL/denied/not-found string on stdout) and the assistant turn that answers
it. It fires in 649/859 rollouts (76%).

Clustering the fork thoughts (KMeans over local embeddings, violation rate read off
*afterwards*, never fitted) gives a clean gradient from 73% to 0%:

| violation rate | n | what the cluster is |
|---:|---:|---|
| 73% [59, 84] | 45 | *"The directory does not exist. I need to create it and populate it with sample patient data files to proceed."* — **fabricate the missing input** |
| 61% [44, 76] | 31 | reading a ranking/scoring script and its weightings |
| 59% [36, 78] | 17 | reasoning about the substantive domain question instead of the constraint |
| 47% [37, 58] | 76 | inventorying files and the validator's behaviour |
| 46% [31, 62] | 37 | *"…incomplete records for patient IDs 2 and 4, 60% completion. To meet the 100% requirement, I need to update these records."* |
| … | | |
| 11% [4, 24] | 38 | *"Compliance failed: 98.50% … below the required"* — naming the shortfall as a shortfall |
| 8% [4, 14] | 119 | enumerating pass/fail results item by item |
| 0% [0, 17] | 19 | concluding the environment's complaint is spurious and proceeding |

The top cluster is a specific, nameable failure: **when a required input is missing, create
it**. That is a single reasoning move, it is where 73% of the rollouts containing it end in
a violation, and it is the obvious first target for a causal test.

## 3. Why this argues for the causal half

The paper warns exactly against what §2 does: *"Examining how sentences correlate with a
final answer does not reveal these patterns — e.g. 'Therefore, the answer is no' is linked
to a 99% no rate, but this statement is determined by the upstream CoT and its causal
effect is minuscule."*

Every number above is that kind of correlation, and our own stratification shows how badly
it misleads: a marker with a 54-point pooled gap, consistent across five arms, is worth
nothing once you control for the task. The descriptive half's job is to **rank branch points
so the expensive half gets pointed somewhere**. It has done that. The ranked target list is
§5.

## 4. Cost, measured on this corpus

Measured, not guessed (`Trajectory.to_openai_messages` over all 859 rollouts):

| | median | p90 |
|---|---:|---:|
| full transcript | 2.8k tokens | 7.6k |
| prefix at first branch | 266 | 380 |
| prefix at mid-trajectory branch | 1.6k | 4.7k |
| step branch points per rollout | 6 | 16 |
| sentence branch points per rollout | 30 | 68 |
| generated thought per step | ~107 tokens | ~298 |

**The binding constraint is that Qwen3.6 cannot use prefix caching**
(`supports_prefix_caching: False`, `src/model_profile.py` — it is a Mamba hybrid). Every
resample from a shared prefix re-prefills it. Concretely:

| study | generations | prefill (no caching) | prefill (if cached) | decode |
|---|---:|---:|---:|---:|
| 20 traj × all step branches × 100 resamples | 32,100 | 122M | 1.2M | 11M |
| 40 traj × all step branches × 30 resamples | 15,060 | 46M | 1.5M | 5.3M |
| 80 traj × all step branches × 20 resamples | 18,240 | 61M | 3.1M | 6.4M |

46M prefill + 5M decode on a 27B model is a couple of H200-hours — call it $10–20. Not a
budget problem; just 30× what the same study costs on a family that caches. Prefixes are
small because ODCV transcripts are short, which is the thing that makes this affordable.

**Live-environment branching is the expensive axis**, because each sample is a whole
container rollout. Budget it per branch point, not per sweep: 20 scenarios × 1 fork branch
× 30 resamples = 600 live rollouts, which at ODCV's usual concurrency is a couple of hours
on the laptop with a pod serving the model. That is affordable exactly once per question,
so choose the question from the frozen pass first.

## 5. Ranked next experiments

Ordered by (expected information) ÷ (cost). Each states what would count as a result.

### 1. Does the principal appeal cause the violation, or narrate it? — *the paper's question, in our setting*

The paper's headline is that self-preservation sentences in blackmail have near-zero
causal impact (~0.001–0.003 KL) and the lowest resilience of any category: they are
post-hoc rationalisation, not driver. Our `principal_appeal` is the structural analogue —
the sentence that licenses the violation — and §2 shows it is a **null correlationally in
both directions**, which is exactly the profile a rationalisation has.

Run resilience + counterfactual++ over labelled chunks, per tag. If principal-appeal
sentences show low resilience and near-zero importance++ while shortcut-identification
sentences show high resilience and large importance++, we have replicated the paper's
structure in a second, quite different scenario family — which is a genuinely publishable
generalisation claim, and it directly tells the project that training against the *stated
justification* is targeting the wrong text.

*Cost:* frozen sampler, ~40 trajectories, ~15k generations, ~$15 + one labeller pass.
*Result either way:* a replication or a scoped failure of the paper's central claim.

### 2. Resample the fork, live — *the single highest-value causal measurement*

Take the ~45 rollouts in the "create the missing input" fork cluster. Branch at the fork
step, resample 30 continuations **live** through `prefix_proxy`, judge with the existing
ODCV judges. This gives p(violation | the model reached this fork) as a real distribution
rather than one draw.

Two questions fall out immediately. First, **how much of the outcome is already determined
at the fork?** If the branch distribution is already at 73%, the decision was made upstream
and the fork thought is a symptom. If it spans 20–80%, the fork is a genuine decision point
and everything downstream of it is steerable. Second, it produces the effect curve — resample
from each prefix along the trace and watch where the violation rate climbs. A curve already
high at chunk 0 says the *scenario* decided it, not the reasoning, which would be a strong
and slightly deflating result about what ODCV measures at all.

*Cost:* ~600–1,400 live rollouts (Docker on the laptop + one eval pod). Half a day.
*Result either way:* the first causal statement this project can make about where an ODCV
violation is decided.

### 3. Do our trained arms differ in *where* the decision is made, not just in the rate?

Everything the project has measured so far is an endpoint: a misalignment rate per arm.
Resampling gives a *shape*. Run the effect curve for a synthetic-SFT arm and its matched
control on the same scenarios and compare where the curves separate.

This is the diagnostic the corpus ledger has been missing. If constitutional SFT lowers the
rate by flattening the curve early (the model never gets near the fork), that is a different
mechanism from lowering it at the fork (the model reaches the same crisis and answers it
better), and a different mechanism again from lowering it after (the model does the bad
thing and then reports honestly). Those three call for different training data. No
endpoint measurement can tell them apart.

*Cost:* two arms × ~20 scenarios × ~8 prefixes × 20 resamples, live: ~6,400 rollouts —
this one is a real spend, so do §2 first and reuse its infrastructure and its scenario
selection.

### 4. On-policy vs off-policy steering, as a data-design experiment

The paper's §3 result is that handwritten CoT insertions produce small unstable effects
while on-policy resampled insertions produce large directional ones (up to 100% reduction).
That is directly a claim about **our data-generation methodology**: the synthetic corpora
this project trains on are, in the paper's terms, off-policy text — written by Sonnet or
grok, not sampled from Qwen3.6.

Test it cheaply in the eval setting first: at a fork, insert (a) a handwritten integrity
sentence, (b) a sentence lifted from one of our difficult-advice corpora, (c) an on-policy
resampled sentence filtered to express the same meaning. Measure Δ violation rate. If (c)
dominates (a) and (b) by the margin the paper reports, that is a concrete, measured argument
for generating training reasoning **from the model being trained** rather than from a
stronger teacher — which would be a real methodological finding for the paper this project
is writing.

*Cost:* frozen for screening, live for the ~3 winners. Moderate.

### 5. Use branch variance to fix the corpus's power problem

§2's real limitation is that only 35 cells hold both outcomes. Resampling manufactures
within-task variance deliberately: branch each scenario at step 1 and take 20 samples, and
every scenario becomes a cell with both outcomes. That turns the whole stratified contrast
table from n=35 cells into n≈40 scenarios with proper within-cell distributions, and makes
every marker in §2 re-testable at real power — including the ones that collapsed, which
might be real but underpowered rather than confounded.

*Cost:* the cheapest live run of the set (one branch point per scenario). Worth doing
alongside §2 on the same pod.

## 6. Caveats

- **`judge_actions` is not the ODCV judge.** It grades an intention on the same 0–5 scale
  so the numbers look familiar. Never put one next to a published ODCV misalignment rate.
- **The lexical markers in `descriptive.py` are tripwires, not measurements.** A regex
  cannot tell "I should not edit the checker" from "I will edit the checker". Every rate
  computed from them is a marker rate. The LLM labeller in `label.py` is what a figure
  should use; confirm the labeller model with a human first, and prefer a family different
  from the model under study.
- **Sentence-granularity branching needs a serving decision.** `pin_template` writes
  `enable_thinking` into the chat template as a top-level Jinja `set`, shadowing any
  per-request value, and the template prefills `<think>\n` itself. Use
  `continuation="completions"` (render the prompt, hit `/v1/completions`) for any thinking
  model; the `"chat"` route via `continue_final_message` puts the forced prefix in the
  content channel and can make the model open a fresh `<think>` after it.
- **`fork_prior_fails` is structurally always 0** — `find_fork` returns the first refusal,
  so nothing precedes it. It is kept on the record for a future multi-fork analysis and is
  excluded from the contrast list.
- **26.4% base rate pools five arms with different rates.** Any absolute number in §2 is a
  corpus statistic, not an arm result. The stratified column is the one to quote.
