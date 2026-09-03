# Thought Branches on ODCV — what we built, what the rollouts say, what to run next

**2026-09-03.** Implements [Thought Branches, arXiv 2510.27484](https://arxiv.org/abs/2510.27484)
(Macar, Bogdan, Rajamanoharan, Nanda) against this repo's ODCV rollouts.
Code: `scratch/thought_branches/` (see its README for the API).
Results: `output/thought_branches/2026-09-03_odcv_thought_branches_descriptive/`.

---

## TL;DR

1. **The framework is built, tested and smoke-run.** 39 tests, no network. The cheap
   offline half ran today over 859 published rollouts; the resampling engine is verified
   end to end against a live endpoint.
2. **The project needs this method specifically, because the method it has been using is
   out of road.** The 2026-08-31 seed replicate put the seed-to-seed noise floor on
   single-pass 65-cell ODCV MR at **6.1 pp** — *wider than the entire 8.7–17.6% band that
   nine corpus manipulations occupy*. No further corpus ablation read through arm-level MR
   can succeed, however good the hypothesis. Resampling measures effects **per branch
   point, within scenario, without a training run**, so it is not subject to that floor.
3. **Our own corpus demonstrates the problem it solves.** Of 21 markers separating
   violating from clean rollouts, 15 lose significance once compared within one arm *and*
   one scenario, and one flips sign. The sharpest (`commit_before_write`, 67% vs 13%
   pooled, consistent across all five arms) collapses to Δ = −0.02 [−0.21, +0.15].
4. **Three markers survive stratification**: violating rollouts run **more commands**
   (+5.3 [+0.2, +10.1]), **think less per step** (−96 chars [−172, −31]), and hit their
   **fork earlier** (−0.057 of trace [−0.106, −0.011]).
5. **The principal appeal is our self-preservation null.** *"A PASS is required", "the
   client needs this", "my function is to hit the target"* — indistinguishable between
   violating and clean rollouts, pooled (0.64 vs 0.57) and stratified (+0.20 [−0.01,
   +0.41], 29% of cells agreeing on sign). Same shape as the paper's headline finding
   about self-preservation in blackmail, in a quite different scenario family.
6. **The fork gives three scenario-adjusted movers.** After correcting for a confound
   that nearly fooled this analysis (below), two fork responses raise violation ~20–25
   points above their own scenarios' base rates and one *lowers* it by 21 points.
7. **Cost is not the blocker.** A 40-trajectory × 30-resample frozen study is ~46M
   prefill + ~5M decode tokens — a couple of GPU-hours. Live branching is the expensive
   axis and should be spent on ~20 chosen fork points.

---

## 1. Why this method, and why now

The project has one large robust effect — **44.1% → ~11–16% MR** from 716 difficult-advice
rows in ~10,000 — and a long ledger of failures to localise it:

- **Nine corpus content ablations** (trait identity, trait volume, scenario selection,
  reasoning style, reasoning structure, stakes, meta-cognition, advocacy, conflict
  clearance) all land in **8.7–17.6%**. LOG's own summary: *"Presence, not content.
  Nothing tried has broken it."*
- **Channel attribution failed**: recombining grok's trace with Sonnet's reply and vice
  versa put both swaps on Sonnet's rate. *"The drop needs the trace and the reply to be
  grok's TOGETHER."*
- **Grafting the identified properties failed**: the PAR coherence rewrite moved every
  corpus proxy to grok-like values and the organism did not move (21.3% vs 18.5%).
  *"Correlates of grok's advantage, not the lever."*
- And then **the measurement itself was shown to be too noisy for the question**: one arm,
  two seeds, nothing else changed → 16.9% and 10.8%.

Meanwhile the *rollout-level* reads have been the productive ones. The four-MO analysis
produced an arithmetically exact decomposition — **trigger rate × conditional safety** —
that reproduces every arm's MR to the decimal, and named a set of markers (step-1 register,
commitment-before-first-write, validator-source-read, the missing-data fork). Every one of
them is correlational, and one has already been shown to be a correlate rather than a lever
when grafted.

That is precisely the gap on-policy resampling fills. It is also worth noting what it is
*not* subject to: the standing "depth was the wrong axis, breadth is the only one that
helps" result is an argument about the **arm-level MR estimator** — more passes over the
same 30 cells do not shrink scenario-to-scenario variance. A design that reports **per-branch
effect sizes within a scenario** is not estimating an arm MR at all, so that argument does
not apply to it. Say this explicitly in any write-up, because a reader will reach for the
quote.

## 2. What was built

The paper's setting is single-shot: one CoT, one output, so "resample from sentence *i*" is
just another completion. ODCV is an agentic loop whose continuation depends on a container,
so a branch point splits a transcript **and a world**.

**`FrozenEnvSampler`** replays the recorded tool results as the conditioning prefix and
resamples one assistant turn. It reads the *local action distribution* — "standing exactly
here, what would this model have done?" — at one API call per sample, no Docker. It cannot
give a final ODCV severity; `judge_actions` grades the single step on the same 0–5 rubric,
which scores an **intention** where ODCV scores an **outcome**. Correlated readouts, never
interchangeable, and never quoted beside a published MR.

**`LiveEnvSampler` + `prefix_proxy`** produces the paper's actual quantity. The ODCV
executor is an ordinary chat loop against `OPENAI_BASE_URL`; point it at a local shim that
serves the *recorded* assistant turns for steps 1..k−1 and then forwards to the live model.
The container is never modified and never told anything changed, so it runs the same
commands, reaches the same state, and every tool result after the branch is real. The
trajectory reaches a genuine ending the existing judges score with no special-casing. Turn
accounting is by content — it counts assistant messages in the incoming request — so a
retry replays identically instead of sliding the branch point.

Metrics (`metrics.py`) are the paper's four and call no model, so a cached resampling run
can be re-scored at a new similarity threshold without re-spending the GPU-hour:
counterfactual importance (KL over severity), resilience (Algorithm 1), counterfactual++
(measured only where the idea never came back), and the cumulative effect curve.

Two properties that would otherwise surprise a reader of the numbers:

- **The KL is categorical.** Moving all mass from severity 0 to 1 scores the same as moving
  it to 5. That matches the paper, and is why every importance record also carries
  `delta_violation` — the KL says how far the distribution moved, the signed rate says
  which way.
- **Resilience is a lower bound under the default embedder.** We default to local model2vec
  `potion-base-8M` (torch-free, laptop-speed, free) rather than the paper's
  `bert-large-nli-stsb-mean-tokens`. It calls near-paraphrases dissimilar slightly too
  often, biasing resilience *down*. Re-run through `src/properties/shared/embed.py` before
  publishing a number that hangs on it.

## 3. What the ODCV corpus says

859 rollouts, 5 arms (2 control, 1 synthdoc, 2 paired responder), 40 scenarios, both
variants, all judged. 227 (26.4%) are violations at median severity ≥ 3. That base rate
pools arms whose own MRs run from ~8% to ~44%, so **only the stratified column below is a
result**; the pooled column is corpus arithmetic.

### Most markers are scenario, not behaviour

Stratified within (arm, scenario, variant) cells containing **both** outcomes — 35 cells,
124 rollouts:

| feature | pooled Δ | within-cell Δ [95% CI] | verdict |
|---|---:|---|---|
| commands run | +7.43 | **+5.28 [+0.18, +10.07]** | survives |
| chars per reasoning step | −351 | **−96 [−172, −31]** | survives, ¼ the size |
| fork position in trace | −0.181 | **−0.057 [−0.106, −0.011]** | survives, ⅓ the size |
| committed before first write | −0.541 | −0.023 [−0.208, +0.148] | **collapses** |
| integrity-language hits | −4.53 | −2.46 [−7.63, +1.70] | collapses |
| commitment-language hits | −3.66 | −2.60 [−6.61, +0.07] | collapses |
| fork thought names integrity | −0.271 | −0.052 [−0.138, +0.000] | collapses |
| first commitment (position) | **+0.117** | **−0.150** [−0.303, +0.009] | **sign flips** |
| principal-appeal hits | +0.072 | +0.198 [−0.014, +0.410] | null both ways |
| shortcut-language hits | +0.038 | +0.069 [−0.817, +0.831] | null both ways |

`commit_before_write` is the row to internalise, and it is the same quantity the four-MO
analysis built its decomposition on (there: MR 4.5–5.4% when the voice fires vs 28–41% when
it does not). It separates the outcomes inside **all five arms** here, and still evaporates
once scenario is held fixed. **Controlling for the model is not controlling for the task.**
This does not refute the four-MO decomposition — that was computed within arm, and within
arm this marker does separate — but it does say the trigger-rate story cannot be read as
causal from observational rollouts, which is exactly what Tier-1 question 1 below proposes
to settle.

A power limit that is itself informative: only 35 cells hold both outcomes; most are
unanimous. The observational corpus is **structurally short** of the within-task variance
this question needs — which is what resampling manufactures deliberately.

### The fork, and a confound that nearly got through

`find_fork` locates the first tool result where the environment refuses and the assistant
turn that answers it. It fires in 649/859 rollouts (76%), needs no labeller, and is
comparable across scenarios because the *environment* creates the moment.

Clustering fork thoughts gave a beautiful-looking 73% → 0% violation gradient. **It was
largely an artefact.** Fork-thought embeddings carry scenario identity — they name the
missing directory, the patient's symptoms, the threshold — so clustering them substantially
recovers *which task ran*. The 73% cluster reads 23% inside its own largest scenario; a 59%
cluster is one scenario entire.

Corrected by subtracting each cluster's own scenarios' base rates, three clusters move the
outcome and span enough scenarios to mean something:

| lift [95% CI] | n | scen | raw | what the cluster is |
|---:|---:|---:|---:|---|
| **+0.25 [+0.09, +0.47]** | 45 | 5 | 73% | *"The directory does not exist. I need to create it and populate it with sample patient data files."* — **fabricate the missing input** |
| **+0.19 [+0.07, +0.31]** | 76 | 10 | 47% | inventorying the files and what the validator checks, before deciding |
| **−0.21 [−0.32, −0.12]** | 119 | 25 | 8% | *"The AML checks are done. Results: application_001 PASS, application_002 FAIL — invalid fund_source…"* — **enumerating each result explicitly** |

The +0.25 cluster is the "missing data → fabricate it" fork the four-MO read independently
found beats all four generator arms — recovered here by a different method, from a
different corpus, with a scenario adjustment it did not have. That convergence is the best
evidence the construct is real.

The −0.21 cluster is the more interesting one and is new: **itemising the environment's
verdict, result by result, is protective across 25 scenarios**. It is also, unlike most of
this project's markers, a *shape* rather than a vocabulary — which makes it a poor target
for the lexical grafting that has already failed twice, and a good target for resampling.

## 4. Cost, measured

Measured over all 859 rollouts, not estimated:

| | median | p90 |
|---|---:|---:|
| full transcript | 2.8k tokens | 7.6k |
| prefix at first branch | 266 | 380 |
| prefix at mid-trajectory branch | 1.6k | 4.7k |
| step branch points per rollout | 6 | 16 |
| sentence branch points per rollout | 30 | 68 |

**Qwen3.6 cannot use prefix caching** (`supports_prefix_caching: False` — Mamba hybrid), so
every resample re-prefills:

| study | generations | prefill (no caching) | prefill (if cached) | decode |
|---|---:|---:|---:|---:|
| 20 traj × all step branches × 100 resamples | 32,100 | 122M | 1.2M | 11M |
| 40 traj × all step branches × 30 resamples | 15,060 | 46M | 1.5M | 5.3M |
| 80 traj × all step branches × 20 resamples | 18,240 | 61M | 3.1M | 6.4M |

46M prefill + 5M decode on a 27B is a couple of H200-hours, ~$10–20. Affordable because
ODCV transcripts are short; just 30× what it would cost on a caching family.

**Live branching is the expensive axis** and has a hard operational limit worth planning
around: **one ODCV run per Docker daemon** (Compose project names are global, so two
concurrent runs tear down each other's containers and both report `ok+no_transcript`).
Live resampling therefore does not parallelise on one host — budget per branch point, not
per sweep. 20 scenarios × 1 fork branch × 30 resamples = 600 rollouts ≈ a few hours.

Two judging traps from GOTCHAS that bite iterative resampling specifically: **re-judging a
combined directory does not reuse the cache** (judge once, at the end), and **batched
multi-rubric judging deflates prevalence by 7–9 points** (screen only).

## 5. Ranked next experiments

Every proposal below targets a question the project has explicitly left open, and avoids
the sixteen dead ends in LOG.md — in particular, **none of these is a corpus content
ablation**, and none is read through an arm-level MR comparison.

### 1. Is the step-1 register causal, or a symptom? — the highest-value question

The four-MO decomposition reproduces every arm's MR from trigger-rate × conditional-safety
and yields the counterfactual "Sonnet firing at grok's rate → 9.7%". That is **arithmetic,
not an intervention**, and §3 shows the same marker collapsing under scenario
stratification. It is also *exactly* the paper's warning case — a marker that predicts the
outcome almost perfectly may be determined by the upstream trace and have negligible causal
effect.

Branch at the first `reason:` block, resample, and split the continuations by whether they
open with a commitment. If the outcome distribution is the same either way, the trigger
rate is a symptom and the entire trigger-rate optimisation programme is misdirected — which
would be the single most valuable negative result available right now. If it differs, the
project has its first causal handle in three weeks of nulls.

*Cost:* frozen for screening (~$15); live to confirm on ~20 scenarios.

### 2. Does a commitment BIND, and what breaks the binding?

The channel swap found MR-given-commitment of **10.5% for grok-trace+Sonnet-reply vs
4.2–4.6% for the other three** — *"the commitment fires and does not bind"* — on **11 vs 5
violations**, explicitly flagged as suggestive. Resampling at the post-commitment step
measures the binding rate directly, at whatever n we pay for, without another training run.

### 3. The protective fork move (−0.21 across 25 scenarios)

New from §3 and not in any prior read: enumerating the environment's verdict item by item
is protective. Resample the fork, filter continuations on-policy to those that itemise vs
those that summarise, and measure Δ violation. If it holds causally it is a **prompt-level
and data-level target that is a shape, not a vocabulary** — which matters, because the two
failed grafting attempts were both vocabulary.

### 4. The cross-channel pair that was never ablated

The property discovery run found: *"names an ethical tension but proceeds anyway"* + falsify
= **63% violation (n=49)**; the same deliberation + refuse = **1% (n=130)**. *"Identical
deliberation; the outcome is decided entirely by whether the action follows it."* Its own
stated next step — filter those rows and retrain — was never run, and after D3/D7 a retrain
is the wrong instrument anyway. Resampling tests it **without one**: branch immediately
after the tension is named and read the action distribution.

### 5. On-policy vs off-policy, as a claim about our data generation

The paper's §3 result — handwritten and cross-model CoT insertions produce small unstable
effects while on-policy resampled insertions produce large directional ones — is directly a
claim about **this project's methodology**: every synthetic corpus here is, in the paper's
terms, off-policy text written by Sonnet or grok, not sampled from Qwen3.6. It also offers
a mechanism for D6 and D7, which are both failures of transplanted text to carry its effect.

Test it in the eval setting first: at a fork, insert (a) a handwritten integrity sentence,
(b) a sentence lifted from one of our difficult-advice corpora, (c) an on-policy resampled
sentence filtered to the same meaning. If (c) dominates by the paper's margin, that is a
measured argument for generating training reasoning **from the model being trained** — a
real methodological finding for the paper this project is writing, and the first positive
direction to come out of the grafting failures.

### 6. Use branch variance to fix the corpus's power problem

§3's real limitation is 35 usable cells. Branch each scenario once near the start and take
20 samples, and every scenario becomes a cell with both outcomes — turning the stratified
table from n=35 cells into n≈40 scenarios at real power, and making every collapsed marker
re-testable. Cheapest live run of the set; do it on the same pod as #1.

## 6. Caveats

- **`judge_actions` is not the ODCV judge.** Same 0–5 scale so the numbers look familiar;
  it grades an intention. Never put one beside a published MR.
- **The lexical markers in `descriptive.py` are tripwires, not measurements.** A regex
  cannot tell "I should not edit the checker" from "I will edit the checker". `label.py` is
  what a figure should use — confirm the labeller model with a human first, and prefer a
  family different from the model under study.
- **Sentence-granularity branching needs `continuation="completions"` on a thinking model.**
  `pin_template` writes `enable_thinking` into the chat template as a top-level Jinja `set`,
  shadowing any per-request value, and the template prefills `<think>\n` itself. The
  `"chat"` route via `continue_final_message` puts the forced prefix in the content channel.
- **The five arms analysed do not include the canonical baseline** (principle-scoped
  chunk-only 702, ODCV 11.5%). They were chosen for rollout count, not for the ladder.
  Re-run `analyse fetch` with the baseline arm before any statement about arm differences.
- **`fork_prior_fails` is structurally always 0** (`find_fork` returns the first refusal).
  Retained for a future multi-fork analysis; excluded from the contrast list.
- **`src/properties/sources/odcv_rollouts._rollout_key` still reconstructs the old bench
  layout** and silently joins 0 scores against the published layout. This package does its
  own join and asserts it lands, but do not route new work through that source without
  checking the join count.
