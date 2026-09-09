<!-- ABOUTME: Frozen v1 decision rules for the first paired nonmoral-deliberation experiment. -->
<!-- ABOUTME: Separates the executable paid pilot from later stages that depend on its measured feasibility. -->

# Protocol v1 — 2026-09-08

Status: local preparation complete for a **24-pair pilot**; awaiting paid launch approval.
No inference or training result exists for this protocol. Canonical intent:
[research brief](research_brief.md). File hashes are recorded in the accompanying local
freeze manifest. Amendments receive a new version and reason; never overwrite the record
of a failed gate. Nothing may be selected using new ODCV outcomes.

## Conditions and questions

| Arm | Definition | Role |
|---|---|---|
| A | Original nonmoral adapter at revision `2225547cec8bd312a1e025f02fb6b3321c047e4a` | Historical recipe, freshly evaluated |
| B | Broad comparative CoT | Revised recipe |
| C | Same prompts, decisions and complete final answers; execution/verification CoT | Paired content control |

Primary question B−C: added value of comparative reasoning within CoT. Secondary B−A:
effect of the entire revised recipe. A is not an isolated dataset-selection intervention.
One new LoRA per B/C, training seed 0. No capability tests, extra training seeds,
stakes experiment or short-CoT arm in this package. No promise of global optimality.

## Pilot: the next paid action

- Exactly **24 candidate pairs**, two from each of the 12 domains in
  `preferences/nonmoral_broad/preferences.md`. Pilot examples never enter training.
- Pipeline config: `configs/data/synth/nonmoral-paired.yaml`. Haiku 4.5 authors scenarios;
  Sonnet 5 authors the shared answer and two traces; Gemini 3 Flash reviews them with
  source labels hidden and trace order determined by an ID hash. Provider pins remain
  Anthropic / Anthropic / Google AI Studio, no fallback. These are constructed traces.
- Full prompt and final answer are fixed before writing either CoT. A pair is generated
  together to hold writing context constant. This is a design choice, not proof that
  other stylistic properties are held constant; inspect both traces for artifacts.
- **$10 maximum**, included in the session's $100 data allocation. The wrapper reserves
  an upper text-token cost before each request, disables transport retry multiplication,
  keeps uncertain charges reserved, and persists the ledger across restarts. One worker,
  no asynchronous provider batch, no remote cache upload during the pilot.
- Offline command: `uv run python -m scratch.nonmoral.pilot`.
  Paid command, ONLY after approval: add `--execute`.
- Failures are retained; never reset the ledger or silently lower acceptance thresholds.
  Validate provider status/prices again at execution. No inferred authorization from a CLI flag.

Pilot review: inspect all 24 complete pairs locally alongside the blinded external audit.
Accept a pair only if every criterion below passes. Full generation can proceed under a
subsequent approved package only if **at least 20/24 pairs pass**, every domain has at least
one passing pair, no systematic moral/constraint failure remains, and the cost forecast fits.
These are operational pilot gates, not estimates of population purity or proof of validity.
Failure means diagnose/revise and record v2 before spending more; it never means relax
the definition to obtain the desired yield.

## Selection and validation rules

1. **Nonmoral:** no choice about wronging someone; stakes magnitude is a separate tag.
   Any detected moral conflict rejects the pair. Harmless costs, uncertainty and taste are allowed.
2. **Decision:** at least two defensible approaches and actual case-specific reasons for
   the choice. Unique correctness is allowed; forced tactics/rote calculations are excluded
   from this first corpus. Mathematical method and game-plan choices remain eligible.
3. **Completion:** one user and one assistant turn, ordinary neutral system optional;
   full bounded deliverable. Respect valid hard constraints. A conditional decision requires
   a real missing preference and complete actionable branches; no follow-up turn.
4. **Accuracy:** no fabricated task facts, unsupported technical claims or contradiction
   between metadata, prompt, traces and output. All code/math/layout/game claims need an
   applicable independent local check; unresolved checks exclude the pair, not silently pass it.
5. **Manipulation:** blinded comparison score must be **2 for B and 0 for C** on the rubric
   in the YAML. Both must be substantive, grounded and compatible with the same final answer.
   Neither may contain padding. Auditor uncertainty/disagreement triggers explicit local review;
   unresolved cases are rejected. Record the original labels and reason for any adjudication.
6. **Length:** tokenize CoT using the pinned Qwen tokenizer. Per-pair B/C token ratio must
   be within **0.8–1.25**, and the final selected corpus mean CoT length ratio within
   **0.95–1.05**. These bounds are an operational confound screen, not exact equality.
   No filler to pass. If valid natural pairs cannot meet the bounds, stop and revise the
   design rather than claiming that a content+length intervention isolates content.
7. **Diversity:** deduplicate prompts and semantic task families jointly across arms and
   held-out material. Inspect reasoning templates as well as surface text. No selection
   based on model performance in ODCV, no mining its tasks, and no condition-specific removal.

Full target, conditional on pilot feasibility: **684 selected pairs = 57 per domain**.
Conditional decisions capped at **34/684**; tag and report the actual count. Math/game
subtypes are recorded, with no forced quota within domain 12. Quotas prioritize balanced
coverage, not an assumed natural domain frequency. Generate candidates in bounded waves;
stop at the data-stage ceiling, not at an unlimited retry count. After validity screening,
choose by fixed hash order (`nonmoral-v1-selection` + scenario ID) within domains, with joint
length/diversity checks. Do not select the most eloquent or most anti-ODCV-looking responses.

Reserve **60 new held-out tasks**, five per domain, from distinct scenario families before
training. Pilot is a third, separate split. Generated IDs and near-duplicate review must
agree on the split. Reject overlap, including paraphrases. The historical 18 unused examples
are not a substitute. The repaired legacy holdout helper remains an override probe, not the
new broad manipulation rubric.

Full-corpus audit: external content ratings on every pair; locally inspect all flagged
cases and a frozen random 60-pair sample of accepted cases. A hard failure in this sample
blocks training until the relevant failure class is screened throughout and re-audited.
Zero detected failures is not proof of zero contamination. Save all rejected rows and reasons.

## Training and evaluation

Retain byte-identical base conversations from the **9,284** historical Table-2 rows;
use the historical nonmoral mixture at revision `6364505df02b0020b030bf379bd42285a14de6a5`
as the pinned extraction source. Both new mixtures contain **9,968 rows**. Keep the
historical base-row thinking convention; correcting it is a separate intervention.
Record row hashes and equality of B/C prompts/final answers. Equal row presentation counts;
do not duplicate examples to token-match. Whole non-empty CoT plus final answer supervised.

Use shared `configs/train/sft.yaml`, Qwen3.6-27B, r64/alpha128/dropout .05, seed 0,
one epoch, global batch 16, LR 1e-4, max length 8192, dynamic batching, no packing.
Pin the same base model/tokenizer revision in both runs. Check full-mixture lengths and
the existing masking gate before any rental; no truncation allowed in the new synthetic rows.

Evaluate A/B/C under `configs/eval/odcv-nonmoral-paired.yaml`: same frozen harness,
base revision, thinking mode, 28,000 context, temperature .7, five complete passes,
40 scenarios × two variants, 400 rollouts per checkpoint. Record actual rollout seeds
if exposed by the harness; otherwise state that nonzero-temperature resampling was used,
without inventing a guarantee of independent deterministic seed assignment.

Primary judge: Gemini 3 Flash, same pinned provider for MR and task progress. MR uses
the repository's severity threshold **>=3**. Report exact numerators/denominators,
severity, submission rate, progress distribution and progress >=3. Submission is a proxy
for completing the task, not a capability or direct over-refusal measurement.

Infrastructure failure policy: preserve failures, recover from logged evidence where the
existing harness supports it, otherwise retry failed cells by the same fixed rule. Never
drop difficult cells or incomplete passes and publish a full-protocol headline. If the
complete grid cannot fit the cap, report the comparison incomplete. No extending the run
because an early result is close to significance, and no stopping because one arm looks good.

Preselect a secondary MR judge audit before any new scores: one hashed pass from each
scenario in each arm, alternating variants by scenario hash (**40 per arm, 120 total**).
Use Grok 4.20, available in the provider registry. If binary MR disagreement exceeds
**10%** on this sample, mark primary rankings judge-sensitive and do not claim a settled
ordering. Preserve both labels; no cherry-picked replacement. A full rejudge is future
work unless separately budgeted before the primary results are examined.

In-domain manipulation check: A/B/C answer the same 60 held-out tasks once with the same
decoding parameters. Judge comparison content, grounding, decision/completion and correctness
with source labels hidden. These are targeted intervention checks, not capability benchmarks.
If B/C show no measured behavioural separation, an ODCV null cannot settle whether an
effective change in deliberation would transfer. Record the manipulation outcome separately.

## Analysis and claims

Freeze B−C as primary and B−A as secondary. Compute per-scenario rates averaging the
two variants and all five passes. Bootstrap **10,000 paired scenario resamples**, analysis
seed **80085**, to estimate each difference. Use **97.5% intervals** for claims across
the two planned comparisons (Bonferroni family allocation); show ordinary per-arm 95%
intervals from the repository alongside them. This quantifies scenario uncertainty for
the fixed checkpoints; it does not measure training-seed variation.

Lower MR with degraded task completion/progress is not a success. Any observed negative
completion difference triggers inspection and a provisional result, not a tolerated-loss
margin. Sampling noise prevents proving literally zero degradation from finite runs.
Capability preservation remains unestablished until the user's later capability tests.
An interval crossing zero is unresolved, not equivalence. No post-hoc domain-based recipe
optimization; domain breakdowns of the training audit are quality checks, not ODCV tuning.

Deliver: public HF corpora, mixtures, adapters and results under LASR-Callum; exact result
table, MR/difference/task-progress charts and their source data, selection/audit report,
protocol hashes, and a short supported/unresolved/follow-up answer. Pilot raw candidates
remain local until packaged with an explicit unvalidated status; never publish them as
approved SFT data. No Git main merge in this session.

## Budget, autonomy and stage gates

Session proposed maximum **$290**: data/pilot $100; two SFTs $50; primary ODCV $80;
manipulation/secondary judge audit $20; failure/cleanup reserve $40. Spending is not
authorized by this document. The next approval request is the $10 pilot, which supplies
the missing measured yield, lengths and cost before a full-session launch decision.

A full-session approval would permit routine choices within the agreed scope after the
30-minute response window, with reasons recorded. It would not permit exceeding the cap,
running capability tests, changing the scientific contrast or modifying selection after
new ODCV results. Park affected work if no valid choice fits. No scheduled autonomy has
been installed. GPU stages need owned-resource lifetime caps, checkpointing and verified
teardown; Docker working today is not proof of all overnight failure modes being handled.

## Local preflight evidence

- Docker engine 29.6.2, Compose and temporary-network create/remove passed.
- SSH keypair and HF/OpenRouter/RunPod credential presence checked without printing values;
  this does not establish live account permission or available balance.
- All three pilot provider endpoints returned HTTP 200; selected pins active and listed
  prices matched the registry (Haiku $1/$5, Sonnet $2/$10, Gemini $.50/$3 per million
  input/output tokens). Snapshot: `output/nonmoral_investigation/20260908/provider_inventory.json`.
- 93 focused existing checks passed; six new offline pilot/split checks passed, including
  the whole 24-pair pipeline with injected fake responses. Fake responses test wiring only.
- Watchdog Windows liveness/detachment fixed. Historical holdout ID extraction fixed and
  refuses missing/conflicting IDs. No model calls, rentals or model capability tests run.
- Initial protocol revisions and full generation-run artefacts must stay distinct:
  approval now concerns the measured pilot; the complete remote training/eval path can
  only be validated on its authorized staged smoke runs later.
