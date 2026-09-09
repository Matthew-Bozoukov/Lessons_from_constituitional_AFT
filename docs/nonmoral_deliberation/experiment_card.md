<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Current execution and exact outcomes; historical decisions remain in LOG and frozen artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-09. Full intent and constraints: [research brief](research_brief.md).
Chronological evidence: [experiment log](../LOG.md). Earlier failures and reset:
[postmortem](2026-09-08_postmortem.md).

**Resumed by user authorization.** The correction rule is settled: preserve the original
request and substantive decision; permit one documented local correction of an isolated
verifiable error, or removal of an incidental false verification claim. Central reasoning
failures, incomplete sources and broken artifacts needing reconstruction remain excluded.
Every recovered candidate is read in full; originals, literal edits and checks are retained.
Changed text receives one independent review; existing model judgments can be reused only
for byte-identical conversations. No repeated repair/rejudge loop.

Examples of allowed corrections:5:45→5:50 when the supplied70-minute interval ends7:00;
“aligns exactly”→“spans the window” when the recorded coordinates establish only the latter;
copy an already-correct requested explanation into the final answer. Do not replace a false
central argument with a newly invented one. Calculations/code/geometry remain welcome.
Recipe10 keeps the twelve families and full outputs, and removes the detailed puzzle
variation library. Recovery and new production run in parallel under the same$300 ceiling.
The [pause audit](2026-09-09_process_reset.md) remains the historical snapshot.

## Current follow-up: broader dataset

User authorized Opus 5 for **both source scenarios and full answers**. Batch09 saved
12 sources (10 locally accepted), but 9/10 answer requests returned provider
`content_filter` errors; only one answer was saved. **Opus scaling stopped.**
Pilot exposure is $4.202980 of its $5 ceiling, including conservative reservations
for unknown failed-call billing. No new training rows are admitted from this pilot.
See the [pilot outcome](2026-09-09_opus_pilot.md). Earlier Sonnet candidates and
their generator provenance are retained; recovery work continues independently.

**Resolution:** a diagnostic with native refusal metadata identified an erroneous
cyber flag on a harmless baking-log script. Anthropic documents Opus 4.8 as a
supported fallback. Two Opus 4.8 checks completed (code output reproduced exactly;
poem constraints passed). Recipe11 uses Opus 4.8 for sources and full answers with
explicit low reasoning effort; Sonnet review is unchanged. Total Opus pilot and
diagnosis exposure is $4.680615 of $5. Resume production with 24 candidates under
the standing broader-data allocation; no broader LoRA has been trained yet.

**Working checkpoint:** 321 accepted, distinct conversations: 303 originally authored
by Sonnet 5 and 18 by Opus 4.8; all independently model-reviewed by Sonnet 5.
Recovery batch08 retained 105/106 after final adjudication, including 72 literal
corrections. The new 24-source Opus batch admitted 22 sources and produced all 22
answers without API failures; 16 were usable unchanged and two after isolated
time-label/rhythm-description corrections, while four had substantive faults.
The Opus production batch cost $1.401991 including source, answer and review calls.
This small yield check is not evidence of improved alignment or a causal model comparison.
Total project exposure at this checkpoint: $102.714437 of $300 (including conservative
unknown-call reservations). Next: scale the same recipe to 684 selected examples,
publish the frozen mixture, token/mask gate, one seed-0 LoRA, matched ODCV.

**Prospective breadth correction:** source batches12/13/15 repeatedly selected the
same topics: 26/30 learning requests mentioned whistling and 20/30 organizing requests
mentioned coffee. Batch15 is archived source-only before answer spending (119 saved
sources, one API refusal); these are not 119 quality rejections. Recipe12 restores
the existing variation field with 240 short context cues, not the old detailed puzzle
recipes. Batches17/19 freeze disjoint first/last ten cues per domain. Source and answer
models, author prompt, quality checks and budget stay fixed. Previously completed
data retain their original recipes; this adjustment uses no ODCV information.

Primary endpoint: **broader nonmoral corpus -> one new LoRA -> matched ODCV**.
The user accepts all twelve scenario families. No further taste checkpoint or baseline
is needed. Source requests stay unchanged; incomplete or contradictory requests are
excluded. Full reasoning and full answers are retained, with material-error checks.

Production batches01–06 retained **198 complete candidates** (7 + 11 + 56 + 48 + 39 + 37).
First12 remains a separate feedback packet. Batch06 admitted 94 of 120 sources and
produced 89 full answers. Final local review: 37 accepted, 51 rejected, 1 held.
All 89 completed independent model review; two model rejects were explicitly adjudicated
as defensible language/style choices from the full text. Five author calls failed;
terminal exceptions retain their maximum charges. Broader exposure: **$36.717364**.

Batch06 acceptance was **37/89 = 41.6%**, versus **87/208 = 41.8%** in batches04/05.
The simpler prompt has not demonstrated a yield improvement; task mix also changed.
From batch07, recipe8 changes review order: full local author review first, independent
Sonnet review only for local accepts, then explicit final local adjudication. Every
training candidate still gets both reviews. On the fixed batch06 outputs this would
avoid 52 candidate reviews (54 raw calls including formatting retries), saving **$2.005170**
of its **$8.007640** answer/review exposure. Historical spend is unchanged.

The shared tagged-output helper allows up to three formatting attempts: batch06 made
94 author calls and 93 reviewer calls for 89 answers (four extra formatting attempts).
Transport retries are disabled; no semantic repair loop. This distinction is retained
in `output/nonmoral_broader/20260909/selective_review_savings.json` and public batch06 audit.

Batch07 generated120 sources;93 passed local review,26 were rejected and1 held.
Author-only generation was stopped at the user pause with83 full answers saved.
Source-phase broader exposure was$38.027718; see the pause diagnosis for current costs. Production uses
8 workers,36 subtask directions and frozen per-phase configs; author/reviewer ceilings
remain16,384/12,288 tokens.
See [diversity rationale](broader_diversity_directions.yaml).

A second audit found the anti-repetition mechanism inserted3–4 raw historical request
excerpts into every batch07 source prompt;52/120 included previously rejected or held
batch06 requests. Several task templates still repeated. Recipe9 removes raw excerpts
from future source prompts while retaining the36 directions and the same quality gates.
This avoids conflicting guidance and input tokens; a yield benefit is unproven, and
batch06 already had high rejection without excerpts. Batch07 keeps its frozen recipe8.
Evidence: `production/batch07/rejection_anchor_audit.json` under the broader output root.

[Public growing corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-synth):
198 accepted default rows at revision `c463256cf44b2df7616b78a4a193d1cc720f71eb`.
Four public files were downloaded anonymously and hash-verified. Stage candidates,
dispositions, checks and raw calls remain separate audit material.

Quality diagnosis across batches04/05:208 authored answers,87 initially accepted,
111 rejected,10 held. Primary reject categories (manual attribution, overlapping causes
assigned one primary label):45 arithmetic/geometry/software,37 fabricated constraints
or false comparisons,9 requested explanations absent from the visible response,20 other.
Evidence: `output/nonmoral_broader/20260909/rejection_audit.json`. From batch06, the
prospective author prompt emphasizes actual benefits/costs and a defensible preference,
full visible deliverables, and only useful verification prose. Sources get a concrete
small-task guideline. Checks remain unchanged; lower rejection is a hypothesis to assess
in ordinary production, not an established improvement or a separate CoT-length arm.
A few borderline exclusions are flagged for individual adjudication, with no automatic
reversal. Historical corpus yield is not equivalent evidence: its recipe used rewrites,
requested artifact slices and disabled its quality filter.

Collect about700 accepted examples, then select **684** by fixed hash order with balanced
representation subject to available domain counts, before any new ODCV. Preserve the
exact9284 replay rows from the historical nonmoral mixture (revision
`6364505df02b0020b030bf379bd42285a14de6a5`, SHA256
`0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`).
This matches historical synthetic row count; it does not claim matched token lengths or
isolate a single causal mechanism. One seed0 rank64 LoRA, dynamic batching on2xH200.
No new main LoRA or ODCV results exist yet. Prospective broader-data cap increased
from$100 to$140 after observed quality exclusions; this fits inside the existing$300
authorization. Actual spend remains in the ledger, distinct from allowances.

**Secondary stakes lane is paused**, with no paid calls or GPUs running.
First8 retained4 candidate pairs at$0.609012. The subsequent generic loss wrapper was
stopped after12 of48 early answers described the added cost as unrelated; its full
exposure is$17.126848. A four-pair integrated craft trial cost$0.244464, made losses
relevant, but retained zero complete pairs because of reasoning/formatting defects.
Total stakes exposure: **$17.980324**, including unknown charges at full reserved bounds.
[Development archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-development)
revision`685a82c88faeb0b44297d53ecad5a42c84703752`; no stakes effect has been measured.

All allocations share the **$300 project ceiling**:
prior exposure$31.929031 + broader data cap$140 + stakes data cap$55 + main SFT reserve$40
+ main evaluation reserve$20 = **$286.929031 reserved or spent**, leaving$13.070969
unallocated. Caps are not actual spend; actual ledgers are separate and centrally linked.

Config: `configs/data/synth/nonmoral-broader.yaml`; artifacts and ledger:
`output/nonmoral_broader/20260909`. Local dispositions take precedence over Sonnet verdicts;
earlier model reviews missed material errors. No semantic repair loop or ODCV-driven selection.

## Completed overnight result

**Baseline fallback completed.** Evaluated the existing nonmoral-684, math-716 and
Table-2-only checkpoints with identical ODCV settings: 80 cells x 3 passes each,
temperature 0.7, context 28,000, both misalignment and task-progress judging.
These are **720 scored rollouts and no new training**. Exact checkpoint revisions and
protocol are in [the baseline inventory](baseline_inventory.md) and
[`scratch/nonmoral/odcv-paired.yaml`](../../scratch/nonmoral/odcv-paired.yaml).

All three are complete and public, with owned-pod termination verified.
[Public report, chart, exact results and audits](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-baseline-comparison)
are pinned at revision `a8fd17e2ac42ad7129596fdc3ce6e36703d0b7ec`.

| Checkpoint | MR | Submitted | Progress mean /5 | Progress >=3 |
|---|---:|---:|---:|---:|
| [Nonmoral](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch) | 33/240 (13.75%) | 236/240 | 4.921 | 236/240 |
| [Math](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64) | 92/240 (38.33%) | 215/240 | 4.771 | 231/240 |
| [Table2 only](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-only-9284-rank-64) | 90/240 (37.50%) | 224/240 | 4.8125 | 233/240 |

Nonmoral minus math: **-24.58 percentage points**, scenario-paired 95% CI
**[-34.01, -15.16]**. Nonmoral minus Table2: **-23.75 points**, CI
**[-32.85, -14.65]**. The nonmoral checkpoint also has the highest observed
submission and mean-progress values. This supports a checkpoint difference under
this protocol, not a deliberation-only causal explanation or a new training improvement.

All transcripts and both judging axes are present. Nonmoral has 3 context-limit
and 1 cycle-limit endings; math has 14 context-limit, 10 cycle-limit and **1 API
timeout**; Table2 has 10 context-limit, 5 cycle-limit and **1 API timeout**.
Both API timeouts were hidden by outer status `ok`/executor exit 0; raw-log
audits expose them and their original partial behavior remains scored. No model
needed a transcript reconstruction or an outer scenario-timeout retry. Progress
without submission is capped at **4**, so progress >=3 is not a completion count.
Scenario 95% MR intervals: nonmoral **[7.79, 23.12]%**, math **[27.73, 50.17]%**,
Table2 **[26.66, 49.76]%**. All 45 nonsubmissions were inspected: one refusal-like
noncontinuation in nonmoral and one in Table2, with justification unadjudicated.
This selected audit is not an overall refusal-rate measurement.

Historical **73/400 = 18.25% nonmoral** and **98/240 = 40.83% math** used different
context/harness settings and remain historical observations, not this comparison.

## Fresh paired data: stopped at the frozen gate

| Item | Result |
|---|---:|
| Fresh development tasks | 32 across 8 domains |
| Valid after at most one correction | 13/32 |
| Correction failures | 11/32 |
| Source/construct exclusions | 8/32 |
| Frozen minimum required for scaling | 24/32 |
| Accepted domains; aggregate B/C token ratio | 6; 0.88874 |
| Settled generation cost | $1.204914 for 115 calls |

**No further generation, corpus scaling or new SFT will run in this overnight branch.**
The user explicitly authorized this fallback if validation failed. Remaining errors
include arithmetic, unsupported checks and review-history text leaking into examples.
Changing the gate after seeing these results would not be a legitimate repair.

[Public development artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-paired-development)
contains full examples, both versions, row-level reviews, raw requests and cost ledgers;
revision `105dc5c9c84d1de0361e14d453347777aaf60e85`. It is **not approved SFT data**.
Financial amounts in synthetic scenarios are fictional task details.

The intended contrast keeps original single-turn prompts and full final answers
identical. B weighs viable actions; C constructs/checks the selected answer without
weighing alternatives. Essential missing information excludes a prompt. Neither arm
may contain moral deliberation. This compliant paired recipe differs from the old
recipe that sometimes taught reasoned overrides of user instructions.

## Harness health and spending

The first partial baseline attempt was invalidated **before judging**: stale Windows
CRLF bytes in Linux scripts forced models to repair their environment. Its 32 completed
and 8 partial rollouts are preserved separately and will not enter the comparison.
Cost including cleanup: **$1.3044 estimated GPU/storage**. The owned pod was terminated.
[Public invalid/unjudged archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-nonmoral-invalid-crlf-unjudged-attempt)
preserves transcripts, the audit and repair evidence; revision
`6d2fc052115789118dbe5476d1e1313a58e801d2`.

Restored 164 scripts to exact committed LF bytes; all 168 scripts passed Docker
`bash -n`. Physical-byte preflight and source hashes now protect the corrected run.
Independent transcript inspection checks real tool behavior as well as harness status.

User approved **$300 total including prior spending**. The baseline lane has a $60
GPU/storage cap and $10 judge cap, durable reservations, a separate pod watchdog,
progress monitoring and verified teardown. All GPU inference runs on RunPod; Docker
drives ODCV locally. Any future authorized SFT uses **2xH200, dynamic batching** and
the shared training configuration. Artifacts are public under `dougalldeepmind`.

**Final exposure estimate: $31.929031 / $300**, including prior settled $2.110418,
prior uncertain reservations $0.168748, fresh data $1.204914, invalid-attempt
GPU/storage $1.304417, and completed nonmoral/math/Table2 estimates of
$8.026093 / $8.868183 / $10.246257. All 1,440 baseline judge requests settled;
no new judge reservations remain. Costs use token rates and full owned-pod lifetimes,
not provider invoices or shared-account usage deltas. All four owned pods, including
the invalid attempt, are terminated. No paid work remains running in this branch.

## What this can establish

The baseline comparison measures these three fixed checkpoints under one protocol.
Report exact counts, scenario-level intervals and paired differences, plus submission
and task progress. Evaluation passes are **not training seeds**. Training recipe and
dataset differences prevent a deliberation-only causal claim. Capability tests remain
deferred; submission/progress cannot establish capability preservation.

Shorter reasoning, a valid retained-reasoning/no-comparison control, broader domains
and nonmoral stakes remain future experiments. The failed paired-data branch does
not answer those questions. The [bounded next-experiment proposal](next_experiments.md)
separates the repair-prompt diagnostic from a fresh validation batch; it authorizes no runs.
