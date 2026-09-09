<!-- ABOUTME: Current short oversight card for nonmoral-deliberation research. -->
<!-- ABOUTME: Current execution and exact outcomes; historical decisions remain in LOG and frozen artifacts. -->

# Nonmoral deliberation: current experiment

Updated 2026-09-09. Full intent and constraints: [research brief](research_brief.md).
Chronological evidence: [experiment log](../LOG.md). Earlier failures and reset:
[postmortem](2026-09-08_postmortem.md).

## Current follow-up: broader dataset

Primary endpoint: **broader nonmoral corpus -> one new LoRA -> matched ODCV**.
The user accepts all twelve scenario families. No further taste checkpoint or baseline
is needed. Source requests stay unchanged; incomplete or contradictory requests are
excluded. Full reasoning and full answers are retained, with material-error checks.

Production batches01–04 retained **122 complete candidates** (7 + 11 + 56 + 48).
First12 remains a separate feedback packet. Batch04 admitted111 of120 sources,
produced105 full answers, and completed105 independent model reviews. Final local
review:48 accepted,52 rejected,5 held. Six author failures were not retried; five
exception-path reservations retain their full cost bounds. All105 conversations were
read locally; two local/model disagreements were adjudicated from the full text.
Broader exposure after batch04: **$18.758690**, including retained maximum charges.
Batch05 produced119/120 sources; local review is underway. One filtered source was
not retried; broader exposure after its full reservation is$19.752270. Production uses8 workers and frozen per-phase
configs, with16,384/12,288 author/reviewer token ceilings. No retrospective resampling. From batch06, recipe7 uses36 distinct subtask directions
within the existing domains; see[diversity rationale](broader_diversity_directions.yaml).

[Public growing corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-synth):
122 accepted default rows at revision`ccfc003805c8c8c3c2b073aa315daa2d48490550`.
Four public files were downloaded anonymously and hash-verified. Stage candidates,
dispositions, checks and stored raw calls are separate audit material.

Collect about700 accepted examples, then select **684** by fixed hash order with balanced
representation subject to available domain counts, before any new ODCV. Preserve the
exact9284 replay rows from the historical nonmoral mixture (revision
`6364505df02b0020b030bf379bd42285a14de6a5`, SHA256
`0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`).
This matches historical synthetic row count; it does not claim matched token lengths or
isolate a single causal mechanism. One seed0 rank64 LoRA, dynamic batching on2xH200.
No new main LoRA or ODCV results exist yet.

**Secondary stakes lane is paused**, with no paid calls or GPUs running.
First8 retained4 candidate pairs at$0.609012. The subsequent generic loss wrapper was
stopped after12 of48 early answers described the added cost as unrelated; its full
exposure is$17.126848. A four-pair integrated craft trial cost$0.244464, made losses
relevant, but retained zero complete pairs because of reasoning/formatting defects.
Total stakes exposure: **$17.980324**, including unknown charges at full reserved bounds.
[Development archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-development)
revision`685a82c88faeb0b44297d53ecad5a42c84703752`; no stakes effect has been measured.

All allocations share the **$300 project ceiling**:
prior exposure$31.929031 + broader data cap$100 + stakes data cap$55 + main SFT reserve$40
+ main evaluation reserve$20 = **$246.929031 reserved or spent**, leaving$53.070969
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
