<!-- ABOUTME: Outcome of the authorized, bounded full-response grounded revision pilots. -->
<!-- ABOUTME: Records stopped gates, exact cost, local disagreements and preserved evidence; no model was trained. -->

# Grounded revision: stopped after two pilots

The user authorized [the execution plan](2026-09-10_execution_plan.md) on September10.
The candidate stopped at its predefined pilot gates. **No production generation,
SFT, ODCV or stakes run was launched.** There is no new alignment result.

| Measure | Pilot01 | Corrected pilot02 | Required to scale |
|---|---:|---:|---:|
| Existing source prompts, unchanged |16|16|16|
| Complete Opus revisions |16|16|—|
| Complete independent Sonnet reviews |16|16|—|
| Sonnet accepts |16|16|Not sufficient alone|
| Local usable, textually changed revisions |2|10|≥12|
| Local concrete improvements among usable revisions |1|3|≥4|
| Recorded API cost |$1.362485|$1.301459|$50 entire data phase|

There were64 paid calls total, all settled in the token-price ledger:32 Opus4.8
author calls and32 Sonnet5 review calls. No transport failures, parser retries,
additional semantic revisions or literal-repair calls occurred. The four paid
phases took approximately62,81,58 and62seconds. Wall time between the first
paid dispatch and the last review was about7.7minutes; local inspection and
preparation are separate from those generation times.

**New exposure: $2.663944. Cumulative project exposure: $167.894061/$300.**
The cumulative figure includes conservative reservations from previous work;
it is not a shared-account balance difference. The approximate production
forecast was **$71.61**, computed as spent cost plus652 unattempted sources at
pilot02 mean author/review cost, multiplied by1.30. This exceeds the frozen
$50 data cap even before extra replacement/repair costs. Forecast uncertainty
does not justify raising that cap after seeing the pilot.

## What happened

Pilot01 anchored strongly on the supplied old answer. Its reasoning frequently
discussed editing a prior assistant response that would not exist in the training
conversation. Sonnet accepted every row, sometimes explicitly noticing these
inconsistencies while excusing them because the final artifact was usable.

The one permitted prompt correction made the intended training context explicit:
standalone reasoning about the original task, editing notes confined to metadata,
and independent source checks. The fresh16 sources were disjoint from pilot01
and the six earlier plan-development fixtures. Model identities, temperatures,
reasoning settings and source prompts stayed fixed. This removed the obvious
editing commentary, but mostly produced paraphrases instead of substantive changes.

Three locally supported improvements in pilot02:

- `broader_batch10_021`: explicit empty stove-side shelf slots make the promised
  center/right placement unambiguous; all12 jars remain within shelf capacities.
- `broader_batch19_054`: explicit echo-return path derivation supports the previously
  asserted counts. An independent image-source distance calculation reproduces them.
- `broader_batch17_108`: corrects addressing the user as the opponent in the dice
  explanation. Exhaustive enumeration confirms10points over two lanes is optimal.

Representative unresolved defects:

- `broader_batch12_043`: baking/oatmeal usage does not establish how frequently
  cinnamon is used, yet it is assigned an unmarked occasional-use classification.
- `broader_batch07_106`: a valid melody is justified by dismissing repetition's
  catchiness, the precise benefit the user explicitly wanted weighed.
- `broader_batch12_051`: the target has a pause between the two notes, while the
  assembled practice pattern places its explicit pause after each high/low pair.
- `broader_batch12_001`: unsupported fixed time/price claims dismiss decanting,
  and Italian is assumed to be the most-cooked cuisine without source support.

These are local judgments about this development sample, not population error-rate
estimates. Subjective choices, mathematics, code and geometry were not rejection
criteria. Valid but unchanged or merely restated answers were not counted as useful
improvements. Sonnet's acceptance rate is not a correctness measurement.

An initial local suspicion about pilot01 receipt character counts was **wrong**:
running the supplied Python verified the counts and displayed widths. That claim
was retracted in chat and in the saved local audit. Other concerns were assessed
separately; the receipt was not rejected for those correct counts.

## Preservation and downstream status

Exact original inputs, both frozen prompts,64 raw requests/responses, stage
snapshots, model reviews, local per-ID decisions, executable checks and costs
are under `output/nonmoral_grounded_revision/20260910`. The final `pilot_gate.json`
records both failed gates; authorization is stopped and a STOP_DISPATCH marker
prevents further requests. No GPU owner was created. The thread heartbeat is paused.

The public pilot audit is published through `broader_data.py --revision publish-audit
--execute`, using `src/naming.py::artifact_name` and the shared HF card/push gate.
It is explicitly audit-only:32 candidate pairs, all marked not approved for training.
There is no training mixture or model publication for this candidate. This audit
preserves useful examples and failure evidence without presenting them as a completed
training corpus.

[Public audit](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-grounded-revision-pilot-audit)
revision `58f80cd4ec9d2f7749d66a1e92153399b301ee38`: candidate JSONL, manifest,
results and snapshot-hash index were downloaded at this pin and matched local bytes.
RunPod inventory at14:45:31UTC contained no matching experiment pods; five unrelated
shared-account pods were left alone. No local generation/training/evaluation owner
remained running. Code and findings are pushed on `codex/nonmoral-next-plan`.

The result rules out scaling **this frozen revision recipe under this budget**.
It does not rule out nonmoral deliberation or establish why the earlier broader
model underperformed. More retries on this recipe are not the next step. Any next
experiment should specify a substantive change in the decisions being learned,
and demonstrate its cost on a small sample before reserving a full corpus.
