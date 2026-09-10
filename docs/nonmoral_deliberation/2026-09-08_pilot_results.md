<!-- ABOUTME: Results of the first approved $10 paired nonmoral pilot, including failed gates. -->
<!-- ABOUTME: Reports candidate validity and auditing limitations; no model-training or alignment result. -->

# Paired nonmoral pilot v1: failed

**Only 1/24 candidates passed individual checks; the gate was at least 20/24.**
Generation and automatic auditing completed. No full dataset, LoRA, GPU rental or ODCV
run followed. This tests the data-generation process, not whether nonmoral deliberation
improves alignment. [Frozen protocol](protocol.md); [canonical research brief](research_brief.md).

| Outcome | Count |
|---|---:|
| Original candidates, two per domain | 24 |
| Complete shared answers | 19 |
| Complete pairs, all externally audited | 15 |
| Generation failures: answer / trace stage | 5 / 4 |
| Completed pairs rejected for local content failures | 10 |
| Completed pairs excluded for unresolved claims | 4 |
| Pass individual content, contrast and length checks | **1** |
| Domains with a qualifying pair | **1/12** |

The surviving case is the complete integer-sum proof (`pilot_t12_b00_s000`). Its B/C
CoT length ratio is 1.159, within the individual 0.8–1.25 bound. A selected set consisting
only of this pair misses the separate 0.95–1.05 corpus-mean bound. The pilot therefore
fails yield, coverage and mean-length gates; systematic quality failures also remain.
The single survivor is **not a training-approved dataset**.

## What failed

The generator made fluent answers that quietly changed facts or constraints. The
schedule allocated only 25 minutes to two hours of sorting/folding; another left no
time to eat dinner. The policy rewrite invented a completion time. Both report-generator
implementations passed normal inputs but failed a valid category named `__proto__`.
A toy population-model answer predicted a period-two cycle where direct iteration gives
period four. Traces often repeated or expanded those mistakes.

Other exclusions are uncertainties, not demonstrated failures: the exact cookie/rice
texture claims, the Catalan idiom premise, and the pasta timing/technique claim. These
remain separately labeled. Personal inconvenience or monetary/plant loss was not treated
as morality. One local moral-deliberation flag concerns explicitly weighing whether to
omit information about consequences because doing so would be evasive.

**The external judge was too permissive about accuracy.** Gemini marked every completed
pair accurate, complete, grounded and requiring no external check. Its comparison scores
allowed 9/15 pairs through; adding length left 8. Local review rejected 7 of those 8
(five content failures and two unresolved cases). This is a diagnostic disagreement
count, not a human-validated estimate of judge error. Local review was by the Codex agent,
not blinded to condition; judgments and source evidence are preserved for inspection.

The contrast itself often worked: Gemini rated B as comparison=2 in 15/15 cases and C
as comparison=0 in 9/15. Local/external comparison-score disagreements are retained;
none was used to rescue a candidate. Implementation checks involving odd n or comparing
measured trial results were distinguished from choosing between competing methods.
The control sometimes leaked genuine option evaluation, including choosing one English
phrase over another and justifying hand-grating against pre-grated cheese.

**Length:** 12/15 pairs met the individual bound. Across all completed pairs, B averaged
498.6 tokens and C 527.9 tokens (ratio 0.9446; pinned Qwen tokenizer). This is a manageable
confound to monitor, but cannot be declared solved from this failed sample.

## Cost and provenance

72 API dispatches: 12 scenario calls, 24 answer calls, 21 trace calls (including two
JSON-parse retries), and 15 audit calls. Recorded settled token cost: **$1.300936**.
Three empty length-limited responses retain **$0.168748** in conservative reservations;
total recorded exposure is **$1.469684**, below the authorized $10 cap. These are token
costs at verified provider rates, not a reconciled account invoice.

The original zero-failure pipeline stopped after each failing stage. Before any audit
labels existed, a documented continuation policy preserved the abort manifests and
resumed downstream work on successful checkpoint rows only. No failed candidate was
replaced after either abort; all nine remain in the original denominator. Original
prompts, models, thresholds and cumulative ledger were retained. The final engine
manifest reports only the last resume's $0.0281 audit cost; **use `pilot_spend.json` and
`final_summary.json` for the whole pilot**, not that per-invocation figure.

Models: Haiku 4.5 scenarios; Sonnet 5 shared answers/traces; Gemini 3 Flash audit.
Config: `configs/data/synth/nonmoral-paired.yaml`. Code started from Git `42b6acff`
on `codex/nonmoral-deliberation-controls` with recorded uncommitted changes. The prelaunch
freeze and uploaded source snapshots preserve those changes. Ten focused pilot/watchdog
tests passed after adding resume support; independent checks are reproducible from the
bundled script. All raw candidates and original labels are retained, including rejections.

## Recommended next step

Prepare v2 around **verified premises and answer validation before trace generation**:

1. Bound the actual deliverable and provide all required facts, durations, dimensions
   or code contracts. Remove invented source-language facts and forced-tactic cases.
2. Check the complete answer first, using arithmetic/code checks where applicable and
   explicit unknowns elsewhere. Reject invalid answers before paying for two traces.
3. Separate factual/constraint auditing from contrast scoring. Test the revised auditor
   against this pilot's known errors and clean controls before trusting acceptance labels.
4. Fix output-limit handling explicitly, then freeze a new balanced pilot and its cap.
   Preserve v1 failures; use new cases to test generalization beyond repaired examples.

No credible full-corpus cost forecast follows from the current recipe: 11 domains have
zero qualifying pairs. Re-cost after v2. Do not change the scientific gates to inflate
yield, select using ODCV, or infer a capabilities/alignment result from this pilot.

## Artifacts

Local run: `output/nonmoral_paired_pilot/20260908_141101/`.
The complete bundle is staged at
`output/nonmoral_paired_pilot/publish/2026-09-08-nonmoral-paired-synth/`. **HF publication
is blocked:** repository creation returned 403; the configured write token belongs to
`nikakogho`, which lacks access to `LASR-Callum`. No alternate configured credential
was available. Publication requires a token with access to the requested namespace;
no fallback upload to a different account was made. The intended HF ID is
`LASR-Callum/2026-09-08-nonmoral-paired-synth`.

The local bundle contains
`candidate_decisions.csv` (all 24), `dataset.jsonl` (15 raw complete pairs), original stage
checkpoints, `local_review.json`, `review_packet.md`, independent checks and sources,
cost ledger, configurations, provenance, and `pilot_outcomes` / `cot_lengths` PNG/SVG
figures. The bundle is explicitly a **failed diagnostic pilot, not training data**.
