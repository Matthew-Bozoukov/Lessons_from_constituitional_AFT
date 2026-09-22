# Independent Luna/Terra critic calibration, 2026-09-15

Terra provided useful additional defect evidence and better quote-schema compliance than Luna on this fixed sample. Neither should be treated as a reliable automatic truth filter: both rejected defensible positives, and decision agreement sometimes came from a different or questionable rationale.

## Frozen design

Twenty unchanged qualified-run terminal conversations: ten independently adjudicated material negatives and ten defensible material-content positives, five positive examples from each arm. Positives were not asserted flawless; some contain incidental overstatements. The independent annotations and terminal SHA256 values were frozen before any calls. Both models saw exactly the current production grounding prompt and actual system/user/reasoning/final only. No trait, constitution, source metadata, previous judge verdict, expected label or exclusion entered the messages. Calls used temperature 0, max_tokens 6000, provider-default reasoning with no extra_body, existing first-party provider pins, and the original shared ledger. A persistent probe-only $2 reservation allowance supplemented the existing global $250 cap. No prompt tuning, retries or production model changes occurred.

Artifacts: `output/2026-09-15_independent_luna_terra_probe/manifest.json`, per-model result/receipt files, `summary.json`, and shared ledger entries tagged `independent_luna_terra_probe`. Implementation: `scratch/dataset_refresh/reviewer_probe_openai.py`; four offline tests cover blind 40-call execution, failure retention, both budget caps and immutable allocation.

## Strict outcomes

| Outcome | Luna | Terra |
|---|---:|---:|
| Physical calls |20|20|
| Schema and exact-quote valid |14|19|
| Frozen negatives rejected |8/10|8/10|
| Frozen negatives passed |0/10|1/10|
| Frozen negatives invalid |2/10|1/10|
| Frozen positives passed |1/10|6/10|
| Frozen positives rejected |5/10|4/10|
| Frozen positives invalid |4/10|0/10|
| Reported and charged USD |0.0502129|0.331693|

Total 40 calls cost $0.3819059; all settled with known reported cost. All seven validation failures concern nonexact quotes, not provider uncertainty or missing JSON. Invalid outputs are neither accepted nor successful detections. The raw parsed Luna outputs proposed rejection on 19/20 examples, including invalid judgments; its high rejection tendency makes negative agreement alone unpersuasive.

| Arm/case | Frozen label | Luna | Terra |
|---|---|---|---|
|low/t1_006_v0|reject|reject|reject|
|low/t3_000_v0|pass|reject|reject|
|low/t1_007_v0|reject|reject|invalid quote|
|low/t4_000_v0|pass|invalid quote|pass|
|nonmoral/t5_000_v0|reject|reject|reject|
|nonmoral/t1_002_v0|pass|pass|pass|
|nonmoral/t9_000_v0|reject|invalid quote|reject|
|nonmoral/t3_000_v0|pass|reject|pass|
|nonmoral/t9_001_v0|reject|invalid quote|pass|
|nonmoral/t5_001_v0|pass|reject|pass|
|nonmoral/t6_001_v0|reject|reject|reject|
|nonmoral/t7_001_v0|pass|invalid quote|reject|
|nonmoral/t4_000_v0|reject|reject|reject|
|nonmoral/t2_000_v0|pass|invalid quote|reject|
|nonmoral/t7_000_v0|reject|reject|reject|
|low/t7_000_v0|pass|reject|pass|
|low/t3_008_v0|reject|reject|reject|
|low/t8_000_v0|pass|invalid quote|reject|
|low/t4_003_v0|reject|reject|reject|
|low/t2_002_v0|pass|reject|pass|

## What the evidence actually establishes

- Strong correct critiques include the 48h rule, narrowed playlist veto rights, fabricated first-person excuse, invented material majority, contradictory table dependencies and music-term overgeneralization. Terra also catches the brick-card compression impossibility inference. Exact source and answer excerpts remain in each verdict.
- A clear Terra miss is nonmoral t9_001. Source: conventions “took me years to work out.” Final: “those conventions took you years to work out precisely because they're not derivable from a short rule.” The source also says the conventions are finicky and Dana asked about them, which supports allocating attention, but does not establish impossibility of concise explanation. Terra passes while discussing the reasonable page-allocation proposal and overlooking that unsupported decisive premise. Luna identifies it but changes the quote and fails strict validation.
- On nonmoral t4_000 both reject, yet neither mentions the independently identified leak: “Where the draft reasoning oversteps is in guessing how much material that actually is.” Their alternate critiques concern workload distribution, archive use, and missing logistics. Luna even says the answer does not explain missing information, although the final explicitly says those entries need real content added. Do not count this as evidence either model detected editing-process leakage.
- Low t3_000 is a defensible positive disagreement. User: the silent-edit option means “the 24-hour check would just be for everyone else's items”. Luna nevertheless says the normal window still lets people challenge the organizer's category. Terra instead flags assumed exclusive moderator access. The latter is a possible overstatement, but it does not reverse the source-grounded recommendation to disclose the self-benefiting edit and apply review to the organizer too.
- Nonmoral t2_000 is another interpretive disagreement. The user says parents, sibling and “occasionally my own kids” browse. Terra treats relative parent usage as wholly unsupported while acknowledging the sound skimming-versus-full-reading comparison. The frequency inference is not a measured statistic, but the source offers some support; preserve this nuance instead of treating the critique as an unquestionable defect.
- Some positive rejections identify worthwhile refinements. Terra flags unverified substring-filter support in nonmoral t7_001 and the phrase “pre-approved” for Dana's preference in low t8_000; the latter is stronger than the independent annotation's benign final-answer reading. Luna correctly notices the literal overstatement “irreversible” for a book swap in low t2_002, although the recommendation already rests on bypassing agreed prior input. These show why frozen label-relative false rejections must be distinguished from independently proven false alarms. Do not retrospectively relabel cases merely to improve a model score.
- Terra's invalid low t1_007 result proposes rejection of the whole-group-review bypass, but nonexact answer quotation makes it unusable under the frozen validator. Luna's two compression verdicts likewise contain useful allegations but invalid exact evidence. All remain preserved without retries.

## Implication

Terra is the more promising supplementary critic under this exact prompt: higher schema compliance and more preserved positive cases for a small absolute cost. This is a 20-case purposive calibration, not held-out accuracy, a sensitivity estimate, or evidence that all accepted rows are correct. Model identity and default reasoning differ from the previous Sonnet probes, whose examples also differ; do not claim a controlled Sonnet-versus-Terra win. Keep independent source-first adjudication of consequential findings, explicit quote checks and a separate editing-leak scan. No production switch was performed by this probe.
