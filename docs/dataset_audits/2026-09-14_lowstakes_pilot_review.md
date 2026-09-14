<!-- ABOUTME: Independent full-conversation review protocol for the low-stakes 18-candidate pilot. -->
<!-- ABOUTME: Predeclares evidence, row dispositions and stop conditions without delegating trust to the model judge. -->
# Low-stakes pilot independent review

This is the review protocol, not an approval or evidence that generation has run. Read the eventual frozen config and all18 candidate outputs under the root-provided run path. Include every failed or rejected candidate; never replace it silently in the pilot denominator.

For each candidate, read the original pinned source scenario and available source system/user, its exact target principle, generated system/user, entire draft if relevant to failure, entire final reasoning and entire final response. Read the author changes and judge verdict only after forming a provisional view. Failed provider or parser/lint outputs remain failed rather than being inferred acceptable. Preserve the raw evidence paths and candidate ID.

Assign independently:

- PASS: the final conversation satisfies every semantic requirement and deterministic lint/schema requirement.
- REJECT: complete conversation fails a requirement, with the exact clause or passage and concrete consequence.
- FAILED: no complete usable conversation, with the recorded technical/provider/length reason.

Record the actual domain, claimed versus assessed stakes0-3, human versus assistant decision actor, unchanged versus relocated domain, and whether the parent was exported. For a sourced row, describe the exact ethical mechanism that survived or changed. A shared trait ID alone is not evidence of mechanism preservation.

Check the whole conversation for plausible physical/animal harm, recurring or consequential financial costs, durable reputation or relationship loss and misleading worst_outcome summaries. Calibrate moral wrongness separately from consequence magnitude. Check that legitimate alternatives remain feasible and carry a modest meaningful cost; a harmless activity should not receive an invented moral refusal. Advice must leave execution and authority with the human; drafting useful text can be part of advice, but autonomous sending, editing, moderation or other tool action is not. Confirm it actually answers the user's request, obeys feasible hard constraints, uses available facts, and does not fabricate missing materials. Reject repetitive length-padding even if it passes the700-character floor.

Use a row ledger with candidate_id, trait_id, independent_disposition, model_disposition, actual_domain, assessed_stakes, actor, parent_exported, mechanism_preserved, defect_family, exact_evidence, rationale. When disagreeing with the model judge, preserve both verdicts and explain the difference rather than overwriting its saved review.

Approval requires at least16 PASS of18, at least1 PASS per each of9 traits, and no repeated material defect family affecting3 or more candidates. Up to2 REJECT/FAILED candidates are allowed; they are not admitted into the final corpus. No row gets repeated judge calls until it passes. Check similar defects even where the model accepted them. If the gate fails, retain the stopped pilot and describe the fix before the single permitted explicit design revision; subsequent failure is a stop/report, not a wider retry loop.

Write the independent row ledger and summary into the run's arm directory with dated names. Root owns the machine-readable pilot_gate.json approval decision after inspecting the independent report. This workflow does not authorize training or evaluation.
