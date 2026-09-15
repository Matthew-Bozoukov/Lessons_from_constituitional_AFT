<!-- ABOUTME: Document the distinct saved-answer offline adoption and publication interface. -->
<!-- ABOUTME: Keep old model reviews, fresh independent review and root acceptance separate and hash-bound. -->

# Saved-answer offline acceptance

This helper makes no API calls and never changes an origin row or exclusion. A prepared dossier is **not accepted**. Acceptance requires a separate independent Codex agent's full source/answer review, the frozen full new09 and qualified craft contract, exact native Qwen token/masking evidence, a duplicate adjudication artifact, and explicit root approval. Original automatic judge outcomes are retained as historical evidence; they do not certify a revised answer.

The implementation must be frozen before preparation/adoption. Each dossier binds its helper SHA and portable validation refuses a changed implementation. The original run, per-row engine and pilot remain untouched.

## Prepare

Run `uv run python -m scratch.dataset_refresh.offline_acceptance prepare --spec SPEC.json --output NEW_DIRECTORY`. The new directory must be outside the immutable origin. Preparation checks current frozen source identity/eligibility and the actual successful Sonnet call against the original shared ledger; it then copies all original row JSON, receipts and exclusion/failure archives plus exact author/input/config/review evidence. No ledger copy is used to authorize new spending; this helper does not spend.

Required JSON spec fields:

```json
{
  "source_ref": {"root": "ABSOLUTE_ORIGIN", "arm": "nonmoral-advice", "candidate_id": "t1_004_v0", "result_sha256": "FULL_SHA"},
  "author_kind": "single_saved_revision",
  "review_config_path": "QUALIFIED_CONFIG.json",
  "review_config_sha256": "FULL_SHA",
  "review_contract_path": "independent_review_contract.json",
  "review_contract_sha256": "5d1ec43313b10807e50f245c6a342e093f8a324f20889a00803e5276c22860e2",
  "input_path": "CANDIDATE.input.json", "input_sha256": "FULL_SHA",
  "result_path": "CANDIDATE.result.json", "result_sha256": "FULL_SHA",
  "raw_path": "CANDIDATE.raw.json",
  "physical_receipt": {"call_id": 11605, "request_sha256": "FULL_SHA", "raw_sha256": "FULL_SHA", "ledger_entry_sha256": "FULL_SHA"},
  "ledger_path": "ABSOLUTE_ORIGINAL_BUDGET/spend.json"
}
```

`single_saved_revision` consumes the frozen pilot-compatible input/result checkpoints. Campaign inputs additionally bind `case_key=origin_directory_name::candidate_id` and the exact `source_ref`; that namespaced key must match the physical ledger entry. The input must contain the full working preference and actual unchanged system/user in the *submitted request*, not just in metadata. Returned reasoning/response must match the settled Sonnet raw output exactly after the frozen tag parser's whitespace handling.

For `author_kind="untouched_saved_final"`, replace input/result fields with `saved_stage` and `saved_stage_sha256`. Point `raw_path`/`physical_receipt` to the exact original successful author call. The original terminal may be failed, rejected or independently excluded; those states are preserved. A receipted source terminal with its original record, identity, scenario and eligible preflight is required. Pending rows without that evidence are unsupported, not fabricated into accepted terminals. Old working preferences remain in original metadata; the separately bound full qualified preference and full new09 are reviewed afresh.

## Review and adopt

The independent review follows the existing pilot review JSON schema: accepted/decision, all ten content gates, empty issues, exact source/result/conversation/request/receipt/full-review/preference hashes, exact source and answer quotes, substantive eligibility/craft/constitution findings, and `reviewer_provenance={"kind":"independent_codex_agent","task":"/root/REVIEW_TASK","human_review":false}`. This is agent review, not human review or an automatic Sonnet judge pass.

Root approval is a new JSON file with these exact fields:

```json
{
  "approved": true, "route": "independent_offline_full_read_v1", "root_actor": "/root",
  "reason": "Concrete root adjudication of this answer and remaining caveats.",
  "dossier_sha256": "FULL_SHA", "conversation_sha256": "FULL_SHA",
  "independent_review_sha256": "FULL_SHA", "native_audit_sha256": "FULL_SHA", "native_input_sha256": "FULL_SHA",
  "duplicate_check_passed": true, "duplicate_evidence_path": "ACTUAL_ADJUDICATION.json", "duplicate_evidence_sha256": "FULL_SHA"
}
```

Then run `uv run python -m scratch.dataset_refresh.offline_acceptance accept --dossier DIR/dossier.json --review REVIEW.json --approval APPROVAL.json --native TOKEN_MASK_AUDIT.json --native-input ACTUAL_AUDITED_ROWS.jsonl`. It copies the review, approval, native input/audit and duplicate evidence into the dossier and creates immutable `acceptance.json` plus receipt. It does not mutate the original terminal, generate another response or publish anything.

Native evidence must bind the exact full JSONL bytes, contain the actual conversation exactly once, and report untruncated `Qwen/Qwen3.6-27B` training tokens at most8192 with positive assistant reasoning/final supervision. Duplicate clearance remains a substantive independent/root decision. This validator verifies the artifact bindings; it does not claim semantic duplicate detection is infallible.

## Publication integration

Use `validate_accepted(acceptance_path, expected_sha256) -> (training_row, audit)` alongside the original publication validator for legacy-route selections. Its metadata preserves original fields, removes trained/audit text such as `rewrite_changes`, namespaces the new scenario ID by exact conversation SHA, and records the true offline route, author receipt, original status and independent reviewer. It never invents `accepted_attempt` or old judge passes.

Preserve the **whole dossier directory**, including every file named by both dossier and adoption manifests, in the published generation audit. Add its acceptance SHA and origin reference to the final selection manifest. The returned training row contains only system, user, assistant content and assistant `reasoning_content`; adoption evidence is untrained metadata/audit material.

After individually validating every original/offline selected row, call `validate_release(rows)` for the common exact716, nine-trait quota, unique scenario-ID and normalized user-prompt gate. This does not replace the final joint semantic/lexical census, corpus and native audit, pinned replay comparison or complete publication snapshot. A four-row accepted pilot is not a716-row release.

Validation:32 focused offline tests plus21 frozen pilot tests pass. Four actual pilot dossiers, their independent reviews and their native evidence passed preparation-only preflight; no adoption was performed by this helper's implementation task. The current preparation is `offline_acceptance_preflight_v2` under the pilot review output; the earlier unadopted preparation is retained as superseded evidence.
