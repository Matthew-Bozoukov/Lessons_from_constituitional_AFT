# Preserve an incomplete refresh without claiming a training release

This is the fallback if either arm lacks a defensible final 716-row selection when the shared $250 limit is reached. It does not reduce quotas, admit rejected rows, build a smaller substitute mixture, or change `publish_composite`'s release checks. A completed arm can still use the normal release workflow independently; the audit bundle preserves all work across both arms.

## Identity and readiness

Build the name using `src.naming.artifact_name('dataset-refresh-incomplete-audit', date=<snapshot UTC date>)`. This is a general research artifact, not a `*-synth` corpus or `*-mix`. Use the configured HF organization through the existing `push_run_dir` helper. `CLAUDE.md` requires bulk artifacts to live on HF and every upload to carry a complete card; it does not require an incomplete artifact to masquerade as a completed dataset.

The top-level README begins **INCOMPLETE RESEARCH AUDIT — NOT A TRAINING DATASET**. `readiness.json` contains `train_ready:false`, `mixture_ready:false`, `target_rows_per_arm:716`, exact effective accepted counts and per-trait shortfalls, unresolved holds, remaining partial checkpoints, final budget exposure and its uncertainty. Distinguish automatic accepts, independently cleared accepts, and actually selected release rows. Never substitute one count for another.

Do not place `dataset.jsonl`, `mixture.jsonl`, or a train split at the top level. If a Hub data configuration is supplied, it exposes only `audit_index.jsonl` as an explicitly named **audit** split: each row describes an archived phase and its hashes/counts, not a training conversation. The reusable conversations stay inside the original byte-preserving phase archives with their statuses and reviews.

## Exact source manifest

`output/2026-09-15_dataset_refresh_quality_screen/partial_preservation_source_plan.json` lists the five immutable run roots and the supporting evidence directories currently known. This is a proposed intake manifest, not a prepared snapshot or an approved upload. Before preparing, root must check it against the final experiment inventory, add any newly created final-selection/audit evidence, and freeze its SHA.

Include all of the following:

- Original pilot, revised pilot, first all-Sonnet phase, qualified phase, and diverse low-stakes phase. Keep their actual historical model settings; the audit archive can contain earlier pilot models and must not claim that every archived call used the final author recipe.
- Every frozen config, source snapshot, candidate schedule, run metadata/receipt, immutable execution phase/outcome, terminal, source review, eligibility decision, author draft/final, critic output, exclusion, adoption, failed-stage archive, recovery receipt, and independent audit. Empty operational lock files can be indexed rather than treated as substantive evidence.
- The entire closed shared ledger prefix and **every** raw physical call in that prefix, including unscoped original calls and separate reviewer calibration calls. Do not use `publish_completed.scoped_ledger` alone: its deliberate run-scoping omits early calls without `run_root`. An explicit index should map those original unscoped entries to the original root only when the frozen evidence supports that mapping; otherwise label them unscoped rather than guess.
- All billing-reconciliation archives, their original raw/accounting records, exact provider GET responses, catalogue canonical-model mapping, plans and applied receipts. `billing_verified_failure` remains failed content. Unknown costs remain reserved; a closed snapshot may preserve unknown outcomes but must state them honestly and never treat them as zero.
- All calibration fixtures/prompts/labels/verdicts and reports. Include the final **23-case** short-draft accounting/audit bundle (`short_draft_authorized_all23_completed.json` and its `_audit` directory), not just the superseded first 18. Include the final full-answer audit of the 20 nonmoral billing-retry accepts and its applied exclusions/adjudications.
- Final census, domain/template checks, lexical/semantic adjudication, tokenizer/mask evidence, any proposed selection manifests and policy decisions. Historical/provisional audits retain their actual input SHA and must not be relabeled as final-selection audits.

## Closed local bundle preparation

Preparation is read-only with respect to every origin. It must acquire all relevant run execution locks and the shared spend lock, require no active `reserved` calls, and require an explicit cutoff equal to the then-current ledger length. Preserve unresolved failures; do not reopen them. Reject an exposure above $250 or invalid/nonfinite amounts. Validate corrected charges with `billing_evidence.validate_billing_call`.

Use one byte-preserving archive per origin and a separate shared-budget archive. Exclude the original root's nested budget directory from its origin archive to avoid duplicating the shared budget. Include supporting quality/calibration evidence in separate named archives. Reject symlinks/junctions and resolved path escapes. Inventory every source file before copying, hash every archived byte stream, and verify source inventories again afterward; a changed file makes the snapshot incomplete and nonpublishable. JSON audit records go through the existing transport-credential check. Never collect `.env`, credentials, or an arbitrary parent directory.

A source-code archive names the full committed repository SHA. Include all generating/validating/recovery/publication helpers, source adapters, resolved source configs, constitution and craft specifications, and lockfile. Existing `freeze_code` can preserve current code, but old phases retain their original recorded core/config hashes and exact resolved prompts: do not imply that the latest commit was the historical generating commit for every phase.

The bundle also contains the exact source manifest, per-archive/per-file hash inventory, original absolute-to-archive path map, final status census, ledger cutoff, card fields/front matter, and a preparation receipt identifying the source commit, helper SHA, command and timestamp. This preserves enough evidence to reconstruct original paths, inspect failures, and resume explicit pending candidates under a separately authorized future budget; it does not silently authorize more calls or reset the old ledger.

## Review and upload boundary

Root reviews the concrete prepared folder, its exact file hashes, target HF name, complete README and readiness/shortfall summary before authorizing upload. Use `src.infra.huggingface.push_run_dir` with all required card fields (`experiment`, `date_generated`, `constitution`, `source_repo`, `models`, `generation_config`, `schema`, `provenance`). The card records the actual new constitution SHA, original and operational craft hashes, nosynth pin intended for the unfinished mixtures, all phase model settings, seed/sampling/budget settings, exact reconstruction command and the fact that training/evaluation were not run.

After upload, resolve and record the full HF commit, verify the uploaded file inventory/hashes, and put the immutable HF link in the repository's allowed experiment report. Do not change `CLAUDE.md` or `docs/TODO.md`. Keep the local work and evidence until remote verification succeeds.

The implementation is `scratch/dataset_refresh/preserve_partial.py`. After root adds readable final counts/shortages/review limits as `final_summary` in the source manifest, closes all source-writing work and commits all helpers, prepare with:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.preserve_partial --manifest <final-source-manifest.json> --destination <new-audit-snapshot-directory> --source-commit <full-commit> --ledger-end <closed-exclusive-cutoff> --date <UTC-snapshot-date>
```

The helper does not upload. Zero-byte or one-byte operational OS lock markers are indexed separately because Windows can make the locking byte unreadable; substantive files are preserved exactly. Missing historical metadata receipts are explicitly labeled absent, while present receipts must verify. The complete README, readiness record, archive inventories and file manifest are ready for root review before any upload.

No bundle preparation or upload has been performed by this plan.
