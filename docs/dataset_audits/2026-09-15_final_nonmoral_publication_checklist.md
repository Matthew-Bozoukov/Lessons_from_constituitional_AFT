<!-- ABOUTME: Give the root an exact final716 nonmoral publication sequence with immutable quality and replay pins. -->
<!-- ABOUTME: Preserve historical author routes and inherited replay bytes; this checklist itself executes nothing. -->

# Final nonmoral publication checklist

Work in `C:/Users/nikak/source/repos/LASR/teaching_claude_why_replication_dataset_refresh`. Use `uv` and `$env:PYTHONUTF8='1'`. Complete independent full reads and root adoptions first. The release gate is650 fixed corrected-base rows plus66 valid separate offline acceptances, totaling80 each for t1–t5 and79 each for t6–t9. Review verdicts, source identity, old exclusions and first attempts remain unchanged. An authorized second correction does not erase its unsuccessful first answer.

1. Root freezes an explicit JSON selection with `arm: nonmoral-advice`, the base selection path below and its exact SHA, and exactly66 `offline_entries`, each containing the absolute `acceptance_path` and actual `acceptance_sha256`. No provisional or awaiting-review result qualifies.

   Base: `output/2026-09-15_dataset_refresh_correction_review/final_selected_nonmoral/pool_selection.json`; SHA256 `5e52a18a45dcef2065428c3a9b38d4a1f1c1c79b75a7f34817c572c0c9ebf0aa`.

2. Choose new output directories and run the exact publication transformation and local checks:

```powershell
$nonmoralSelection = 'ABSOLUTE_FINAL_SELECTION_JSON'
$nonmoralPreview = 'ABSOLUTE_NEW_PREVIEW_DIRECTORY'
$nonmoralEvidence = 'ABSOLUTE_FINAL_INDEPENDENT_EVIDENCE_DIRECTORY'
$nonmoralQuality = 'ABSOLUTE_NEW_SEALED_QUALITY_DIRECTORY'
$nonmoralSnapshot = 'ABSOLUTE_NEW_SYNTHETIC_SNAPSHOT_DIRECTORY'
uv run python -m scratch.dataset_refresh.publish_offline_composite validate --selection $nonmoralSelection
uv run python -m scratch.dataset_refresh.publish_offline_composite preview --selection $nonmoralSelection --output $nonmoralPreview
```

The preview uses cached tokenizer and embedding assets without network calls. Require the exact716 native Qwen8192 untruncated token/mask diagnostics and inspect all final semantic/lexical/literal candidates. Prior round checks are supporting evidence; they do not replace a final joint census after all66 additions are selected. Keep source-versus-answer duplicate distinctions and any reversed false positives explicit.

3. Root's final evidence directory contains the actual supporting reviews and `adjudication.json` with the exact preview `dataset_sha256`, nonempty `scope`, `release_approved: true`, and `unresolved_material_issues: []`. Bind final length/domain/token comparability and realistic limitations. Seal it:

```powershell
uv run python -m scratch.dataset_refresh.preview_release seal --preview $nonmoralPreview --evidence $nonmoralEvidence --output $nonmoralQuality
```

The seal hashes both automatic and independent evidence. Root should include the separate `output/2026-09-15_nonmoral_proposal_audits/utf8_replacement_character_investigation.json` correction: t1_040 stored text has two valid U+2014 em dashes, zero replacement characters. The earlier visual U+FFFD claim was output transcoding. No source edit is needed.

4. Close all generation/execution/evidence writes. Confirm every shared ledger entry is settled or billing_verified_failure, zero reserved/uncertain, and conservative charged total<=270. Freeze exact current Git commit and actual ledger exclusive cutoff; do not use an old count. Commit all new helper code before preparation. Initialize these variables only after closure:

```powershell
$releaseCommit = (git rev-parse HEAD).Trim()
$closedLedgerEnd = ACTUAL_FINAL_EXCLUSIVE_INTEGER
uv run python -m scratch.dataset_refresh.publish_offline_composite prepare --selection $nonmoralSelection --destination $nonmoralSnapshot --source-commit $releaseCommit --ledger-end $closedLedgerEnd --date 2026-09-15 --quality-dir $nonmoralQuality
```

Preparation revalidates the entire selection and freezes code, base origin records/receipts, all accepted offline dossiers, whole new author execution roots (including failed calls), first-attempt ancestry of every selected second correction, scoped raw calls, shared ledger, billing proofs and final quality. A failure preserves `PREPARATION_FAILED`; never reuse or push that directory. Root reviews the concrete README, publication manifest/file hashes,716 quotas, distinct author/review routes, training-only fields and cost accounting.

5. After root approves that concrete snapshot:

```powershell
uv run python -m scratch.dataset_refresh.publish_offline_composite push --destination $nonmoralSnapshot
```

Record the actual full HF commit from the resulting repository and independently verify remote file inventory/hashes plus fresh README/provenance/default train payload at that immutable pin. The intended repository is `dougalldeepmind/2026-09-15-nonmoral-advice-synth`; no moving or guessed revision is permitted. Keep receipts outside the snapshot. This step performs publication only, no training/evaluation.

6. Build the nonmoral mix using the already frozen low replay reference. The helper writes a no-upload config: its YAML omits `hf`; leave it that way so root can verify the actual mixture before publication.

```powershell
$nonmoralSynthRevision = 'ACTUAL_FULL_40_HEX_HF_COMMIT'
$nonmoralMixPrep = 'ABSOLUTE_NEW_MIX_PREPARATION_DIRECTORY'
$lowReplayReference = (Resolve-Path 'output/2026-09-15_low_mixture_release/replay_reference.json').Path
uv run python -m scratch.dataset_refresh.prepare_mixtures prepare --output $nonmoralMixPrep --single-arm nonmoral-advice --nonmoral-repo dougalldeepmind/2026-09-15-nonmoral-advice-synth --nonmoral-revision $nonmoralSynthRevision --replay-reference $lowReplayReference
uv run mix --config "$nonmoralMixPrep/nonmoral-advice.yaml"
```

Use the actual timestamped build directory printed by `mix`, not an assumed timestamp:

```powershell
$nonmoralBuild = 'ACTUAL_TIMESTAMPED_MIX_BUILD_DIRECTORY'
$nonmoralMixAudit = "$nonmoralMixPrep/mixture_validation.json"
uv run python -m scratch.dataset_refresh.validate_mixtures --single-arm --mixtures "$nonmoralBuild/mixture.jsonl" --configs "$nonmoralMixPrep/nonmoral-advice.yaml" --replay-reference "$nonmoralMixPrep/replay_reference.json" --output $nonmoralMixAudit
uv run python -m scratch.dataset_refresh.publish_offline_composite mixture-card --config "$nonmoralMixPrep/nonmoral-advice.yaml" --mixture-dir $nonmoralBuild --audit $nonmoralMixAudit --synthetic-provenance "$nonmoralSnapshot/generation_provenance.json"
```

Require716 synthetic +9284 replay, seed0, all10000 untruncated Qwen8192/mask passes, exact pinned synthetic payload multiplicity, and exact reference replay positions/payloads. Replay quotas: no_robots2580, tulu3_if1366, numinamath_cot987, self_oss_instruct988, smol_constraints979, apigen_function_calling978, smol_summarize914, lima291, longalign201. Base pin remains `dougalldeepmind/2026-09-08-nosynth-mix@7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`.

The replay user row at mixture index77 contains five literal U+FFFD inherited exactly from base row4035; retain it unchanged in both arms. All newly selected synthetic fields must still be checked normally. The user requested this pinned replay source; silently cleaning it would break the controlled replay equality.

7. For an observed two-arm equality receipt, additionally run the paired audit after the single-arm audit/card gate. Keep the single-arm validation file because the prepared single-arm card explicitly requires it:

```powershell
uv run python -m scratch.dataset_refresh.validate_mixtures --mixtures output/2026-09-15_low_mixture_release/builds/da-lowstakes-refresh/20260915_122305/mixture.jsonl "$nonmoralBuild/mixture.jsonl" --configs output/2026-09-15_low_mixture_release/da-lowstakes-refresh.yaml "$nonmoralMixPrep/nonmoral-advice.yaml" --output "$nonmoralBuild/paired_mixture_validation.json"
```

This reads/downloads only the already pinned public datasets if not cached; no model calls. Before mixed publication root reviews the actual10000-row payload hash, both validation files, inherited reasoning/backfill metadata, full synthetic pin and honest review-route card. Existing `src.infra.huggingface.push_run_dir` is the repository upload entrypoint; there is no separate new mixed-publication CLI. Call it on the reviewed build directory with `publication_plan.json.name`, `card_fields.json`, `card_front_matter.json`, `private=False`. Do not rerun `mix` with an `hf` override, which would create another build and generic card.

After upload, independently verify the exact remote commit/payload and source/replay pins, save a publication receipt outside the build, and report actual716/9284 counts and token-share differences. Rounded7 in the canonical repository name means exact7.16% of rows, not7% of tokens or loss weight. Training and evaluation still wait for the user's later confirmation.
