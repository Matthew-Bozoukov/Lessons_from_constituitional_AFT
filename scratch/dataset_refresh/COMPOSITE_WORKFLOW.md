# Composite synthetic publication

`publish_composite.py` consumes an explicit selection supplied by the run owner. It does not choose, generate, repair, or independently accept examples. Prepare each arm separately after all included source phases are idle and independently audited.

The selection JSON declares `arm` (`da-lowstakes-refresh` or `nonmoral-advice`) and exactly 716 ordered `entries`. Each entry has exactly these fields:

```json
{
  "root": "absolute/or/repository/relative/immutable/run/root",
  "arm": "da-lowstakes-refresh",
  "candidate_id": "t1_000_v0",
  "result_sha256": "actual 64-character SHA256 of the selected terminal result.json"
}
```

The helper checks the 80/80/80/80/80/79/79/79/79 trait quotas. Each result must remain accepted under `load_result`, match its origin candidate/identity/config, and pass the origin's actual eligibility, quality and exact-quote grounding receipts. It checks immutable source snapshots, current implementation versus bound execution phases, source lineage and duplicate prompts across phases. Adopted independent repairs additionally require the bound proposal, adoption reason, archived original and exclusion, repair manifest, and unchanged system/user.

Run a read-only validation first:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.publish_composite validate --selection <explicit-selection.json>
```

Commit all current validation/publication helpers before preparing the snapshot. The explicit commit must contain the implementation that validates every selected phase. Obtain `ledger-end` as the exclusive final shared ledger length; it cannot omit later calls belonging to a selected root/arm. Use the actual publication date.

```powershell
uv run --no-sync python -m scratch.dataset_refresh.publish_composite prepare --selection <explicit-selection.json> --destination <new-snapshot-directory> --source-commit <full-git-commit> --ledger-end <exclusive-index> --date <YYYY-MM-DD> --quality-dir <quality-directory>
```

Preparation holds each origin execution lock. The snapshot contains:

- `dataset.jsonl` as the only default training split, with exactly 716 native-reasoning conversations.
- `selection.json` and the untouched `explicit_selection.json`, binding output order and hashes.
- `origin_phases.json` and `generation_provenance.json`, preserving every exact origin configuration and phase selection count.
- Phase-prefixed scenario IDs, `original_scenario_id`, the original source metadata, and explicit origin root/config/candidate/result hashes.
- Complete flattened JSON checkpoints for selected source arms, including failures, exclusions, repairs and receipts. Empty operational lock-file hashes are listed separately.
- Root/arm-scoped physical raw calls and their original hashes, plus the cumulative shared accounting snapshot.
- The supplied selected-corpus quality bundle, with exact dataset and artifact hashes. Arm-root independent reviews remain in the origin archives as well.
- Source-code snapshot, full dataset card, and a file-hash publication manifest.

Review the frozen snapshot, then publish using the existing hash-checked push implementation:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.publish_composite push --destination <snapshot-directory>
```

The existing `src.naming.synth_name` supplies the usual dated `da-lowstakes-refresh-synth` or `nonmoral-advice-synth` name. No phase-specific alternate dataset style is introduced. Obtain the full published HF commit before preparing mixture configs.

For the final mixture card, pass the composite snapshot's `generation_provenance.json` to `prepare_mixtures.py card --synth-config`. This self-contained bundle validates every origin recipe and records models by phase. The separate mixture still uses exactly 716 synthetic plus the common 9,284 replay rows; no additional generation or replay backfill occurs during mixing. Follow `MIXTURE_WORKFLOW.md` for the mandatory two-arm replay-position and Qwen 8,192-token audit.

These commands perform no training or evaluation. No actual selection, snapshot preparation, or publication was performed while implementing this helper.

## Final selected-corpus evidence

Before preparation, the run owner's selector writes the selected preview using `validate_selection` and the same `run.write_rows` serializer as publication. Do not hand-edit it. Run this against that preview:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.audit_corpus --dataset <selected-preview.jsonl> --output <quality-directory>
```

Independently adjudicate the resulting duplicate candidates. Similarities are candidate evidence, and domain counts are assigned labels, not independently certified labels. Include the selection policy, lexical/semantic adjudications and Qwen length/mask report in the quality directory. Reports describing the selected corpus should carry its exact `input_sha256`, `dataset_sha256`, or `selected_dataset_sha256`. Historical/calibration reports can be included without claiming they audited these exact rows.

Freeze the quality directory with `quality_manifest.json`, using actual SHA256 values for the preview and every other file, including prose reports:

```json
{"dataset_sha256": "<physical SHA256 of selected preview dataset.jsonl>", "files": {"corpus_audit.json": "<SHA256>", "selection_policy.md": "<SHA256>"}}
```

The optional `--quality-dir` intake verifies exact dataset binding, every file hash, no symlink/junction traversal, and no changes during copying. It runs no new audits and does not turn model or agent judgments into guarantees. Use it for both final releases; omission is explicitly disclosed in the card. All included source phases and independent repair jobs must be idle. Do not add files to the publication snapshot after preparation: the push hash gate refuses any change.

Accounting cutoff is an exclusive shared-ledger index, not the sum of selected rows' calls: include rejected candidates, pilots, uncertainty reserves and independent calibration in the cumulative prefix. Physical requests in this publication are scoped only to its selected origin root/arms. The card distinguishes their reported-cost subtotal from charged/reserved exposure. Unknown-cost failures remain reserved; they are not zero-cost completions. The helper refuses invalid/nonfinite exposure, a prefix exceeding $250, an active selected-arm reservation, or a cutoff hiding later selected-arm calls. Use one final settled cutoff for both releases when practical.
