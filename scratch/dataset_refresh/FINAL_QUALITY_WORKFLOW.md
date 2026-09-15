# Final selection, quality and publication

Use explicit final selections with exactly 716 accepted rows and quotas t1–t5=80, t6–t9=79. A partial pool is an honest intermediate artifact; it cannot pass the release preview. Do not make training/evaluation calls in this workflow.

The following commands are executable templates; replace angle-bracket paths. Preview and seal use no API calls. Preview requires cached Qwen tokenizer and the pinned potion-base-8M embedding snapshot; missing cache fails instead of downloading. Use a new output directory outside every frozen source root.

```powershell
uv run --no-sync python -m scratch.dataset_refresh.preview_release preview --selection <ordered-selection.json> --output <preview-directory>
```

This uses `publish_composite.validate_selection` and its actual serialization. It validates origin hashes, receipts, immutable recipes, adoption chains, quotas and prompt uniqueness. It then writes exact `dataset.jsonl`, runs the untruncated Qwen 8192-token limit and native-reasoning mask checks, screens literal leakage/repetition candidates, and saves full census plus lexical/semantic candidate pairs. Automatic checks do not certify substantive answer quality. Domain fields remain author/source assignments until independently adjudicated.

Independently inspect the final selected conversations, outstanding literal flags, all semantic pairs above 0.9 and the top pairs, and actual domain/template balance. Preserve the selection policy, exclusions and dispositions in a separate evidence directory. Its `adjudication.json` must contain the exact preview `dataset_sha256` and a nonempty `scope` describing what was reviewed. The helper does not invent approval or turn candidate similarity into an automatic defect.

```powershell
uv run --no-sync python -m scratch.dataset_refresh.preview_release seal --preview <preview-directory> --evidence <independent-evidence-directory> --output <quality-directory>
```

Seal refuses changed dataset/selection/audit bytes, incomplete previews, failed token/text checks, mismatched independent input hashes and symlinks. It freezes every automatic and supplied independent artifact into the existing exact `quality_manifest.json` intake contract. Its temporary intake check invokes the actual publisher validator.

Commit all validating helper changes. Stop generation before selecting the final shared ledger cutoff; reconcile only independently verified charges with their full failed-call evidence. Then prepare (still no upload):

```powershell
uv run --no-sync python -m scratch.dataset_refresh.publish_composite prepare --selection <ordered-selection.json> --destination <new-synthetic-snapshot> --source-commit <full-commit> --ledger-end <exclusive-call-index> --date <YYYY-MM-DD> --quality-dir <quality-directory>
```

Verify the snapshot `dataset.jsonl` SHA equals the preview SHA. Publication archives ordinary and corrected failed-call accounting separately; billing correction never changes failed content into accepted content. The release name continues to use each existing style. Follow `COMPOSITE_WORKFLOW.md` for explicit snapshot publication, then `MIXTURE_WORKFLOW.md` to prepare the two full-commit-pinned synthetic sources, build exact 716+9284 mixtures, jointly verify identical replay multiplicities/positions and synthetic payload preservation, and write their complete cards. Those later build/upload commands require the actual published corpus revisions and are not run by this helper.
