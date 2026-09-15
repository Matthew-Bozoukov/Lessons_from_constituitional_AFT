<!-- ABOUTME: Describe the exact mixed-acceptance nonmoral release and shared-replay workflow. -->
<!-- ABOUTME: Keep preparation, independent final adjudication, publication and subsequent mixing distinct. -->

# Mixed nonmoral release

`publish_offline_composite.py` composes the exact corrected650-row base with66 separately accepted saved answers. Those66 can come from multiple campaigns, the four-case pilot, or unchanged earlier Sonnet finals; they need valid `offline_acceptance` dossiers with their real source/author/current-review evidence. They are not required to share a campaign or to be newly generated. Original automatic passes are never assigned to the new route.

The fixed base is `2026-09-15_dataset_refresh_correction_review/final_selected_nonmoral/pool_selection.json`, SHA256 `5e52a18a45dcef2065428c3a9b38d4a1f1c1c79b75a7f34817c572c0c9ebf0aa`. Its original650 source records, receipts, source identity, automatic checks, correction histories, any adopted repairs and exclusion status are revalidated. Every addition calls the frozen `offline_acceptance.validate_accepted()` independently. The common gate then enforces716 rows,80 each for t1–t5 and79 each for t6–t9, unique source identities and normalized user prompts. Final semantic adjudication remains necessary.

Create a caller-owned ordered JSON manifest:

```json
{
  "arm": "nonmoral-advice",
  "base_selection": {
    "path": "ABSOLUTE_FINAL_CORRECTION_SELECTION.json",
    "sha256": "5e52a18a45dcef2065428c3a9b38d4a1f1c1c79b75a7f34817c572c0c9ebf0aa"
  },
  "offline_entries": [
    {"acceptance_path": "ABSOLUTE_DOSSIER/acceptance.json", "acceptance_sha256": "FULL_SHA256"}
  ]
}
```

There must be exactly66 entries, each with its real acceptance hash. No automatic selection or quota filling occurs here.

## Exact-byte preview and final quality

```powershell
uv run python -m scratch.dataset_refresh.publish_offline_composite validate --selection SELECTION.json
uv run python -m scratch.dataset_refresh.publish_offline_composite preview --selection SELECTION.json --output NEW_PREVIEW
```

Preview writes the exact intended `dataset.jsonl`, loads cached Qwen3.6 tokenizer and cached pinned8M embedding model locally, and produces native8192 token/masking checks, corpus/length/domain census, lexical/semantic pair candidates and contextual literal flags. It makes no network/model calls and does not accept or publish a row.

Read the final surfaced pairs/flags and write a separate independent evidence directory. Its `adjudication.json` must bind `dataset_sha256`, state a nonempty `scope`, set `release_approved:true`, and have `unresolved_material_issues:[]`. Preserve the actual supporting pair decisions, corrected false positives and remaining limitations. These flags express the root's reviewed decision; passing the script does not prove semantic correctness.

Seal using the existing unchanged quality helper:

```powershell
uv run python -m scratch.dataset_refresh.preview_release seal --preview NEW_PREVIEW --evidence INDEPENDENT_EVIDENCE --output NEW_SEALED_QUALITY
```

The sealed `quality_manifest.json` hashes every supplied artifact and binds the exact dataset. Final preparation separately rechecks native716 rows/IDs and all token/masking bounds, census input identity, required semantic/lexical files and explicit final adjudication.

## Prepare and publish only after review

Close all relevant source/evidence writes and API calls; freeze the latest helper/recipe commit. The shared ledger must contain only settled calls or verified billed failures, with an exact exclusive cutoff and conservative total at most the user-approved$270. A billed failure is accounting evidence, never successful author content.

```powershell
uv run python -m scratch.dataset_refresh.publish_offline_composite prepare --selection SELECTION.json --destination NEW_SYNTH_SNAPSHOT --source-commit FULL_GIT_SHA --ledger-end EXACT_CLOSED_COUNT --date YYYY-MM-DD --quality-dir NEW_SEALED_QUALITY
```

Preparation preserves the whole base origin arm, root/phase metadata, every accepted offline dossier, complete new execution roots including unselected/failed outcomes, exact scoped physical calls, shared ledger, billing reconciliation proofs, final quality artifacts and a committed code archive. All tar members have deterministic timestamps and hash inventories. New dossiers remain directly accessible under `audit/offline`; historical absolute paths are provenance identifiers. Only `dataset.jsonl` is the default train split.

The canonical synthetic name remains `YYYY-MM-DD-nonmoral-advice-synth`. The card distinguishes650 original automatic acceptance routes from66 independent Codex-agent/root adoptions, reports actual author kinds/settings, preserves old craft text versus current review meaning, and states that no training/evaluation occurred.

Root reviews the concrete card, counts, provenance inventory, dataset hash and closed budget before calling:

```powershell
uv run python -m scratch.dataset_refresh.publish_offline_composite push --destination NEW_SYNTH_SNAPSHOT
```

`push` uses the existing snapshot-file hash and exact716 export validator. A failed preparation retains a failure marker and cannot publish.

## Same replay mixture

After successful synthetic publication, supply its **actual full HF commit**, never a guessed or moving pin. Use the already frozen low-arm replay reference:

```powershell
uv run python -m scratch.dataset_refresh.prepare_mixtures prepare --output NEW_MIX_PREPARATION --single-arm nonmoral-advice --nonmoral-repo ORG/YYYY-MM-DD-nonmoral-advice-synth --nonmoral-revision FULL_HF_SHA --replay-reference LOW_RELEASE_REPLAY_REFERENCE.json
uv run mix --config NEW_MIX_PREPARATION/nonmoral-advice.yaml
uv run python -m scratch.dataset_refresh.validate_mixtures --single-arm --mixtures ACTUAL_BUILD/mixture.jsonl --configs NEW_MIX_PREPARATION/nonmoral-advice.yaml --replay-reference NEW_MIX_PREPARATION/replay_reference.json --output MIXTURE_VALIDATION.json
uv run python -m scratch.dataset_refresh.publish_offline_composite mixture-card --config NEW_MIX_PREPARATION/nonmoral-advice.yaml --mixture-dir ACTUAL_BUILD --audit MIXTURE_VALIDATION.json --synthetic-provenance NEW_SYNTH_SNAPSHOT/generation_provenance.json
```

The exact mix contract remains716 synthetic +9284 replay, seed0, strict8192, pinned `dougalldeepmind/2026-09-08-nosynth-mix@7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`, exact replay source quotas and payload/position agreement with the existing low-arm reference. Native reasoning/backfill is inherited; no new backfill or paid generation occurs. The card adds the truthful offline review distinction to the existing pinned-source and masking validation. The rounded7 repository suffix remains despite the exact7.16% row share. Token share and training loss weight remain separate quantities.

The implementation task ran no API calls, uploads, generation or final release selection. Actual base650 validation passed against the current immutable correction selection.
