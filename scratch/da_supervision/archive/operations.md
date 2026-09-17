<!-- ABOUTME: Condensed operational history of the completed September DA supervision campaign. -->
<!-- ABOUTME: Original logs are retained locally and in HF archives; this is not an active run plan. -->

# Completed campaign operations

All dates below are 2026. Final results and immutable artifact pins are in the
[report](../../../docs/da_supervision_rerun_2026-09-16.md).
The original branch through `9922f8fe` preserves the complete incremental journal.

## Training, September 16

- Initial single-GPU rentals were corrected to three separate dual-H200 pods before
  full training. Setup/smoke evidence and empty checkpoint inventories were preserved.
- Final pods: CoT `rsq31hkrf0elpn`, answer `xiw8uh2etoa1hy`, empty `266cqfrfcyxxgs`.
  Names used the `nika-` prefix. Provider allocation failures were reconciled against
  inventory before retrying. No unrelated pod was changed.
- CoT code revision `68ab0cb7`; the other two used `23a13288`. Trainer/model/config
  files were identical; differences concerned placement and diagnostics.
- Each pod cost $9.18/hour with a $9.50/hour ceiling and five-hour deadline,
  including a 45-minute preservation reserve. Owner-death and deadline guards were
  independent. CUDA checks touched both devices; longest-row/DA smoke checks preceded
  training. Global batch remained 16 under two-rank torchrun.
- A slow bootstrap was recovered on its existing pod with the original deadline.
  `resume_boot.py` records that specific handoff; `freeze_single.py` and
  `start_dual.py` record the retired initial-placement correction.
- All three completed 628 optimizer steps. Each full archive was approximately
  14.13 GB. Direct pod-to-HF upload avoided a slow laptop transfer; remote byte count
  and SHA256 were verified before termination. Model repositories retain the archives.
- Estimated training cost was $95.03, including setup/preservation and $1.29 for
  superseded initial attempts. This is not a settled provider invoice.

## ODCV, September 16 UTC / September 17 UK time

- Original single-H100 pods: CoT `f05zyy03ba3ndz`, answer `mg46cd376sn6ba`,
  empty `sq8hhs4lhgxfx7`. All used four-hour deadlines and $3.49/hour GPUs.
- Windows health requests initially used the wildcard listening address. Separating
  the `0.0.0.0` listener from the `127.0.0.1` client address recovered the same pods.
- Answer/empty then failed before any rollout: their process working directories
  were 271/266 characters and Windows CreateProcess refused them. Verified short
  aliases fixed the Compose calls. The failed pods were automatically terminated.
- CoT's pilot was preserved and counted once. Recovery used the same pass/cache and
  original pod deadline. It completed 80 cells and published at about 21:06 UTC.
- The assistant unnecessarily delayed the other arms behind a self-imposed
  replacement-approval restriction in a scheduler prompt. This was not a repository
  requirement. That scheduler was deleted; the user then explicitly requested both
  remaining evaluations and scheduler deletion on completion.
- Replacement pods: answer `hkql4pvemzwtjq` (`nika-da-answer-only-odcv-r2`), empty
  `wvaxjpkc7vltux` (`nika-da-empty-cot-odcv-r2`), allocated 22:16–22:17 UTC.
  A failed HTTP500 answer allocation was checked against inventory before retry.
- Both completed 79 new rollouts plus their counted pilot. GPUs were released before
  local judging. Empty publication initially refused pilot files at the package root;
  moving those two files under `metadata/` with byte verification repaired publication
  without regeneration or rejudging. Answer's files were moved before its publisher ran.
- All 160 new transcript hashes and result/provenance files were verified on HF.
  Both owners and the awake helper exited; fresh inventory confirmed all campaign
  pods absent. Scheduler `finish-remaining-da-odcv-evaluations` was deleted.
- New GPU upper estimates: $3.77 answer plus $3.26 empty, $7.03 total. Including
  the first evaluations gives approximately $13.65. These exclude storage/API costs
  and overcount judging after GPU release. Shared OpenRouter usage deltas cannot be
  attributed to this run; HF metadata/card caveats preserve that limitation.

## Evidence locations

Original worktree ignored outputs:

- `output/da_supervision/2026-09-16/runs/`: training owner/status/backup receipts.
- `output/da_supervision/odcv_2026-09-16/`: initial failures and CoT completion.
- `output/da_odcv_remaining_2026-09-16/`: replacement runs and publication receipts;
  `completion_receipt.json` records both publications, cleanup and scheduler deletion.

These local paths are operational evidence, not canonical result storage. HF model
archives and evaluation repositories contain the durable artifacts.
