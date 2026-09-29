<!-- ABOUTME: Reproduction and interpretation notes for the September 29 DA corpus investigation. -->
<!-- ABOUTME: Audit only: no GPU rentals, training, evaluation reruns or source-data edits. -->

# DA corpus investigation

Compare the exact selected DA rows from the September 25 and September 28
mixtures, pinned in the published `old_pins.json` and `new_pins.json`. Results and
raw evidence are auxiliary metadata on the existing refreshed ODCV result repo;
they do not replace its evaluation outputs. See
`docs/training/2026-09-29_da_refresh_investigation.md` for conclusions and limitations.

`audit.yaml` records the rubric, judge models, seed and $30 ceiling. API requests
use repository provider pins and pricing snapshots, with a conservative reservation
before each call. The runner's lock protects threads within one process only:
**never run multiple paid stages/processes concurrently against this ledger**.
Existing receipts are skipped; unknown-cost reservations remain charged against
the ceiling. Do not delete receipts or the ledger to rerun a stage.

The completed sequence was pilot (cached inside primary), primary (all 1,245
rows), secondary (108 seeded stratified rows), four synthetic calibration cases,
and a nonrandom 19-case challenge set. Only two truncated judgments were retried
with 8,192 output tokens; manifests and both original failures are retained.
Challenge and recovery calls used `judge.run_one` with the recorded stage/model
and token override. They are not prevalence estimates. Manual agent samples were
unblinded and are not human annotation.

Local inputs `output/da_refresh_investigation/{old,new}_selected.jsonl` contain
`id`, `index`, `fingerprint`, `messages` and `source_row`; the latter contains the
original source metadata. Published `selected_rows/*.jsonl` retain the full
conversations and source metadata without duplicating each conversation twice.
Row IDs are zero-based DA-only mixture order. Identical scenario IDs across
generation runs are not matched scenarios.

Commands from the repo root:

```powershell
# These completed paid stages must not be repeated without their existing receipts.
uv run --frozen python scratch/da_refresh_investigation/judge.py --stage pilot
uv run --frozen python scratch/da_refresh_investigation/judge.py --stage primary
uv run --frozen python scratch/da_refresh_investigation/judge.py --stage secondary
# Local aggregation and figures only; no API spend.
uv run --frozen python scratch/da_refresh_investigation/summarize.py
# Narrow, additive HF publication; refuses an existing destination prefix.
uv run --frozen python scratch/da_refresh_investigation/publish.py
# Readback of the completed publication, with no new upload or paid requests.
uv run --frozen python scratch/da_refresh_investigation/publish.py --verify 6aeebc9b1b9cf788c6e1c488d2145407e714f1fb
```

`structure.py` measures actual supervised-token masks with the pinned tokenizer.
`summarize.py` preserves automated counts with explicit measurement-failure
warnings. Primary ceiling scores and zero fabrication flags are not trustworthy
quality estimates. Sonnet raw flags also require adjudication; negative rows
were not fully reaudited. Figures stay local under `output/figures/`, per repo policy.
