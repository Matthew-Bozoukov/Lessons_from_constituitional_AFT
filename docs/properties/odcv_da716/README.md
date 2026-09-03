# da716 ODCV rollout properties — 2026-08-19

275 judged ODCV-Bench rollouts from the Table2-9284 + difficult-advice-716 LoRA.
48 properties, 31 surviving Benjamini-Hochberg. Lifts are computed WITHIN ODCV
condition (incentivized 23.2% base violation, mandated 12.4%), not pooled.

Small text artifacts only. The browsable `dashboard.html` and the
record->property join table `members.jsonl` are on the Hub:

    LASR-Callum/2026-08-19-difficult-advice-716-odcv-rollout-properties

Write-up: `dashboard/content/findings/2026-08-19-da716-rollout-properties/`.
Regenerate:

    uv run python scripts/properties/discover.py \
      --config configs/properties/discover_odcv_da716.yaml
