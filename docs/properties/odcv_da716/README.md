# da716 ODCV rollout properties — 2026-08-19

Small text artifacts only. The full run — including `dashboard.html` (browsable,
every property expanding to the records that carry it) and `members.jsonl` (the
record→property join table) — is on the Hub:

    LASR-Callum/2026-08-19-da716-odcv-rollout-properties

Write-up and metrics: `dashboard/content/findings/2026-08-19-da716-rollout-properties/`.
Regenerate with:

    uv run python scripts/properties/discover.py \
      --config configs/properties/discover_odcv_da716.yaml
