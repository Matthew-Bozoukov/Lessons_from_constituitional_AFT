# Targeted final top-ups

Use `execute_imported` for an ordinary batch. Use `execute_targeted` when the final composite has enough rows in some traits and needs work only in others. Both import the unchanged runner; they do not run `run.py` as `__main__`, so the generation stages and executor share the same `BudgetStop` class.

The run owner supplies a JSON plan with the actual root, arm, cumulative spending ceiling, workers, and exact candidate-attempt counts for all nine traits:

```json
{
  "root": "output/2026-09-15_dataset_refresh_diverse",
  "arm": "da-lowstakes-refresh",
  "ceiling": 225,
  "workers": 8,
  "candidate_counts": {"t1": 0, "t2": 0, "t3": 0, "t4": 0, "t5": 0, "t6": 0, "t7": 0, "t8": 0, "t9": 0}
}
```

The zero counts above are a schema example, not a proposed generation plan. Replace them with the actual composite deficits or explicitly authorized attempt counts. Counts mean candidate attempts, not promised accepted rows. Nothing spends beyond the supplied cumulative ceiling or creates a new ledger. Alternatively replace `candidate_counts` with `candidate_ids`, an explicit unique list of frozen candidate IDs; never supply both modes.

Preview without any API calls:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.execute_targeted --plan <owner-plan.json> --output <outside-root-preview.json>
```

For count plans, selection prefers candidates with more saved paid-stage checkpoints, then the frozen candidate order. It never reopens a terminal, substantive exclusion or missing/invalid receipt. Existing failed BudgetStop terminals must first pass the separate approved recovery helper; incomplete rows from the imported runner need no terminal reopening. Exact-ID plans preserve the supplied order. The helper refuses unavailable counts, repeated required-unique parents and potential phase trait-quota overflow.

After reviewing the owner plan and preview, dispatch explicitly:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.execute_targeted --plan <owner-plan.json> --execute
```

Execution holds the root lock, resolves the plan against current saved state, records exact chosen IDs/candidate hashes/saved-stage hashes/config hash/core and wrapper hashes/git commit/caps in an immutable phase receipt, then calls the same imported `run.generate_one`. All original eligibility, lint, reviews, repairs and acceptance checks run normally. A pre-dispatch stop raises the shared class, stops queued jobs, preserves partial stages, and does not create a spurious failed terminal. The outcome and actual status are saved separately. CLI exit2 indicates a budget stop; it does not imply all planned jobs finished.

Final quality review and composite selection remain separate. This wrapper performs no mixture creation, upload, training or evaluation. Keep the wrapper bytes unchanged after dispatch and include its committed source plus phase receipts in publication provenance.
