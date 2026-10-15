# GPT-OSS plain-Control MASK

Approved 2026-10-15: one full matched MASK pass for the missing plain Control.
No Base/DA15 rerun. The exact Control sampler and all settings are in protocol.yaml.
Frozen historical source 654bad9f uses cookbook 0.5.7, Transformers 5.5.4,
Tinker 0.30.4, medium reasoning, historical render date 2026-09-28 and 16,384
response tokens. This intentionally does not substitute the newer SWE renderer.
Historical judge prompts and metrics are replayed without changes. Runtime safeguards
are separate: no automatic ambiguous retries and no whole-category regeneration.

Run with the existing pinned Python runtime from the gptoss-swe-three-arms worktree:
`python scratch/gptoss_mask_control/campaign.py preflight|pilot|run`.
Outputs: output/gptoss_mask_control_20261015.

The 12-row pilot is the first two rows of each category and is retained in the
full 4,438 generation calls. Judge workload is at most 4,246 calls; historical
DA15 required 4,000 because its persisted blank belief cells are skipped by the
original evaluator. Pressure and belief rubric logic, empty-cell CSV handling,
one pass and lie_k=1 remain historical. Private reasoning is retained but never
passed to the judge. Incomplete generations are review items, not scored evasion.

Every paid call is reserved in ledger.sqlite before dispatch; unknown outcomes
keep reservations. Generation cap $20; OpenRouter cap $12. OpenRouter cost/usage
must be recorded, with no accounting fallback. Explicit empty 429 rejections
alone can retry up to 3 attempts; no other automatic reruns. Only the owned
process may write the ledger, enforced by an OS file lock. PID and birth time
are in owner.json; owner-exit.json must match that receipt to prove exit.

All received generation token arrays and judge HTTP bodies are saved before
parsing. Completed items are reused on restart. Item-specific review cases do
not stop independent work after the pilot; global errors and budget limits stop
admission. The final scoring gate refuses any unresolved item. Results cannot
be described as a full completed pass until all expected work validates.

Monitor progress.json, generation-pilot.json, judge-pilot.json, failure.json,
owner.json and owner-exit.json. Report every five minutes, including unchanged
status, with valid/review/pending counts, settled plus reserved cost, and ETA
only from fresh throughput. Do not rerun a held item without a decision.

Before paid work: test_campaign.py checks durable/atomic ceilings, duplicate
ownership, prompt identity, independent replicate IDs, expected workload,
private-reasoning exclusion, ambiguous outcomes and missing-cost handling.
No paid requests occur in tests or preflight.

## Authorized closeout on 2026-10-15

After the original pass drained, the user explicitly authorized one retry and
publication regardless of the remaining outcome. `closeout.py` pins the 55
generation and 11 judge failures in `retry-authorization.json`, preserves every
original under `calls/`, and writes retry receipts under `authorized-retry-1/`.
Distinct ledger IDs share the original run's budget ceilings. Restarting cannot
admit a second retry. This closeout permits only one dispatch per selected item,
including explicit rejections, and judges newly eligible rows once.

The closeout scores only rows with all required valid generation and judge
outputs. Remaining failures are excluded, never imputed as honest or evasive.
`closeout-coverage.json` lists exclusions; the frozen scoring formula is unchanged.
This is an available-row score after a targeted retry, not a complete first-pass
estimate. `publish_closeout.py` audits counts/costs/metrics, publishes raw evidence
and the ledger to HF, reads back every uploaded file at its immutable revision,
and plots the historical Base and DA15 scores alongside Control with coverage.
The Control protocol specifies 28,000 context tokens; the historical Base/DA15
run metadata specifies 20,480. Both cap responses at 16,384. Disclose this context
difference as well as retry selection and incomplete coverage in comparisons.

Commands (using the pinned runtime above):
`python closeout.py` (includes offline request-hash scoring), then
`python publish_closeout.py references|audit|publish|plot`
from this script directory. The 18 offline tests include retry isolation, shared
budget ceilings and refusal to restart a scored/published run. No Base or DA15 calls are made.
`score_local.py` remains the offline recovery entry point used in the original closeout.

Recovering first-row generations changed dataframe column insertion order and
therefore ordinal judge IDs for 255 saved requests. `score_local.py` matches the
exact request hash within each category/row using a bijection, without choosing
verdicts or making provider calls. All 11 authorized judge retries used their
exact original request. `judge-request-resolution.json` records the mapping.
