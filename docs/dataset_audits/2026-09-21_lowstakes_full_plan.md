<!-- ABOUTME: Frozen full-generation plan following explicit approval of the $120 cap. -->
<!-- ABOUTME: Records selection, accounting and stopping rules before any paid dispatch. -->
# Low-stakes full generation: 2026-09-21

The user approved one full batch after the estimate of $80-110 and recommendation
of a $120 ceiling. This is an additional generation allowance; the $17.565140 of
earlier development is recorded separately. No training, evaluation or automatic
replacement batches are included.

Use the unchanged final domain-smoke recipe, nine complete principles from
`constitutions/claude_distilled_09_principles/constitution.md`, and nine registered
domains. Generate 12 scenarios in each of the 81 trait/domain cells (972 total).
All paid roles use first-party-pinned Sonnet 5. Source requests contain one trait
and one domain, never old examples. Four concurrent API calls; no GPU is required.
Admission remains stakes and text-advice scope; domain-fit notes are diagnostic.
Normal DA drafting/refinement and lint remain unchanged. No paid quality topups.

Run `scratch.dataset_refresh.run_native_smoke` with
`scratch/dataset_refresh/native_lowstakes_full.yaml` (explicit full mode).
The atomic per-physical-call reservation ledger is authoritative, including open
reservations. Cap $120 and 6,000 physical calls; normal work should be below both.
Transport retries are disabled; native JSON/tag parse attempts remain bounded.
Unknown transport/billing failures stop new dispatch. Saved requests and stage
checkpoints permit explicit recovery without paying again for completed work.

Before results, freeze selection: traits t1-t5 receive 80 rows each, t6-t9 79.
Within each trait, rotate the registered domain order by the trait index, then
draw round-robin from nonempty cells. Within a cell rank by SHA256 of
`0:<scenario_id>`. Require all 81 cells to have an accepted row and each trait to
meet its quota. If this fails, report the exact shortfall; do not relax gates or
launch replacements. Preserve every automatic export and every rejected stage.
The selected 716 are unchanged message subsets, not rewritten examples.

Review the final coverage, parse/lint failures and judge distribution. Inspect
examples across all cells and consequential diagnostic flags. Shared normal-DA
factual/rationalization flaws remain disclosed; do not introduce an uncalibrated
post-hoc factual-perfection gate. Native stages publish during execution. Publish
the selection, frozen code/config, raw receipts and clear acceptance limitations
after reviewing outcomes. Selection success does not establish ODCV performance.

Prelaunch validation: 81 offline tests passed, including full 972-cell allocation,
guarded recovery, deterministic 716 selection, missing-cell/quota failure and
ceiling enforcement. Live first-party prices matched the reservation registry;
shared account credit was $324.56 before launch.
