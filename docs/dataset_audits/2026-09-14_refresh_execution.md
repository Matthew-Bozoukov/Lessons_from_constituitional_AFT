# ABOUTME: Approved execution contract for the two September dataset refreshes.
# ABOUTME: Separates source-grounded pilot acceptance, generation, mixing, and later training.

# Approved execution, 2026-09-14

The user approved two human-advice corpora: moral low stakes and nonmoral craft
deliberation, each 716 synthetic examples mixed with the same 9,284 replay rows.
The replay source is `dougalldeepmind/2026-09-08-nosynth-mix` at
`7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`. Preserve selected messages, reasoning,
tools and supervision without enrichment or filtering. Synthetic quotas are80 for
t1–t5 and79 for t6–t9. Mixture names use the existing rounded7 convention; cards
must state716/10000=7.16% exactly and distinguish token exposure from loss weight.

The moral arm adapts scenario mechanisms from the newly supplied DA at
`013886238fca238c4d54ace96530f444bb2b2f02`. Its response stages use that DA's
principle-only new09 constitution and author models/settings. Transformations may
change actor and domain; ancestry is not a claim of one-factor matched pairs.
The nonmoral arm preserves the original nine craft tensions, artifact diversity,
and varied resolutions while making the human the actor. Its ethical constitution
is used in separate compatibility review, not as a craft generation prompt.
See the companion nonmoral spirit review for45 concrete examples.

Each arm first generates18 candidates, two per target, with both arms interleaved.
Both independent source-grounded inspection and the saved judge rubric matter:
at least16/18 hard passes, at leastone per target, and no recurring material defect
on three or more rows. A judge pass can be vetoed with written evidence. Pilot
approval binds the frozen recipe hash. One explicit recipe revision with a disjoint
pilot is permitted; persistent design failure stops that arm. Rejected or failed
rows and API failures remain in the record. No teacher switch, relaxed acceptance,
or repeated padding to pass a length floor is authorized. The copied stage lint
retry settings are descriptive of DA; this runner uses one attempt and rejects
lint failures instead of asking for padding.

The initial combined pilot ceiling is$20. The overall target is at most$200;
the user allows continued necessary work up to a hard combined$250. Every physical
API call reserves a conservative bound in a file-locked shared ledger before
dispatch; uncertain failures retain their reservations. No automatic SDK/provider
retries. All recipe revisions must share this same ledger. A run execution lock
prevents duplicate concurrent candidate dispatch; stage receipts detect changed
checkpoints. Full requests/responses and actual provider costs are retained.

After pilot approval, generate bounded batches toward exact quotas, inspect
acceptance and failures, audit near duplicates and domain/decision variety, then
freeze selected IDs. Validate the training-visible rendering, reasoning/final
supervision and length/truncation distributions without launching training.
Publish source-pinned synth stages, corpora, failure ledgers, resolved recipes,
cards and exact replay-identical mixtures under `dougalldeepmind`, with names
constructed by `src.naming`. Download and verify the final published bytes.
Commit and push the dedicated branch; do not merge main. Training and evaluation
require the user's later confirmation.
