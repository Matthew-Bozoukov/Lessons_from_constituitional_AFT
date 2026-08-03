<!-- ABOUTME: Provenance and known caveats for the 12-principle specgen arm. -->
<!-- ABOUTME: Machine-distilled by scratch/specgen from the pinned published Claude constitution. -->

# Rationale — claude_distilled_12_principles_mid

This constitution was not hand-authored: it is the `mid` arm of the granularity
experiment, distilled by `scratch/specgen/` from Anthropic's published Claude
constitution (CC0, ~30k words, source sha `69198700ea7b`). The single
independent variable across the three arms is granularity (4 / 12 / 24 principles);
all arms partition the same locked inventory of 664 atomic normative claims
(sha `8cef061b6e0f`), so content coverage is identical by construction.

**How it was made:** claims extracted per source section (fable subagents) ->
partitioned into exactly 12 clusters with exact claim-ID accounting (fable) -> each
principle unit written in an isolated call seeing only its cluster's claims (opus),
with one measured revision round enforcing the token budget and the >=60% explanation
share. Preamble and closing are hand-written and byte-identical across all three arms.
Prompt hashes, model aliases and the full config are in the run's `meta.json`
(timestamp 20260803_174937); every draft and revision is preserved on HF
(`LASR-Callum/2026-08-03-specgen-constitution-granularity`, prefix `mid/seed0/20260803_174717`).

**Selection:** seed 0, single-seed pilot — promoted without the 5-seed cross-seed
stability (ARI) check or the pre-registered selection rule. Rerun seeds 1-4 and
re-select before treating cross-arm comparisons as final.

**Known caveats:** Unit 10 partially duplicates the preamble's priority ordering (the source's own ordering claims are in the inventory, so every arm must represent them). Tokens (5,679) sit ~9% above the pre-registered band; explanation ratio holds, length is reported as a covariate.

**Regenerate:** `uv run scratch/specgen/cli.py generate --config scratch/specgen/specgen.yaml --arm mid`
