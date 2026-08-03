<!-- ABOUTME: Provenance and known caveats for the 4-principle specgen arm. -->
<!-- ABOUTME: Machine-distilled by scratch/specgen from the pinned published Claude constitution. -->

# Rationale — claude_distilled_4_principles_coarse

This constitution was not hand-authored: it is the `coarse` arm of the granularity
experiment, distilled by `scratch/specgen/` from Anthropic's published Claude
constitution (CC0, ~30k words, source sha `69198700ea7b`). The single
independent variable across the three arms is granularity (4 / 12 / 24 principles);
all arms partition the same locked inventory of 664 atomic normative claims
(sha `8cef061b6e0f`), so content coverage is identical by construction.

**How it was made:** claims extracted per source section (fable subagents) ->
partitioned into exactly 4 clusters with exact claim-ID accounting (fable) -> each
principle unit written in an isolated call seeing only its cluster's claims (opus),
with one measured revision round enforcing the token budget and the >=60% explanation
share. Preamble and closing are hand-written and byte-identical across all three arms.
Prompt hashes, model aliases and the full config are in the run's `meta.json`
(timestamp 20260803_174711); every draft and revision is preserved on HF
(`LASR-Callum/2026-08-03-specgen-constitution-granularity`, prefix `coarse/seed0/20260803_174556`).

**Selection:** seed 0, single-seed pilot — promoted without the 5-seed cross-seed
stability (ARI) check or the pre-registered selection rule. Rerun seeds 1-4 and
re-select before treating cross-arm comparisons as final.

**Known caveats:** Hard-language density per token is highest of the three arms (0.79) — coarse statements compress ~166 claims into short absolutist sentences. Intrinsic to the granularity condition; claim-level modality is identical across arms.

**Regenerate:** `uv run scratch/specgen/cli.py generate --config scratch/specgen/specgen.yaml --arm coarse`
