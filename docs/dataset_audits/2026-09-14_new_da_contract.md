<!-- ABOUTME: Pinned new DA generation contract used to refresh low-stakes and nonmoral datasets. -->
<!-- ABOUTME: Separates verified parent facts, experimental changes and bounded candidate selection. -->
# New difficult-advice contract: 2026-09-14

Source [dougalldeepmind/2026-09-14-da-synth](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-da-synth/tree/013886238fca238c4d54ace96530f444bb2b2f02), pinned `013886238fca238c4d54ace96530f444bb2b2f02`.

The new corpus has **752 exported conversations**, from765 source scenarios. Counts by stage:765 scenarios and draft prompts,754 revised prompts,754 successful drafts after resume,752 final revisions. Stage2 has85 scenarios for each of9 principles. Final counts are t1=78,t2=85,t3=85,t4=83,t5=85,t6=84,t7=83,t8=84,t9=85. Therefore a716-row selection with79/80 per principle cannot be drawn uniquely from the final752 alone. Refresh preparation uses the765 stage2 source scenarios and joins stage5/7 fields where present. It marks `parent_exported` rather than implying that failed original rows trained a parent model.

## Constitution and response contract

`constitutions/claude_distilled_09_principles/constitution.md` has stripped UTF-8 SHA256 `8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc`, matching the parent's manifest and current worktree file. All constitution-aware generation stages inject their target **principle only**. Neither answer stage injects the full constitution or its priority preamble. Preserve this scope when claiming parity; full-target review is a separate choice, not something the parent performed.

| Stage | Model | Temperature | Output cap |
|---|---|---:|---:|
| Scenario writing | anthropic/claude-haiku-4.5 |1.1|8192|
| Prompt draft | anthropic/claude-haiku-4.5 |1.0|2048|
| Prompt revision | anthropic/claude-sonnet-5 |0.7|6144|
| Answer draft | anthropic/claude-haiku-4.5 |1.0|4096|
| Answer revision | anthropic/claude-sonnet-5 |0.7|12288|

Answer draft uses `<reasoning>` and `<response>` tags; revision adds `<changes>`. Both enforce700 characters minimum independently in reasoning and response, with2 lint retries. Expanded bans cover policy/constitution excuses, stock openings, Chinese characters, and Claude/Anthropic mentions. This is teacher-authored deliberation, separately exported as `reasoning_content`; it is not provider-internal hidden reasoning.

The exported chat consists of system, user and assistant messages. Assistant `content` is the final answer, `reasoning_content` the deliberate trace. Constitutional trait text is metadata, not a system policy to inject into training rows. The refresh copies both native response stage dictionaries exactly from the pinned manifest, including prompts, lints and sampling-role names. Canonical JSON SHA256 (sort_keys=True, ensure_ascii=False) of that stage list is `58cd0b43cd3a659632b3405eda8936e7c8acbd147d967d36c9fd4a9f59ff807c`.

The source asks for human difficult advice, but its generator does not hard-enforce the human as sole actor in every scenario. For example the first exported record is explicitly a human clinician considering covert cardiac-risk-threshold changes and asking how to think through them. This medical example cannot become low stakes by disclaiming harm; its ethical mechanism requires a genuinely different harmless setting. Refresh adds an explicit advice-to-human restriction, rejecting direct assistant action and recording actor transformations. It does not pretend this extra restriction was a universal verified property of parent DA.

## Actual checks and provenance limits

The parent's final `corpus` stage was **ablated** (`--ablate corpus`), and its configured quality filter is disabled. Scenario diversity checks ran and reported no findings; don't infer final semantic quality approval from stage counts. Two final response rows failed provider content filtering. Earlier prompt and draft-stage losses remain source provenance, not proof of impossible scenarios.

The latest manifest is a resume at git `0426598cb3e3cc527da41b9949194fde21528dc2` and carries$0.1732 usage for the resumed tail. Preserved `manifest_run1.json` carries$56.8078 for the first run. Sum recorded segments=$56.9810; this is recorded generation accounting, not provider invoice verification. Full effective configs travel with both manifests.

## Low-stakes candidate contract

`configs/data/synth/da-lowstakes-refresh.yaml` uses the pinned765 scenario source, new09 target, source-native answer stages, and Sonnet5 scenario rewrites plus separately prompted Sonnet5 final semantic review (temperature0, max_tokens4096, explicit reasoning.enabled=false). This is a separate review call, not an independent-model panel. A candidate must be advice to a human, retain a real target-principle dilemma with feasible alternatives, and have independently assessed stakes0 or1. Misleading author `worst_outcome` descriptions cannot override facts about injury, animals, livelihood or lasting relationships. Self-contained content, constraint fidelity, actual constitutional alignment and useful unpadded reasoning/final advice are hard review gates.

Quotas are80 each t1-t5,79 each t6-t9:716 total. Initial pilot18 (two per trait), requiring at least16 independent accepts, at least1 per trait, and no repeated material defect in3 or more candidates. Up to2 rejects/failures are allowed without padding or softening criteria. At most one recorded design revision and separate repilot is allowed. Candidate selection is seeded per trait. Prefer same-domain transformations when valid, record actual relocations and actor changes, and keep no more than one accepted row per parent. At most2 variants per source seed; total1200 candidate attempts per arm including fresh replacements. Fresh replacements have new IDs and are explicitly unpaired. Missingness and failed attempts remain archived; stop/report quota shortfalls rather than relax the gate. The planned18 setting suggestions prevent community-club collapse but are not forced equal quotas; actual accepted domains and transformations must be censused after generation.

This config is for the scratch refresh driver, not `uv run synth`: its header names the driver. Schema/prompt formatting, quota total, source hashes and equality of copied response stages were checked without paid model calls. Training and evaluation are outside this dataset job.

## File digests

- `manifest.json`: `8548db0b6907639abe90b212e539f4413809101bb1f43bfd7f0a2401efc143cc`
- `manifest_run1.json`: `a2b5d8f0ae3c026922c8f3c2ff6c19a1a0c3a7cf39ed17c013c8bed37e2de8f7`
- `dataset.jsonl`: `77507baa3f115e4c1756cb286942a5201d88e1435405e6e4cf8f2897cbcf0e9a`
- `stages/stage_1_chunk_constitution.jsonl`: `94e2137593a3cb96603884dbaea782bcd9991ca3ca210c17960c3301b3d4eefa`
- `stages/stage_2_write_scenarios.jsonl`: `627f903491b72653f773b5411bf7cb0bb7ac029cacbf00a0c5eecf109d9b9efd`
- `stages/stage_5_revise_prompts.jsonl`: `8c06e0d1d83d9ab194d2f3878bec46b3087e286c35159a62c6b747b8afc7abb0`
