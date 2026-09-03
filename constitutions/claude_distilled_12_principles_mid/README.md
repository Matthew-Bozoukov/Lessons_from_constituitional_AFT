<!-- ABOUTME: Metadata card for the mid specgen arm: status, dates, principle count, source. -->
<!-- ABOUTME: The constitution itself is constitution.md; provenance and caveats are rationale.md. -->

# claude_distilled_12_principles_mid

| field | value |
|---|---|
| status | **experiment arm** (mid, granularity study) and **default constitution for synth data generation since 2026-08-03**; re-cut 12→10 on 2026-08-04, then set byte-identical on 2026-08-05 to the 9-principle generation-time snapshot (`claude_distilled_09_principles_mid_20260804/`) so all synth pipelines share one alignment target |
| principles / traits | 9 (originally 12; folder name kept) |
| source material | Anthropic, *Claude's Constitution* (January 2026, anthropic.com/constitution, CC0) — machine-distilled, not verbatim |
| date generated | 2026-08-03 |
| tokens (Qwen3.6) | 5679 (433/unit; compression 6.26:1) |
| explanation ratio | 0.569 |
| claim coverage | 664/664 (inventory sha `8cef061b6e0f`) |
| siblings | `claude_distilled_04_principles_coarse/`, `claude_distilled_24_principles_fine/` |
| evolution record | HF `LASR-Callum/2026-08-03-specgen-constitution-granularity` |
| consumed by | `configs/data/synth/2026-08-01_difficult_advice.yaml`, `configs/data/synth/2026-08-13_pre_action_deliberation.yaml` (segments into 9 traits) |
