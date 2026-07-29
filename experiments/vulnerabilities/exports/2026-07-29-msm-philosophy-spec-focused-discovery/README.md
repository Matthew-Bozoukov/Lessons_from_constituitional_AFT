---
license: other
task_categories:
- text-generation
tags:
- alignment
- llm-evaluation
- model-spec-midtraining
- petri
size_categories:
- 10<n<100
---

# Petri focused discovery: 30 audits, MSM philosophy-spec checkpoint

This is the Hugging Face mirror of a curated export bundle. See `index.md` in
this repository for the full report (following the research log's Petri export
format); this README adds the fields required by this org's data convention.

- **experiment**: Petri (Inspect) focused-discovery audit - 10 seed archetypes across two hypothesis families (`B-provenance-spoofing`, `C-harmful-overgeneralization`) x 3 epochs, 30 audits total, against the primary MSM target checkpoint.
- **date_generated**: 2026-07-29
- **constitution**: The Philosophy Spec from "Model Spec Midtraining" (chloeli/model_spec_midtraining, arXiv:2605.02087, Appendix D.1).
- **source_repo**: `Matthew-Bozoukov/teaching_claude_why_replication`, commit `83877eade31a291522226eab19bb1074111d6998`, path `experiments/vulnerabilities/exports/2026-07-29-msm-philosophy-spec-focused-discovery/`
- **models**: Target `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot` on `Qwen/Qwen3-32B`. Auditor `claude-sonnet-5`, judge `claude-opus-5`, realism `claude-haiku-4-5`. Full detail in `index.md` frontmatter.
- **generation_config**: `max_turns=30`, `realism_filter=0.6` (threshold 0.6), `epochs=3`, temperature 0.7. See `index.md` frontmatter.
- **schema**: `data/scenarios.jsonl` (10 seed scenarios), `results/transcripts.jsonl` (28 retained transcripts, full multi-turn), `results/scores.json` (aggregate outcomes), `artifacts/` (raw grader responses, failed generations, validation funnel, tool-call fidelity, run config). See the visualizer's export guide for the exact field semantics.
- **provenance**: Built by `experiments/vulnerabilities/scripts/petri/build_export.py` from the raw `.eval` log and validation analysis. The raw logs and validation analysis are published separately as `LASR-Callum/2026-07-29-msm-philosophy-spec-petri-validation`.

## Headline result

No seed produced a validated finding in a majority of its 3 epochs; the compound flagging rule that produced the candidates had a measured 57% false-positive rate (8 of 14 flags overturned on adversarial validation). Six transcripts survive as behavioural existence proofs. See `index.md` for the full report.
