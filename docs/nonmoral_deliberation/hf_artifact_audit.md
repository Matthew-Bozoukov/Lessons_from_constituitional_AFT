<!-- ABOUTME: Inventory of this thread's public HF artifacts, their intended uses, and naming corrections. -->
<!-- ABOUTME: Binds the 2026-09-09 audit to live metadata and reversible migration receipts. -->

# Hugging Face artifact audit — 2026-09-09

**Ten public dataset repositories; no new trained model.** Three evaluation names were wrong because a local Windows directory label overrode the automatic model identity. They have been moved to canonical names with history preserved. Cards now identify the actual checkpoint revisions and sampling settings. No transcript, judgment or result was regenerated.

All links below are under `dougalldeepmind`. Revisions are audit snapshots, not promises that a growing corpus will remain unchanged.

| Artifact | Contents and purpose | Downstream use |
|---|---|---|
| [Nonmoral evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch) | 240 transcripts, Docker logs, cell metadata, MR/progress judgments, pinned launch config and costs; 740 files. | Existing nonmoral checkpoint's measurement baseline. Never training data. |
| [Math evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64) | Same 240-rollout protocol, with original timeout evidence retained; 741 files. | Math control in the fixed-checkpoint comparison. |
| [Table2-only evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-only-9284-rank-64) | Same 240-rollout protocol, with original timeout evidence retained; 741 files. | Replay-only control in that comparison. |
| [Baseline comparison](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-baseline-comparison) | Report, chart, exact counts, paired scenario intervals, frozen input results, audits and cost receipts; 96 files at inspection. | Readable scientific result. A research bundle, not another evaluation or pooled model. |
| [Paired development](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-paired-development) | Earlier pilots/calibration and fresh 32-case attempt, prompts, raw calls, reviews, correction and costs; 289 files. | Failure analysis and recipe provenance. Fresh attempt retained 13/32 after correction, below the frozen 24-case gate; **not approved training data**. |
| [Invalid CRLF attempt](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-nonmoral-invalid-crlf-unjudged-attempt) | 32 complete and 8 partial rollout archives, environmental failure audit and cleanup evidence; 329 files. Unjudged. | Forensic record only. **Exclude from all scientific comparisons and training.** |
| [Broader first12](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-first12) | Frozen twelve-case feedback packet: sources, responses/reviews, stage snapshots, raw calls and costs; 48 files. | Inspect the broader recipe before scaling. This packet is not the growing corpus. |
| [Stakes first8](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-first8) | Eight low/high candidate pairs, responses/reviews, retained candidates, local checks and provenance; 63 files. | Inspect whether stakes can vary while holding the nonmoral decision fixed. No trained stakes comparison or measured stakes effect. |
| [Broader synth](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-synth) | **198 accepted examples** at the inspected revision, default `dataset.jsonl`, twelve source/answer stage snapshots, full audits, raw calls, frozen configs and cumulative spend. | Growing production corpus. Planned next stage: select 684 synthetic rows and combine with unchanged 9,284 replay rows in a **separately published mixture**, then train. Not yet trained/evaluated. |

| [Stakes development](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-development) | Failed general stakes wrapper and four integrated craft pairs: full sources, raw answers, reviews, local checks and costs; 855 files. | Failure analysis only. The wrapper was often irrelevant to the decision; the integrated trial retained zero complete pairs after quality review. **Not training data or a stakes-effect result.** |

## Exact snapshots and migrations

The three moved repositories now use `eval_name("odcv", exact_model_key)`. Their basename lengths are 82, 60 and 52 characters, within the 96-character limit. No legacy-name shortening was needed.

| Old basename → current identity | Before revision → verified corrected revision |
|---|---|
| `2026-09-09-odcv-nonmoral-lf-common-3x` → nonmoral evaluation above | `aaf6f8b09e6ed6b283b85fa6d0d8122a14b6e4c6` → `020266bea9bfdb86bc186e5951fea1a2e8fee1ab` |
| `2026-09-09-odcv-math-common-3x` → math evaluation above | `f823ba43645810da5863c8cf9d2401407bfba454` → `7df8b3fb44b16c6b1132807de1c5f1cdb84165eb` |
| `2026-09-09-odcv-table2-common-3x` → Table2 evaluation above | `b12209ccd86eb201a4ea7d69f432e9346fa6c139` → `c805ba3bc12071d78def64f081bcd94c54eeda28` |

Anonymous reads through all three old URLs still retrieved the original pinned result bytes with matching SHA256 hashes. HF `move_repo` preserved history; subsequent commits added corrected cards and migration metadata. Original publication/cost receipts remain historical records; `canonical_publications.json` adds the current mapping without invalidating their hashes.

Other inspected revisions:

- Comparison: `a8fd17e2ac42ad7129596fdc3ce6e36703d0b7ec`.
- Paired development: `105dc5c9c84d1de0361e14d453347777aaf60e85`.
- Invalid attempt: `6d2fc052115789118dbe5476d1e1313a58e801d2` → card-only correction `eac4da56e03a39f9d52f27ed34320ce55cf03c9b`.
- Broader first12: `e10cf7ae8c7750094dc4017102338b8fe583fd27`.
- Stakes first8: `b951660317dcca7844337ac3a181ca6dca8602dd`.
- Stakes development: `685a82c88faeb0b44297d53ecad5a42c84703752`.
- Broader synth: `ccfc003805c8c8c3c2b073aa315daa2d48490550`.

## Pipeline and card corrections

`src/eval/run_eval.py::_run_repo` previously returned `artifact_name(run_name)` when a local label was supplied. It now always derives published identity from the eval and model key; `run_name` remains a local directory label. The naming error no longer recommends bypassing identity. Overlong future model names fail explicitly.

Ordinary future evaluation cards now read the frozen launch `run_meta.json`: exact target/base revision pins, source commit and actual config. Root-level ODCV sampling settings populate `generation_config` instead of `{}`; existing explicit generation blocks remain supported. Focused regressions cover the local-label bug, length boundary, sampling metadata and launch source revision. Validation: `uv run --no-sync pytest -q tests/test_eval_framework.py tests/test_naming.py` — **75 passed**; `git diff --check` passed.

The three current evaluation cards received the same provenance correction through shared card/gate helpers. The invalid archive keeps its explicit invalid name, but its tags now use the exact adapter key and `mode:think`, replacing generic model and `mode:thinking` tags. Its card prominently excludes it from results. Consumers must honor invalid status; `eval-run` alone does not mean a valid experiment.

The other names are legitimate **research/review archives**, for which generic `artifact_name` is appropriate. They must not masquerade as stage outputs. The correct production-stage identity is `synth_name("nonmoral-broader")` → `2026-09-09-nonmoral-broader-synth`, now published with training-data discovery tags. A generator model belongs in dataset provenance, not automatically in a synthetic corpus's name; model/eval identities follow their own naming functions. First12/first8 remain frozen review packets.

The attempted `LASR-Callum/2026-09-08-nonmoral-paired-synth` upload returned 403 with `uploaded:false`; it is **not an additional upload**. Its failed receipt is preserved in development history. The audit covers metadata and declared use of the stakes packet; its separate material-validity review belongs to the stakes workstream.

Evidence: `output/nonmoral_investigation/20260909/hf_artifact_audit/{live.json,production_live.json,migration_plan.json,migration_receipt.json,invalid_card_receipt.json,canonical_publications.json}`. Current links in the baseline inventory are corrected; older pinned references remain reproducible through redirects. The comparison's frozen input receipts intentionally retain their original identities.
