<!-- ABOUTME: Complete public artifact index for this nonmoral deliberation workstream. -->
<!-- ABOUTME: Distinguishes actual training inputs, new models, evals, comparisons and untrained development archives. -->
# Nonmoral workstream: all 25 public HF artifacts

Verified anonymously on September11. Each revision is immutable; current repository links follow the latest presentation metadata. Numerical results were not changed by the September11 figure-storage/tagging corrections. There are 6 training datasets, 3 new LoRAs, 6 completed ODCV runs, 3 comparison artifacts and 7 development/incident archives.

## Training data

| Artifact | What it contains and downstream use | Verified revision |
|---|---|---|
| [Broader mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-7-mix) | The actual SFT input: 684 synthetic plus 9,284 replay rows. | `f1e61baf643c861920303c7ba1e9844df5f6ed48` |
| [Broader corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-synth) | 705 accepted conversations and generation/audit stages; 684 selected for training. | `a3d266e2f0cc48e26e153caf078a5d641ecbbb5c` |
| [High-stakes mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-high-7-mix) | Actual high SFT input; replay bytes and positions match low. | `51a2e0f6468f5e58a009f3acd0f33503df0fa3b8` |
| [High-stakes corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-high-synth) | The same 684 conversations with higher numerical losses. | `775270ae96acb57e18c436f4cb5f57191d2ceb04` |
| [Low-stakes mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-low-7-mix) | Actual low SFT input: 684 synthetic plus 9,284 replay rows. | `de6fbbd3074c3bb43dbab2484166e537b1efc7ed` |
| [Low-stakes corpus](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-low-synth) | 684 complete matched conversations; lower numerical losses. | `40733eff09d8e6775d645f67a0e61f50fbf6d966` |

## New LoRAs

| Artifact | What it contains and downstream use | Verified revision |
|---|---|---|
| [Broader LoRA](https://huggingface.co/dougalldeepmind/2026-09-09-qwen36-0-nonmoral-broader-7) | Seed0 Qwen3.6-27B adapter; apply to pinned base model for downstream inference/evaluation. | `d52838446ef134088841e8dc436094d93827487f` |
| [High-stakes LoRA](https://huggingface.co/dougalldeepmind/2026-09-10-qwen36-0-nonmoral-stakes-high-7) | Seed0 Qwen3.6-27B adapter trained on the high-stakes mixture. | `af5c9f5356e2b38cc4b065c240a1b0bb950f0e9a` |
| [Low-stakes LoRA](https://huggingface.co/dougalldeepmind/2026-09-10-qwen36-0-nonmoral-stakes-low-7) | Seed0 Qwen3.6-27B adapter trained on the low-stakes mixture. | `e4b2aa199370314e534115eaa5ca08f6e69f2b6c` |

## Completed ODCV

| Artifact | What it contains and downstream use | Verified revision |
|---|---|---|
| [Math control re-evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen3-6-27b-lora-9284-numina-control-716-r64) | 92/240 MR; 240 full rollouts and judgments. Reused an existing math LoRA. | `7df8b3fb44b16c6b1132807de1c5f1cdb84165eb` |
| [Broader ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-0-nonmoral-broader-7) | 76/240 MR; evaluates the newly trained broader LoRA. | `fe7b98403d7efca11764fc94a5e7443b720c77ee` |
| [Original nonmoral re-evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch) | 33/240 MR; 240 full rollouts, both judges and protocol metadata. Reused an existing LoRA. | `020266bea9bfdb86bc186e5951fea1a2e8fee1ab` |
| [Replay-only re-evaluation](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-qwen36-lora-table2-only-9284-rank-64) | 90/240 MR; 240 full rollouts and judgments. Reused an existing replay-only LoRA. | `c805ba3bc12071d78def64f081bcd94c54eeda28` |
| [High-stakes ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-high-7) | 40/240 MR; complete without judge recovery. | `54ee07c989bc8cbdea2f608cae0c1bdb24c769a6` |
| [Low-stakes ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-low-7) | 47/240 MR; six missing progress verdicts recovered, with no rollout rerun. | `9361ec441aeb2e45eab43fc27c61a0a67d5e174d` |

## Comparisons

| Artifact | What it contains and downstream use | Verified revision |
|---|---|---|
| [Baseline controls](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-baseline-comparison) | Exact original nonmoral, math and replay-only results, paired statistics, protocol and failure accounting. | `b6eef613190f589c3466b9f2e67c1d55ee810589` |
| [Broader comparison](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-comparison) | Four-checkpoint results and general visualizer metadata relating dataset properties to model outcomes. | `860ce648efa8579e0876e05284cc2146c2d6a1a2` |
| [Stakes comparison](https://huggingface.co/datasets/dougalldeepmind/2026-09-11-nonmoral-stakes-comparison) | Low/high results, paired uncertainty, measured traits, artifact recovery and cost evidence; visualizer input. | `c8f0e504d6d0f4f184b18ba92f5d355617838ecc` |

## Development and incident archives

| Artifact | What it contains and downstream use | Verified revision |
|---|---|---|
| [Broader first12](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-broader-first12) | The initial 12-example design packet, not the final training population. | `e10cf7ae8c7750094dc4017102338b8fe583fd27` |
| [Paired-deliberation development](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-paired-development) | Early pilots, full candidates, corrections and audits. Fresh32 yielded13 qualifying pairs; no paired SFT/ODCV. | `105dc5c9c84d1de0361e14d453347777aaf60e85` |
| [Stakes development](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-development) | Stopped wrapper/integrated-craft attempts, raw calls, failures and accounting; not a trained-model result. | `685a82c88faeb0b44297d53ecad5a42c84703752` |
| [Stakes first8](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-nonmoral-stakes-first8) | Eight early low/high candidate pairs; superseded by the matched684 experiment. | `b951660317dcca7844337ac3a181ca6dca8602dd` |
| [Invalid CRLF attempt](https://huggingface.co/datasets/dougalldeepmind/2026-09-09-odcv-nonmoral-invalid-crlf-unjudged-attempt) | 32 complete and8 partial contaminated traces; INVALID and UNJUDGED, excluded from every effect estimate. | `1be8548002cb72962b0ddad09512fffedccd6ffa` |
| [Grounded-revision pilots](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-grounded-revision-pilot-audit) | 32 revisions, source pairs, judge/local reviews, costs and failed gates; no production corpus or LoRA. | `58f80cd4ec9d2f7749d66a1e92153399b301ee38` |
| [Matched-stakes pair audit](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-nonmoral-stakes-pair-audit) | All generation/correction stages and original-to-low/high differences; reproduction and quality audit input. | `a5b18d5a7236a814ec1ea5db283761633aa3b3a3` |

## Existing inputs reused, not created here

- [Original nonmoral LoRA](https://huggingface.co/dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch).
- [Math-control LoRA](https://huggingface.co/matboz/qwen3.6-27b-lora-9284-numina-control-716-r64).
- [Replay-only LoRA](https://huggingface.co/dougalldeepmind/2026-08-04-qwen36-lora-table2-only-9284-rank-64).
- [Original nonmoral training mixture](https://huggingface.co/datasets/dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture).

The original18.25% historical MR is prior work; our common-protocol re-evaluation of that checkpoint was13.75%. The missing paired/no-comparison LoRAs are not omitted links: those pilots did not reach training.

## Standards audit

The read-only audit checked public access, required card fields and production dates through `gate_push`, canonical repo IDs, eval tags/files and final-model artifacts. All25 pass after correction. Current eval names include the evaluated model identity; existing migrated aliases redirect and their immutable revisions remain readable. Models/mixtures/corpora use the shared builders; the seed0 appears before the style in the implemented model naming law. Six completed runs each retain240 transcripts.

Four old comparison plot files were copied and hash-verified locally before removal from current HF revisions. Old revisions remain readable; every non-presentation file was verified unchanged. The invalid/unjudged attempt now carries `research-incident` instead of `eval-run`; no missing score or provenance was invented. New chart exports remain local as CLAUDE.md requires.

Local audit receipts: `output/nonmoral_writeup/20260911/hf_audit/corrections.json` and `hf_audit_final/inventory.json`. [Writeup summary and chart guide](2026-09-11_writeup.md).
