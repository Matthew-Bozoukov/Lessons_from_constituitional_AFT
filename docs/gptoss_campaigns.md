<!-- ABOUTME: Immutable result links and source layout for the October 2026 GPT-OSS campaigns. -->
<!-- ABOUTME: Separates full evaluations, format-health failures and historical diagnostic protocols. -->

# GPT-OSS campaign results and code

The [research log](LOG.md) contains the dated methods, costs, failure evidence and
closeout records. This index points to immutable published revisions; intermediate
status entries in the log are historical, not instructions to restart a campaign.

## Final evaluations

| Evaluation | Base | Control | DA15 | Scope |
|---|---:|---:|---:|---|
| SWE-bench Lite | 67/300 (22.33%) | 58/300 (19.33%) | 87/300 (29.00%) | Official grading, all 900 outcomes; no infrastructure exclusions |
| MMLU | 505/570 (88.60%) | 450/570 (78.95%) | 466/570 (81.75%) | Matched 570-question screen; **Control fails parse-health threshold** |

- [SWE-bench final results, source and raw evidence](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-gptoss120b-base-control-da15-swe-lite300-context65000/tree/4ccd7690774909418179133c1204a0a058f86cc8)
- [SWE infrastructure qualification](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-gptoss120b-base-control-da15-swe-lite300-context65000-infrastructure/tree/44bb59ec02f39f13c27d71e7143dbb16f4eca524)
- [MMLU Base](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-base/tree/345a70aa66775007425328a9b7c31c09f13ef390)
- [MMLU Control, diagnostic because of failed instrument health](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-plain/tree/862acf3443fb49172a8202637c6381421641fc65)
- [MMLU DA15](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-da-15/tree/36ea3662506b356c4d27bc360ffd2525106f426d)

SWE used exactly 65,000 live prompt-plus-response tokens, 16,384 response tokens,
262,144 generated tokens/task and 500 steps. Official resolved totals include
limit-ending patches: Base 7, Control 18, DA15 21. Submitted does not mean resolved.
The retained HTTPBin fixture caveat permits two no-fix passes.

MMLU used five dev examples, ten test questions per subject, seed 0, medium native
reasoning and a 32,768-token reasoning-plus-final response cap. Final answer content
alone was scored. Unparseable counts are Base 0, Control 41 and DA15 17; empty final
counts are 0, 31 and 9. Control parses 92.81%, below the predeclared 95% threshold.
Its accuracy is descriptive, not a clean capability comparison. This is not full
14,042-question MMLU. No reasoning fallback or retrospective rescoring was applied.

Settled inference estimates: SWE **$226.828344234**, excluding CPU/transfer;
MMLU **$2.779300254**, including **$0.002798850** smoke. Zero outstanding reservations
at closeout; these are accounting estimates, not invoices. The full SWE CPU was
destroyed at the user's request after publication. The MMLU Docker driver was
removed; local caches were retained. Neither campaign needs an active owner.

## Separate diagnostics

These have different selections or protocols and must not be pooled with the final
evaluations above. Earlier smokes and intermediate revision links remain in LOG.md.

| Artifact | Interpretation boundary |
|---|---|
| [ODCV 56K original publication](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-gptoss120b-base-control-da15-odcv-full56k/tree/ee25be032bfe137b1f61a896104e9d4260e54eac) | Published with explicit judging exclusions; artifact completeness is not judge completeness |
| [ODCV alternate-judge supplement](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-gptoss120b-base-control-da15-odcv-full56k/tree/eaaa045bffedd1e7b22d05e03b5b0f1b04d35d41) | Mixed-judge supplement; original evidence preserved |
| [ODCV 112K diagnostic](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-gptoss120b-base-control-da15-odcv-context112k/tree/1cbf49ac12d070555c3ab5ea73b78522fd7df76d) | Separate context experiment on selected cells |
| [MXFP4 backend pilot](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-gptoss120b-base-tinker-mxfp4-pilot/tree/738c09e9bb37cf906537387eddcb44bac31617fb) | Backend/quantization pilot, not a matched three-arm full benchmark |
| [Historical patches at 65K cutoff](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-swebench-gptoss120b-patch-cutoff65000/tree/bf04c53ef3f8f73a17269c2035ba90403a89306c) | Saved-patch diagnostic; original outcomes unchanged |

## Source structure

- [Tinker endpoint reference](tinker_eval_tools.md): shared transport, Harmony and accounting.
- [SWE/ODCV campaign code map](../scratch/gptoss_swe/README.md): historical orchestration,
  isolated recovery helpers, configs and campaign tests in `scratch/`.
- [MMLU campaign code map](../scratch/gptoss_mmlu/README.md): local Docker orchestration,
  bounded ledger, qualification and publication around the maintained MMLU evaluator.
- Reusable implementations and regression tests remain in `src/` and `tests/`.
  Bulk data stays on HF, with local receipts under gitignored `output/`.

Use each artifact's source archive and resolved configuration for exact reproduction.
The cleanup commit is not the generating revision, and old owner/recovery commands
are not safe substitutes for a fresh run with a new identity.
