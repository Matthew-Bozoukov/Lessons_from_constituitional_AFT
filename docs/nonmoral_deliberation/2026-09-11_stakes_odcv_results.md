<!-- ABOUTME: Completed matched low/high nonmoral stakes SFT and ODCV results with immutable public sources. -->
<!-- ABOUTME: Records exact outcomes, paired uncertainty, run failures, artifact recovery, cost and cleanup. -->
# Matched nonmoral stakes: 19.58% versus 16.67% MR

High stakes had 7 fewer misaligned rollouts than low stakes. The scenario-paired
high-minus-low difference is **−2.9167 percentage points**, 95% CI
**[−7.8770, +2.0437]**, two-sided p=0.2415. This does not establish a stakes effect
or equivalence. There is one seed0 checkpoint per arm; evaluation repeats do not
estimate training-seed variability.

| Outcome | Low stakes | High stakes |
|---|---:|---:|
| Misaligned | 47/240 (19.5833%) | 40/240 (16.6667%) |
| Scenario-level MR 95% CI | [11.0600, 32.2905]% | [9.2572, 28.1659]% |
| Task submitted | 234/240 (97.5000%) | 235/240 (97.9167%) |
| Mean judged progress /5 | 4.937500 | 4.945833 |
| Progress ≥3 | 238/240 | 238/240 |
| Token-limit hits | 5 | 4 |
| Agent cycle-limit hits | 1 | 1 |
| Outer timeouts / terminal API deaths / rollout retries | 0 / 0 / 0 | 0 / 0 / 0 |

All 480 transcripts and both sets of 480 judgments are preserved. Each arm covers
40 scenarios ×2 variants ×3 passes. Partial-status flags were absent for all
transcripts; zero explicit partial markers is not proof of complete work. Submission
is the agent's task_complete signal, distinct from independently judged progress.
Neither establishes preserved general capabilities; formal capabilities testing
remains deferred. Cycle limits occurred in low incentivized Healthcare-Trial-Recruiter
pass2 and high mandated Ai-Hiring-Assistant pass1; both remain scored.

## What the experiment changes

Both mixtures use the same original 684 nonmoral professional-craft scenarios and
9,284 replay rows, with identical replay bytes and positions. Low/high conversations
differ only in numerical loss magnitudes. Recommendations and substantive reasoning
are held fixed; full reasoning and final answers receive supervision. This tests
different stakes exposure during SFT, not how a teacher changes its deliberation
when responding to stakes. Relative loss labels were not independently validated
as perceived stakes. Inherited incomplete prompts and instruction overrides remain
in both arms. [Data construction, audits and pins](2026-09-10_matched_stakes_results.md).

Both Qwen3.6-27B LoRAs trained for one epoch/623 optimizer steps on 2×H200 with
dynamic batching, seed0, rank64/alpha128/dropout0.05, effective batch16, learning
rate1e-4, cosine schedule, warmup0.05, weight decay0.01, maximum sequence8192 and
dynamic token budget8000. ODCV used local Docker/CPU and RunPod H100 inference,
temperature0.7, context28000, concurrency8 and the same Gemini3 Flash MR/progress
judges pinned to Google AI Studio. The original checkpoint's historical18.25%
and later common-protocol13.75% are prior evaluations of one checkpoint; neither
is a new result or training seed here.

## Public artifacts and visualizer

| Artifact | Immutable revision |
|---|---|
| [Low LoRA](https://huggingface.co/dougalldeepmind/2026-09-10-qwen36-0-nonmoral-stakes-low-7) | `e4b2aa199370314e534115eaa5ca08f6e69f2b6c` |
| [High LoRA](https://huggingface.co/dougalldeepmind/2026-09-10-qwen36-0-nonmoral-stakes-high-7) | `af5c9f5356e2b38cc4b065c240a1b0bb950f0e9a` |
| [Low ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-low-7) | `9361ec441aeb2e45eab43fc27c61a0a67d5e174d` |
| [High ODCV](https://huggingface.co/datasets/dougalldeepmind/2026-09-10-odcv-qwen36-0-nonmoral-stakes-high-7) | `54ee07c989bc8cbdea2f608cae0c1bdb24c769a6` |
| [Comparison, measurements and incident evidence](https://huggingface.co/datasets/dougalldeepmind/2026-09-11-nonmoral-stakes-comparison) | `c8f0e504d6d0f4f184b18ba92f5d355617838ecc` |

All are public. The general visualizer discovers the comparison through its
`dataset-model-comparison` tag and reads `results/dataset_comparison.json`. It shows
the stakes manipulation, matched task/reasoning/mixture properties, measured response
lengths, outcome counts and signed paired interval. Live discovery and parsing of
the published revision passed. Dataset rows were downloaded at their immutable pins;
numeric-only synthetic equality, scenario/trait IDs and replay equality were rechecked.
Published eval files match their local bytes, including every full transcript.

Local figures and their results note are in
`output/nonmoral_stakes/20260910/experiment/final_comparison/`.
Figures use the shared naming builder and remain local under repository policy.
The existing report/publisher were extended rather than introducing a new pipeline.
47 focused Python checks and 7 visualizer contract/rendering tests passed.

## Failures, recovery and cost

The 2×H200 training pod disappeared around21:03UTC during archive retrieval, after
both trainings and model publications completed. Cause is unknown; both watchdogs
observed it already gone. Final adapters and complete training metrics were recovered
locally and verified against HF hashes. The original complete archive and high's
final optimizer/RNG/scheduler checkpoint were **not** recovered. Low checkpoint623
and high checkpoint500 remain locally verified. This affects resume artifacts, not
the identity of the final evaluated models.

Low judging hit an upstream429 after all MR and234 progress verdicts were cached.
Only the six missing progress judgments were resumed locally with the same
model/provider and one worker; all existing verdicts remained unchanged. No rollouts
were repeated. High evaluation and judging completed without recovery.

| Tracked exposure | USD |
|---|---:|
| Earlier project work, including datasets | 216.655025 |
| Both SFT runs and training storage/retrieval | 36.654498 |
| Low ODCV GPU/storage + judging | 8.745411 |
| High ODCV GPU/storage + judging | 7.260509 |
| Cumulative total / ceiling | **269.315443 / 300** |

These are conservative elapsed-rate/reservation estimates, not provider invoices.
Both inference pods were terminated only after local serving-log verification;
rollouts/judgments were already local. RunPod now reports no active pods, Docker
has no running containers, and all experiment owners/watchdogs have exited.

Further work should address training-seed variability and whether numerical loss
changes alter perceived stakes. No extra runs were launched, and this result does
not establish a new best nonmoral SFT recipe.
