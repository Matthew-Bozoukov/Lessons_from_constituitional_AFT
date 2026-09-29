<!-- ABOUTME: Bounded paired Qwen Arena output-budget study using eight fixed prompts. -->
<!-- ABOUTME: Separates completion, answer usefulness, judge preference and sampling limitations. -->

# Arena output-budget study, September 29

The user approved a small 6,000-versus-12,000-token check after three of eight
nosynth smoke responses exhausted the existing budget. This is a protocol study
on one checkpoint, not evidence of a training treatment effect.

## Frozen design

- Target: `dougalldeepmind/2026-09-22-qwen36-0-nosynth` at
  `633908b72a9799fb3e6b101b0a8a82aec3c3d642`.
- Training base: `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- The same deterministic smoke selection: four hard and four creative prompts.
  All eight are retained regardless of outcomes; no answer is rerolled for quality.
- Thinking enabled, temperature 0, top-p 1, vLLM 0.26.0, context 16,384.
  Both fresh arms use A100 80GB serving concurrency 8. Only output budget changes.
- H200 and H200 NVL provisioning reported no capacity. Therefore both budgets
  are freshly generated on one host. The first H100 allocation failed its GPU
  driver check (`nvidia-smi`: failed to initialize NVML) before any answers, and
  was terminated. With no healthy H100/H200 capacity available, an A100 80GB
  replacement was selected for both budgets. The earlier H200 6,000-token run remains
  a separate replication observation, not the paired reference.
- Primary checks: normal completion, nonempty final answer, observed dialogue
  looping, and concrete prompt compliance. Counts and per-prompt outcomes are
  reported; there is no threshold calibrated from this eight-prompt sample.
- GPT-4.1 additionally compares the final answers in both positions (16 judgments).
  Preference is descriptive. It is not proof of correctness or representative
  benchmark performance. Empty and truncated answers stay in the denominator.
- Review explicitly distinguishes the barber's unrequested dialogue loop from
  the song's requested chorus reprise and the rap's prescribed repeated wording.
- No default changes are assumed in advance. Existing 6,000-token results retain
  their identity; a future budget change requires matched regeneration of arms.

## Fresh 6,000-token reference

The A100 reference completed in approximately 266 seconds after approximately
20 minutes of first-start model loading, compilation and API initialization.
Four of eight responses hit the limit. Rap, Zig and the Spanish poem had empty
final answers; the barber dialogue was nonempty but cycled through recommendations
without reaching the ruined haircut. It repeated "Okay, I'll have to try it
sometime" 41 times across 376 dialogue lines. This repetition is not requested.

The other four responses finished. The song retained its seven sections and
requested chorus reprise; CSS gave parent flexbox centering; the table prompt
retained diagram fields and added structured table fields. The video code was
incorrect despite finishing: `frame_count % 10 == 0` saves every ten frames,
not every ten seconds, and it assumes its output directory already exists.

The earlier H200 6k run had three truncations and two empty finals. This difference
is why the two fresh A100 budgets are the matched comparison; the H200 run is
descriptive only. Hardware/backend/batching differences cannot be separated from
each other by this comparison.

## Results and interpretation

| Observation | Fresh 6,000 | Fresh 12,000 |
| --- | ---: | ---: |
| Prompts | 8 | 8 |
| Normal completion | 4 | 8 |
| Output-limit termination | 4 | 0 |
| Empty final answer | 3 | 0 |
| Generation time, excluding startup | about 266 s | about 498 s |

The 12k run supplied finals for the rap, Zig and Spanish poem, and the barber
stopped after 26 dialogue lines with the ruined haircut and customer departure.
These are improvements in answer availability and observed story completion,
not four established task successes. The rap misses specified verse-line
positions and the requested rhyme structure. Zig uses the wrong diagonal index:
for row 1, column 2 it returns 31916031 instead of the supplied 18749137. It also
assumes two integer input lines and uses `std.debug.print` while claiming stdout;
it was not compiled. The video code hardcodes 30 FPS, so its 300-frame interval
is ten seconds only for that frame rate. Song, CSS and table-prompt answers in
both runs address their core structural requirements.

The rap is not a refusal: the heuristic matched lyrical phrases such as
"I'm just a machine". Likewise, requested chorus repetition and the Spanish
poem's formulaic but varying stanzas are not equivalent to the low barber's
unrequested endless exchange. Raw heuristic metrics remain available; the
qualitative review documents these distinctions.

All eight reasoning trajectories diverged before the 6k limit despite the same
host, settings and greedy decoding. The study therefore compares protocol
outcomes; it does not isolate the causal effect of appending tokens to the same
trajectories. Re-tokenizing saved text with the exact pinned base tokenizer puts
the high rap final after about 10,933 reasoning tokens and Zig after 7,466. Those
particular trajectories need more than 6k to reach a final. The Spanish high
trajectory instead uses about 4,925 reasoning tokens before its final, so its
rescue cannot be explained as needing more than 6k on that trajectory. These are
saved-text token estimates, excluding lost control tokens/whitespace, not exact
provider token usage.

GPT-4.1 completed all 16 judgments without retries. It preferred 12k on rap,
Zig, barber, Spanish poem and video; it preferred 6k on song, CSS and table prompt.
Both orderings agreed for all eight prompts. Descriptive preference scores are
75% for creative writing and 50% for hard prompts (four prompts per slice).
Preference over a blank answer does not make the flawed Zig code correct.
The eight prompts are a smoke selection, not a representative calibration set.

## Recommendation and reproducibility

Use 12,000 as the candidate budget for future matched Arena comparisons, because
it removed truncation and empty finals in this bounded check. Regenerate both
comparison arms under that protocol, retain every failure, and continue to report
completion alongside preference. Do not pool old 6k and new 12k answers or raise
the cap repeatedly to chase favorable scores. The repository default remains
6,000 in this study; adopting a new default is a separate explicit protocol
change. This work does not repair model mistakes by changing prompts or dropping
unfavorable samples.

The comparison driver is `scratch/arena_budget_compare.py`. Its prepare step
checks exact target/base revisions, all eight prompt identities, actual per-row
allowances, extraction and serving settings, then freezes source snapshots with
SHA-256 hashes. Analysis verifies those snapshots. The real prepare, analysis
and judge paths completed successfully, preserving every raw generation and
judge response. No production code changed in this follow-up.

Published artifact: [Arena output-budget comparison](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-arena-hard-output-budget-comparison).
The verified dataset revision is `c8d3f02b716f3f693cf611d683e618747d6753d0`.
Eight key uploaded files were downloaded at that revision and hash-checked against
the local evidence. The artifact
includes the manual review, pinned-tokenizer counts, generation protocols,
infrastructure records and original transcripts, including the separate H200
replication. Source driver revision: `0eb7c76a`.

## Resources and cost

The healthy A100 pod `4vrt4hv3a85bzb` cost $1.59/hour and ran for approximately
53 minutes: about $1.41. Most elapsed time was model startup/compilation:
approximately 20 minutes for the first invocation and 13 for the second.
The failed H100 pod `u4oc7h522310c1` generated no answers and adds approximately
$0.50. GPT-4.1 reported exactly $0.12232 for 16 calls. Total study spend is about
$2.03, with GPU amounts estimated from rate times duration, not an invoice.

Both study pods were verified absent after generation. The teardown recorded
shared account balance $460.925109804 and $31.743/hour for six other tasks' pods;
those resources were left untouched. Shared balance changes are not this study's
cost. These amounts exclude the prior qualification study.
