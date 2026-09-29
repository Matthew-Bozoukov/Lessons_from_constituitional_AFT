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
  Both fresh arms use H100 serving concurrency 8. Only output budget changes.
- H200 and H200 NVL provisioning reported no capacity. Therefore both budgets
  are freshly generated on one H100 host. The earlier H200 6,000-token run remains
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

Results and resource accounting follow after execution.
