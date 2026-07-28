# Research program structure

This note records the assumptions that shape the research log. It is a product
and data-model note, not a claim that the cited results will replicate.

## Research question

The project studies whether synthetic, constitution-oriented training teaches
models merely to reproduce desired language or to generalize the underlying
reasons and traits. Work will include replications, extensions, vulnerability
checks, distribution-shift tests, capability-retention checks, and new training
runs on open-weight models such as Qwen3-32B.

## Intervention lineages

The log must represent branching lineages rather than one universal sequence:

- Existing published checkpoints may follow
  `base -> midtraining -> SFT -> RL`.
- Local runs will usually follow `base -> LoRA SFT`.
- Some local branches may add bounded DPO after SFT.
- Midtraining is a future possibility, not a current assumption.

Each result therefore records the base model, checkpoint, parent checkpoint,
training stage, run, seed, data mixture, and code/data versions when available.

## Evidence layers

The sources motivate keeping these layers visibly distinct:

1. **Data construction** — constitution/spec slice, synthetic document or chat
   format, generation prompts, critique/rewrite pipeline, deduplication,
   mixture ratios, and detected superficial patterns.
2. **Training execution** — method, checkpoint lineage, hyperparameters,
   runtime, cost, failures, and raw artifacts.
3. **Knowledge/recall evaluation** — whether the model can state the taught
   principles.
4. **Behavioral evaluation** — whether the model acts in accordance with them.
5. **OOD/generalization evaluation** — how the evaluation differs from the
   training distribution: single vs multi-turn, tool use, agentic setting,
   adversarial pressure, persona, or task domain.
6. **Vulnerability and regression analysis** — jailbreaks, adaptive audits,
   eval awareness, synthetic-data artifacts, capability loss, tool-use loss,
   and other unintended behavior.
7. **Findings** — interpreted claims linked back to compatible runs and evals,
   with explicit uncertainty and counterevidence.

The UI must not collapse knowledge into internalization, or alignment uplift
into overall success when capability regressions or artifact vulnerabilities
remain.

## Comparison safety

Automatic comparisons group only compatible results: same eval suite, eval
version, and dataset version. Seeds within a compatible group may be summarized
with individual points, a mean, and standard deviation. Cross-version
comparison must be an explicit user action.

## Source-derived design implications

- Model Spec Midtraining describes synthetic documents about a Model Spec
  before alignment fine-tuning and reports that explanations of underlying
  values and specific guidance improve generalization.
- Teaching Claude Why distinguishes direct eval-distribution training from
  principled OOD generalization, and distinguishes constitution knowledge from
  behavioral internalization.
- Google DeepMind's positive-traits update adds practical axes the log should
  preserve: document vs chat format, critique/rewrite quality, baseline-mixture
  ratio, OOD evaluation type, adaptive multi-turn audits, superficial-pattern
  detection, capability retention, and SFT vs bounded-DPO branches.

## References

- [Model Spec Midtraining: Improving How Alignment Training Generalizes](https://arxiv.org/abs/2605.02087)
- [Teaching Claude why](https://alignment.anthropic.com/2026/teaching-claude-why/)
- [Synthetic document finetuning for instilling positive traits](https://www.alignmentforum.org/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)
