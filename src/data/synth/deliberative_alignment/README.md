<!-- ABOUTME: Running deliberative SFT generation from a completed synthetic corpus. -->
<!-- ABOUTME: Documents final-row prompt intake, Alibaba reasoning, publication and safe resumption. -->

# Deliberative SFT generation

```sh
uv run synth run --config configs/data/synth/delib.yaml --smoke
uv run synth run --config configs/data/synth/delib.yaml
uv run synth run --config configs/data/synth/delib.yaml --resume output/synth_delib/<run>
```

The config's `method: deliberative_alignment` selects this pipeline. Existing configs
default to `ours`, with the same `uv run synth run --config ...` command.
The path-invoked `scripts/data/synth/build_dataset.py` delegates to that shared runner.
`delib.yaml` starts from the pinned final corpus
`LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716`.
Its 708 final rows supplied the 702 trait-balanced synthetic examples in the
chunk-only-702 adapter's training mixture; this pipeline uses all 708 source prompts.
The generation constitution is selected independently by `constitution`. Set
`source.repo` and `source.revision` for a different source; every read resolves to an exact SHA.

## Two modules

- `data.py`: read the final prompt pool, construct API conversations, and export SFT rows.
- `pipeline.py`: validate the YAML, call Qwen, checkpoint responses, account for usage,
  resume, and publish. It reuses the existing synth `StageCache` for the HF contract.

There is no scenario/prompt generation, judge, quality filter, or principle ablation.
Only `dataset.jsonl` is read from the source HF repo: its rows already reflect all
upstream deduplication and filtering. Earlier `stages/` snapshots never supply prompts.
Each final assistant is replaced with a freshly generated answer and native reasoning;
all earlier conversation turns stay as context. Row order and multiplicity are preserved.
`limit` selects a prefix only when explicitly configured (two rows in smoke mode).

Generation adds the full constitution and deliberation instructions to a copy of the
original system prompt. The model is Qwen via OpenRouter, with the existing Alibaba pin
and fallbacks disabled. The dataset contains the original context and the new assistant
turn (`reasoning_content`, `content`, and optional `tool_calls`), with `supervise: final`.
Generation-only additions are absent from the training context; references to the
constitution in generated reasoning are retained. Tool schemas and prior tool messages
are preserved; API transport IDs are reconstructed only where unambiguous.

## Outputs and failures

The HF repo is named from the config stem using the repository naming rules:
`<date>-delib-synth` for the supplied config, with a separate smoke name.

```text
dataset.jsonl                     # complete SFT dataset; default HF config
stages/stage_1_prompts.jsonl       # final source contexts, metadata and source row indices
stages/stage_2_responses.jsonl     # native responses, finish reasons, usage and provider
stages/stage_3_export_sft.jsonl    # complete exported rows
manifest.json                    # resolved config, source SHA, constitution hash, usage
run_meta.json                    # repository result-directory provenance
generation_prompt.txt            # exact full augmentation used only at generation time
generations.partial.jsonl        # append-only attempt records for resume
README.md                        # declared dataset/stage configs and provenance card
```

Locally, stage snapshots are flat, matching constitutional SFT. Each completed response
is checkpointed locally, and each worker batch is mirrored. A malformed/incomplete
response or provider error stops the run after the current batch; no source row is silently
dropped and no partial dataset is published. Resume reuses successful rows and retries
failed/missing rows. Config, constitution, prompt snapshot and provider identity must
match; workers and budget may change. A new run refuses to overwrite an existing HF repo.

The budget is checked before each worker batch, so in-flight calls can exceed it.
Reported API costs are used when available; other responses use registry token prices.
Transport retries inside the shared client can incur additional unreported charges.
OpenRouter exposes a model name, not an immutable weights revision: this is self-generation
from the selected Qwen model, not a guarantee that the hosted weights match a local HF SHA.

The existing `uv run mix` consumes the output as `dataset: <org>/<date>-delib-synth`
with `reasoning: native`; existing training renders and supervises reasoning and the final
answer. This command generates data only; it does not launch training or evaluation.
