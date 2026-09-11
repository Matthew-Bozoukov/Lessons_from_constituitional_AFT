<!-- ABOUTME: Running deliberative SFT generation from a completed synthetic corpus. -->
<!-- ABOUTME: Documents prompt intake, best-of-N Qwen reasoning (host per config), the judge filter, publication and resumption. -->

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

## Three modules

- `data.py`: read the final prompt pool, construct API conversations, and export SFT rows.
- `judge.py`: render the judge request, parse its score, and pick each prompt's survivor.
- `pipeline.py`: validate the YAML, sample Qwen, judge, checkpoint, account for usage,
  resume, and publish. It reuses the existing synth `StageCache` for the HF contract.

There is no scenario/prompt generation or principle ablation. Only `dataset.jsonl` is
read from the source HF repo: its rows already reflect all upstream deduplication and
filtering. Earlier `stages/` snapshots never supply prompts. Each final assistant is
replaced with a freshly generated answer and native reasoning; all earlier conversation
turns stay as context. Row order is preserved; prompts the filter rejects are dropped
and listed in the manifest. `limit` selects a prefix only when explicitly configured
(30 rows in smoke mode).

## The quality filter (`filter:`)

The paper's recipe (Guan et al. 2024, "Deliberative Alignment", s2.3.2), which the first
version of this pipeline skipped:

1. `candidates` completions are sampled per prompt.
2. A format gate drops malformed candidates: truncated, no native trace, no answer, tool
   calls without tools, or an answer that cites the constitution or a numbered principle
   to the user (the generation instructions leaking into the training target). A format
   rejection is final; it is never retried.
3. A constitution-aware judge (`filter.judge.model`, never Qwen) scores a prompt's surviving
   candidates SIDE BY SIDE, `runs` times, from the prompt context, each trace and answer,
   and the same constitution. It never sees a teacher answer: the grade is constitution
   adherence, helpfulness and whether the reasoning leads to the answer. A candidate whose
   answer claims to have already taken an action is flagged to the judge for verification
   (the paper's "another AI determined..." hint). A candidate's score is the MINIMUM across
   runs, because any single run may miss a problem. Comparative rather than absolute
   because, scored one at a time on the 2026-09-09 smoke, the judge gave 240/240 a 10 and
   missed fabricated action claims and "ready to execute" bypass payloads.
4. The best candidate at or above `threshold` is exported (ties to the lowest index).
   A prompt with no survivor gets `resample_rounds` further rounds of `candidates`;
   one still without a survivor is REJECTED.
5. Fewer than `min_rows` survivors aborts the run before anything is published, rather
   than lowering the bar. The smoke sets `min_rows: 1` so it can report the survival rate.

`judge_prompt` (config) travels with the artifact; its fields are `{constitution}`,
`{conversation}` and `{candidates}`, and it must end with one `CANDIDATE <k> SCORE: <n>`
line per candidate. Each exported row's `metadata.deliberative_alignment.judge` records the
score, candidate index and how many candidates were judged for that prompt. Every
candidate's fate is in `stage_3_judge.jsonl`.

## Outputs and failures

The HF repo is named from the config stem using the repository naming rules:
`<date>-delib-synth` for the supplied config, with a separate smoke name.

```text
dataset.jsonl                     # complete SFT dataset; default HF config
stages/stage_1_prompts.jsonl       # final source contexts, metadata and source row indices
stages/stage_2_responses.jsonl     # every candidate: native response, usage, provider, format verdict
stages/stage_3_judge.jsonl         # per prompt: each candidate's scores, the min, the selection
stages/stage_4_export_sft.jsonl    # the survivors, exported
manifest.json                    # resolved config, source SHA, constitution hash, usage
run_meta.json                    # repository result-directory provenance
generation_prompt.txt            # exact full augmentation used only at generation time
generations.partial.jsonl        # append-only candidate records for resume
judgements.partial.jsonl         # append-only judge records for resume
README.md                        # declared dataset/stage configs and provenance card
```

Locally, stage snapshots are flat, matching constitutional SFT. Each response and each
judgement is checkpointed locally, and the checkpoint is mirrored to the Hub at most every
five minutes as one commit (best-effort: a Hub error skips the mirror). A rate limit or
other transient error defers the item and retries it at half the workers after a cooldown,
up to three passes; any other error or an unscorable judge reply stops the run after the
current batch; a rejected prompt is
recorded, never silent, and no partial dataset is published. Resume reuses every
checkpointed response and score and retries only errors. Config, constitution, judge,
prompt snapshot and provider identity must match; workers and budget may change. A new
run refuses to overwrite an existing HF repo.

The budget covers generation and judging together and is checked before each worker
batch, so in-flight calls can exceed it.
Reported API costs are used when available; other responses use registry token prices.
Transport retries inside the shared client can incur additional unreported charges.
OpenRouter exposes a model name, not an immutable weights revision: this is self-generation
from the selected Qwen model, not a guarantee that the hosted weights match a local HF SHA.

The existing `uv run mix` consumes the output as `dataset: <org>/<date>-delib-synth`
with `reasoning: native`; existing training renders and supervises reasoning and the final
answer. This command generates data only; it does not launch training or evaluation.
