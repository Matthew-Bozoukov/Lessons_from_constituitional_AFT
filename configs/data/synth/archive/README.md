<!-- ABOUTME: Superseded synth configs, kept only as the generating record of corpora that -->
<!-- ABOUTME: were actually published. Do not start new runs from anything in this folder. -->

# Archived synth configs

**Nothing here is used for new work.** These configs are kept because a published dataset
is only reproducible if the exact config that produced it survives, byte for byte — the
`provenance` field of every dataset card in this project points at a config path, and a
deleted config turns that pointer into a dead reference. Archiving rather than deleting is
what keeps the HF corpora auditable.

Two rules, and they are the whole reason the folder exists:

- **Never edit a file in here.** An archived config is a record of a run, not a template.
  Editing one silently changes what a published corpus claims to have been generated from.
- **Never copy one forward.** Start from the live config it was superseded by (below). A
  copy inherits every reason the original was retired.

Archived 2026-08-13, when the natural-turn recipes replaced the source-run ones; the
two-arm post-action-retrospection recipe joined on 2026-08-25.

| archived config | corpus it generated | superseded by | why |
|---|---|---|---|
| `model_eval_model_self.yaml` | [`LASR-Callum/2026-08-06-model-eval-model-self`](https://huggingface.co/datasets/LASR-Callum/2026-08-06-model-eval-model-self) | `par.yaml` | Its documents put the difficult-advice run's own reply — gold, or minimally perturbed — into the turn under evaluation, so every document also taught that recipe's response shape as untrained context. It also has no `final` constitution-rewrite stage, because its corpus predates the stage, which left the self arm differing from both the other arm and difficult advice by an extra step. |
| `model_eval_model_other.yaml` | [`LASR-Callum/2026-08-07-model-eval-model-other`](https://huggingface.co/datasets/LASR-Callum/2026-08-07-model-eval-model-other) | `pc.yaml` | Same inherited-first-turn problem, plus four fixed transcript wrappers reused across the whole corpus — a structural artifact the model can key on. |
| `model_eval_model.yaml` | **none — never run** | `par.yaml` | The original five-cell scaffold (`hf_repo: null`) that `_self` and `_other` were split out of. Archived as superseded rather than as a record: it produced nothing, and its recipe is the one both arms above moved away from. |
| `post_action_retrospection_two_arm.yaml` | [`LASR-Callum/2026-08-17-post-action-retrospection`](https://huggingface.co/datasets/LASR-Callum/2026-08-17-post-action-retrospection) (576 docs) | `par.yaml` (2026-08-25 rewrite) | Ordinary-request genre with a 50/50 good/flawed arm and the whole constitution in three stages. Superseded 2026-08-25 by the one-arm, difficult-advice-shaped, chunk-only recipe: every record's first reply is a grey-area violation, enforced by a gate rather than declared by a label. |
| `da-sep28.yaml` | [`dougalldeepmind/2026-09-28-da-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-da-synth) (1,194 rows; 1,186 after an error audit) | `da.yaml` | The 2026-09-28 difficult-advice recipe: trait notes kept every scenario human except t6. Replaced 2026-10-01 by the recipe built on the 2026-09-23 one. |
| `da-self.yaml`, `da-otherai.yaml`, `da-explicit.yaml` | [`2026-09-30-da-self-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-da-self-synth), [`-da-otherai-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-da-otherai-synth), [`-da-explicit-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-30-da-explicit-synth) | `da.yaml` | Three arms of the 2026-09-28 recipe (the assistant itself, another AI system, explicit asks). Archived 2026-10-02: their base recipe is retired, and `da.yaml` now deals the self and other-AI lines itself (`rotate.ai`). |
| `da-t6-note.yaml` | [`dougalldeepmind/2026-10-01-da-t6-note-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-10-01-da-t6-note-synth) (645 rows; a second 645-row run was never published, `output/synthdoc_v3/20261001_211644`) | `da.yaml` | The 2026-09-23 recipe with the new writer's ask and a t6-only note, still using the wave list, which put another AI system in 74% of rows. Archived 2026-10-02: `da.yaml` is this recipe with the list off and sector and AI presence dealt. |
| `da-unclear.yaml` | [`dougalldeepmind/2026-10-02-da-unclear-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-10-02-da-unclear-synth) (646 rows; the same rows are the head of [`2026-10-02-da-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-10-02-da-synth)) | `da.yaml` | `da.yaml` plus one sentence in the writer, drafter and reviser: the person is weighing, not demanding, and the right option is not clear. Archived 2026-10-02 when `da.yaml` took the sentence in; the two recipes are now identical. Its mixture config is `configs/data/mixture/archive/da-unclear.yaml`. |
| `da-grok-aug20.yaml` | [`dougalldeepmind/2026-08-20-difficult-advice-grok-716`](https://huggingface.co/datasets/dougalldeepmind/2026-08-20-difficult-advice-grok-716), [`2026-09-14-da-grok-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-14-da-grok-synth) | `da-grok.yaml` | The all-grok arm on the 2026-08-20 recipe (whole constitution in the revise stages, 700-character floor, retries 12, max_fail_pct 15). Archived 2026-10-02 when `da-grok.yaml` was rebuilt on the current `da.yaml`. Was `da-grok.yaml`. |
| `da-qwen-sep16.yaml` | [`dougalldeepmind/2026-09-16-da-qwen-synth`](https://huggingface.co/datasets/dougalldeepmind/2026-09-16-da-qwen-synth) (832 rows used) | `da-qwen.yaml` | The all-Qwen arm on the 2026-09-16 recipe. Archived 2026-10-02 when `da-qwen.yaml` was rebuilt on the current `da.yaml`. Was `da-qwen.yaml`; `configs/data/mixture/da-qwen.yaml` still pins this corpus. |

The replacements differ in what they *evaluate*, not just in wording: the turn under
evaluation is generated by the pipeline rather than lifted from a source run, and in
`par.yaml` the whole exchange is too. See "The natural-turn recipes" in
`src/data/synth/ours/README.md`.

## Reproducing an archived run

The configs still work — they were archived, not broken — so a run is one command with the
new path:

```
uv run synth run --config configs/data/synth/archive/model_eval_model_self.yaml
```

One caveat: a run started from here writes a `provenance` line into its HF dataset card
that names `configs/data/synth/<pipeline>.yaml`, which is now the wrong path. That is
cosmetic for a reproduction and another reason not to publish new corpora from this folder.
