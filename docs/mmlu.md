<!-- ABOUTME: Matched MMLU regression-screen protocol and independent arm execution. -->
<!-- ABOUTME: Use the maintained eval and RunPod commands; completed results live on HF. -->

# MMLU capability screen

`configs/eval/mmlu.yaml` evaluates 570 questions: 10 from each of 57 subjects, seed 0,
with seeded option shuffling and five same-subject dev demonstrations. It uses the
pinned `cais/mmlu` test/dev snapshot, generated final answers and answer-key scoring.
This is a matched regression screen, not the full original MMLU benchmark or a vendor
headline-score reproduction. One checkpoint per arm does not measure training-seed
variation, and 10 questions per subject do not support subject-level claims.

The October 14 protocol uses temperature 1, top-p .95, a per-question sampling seed,
32768 output tokens (reasoning plus final answer), 131072 total context, and 16 requests
in parallel. Qwen-specific extensions must be passed explicitly: top-k 20, min-p 0,
repetition penalty 1. They are optional because other endpoints may reject them.
Presence and frequency penalties are zero. Sampling seeds do not guarantee identical
outputs across different GPU execution schedules. Keep the same settings for every arm.

The primary comparison is DA15 minus plain; plain minus base and DA15 minus base are
secondary descriptive comparisons. Report the subject-clustered paired 95% interval.
The predeclared non-inferiority margin is 3 percentage points: the lower bound must be
at least -0.03. Parse rate must be at least 95%, truncation at most 2%, and unresolved
request failures zero. Preserve and report invalid outcomes; do not reroll completed
wrong, malformed or truncated model answers.

## Independent execution

Provision through `uv run runpod up --eval mmlu --target <target>`; serve and evaluate
through `uv run evals`. Pin the target revision in the eval config. Before rental,
resolve both adapters' base revisions, check funds and prices, validate the serving
plan and prompt lengths, and declare a total dollar/lifetime cap. Use watchdogs and
provider expiry, distinct ports and output directories, and keep a local driver awake.
The per-request timeout is not a campaign budget guard.

Run each arm as soon as its own pod is ready and qualified. A failed allocation,
qualification or publication in one arm must not stop healthy arms. Release each GPU
when its inference is finished and evidence is safely retained; retry publication
on CPU if needed. Only shared dollar/lifetime limits stop the whole campaign. Never
terminate unrelated pods. Verify teardown against the provider and sweep for owned
resources. Do not let repeated qualification or startup retries consume the budget.

For example, after provisioning one arm, the maintained invocation is:

```text
uv run evals --name mmlu --target <target> --server <ssh-host> --ssh-key <private-key-path> --port <unique-local-port> --terminate-pod target_revision=<revision> mode=think generation.extra_body.top_k=20 generation.extra_body.min_p=0.0 generation.extra_body.repetition_penalty=1.0
```

Use ports such as 18201, 18202 and 18203 for concurrent drivers. `mode=think` explicitly
matches the October comparison's full-model default and thinking-trained adapters;
do not apply it to an unrelated non-thinking checkpoint. `--terminate-pod` releases
that invocation's RunPod through the maintained lifecycle. It is not appropriate
for a qualification-only invocation if the same live server will run scored items.

Each completed request is flushed to `records.jsonl` and an append-only `attempts.jsonl`.
The latter records completed SDK calls, including failures after internal retries; it
does not expose every internal HTTP retry. Same-directory `run_arm`/standalone recovery
accepts only matching prompt, target and generation identities, retries transport
failures, and reuses all completed model outcomes. It recovers complete journal rows
after interrupted appends or final compaction. The standard `evals` command creates a
fresh run directory, so invoking it again is a new run, not automatic resumption.

## Completed October 14 comparison

| Arm | Target revision | Published results |
|---|---|---|
| Base Qwen3.6-27B | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | [525/570](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36) |
| October 5 plain | `1565d4f3356aa1b3bea1c503700164faa240ef3a` | [523/570](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36-0-plain) |
| October 5 DA15 on plain | `ce773a6b87db66bd569dd13071ec5e27201462f1` | [521/570](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36-0-da-15) |

All arms passed instrument-health and 3pp non-inferiority checks. See `docs/LOG.md`
for intervals, verified artifact revisions, cost and timing. Each HF artifact retains
the resolved config, manifest, full responses, qualification evidence and comparison.
The one-off campaign and closeout scripts remain in Git history at `02d5d645`; they
were retired after completion because they embed local paths and obsolete coupled
arm orchestration. Do not reuse them as a launcher.
