<!-- ABOUTME: Current entrypoints, published results and boundaries for delegated-harm evaluation. -->
<!-- ABOUTME: Historical dated notes describe the executed protocol; this file describes the maintained code. -->
# Delegated-harm evaluation

The registered eval tests mixed legitimate and improper workplace requests from a
same-checkpoint peer, a same-checkpoint parent, or an unauthorized human requester.
It measures useful completion, refusal, harmful actions, and concerns addressed to
the requester, another human, or the authorized approver separately.

## Standard entrypoint

Run from the repository root:

```powershell
uv run evals --name delegated_harm --target <hf-adapter> --server <owned-ssh-host> --port <unused-port>
```

The config is `configs/eval/delegated_harm.yaml`. It records the exact initial study
settings and checkpoint pins. For a new study, explicitly replace the target pins
and display label in a resolved config; do not silently reuse the Qwen study label
for another model family. Model mode comes from the adapter metadata. The eval
requires the shared vLLM template and therefore does not accept API targets.

The registered runner and benchmark-specific source loading, rubrics and judging
live under `src/eval/misalignment/delegated_harm/`. The runner receives the served
target and never provisions a GPU, loads weights or starts a model server. Judge
requests use the shared OpenRouter client. The upstream source revision and the two
Windows path patches are validated before use; the unpublished egress fixture is
an explicit exclusion, not a replacement scenario.

## Experimental recovery and scaling

Recovery, queue ownership, worker coordination, launchers, scoring repair and plots
live in `scratch/delegated_harm/`. Their tests also live there. Production modules do
not import them. Their serving entrypoint is:

```powershell
uv run python -m scratch.delegated_harm.run_eval --name delegated_harm --target <hf-adapter> --config <resolved-recovery-config> --server <owned-ssh-host> --port <unused-port>
```

This small wrapper supplies a runner callable to `src.eval.run_eval.main`.
The shared entrypoint still performs registered-target checks, naming preflight,
mode/template pinning, serving, output isolation, run metadata, publication and
cleanup. Metadata records the supplied runner's module/function. The ordinary
`uv run evals` runner refuses a recovery/scaling config rather than silently running
a different experiment. `--no-push` suppresses intermediate checkpoint uploads as
well as the final upload; shard workers use it because the coordinator publishes
the merged run once.

All rentals use `src.infra.runpod.up` with an independent watchdog. The scaling
coordinator runs locally and needs the driver computer to remain awake. Claims do
not expire into duplicate attempts. Its cleanup reconciliation verifies a stopped
worker, an absent launcher, an absent pod and every saved result hash before
correcting a stale status. It never terminates another worker or accepts an
incomplete claim as finished. See [live controls](2026-09-11_horizontal_scaling.md).

Results follow `src/eval/layout.py`: `rollouts/`, `results/`, `metadata/`, and a card
with `eval-run`, `eval:delegated_harm`, model and mode tags. Repacking/rescoring uses
the shared card, naming and `push_run_dir` functions. The publishing organization
comes from `HF_ORG`; the historical study was published under `dougalldeepmind`.
Changing the local config does not move or rewrite those existing artifacts.

## Published recovery results (2026-09-11)

| Model | Originally scored | Recovered | Final scored | Still incomplete |
|---|---:|---:|---:|---:|
| Control | 190 | 113 | 303 / 324 | 21 |
| Difficult advice | 255 | 51 | 306 / 324 | 18 |

All 609 completed episodes are scored. The 39 remaining incomplete attempts are
all `benchmark_rescore`: 14 request timeouts, 14 output-token cutoffs and 11 turn
cutoffs. Other included scenarios have complete coverage. The 445 original scored
records were preserved and hash-checked; recovery used revised acceptance/context
limits and is a documented follow-up, not an untouched rerun of the first protocol.

- [Control results and self-contained rollouts, revision 0a8b5e7](https://huggingface.co/datasets/dougalldeepmind/2026-09-11-dh-qwen3-6-27b-lora-9284-numina-control-716-r64/tree/0a8b5e7698d3d6ca6c84bf4b200528541c076f8d).
- [DA results and self-contained rollouts, revision c4d53c0](https://huggingface.co/datasets/dougalldeepmind/2026-09-11-dh-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch/tree/c4d53c04e20dd41845da6d367866fad857413a34).
- [Initial protocol, model pins and limitations](2026-09-11_eval_plan.md).
- [Recovery protocol changes](2026-09-11_recovery.md).

Plotting code reads pinned HF results and writes dated figures, provenance and
Markdown summaries under gitignored `output/`. AI requests differ across adapters;
only human requests hold wording fixed across models. Repeats and paraphrases are
not independent scenarios or training seeds.

## Verification

```powershell
uv sync --locked
uv run pytest -q tests scratch/delegated_harm
```

No GPU rental or fresh model/API evaluation is needed for the contract checks.
The existing paid run was completed before this code relocation; artifact provenance
retains the actual generating commits rather than claiming this refactor generated it.

Merge validation on Windows: the focused eval/lifecycle/queue tests pass. The broader
suite also exposes 23 failures reproducible on clean main at `624179ff`: 22 properties
tests require PyTorch, which the Windows driver environment excludes, and one CTFish
vendored-file byte check fails under this checkout's line endings. These are not
silently skipped or treated as a green full-suite result.

Final integration-tree run: 1,641 passed, 24 failed, 8 skipped. Besides the 23
baseline failures above, the unchanged internalization end-to-end test encountered
its concurrent cache-write failure. Testing that pipeline on clean main also
reproduced `PermissionError: [WinError 5]` while replacing a shared cache file.
The focused checks in the original study checkout total 127 passes, including its
cached upstream-fixture tests. Clean checkouts skip those three fixture-dependent
checks when the pinned source checkout is absent. Mocked provider tests need a
nonsecret `OPENROUTER_API_KEY=offline-test-placeholder` to construct their client;
no real key or paid provider request is needed.
