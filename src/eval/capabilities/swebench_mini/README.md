# SWE-bench: the standardized baseline (`swebench_mini`)

Agentic coding capability, measured with a scaffold that is **not ours**: upstream
mini-SWE-agent v2, pinned, with its official `swebench.yaml` passed through unedited. One
rollout per task, no retries, no reranking, no reviewer model, no extra tools, no planning
layer. Graded by the pinned official SWE-bench docker harness.

Keep this separate from any custom scaffold we build. A different scaffold is a different
measurement and gets its own registry name, config and output tree — the two must never be
pooled or compared as though they were the same benchmark.

## Running it

Two phases, on purpose. Only the first needs a GPU.

```bash
# 1. Rollouts — needs the served model AND docker (the agent works inside SWE-bench images)
uv run scripts/run_eval.py --target Qwen/Qwen3.6-27B --name swebench_mini

# 2. Grading — docker + CPU only; run it after the GPU box is destroyed
uv run scripts/eval/swebench_mini_grade.py --run-dir output/swebench_mini/<key>/<ts>
```

Depth is `subset.fraction` (or `subset.n`) in
[`configs/eval/swebench_mini_verified.yaml`](../../../../configs/eval/swebench_mini_verified.yaml):

```bash
uv run scripts/run_eval.py --target <hf> --name swebench_mini subset.fraction=0.2
```

Subsets are **repo-stratified and nested**: the 10% draw is a strict subset of the 20% draw,
and mini-SWE-agent skips instances already in `preds.json`, so deepening a run costs only the
new instances. Positional slicing (`--slice 0:50`) is deliberately unused — the dataset is
clustered by repository, so a positional slice samples whichever repos sort first rather than
the benchmark.

## Reporting format

```
<model> + mini-SWE-agent <version> (config <sha256>), <dataset>@<revision>
[n/N instances, subset <hash>], pass@1 = X% [CI]
```

`metrics.report_line` builds it; every field is resolved at run time rather than assumed.
Quote the subset size and hash whenever the run was not the full split — the format otherwise
implies all 500 Verified instances.

Read the counters next to pass@1 before believing it. A low score from lost tool-call
formatting, from context overflow, and from genuine inability all look identical in the
headline number and need opposite fixes; `no_tool_call_rate`, `exit_statuses`, `patch_rate`
and `empty_reasoning_rate` are what tell them apart.

## Why two pinned environments

`envs/agent` (mini-SWE-agent) and `envs/harness` (the official harness) are separate uv
projects with **committed lockfiles**. They co-resolve fine against the main project — this is
not a conflict workaround. It is reproducibility: litellm sits in the agent's request path, so
a drifting transitive version silently un-pins a baseline whose whole value is being pinned,
and `uv run --with` re-resolves on every invocation. Splitting agent from harness means
regrading old predictions never needs the agent's stack, and the harness commit is recorded
independently of the scaffold commit. Each `pyproject.toml` carries the full argument.

## Deviations from stock, and why each exists

The official config file is never edited — it is passed with `-c` and layered under a small
overlay (`mini-extra swebench` deep-merges repeated `-c` specs), so `config_sha256` stays
comparable with upstream byte for byte.

| Deviation | Why |
| --- | --- |
| `model.model_kwargs.api_base` | The endpoint swap. The point of the exercise. |
| `environment.run_args` += `--network none` | Approved network isolation; not expressible in the official config. Verified images are self-contained. |
| `MSWEA_COST_TRACKING=ignore_errors` | litellm has no price for a local model and the default mode errors. **Consequence: the config's `cost_limit: 3.0` can never fire, so `step_limit: 250` is the only per-instance bound.** State this wherever the baseline is reported. |
| `MSWEA_GLOBAL_CONFIG_DIR` → empty dir | mini-SWE-agent otherwise auto-loads a machine-global `.env`, letting one developer's leftover settings change the baseline on their machine alone. |
| `serving:` block (context, tool-call parser) | Upstream drives a real bash tool and runs 250 steps; the family default 16384 context and a server with no tool-call parser both fail in ways that look like incapability. |

## Verify a grading host before trusting it

```bash
uv run scripts/eval/swebench_mini_check_env.py --n 2
```

Submits each instance's own reference patch (`--predictions_path gold`) through the pinned
harness. A correct host resolves 100%. Anything less means the environment is broken — wrong
images, docker misconfigured, tests not executing — and every model score produced there would
be wrong in the same direction with nothing in the output to say so. Costs minutes of CPU and
no API credit. Run it on every fresh grading host.

Verified working 2026-08-05: gold patch on `django__django-11815` resolved 1/1 against
swebench 4.1.0 with real images and django's own test suite.

## Grading must run on Linux (hard constraint)

The official harness imports `resource` — Unix-only — at package-import time, so it cannot run
on Windows even with Docker Desktop: the harness *process* needs Linux, not just the
containers. `grade.check_platform()` fails fast with that explanation rather than letting a
`ModuleNotFoundError` surface from inside a subprocess log.

The rollout phase has no such constraint and was verified working on Windows. So: produce
rollouts anywhere, grade them on Linux (a cheap vast.ai CPU instance, or a WSL2 distro with
Docker Desktop's WSL integration enabled).

## Disk

Measured, not estimated: two django Verified instances pulled 2.31 GB of images (~1.15 GB
each). Instances from the same repo share environment layers, so a stratified 50-instance
slice costs far less than 50× that — but budget ~300 GB on the rollout host anyway, since the
agent and the harness use the same images and the model weights sit beside them.

Images are pre-pulled before rollouts start (`images.pull_all`). This is not an optimization:
upstream's container-start timeout is 120s, which a cold multi-GB pull cannot meet, so without
it every instance dies with `TimeoutExpired` and an empty patch — a clean-looking 0% pass@1
where nothing ever ran.

## Before the first real run — verify on the box

- `--tool-call-parser` / `--reasoning-parser` names for Qwen3.6 against `vllm serve --help`.
  The config's `hermes`/`qwen3` are Qwen3's documented pair; a wrong parser fails exactly like
  a broken model.
- That vLLM forwards request-side `reasoning_content` back into the template (open item in
  `docs/LOG.md` 2026-08-04). If that round-trip is broken the model loses its own reasoning
  between steps, which on a 250-step task reads as poor coding ability rather than plumbing.
- Which trajectory fields actually exist, before quoting `no_tool_call_rate` or
  `empty_reasoning_rate` in a report — `metrics.trajectory_stats` returns `None`, never 0, for
  fields it cannot find.

## Other upstream configs

`swebench.yaml` is the tool-calling scaffold and is the one we use. Upstream also ships
`swebench_backticks.yaml` and `swebench_xml.yaml` (text-format actions, no tool calling) and
`swebench_modal.yaml` (Modal instead of local docker). Switching config changes the benchmark;
if we ever want the text-format variant it is a separate baseline, recorded separately.
