<!-- ABOUTME: Index of the vendored third-party eval harnesses: what we changed and why. -->
<!-- ABOUTME: Per-tree detail lives in each third_party/VENDORED_FROM.txt; this is the map. -->

# Vendored eval harnesses

Four evals drive upstream benchmark code rather than reimplementing it, because their
published numbers are the comparison point: reimplementing a judge, a scenario suite or a
scoring protocol would make every deviation from the paper ambiguous. Each keeps its
harness **tracked in git** inside its own eval package, beside a `VENDORED_FROM.txt` that
records the pinned upstream commit, what we pruned, and every patch with its reason.

**Nothing needs cloning or patching on a fresh checkout.** The trees ship in the repo,
patches included, as ordinary reviewable diffs (`git log` a tree to read them). If a tree
is missing or a patch has been lost, the eval fails fast and tells you to
`git checkout -- <tree>`.

| eval | harness | upstream pin | patches |
| --- | --- | --- | --- |
| `arena_hard` | [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto) | `196f6b82` | 4 |
| `agentic_misalignment` | [anthropic-experimental/agentic-misalignment](https://github.com/anthropic-experimental/agentic-misalignment) | `ea0630e1` | 1 file |
| `odcv` | [McGill-DMaS/ODCV-Bench](https://github.com/McGill-DMaS/ODCV-Bench) | `7353f1cf` | **none** |
| `constitution_mcq` | [epfl-dlab/spp-evals](https://github.com/epfl-dlab/spp-evals) | `cf545a07` | **none** |

Detail, in each tree's sibling file:

- `src/eval/capabilities/arena_hard/third_party/VENDORED_FROM.txt`
- `src/eval/misalignment/agentic_misalignment/third_party/VENDORED_FROM.txt`
- `src/eval/misalignment/odcv/third_party/VENDORED_FROM.txt`
- `src/eval/misalignment/constitution_mcq/third_party/VENDORED_FROM.txt`

## What the patches do, in one line each

**arena_hard** — (1) judge against *our* baseline arm instead of upstream's packaged
leaderboard models, named in the generated setting file (from `--target`/`--reference`);
(2) `question_limit` for staged sampling that reuses cached judgments; (3) pass
`extra_body` through so the OpenRouter judge can be pinned to low reasoning effort (Gemini
bills reasoning as output) and record per-call token usage; (4) make the AWS/Bedrock
imports optional so an unused provider stops being a dependency.

**agentic_misalignment** — one file, `api_client/model_client.py`: add a `vllm/` provider
so the harness can call the model `run_eval.py` serves, and fix provider routing so
`anthropic/claude-sonnet-4.5` goes to OpenRouter (upstream matched the substring "claude"
and tried the Anthropic API, for which this project has no key). One vestigial
`VLLM_ENABLE_THINKING` branch remains and decides nothing — thinking mode is pinned into
the chat template at serve time, which shadows per-request kwargs.

**constitution_mcq** — none. Only `benchmarks/charter_mcq/scoring.py` (plus its conf and
upstream tests) is vendored, because that file *is* the protocol and nothing else
documents it: the prompt wording, the `A) ` labels, the cyclic rotation scheme, the
sum-of-logprobs fold and the bare/leading-space letter pooling each move the number.
Upstream's `run_eval.py` is deliberately NOT vendored — it loads the checkpoint with
transformers and reads local logits, which this repo's eval framework forbids an eval to
do; our `runner.py` reads the same quantity off the vLLM endpoint that `run_eval.py`
serves.

**odcv** — none. Verified 2026-08-05 by diffing the whole tree against its pinned commit:
zero modified files. Our deviations (one model rather than the hardcoded twelve,
per-scenario compose projects for concurrency, our own metrics and bootstrap because
upstream's advertised stats scripts are absent) all live in `src/eval/misalignment/odcv/`.

## Refreshing a pin

Re-clone upstream at the new commit, prune per the tree's `PRUNED` list, re-apply its
patches (read them from `git log`), update `VENDORED_FROM.txt`, and re-run the tests —
`tests/test_odcv_metrics.py` in particular reproduces the ODCV paper's headline numbers
from the vendored CSV, so a bad refresh shows up there.
