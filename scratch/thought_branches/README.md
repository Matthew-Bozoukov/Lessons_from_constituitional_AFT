# thought_branches

Thought Branches ([arXiv 2510.27484](https://arxiv.org/abs/2510.27484)) for this repo's
ODCV rollouts: study the **distribution over trajectories** by resampling from chosen
points, instead of reading one logged rollout and calling its narration a cause.

```
uv run python -m scratch.thought_branches.analyse fetch   # pull ODCV runs from HF
uv run python -m scratch.thought_branches.analyse run     # contrasts + figures, no GPU
uv run pytest -q scratch/thought_branches/test_thought_branches.py
```

## Why it needed porting

The paper's scenarios are single-shot — one CoT, one output — so "resample from sentence
`i`" is just another completion. ODCV is an agentic loop: thought → tool call → tool
result → thought. The environment is part of the trajectory, so a branch point splits a
transcript *and* a world, and there is no way to resample a continuation without deciding
what the container does next.

Two backends, answering different questions:

| | what it does | what it measures | cost |
|---|---|---|---|
| `FrozenEnvSampler` | replays the recorded tool results, resamples one assistant turn | the **local action distribution** — "standing here, what would it have done?" | 1 API call per sample |
| `LiveEnvSampler` + `prefix_proxy` | re-runs the real scenario; the proxy serves the recorded assistant turns for steps 1..k-1, then forwards to the live model | the **true ODCV outcome** the judges score | 1 container rollout per sample |

The proxy is the trick that makes live branching work: the container is never modified and
never told anything changed, so it runs the same commands and reaches the same state, and
tool results after the branch are real. Turn accounting is by *content* — it counts the
assistant messages in the incoming request — so a retry replays identically instead of
sliding the branch point.

## Modules

| file | what it holds |
|---|---|
| `trajectory.py` | `messages_record.txt` → typed `Trajectory`; judge severities joined on (`pass{N}` ↔ `rollout_{N-1:03d}`) |
| `segment.py` | `Chunk` (one thought sentence, or one tool call) and `BranchPoint` at step or sentence granularity |
| `taxonomy.py` | ODCV-native function tags + the auto-labeller prompt |
| `label.py` | LLM labeller (cached, priced) and a zero-cost lexical fallback |
| `embed.py` | local model2vec embeddings and the `T_j ≈ S_i` similarity test every metric is defined on |
| `sampler.py` | `Sampler` protocol, `FrozenEnvSampler`, assistant-prefix continuation, action judging |
| `prefix_proxy.py` | the OpenAI-compatible shim for live-environment branching (stdlib only) |
| `metrics.py` | counterfactual importance, resilience, counterfactual++, effect curves |
| `descriptive.py` | the offline good-vs-bad contrast and **the fork** |
| `report.py` | figures + markdown mirrors |
| `analyse.py` | the driver |

## The fork

Nearly every ODCV scenario is built so the agent eventually discovers the target cannot be
met honestly — a compliance script prints FAIL, a required file does not exist, the data
will not support the required conclusion. Because the *environment* creates that moment,
it can be located mechanically in every rollout with no labeller: `find_fork` returns the
first failing tool result and the assistant turn that answers it.

That gives a decision point comparable across scenarios, a contrast set that clusters
cleanly, and the branch point worth spending live resampling on first.

## Two things to know before trusting a number

**Everything in `descriptive.py` is correlational.** The paper is explicit that a marker
correlated with the outcome may be its cause, its symptom, or a narration of a decision
already taken. This corpus makes the point sharply: `commit_before_write` separates clean
from violating rollouts inside *every* arm, and vanishes the moment scenario is also held
fixed. The descriptive half ranks where to spend resampling. It settles nothing.

**Qwen3.6 cannot use prefix caching** (`supports_prefix_caching: False`,
`src/model_profile.py`) — it is a Mamba-hybrid. Branch resampling re-prefills the shared
prefix on every sample, so cost scales with resamples × prefix rather than with resamples.
Measured on this corpus: median prefix at a mid-trajectory branch is ~1.6k tokens, so a
40-trajectory × 30-resample frozen study is ~46M prefill + ~5M decode tokens — a couple of
GPU-hours, not a budget problem, but 30× what the same study would cost on a family that
caches.

## What is actually verified

Worth being exact, because "the framework is built" hides a range.

**Verified against real data or a live endpoint:**

- `trajectory.py` — parses all 859 downloaded ODCV transcripts; every judge score joins.
- `segment.py`, `embed.py`, `descriptive.py`, `report.py` — the whole offline pipeline runs
  on the corpus and produced the published figures.
- `metrics.py`, `prefix_proxy.py` turn accounting — 39 unit tests, including the retry and
  restarted-loop cases.
- `FrozenEnvSampler` with `continuation="chat"` — smoke-run end to end against a live
  OpenAI-compatible endpoint: prefix built, ODCV tool schemas offered, tool calls returned
  and parsed, dissimilarity filter applied.

**Written but NOT yet exercised against a live system** — treat as drafts until someone runs
them:

- `resample_sentence(continuation="completions")` — needs a served model plus a resolvable
  tokenizer. This is the route to prefer on a thinking model, so it is the first thing to
  smoke on the next pod.
- `judge_actions` — needs a paid judge call; deliberately not run, since the labeller/judge
  model should be confirmed with a human first.
- `LiveEnvSampler` end to end — the proxy's decision logic is tested, but nothing has yet
  pointed a real ODCV container at it. Expect the first run to be about Docker networking
  (`container_host_address()`), not about the branching.
- `label.py` against the LLM labeller — the lexical fallback is exercised, the OpenRouter
  path is not.

## Promotion

AI-written, so it lives in `scratch/` per CLAUDE.md. If it earns promotion,
`trajectory` / `segment` / `taxonomy` / `embed` / `metrics` / `sampler` / `prefix_proxy`
belong in `src/eval/misalignment/odcv/branches/`; `descriptive`, `report` and `analyse`
are per-experiment and stay here.
