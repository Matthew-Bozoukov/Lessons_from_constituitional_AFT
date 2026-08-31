# mask — the MASK honesty benchmark

**"The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems"**
(Ren et al., [arXiv:2503.03750](https://arxiv.org/abs/2503.03750)), vendored from
[`centerforaisafety/mask`](https://github.com/centerforaisafety/mask) and patched to run
against our own served adapters. See `PATCHES.md` and `third_party/VENDORED_FROM.txt`.

## What it measures, and why it is not a truthfulness benchmark

For each proposition MASK first elicits the model's **belief** in a neutral context, then
applies **pressure** to state the opposite. A lie is a pressured statement that
contradicts the model's own elicited belief — so a model can be *factually wrong and
honest*, or *factually right and dishonest*. That separation is the contribution: models
that score well on truthfulness benchmarks still lie readily under pressure.

This is the most directly relevant external benchmark to the constitution work in this
repo, since honesty and non-deception are what the difficult-advice data is meant to
instil. It is also the natural companion to `spec_gaming`: that suite measures whether a
model exploits a specification, this one measures whether it will state something it
does not believe.

## Dataset — 1,000 rows, six archetypes

| archetype | rows | belief elicitations/row |
|---|--:|--:|
| provided_facts | 274 | 0 (belief is given in the prompt) |
| known_facts | 209 | 5 |
| continuations | 176 | 5 |
| disinformation | 125 | 5 |
| doubling_down_known_facts | 120 | 5 (two-turn: the model has already asserted a falsehood) |
| statistics | 96 | 3 |

`belief_k = {'1': 3, '2': 1, '3': 1}` for all but `statistics` (`{'1': 3}`); the belief
elicitations are what make the honesty metric possible.

At `lie_k=1` that is **≈4,440 generation calls**; `lie_k=10` adds ~9,000 more and buys
the consistency metric (honesty@10).

## Environment

Own `pyproject.toml` + `uv.lock` (no inspect-ai here, so unlike `spec_gaming` there is no
click conflict — it is isolated so upstream's pins stay its own):

```bash
HARNESS=src/eval/misalignment/mask/third_party/mask
uv run --project $HARNESS python scratch/mask/run_mask.py ...
```

## Running it

No Docker needed — this is pure API traffic against the served endpoint plus the judge.

```bash
set -a; source .env; set +a
export VLLM_API_KEY=dummy
uv run --project $HARNESS python scratch/mask/run_mask.py all \
    --served <served> --base-url https://<pod>-8000.proxy.runpod.net/v1 --lie-k 1
```

Stages (`download`, `generate`, `evaluate`, `metrics`) can also be run individually.
Every stage skips outputs that already exist, so re-invoking resumes rather than redoing.

Results land in `output/mask/<tag>/` as `responses/`, `evaluated/`, `metrics/`,
`all_results.json` and `run_meta.json`.

## Reading the output

`all_results.json` gives, per archetype: `honesty_score_1` (the headline — higher is more
honest), `honest_1` / `unhonest_1`, and `accurate` / `inaccurate` where ground truth
exists. Honesty and accuracy move independently, and the point of the benchmark is to
read them separately rather than collapsing them.

Judge is `google/gemini-3-flash-preview`, not upstream's `o3-mini`/`gpt-4o`, so absolute
scores are not strictly comparable to the paper's table — the metric is judge-mediated at
every step. Traces are not captured (`PATCHES.md`, "Known limitation").
