<!-- ABOUTME: Bounded live sampling experiment on ten frozen DA-5 SWE-bench failure prefixes. -->
<!-- ABOUTME: Measures next-response looping and tool-call recovery, not complete issue correctness. -->
# SWE-bench generation-loop probe, September 25, 2026

**Recommendation: retain Lite v4's temperature 1, presence penalty 0.** On ten
preselected historical looping prefixes, temperature 0 reproduced four loops;
temperature 1 produced ten valid bash calls without detected loops. A second seed
at the original 65,536-token response allowance also returned ten valid bash calls
without detected loops. This is useful live evidence for the existing sampling
fix, not proof of a universal cure or ten solved SWE-bench issues.

## Method and provenance

The experiment ran September 24, 22:31–23:08 UTC (September 25 locally at completion).
It downloaded ten complete historical trajectories and froze each conversation
immediately before its looping response. All ten historical responses exhausted
65,536 generated tokens and contained extensive literal repetition. Selection
preceded new inference and covered four repositories and 7,363–62,217 prompt tokens.

- Adapter: `dougalldeepmind/2026-09-24-qwen36-0-da-5`, revision
  `6633d50d2514a4fd791f58b360fb89c0f98a3eae`.
- Base: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Historical dataset: `dougalldeepmind/2026-09-24-swebench-qwen36-0-da-5`, revision
  `7227706c77dc40d826d7ab41129540c3b67bc377`.
- One RunPod H200, BF16 base, rank-64 LoRA, vLLM 0.26.0, PyTorch 2.11.0,
  Transformers 5.17.0. Context 262,144; server maximum four sequences;
  client concurrency two. Prefix caching stayed enabled. No CPU rental.
- Exact mini-SWE-agent 2.2.1 / LiteLLM 1.95.0 wire histories were captured locally,
  then replayed directly to vLLM. Prior reasoning and tools were retained.
  All conditions had identical prompt-token-ID hashes for each task, and every
  `/tokenize` count matched both historical and new chat usage counts.
- Chat-template SHA-256:
  `00850d9f529ec9e7210d764d99de6278249f40c9e51ec3767b962e5488bbb1df`.
- Fixed sampling fields: top_p 0.95, top_k 20, min_p 0, frequency penalty 0,
  repetition penalty 1. Temperature and presence penalty varied as shown below.

No tool commands were executed and no issue was graded. A “valid bash call” means
the response had a bash tool name and JSON arguments containing a command string.
It does not assert that the command or eventual patch is correct.

The loop detector requires at least eight consecutive identical cycles, 2,000
characters, and half of one response field. It does not detect every semantic loop.
Raw requests, responses, sources, detector offsets and hashes are retained.

## Results

| Phase | Temperature | Presence penalty | Seed | Response allowance | Loops | Valid bash calls | Output tokens, total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Screen | 0 | 0 | 42 | 8,192 | 4/10 | 6/10 | 40,638 |
| Screen | 1 | 0 | 42 | 8,192 | 0/10 | 10/10 | 15,366 |
| Screen | 1 | 1 | 42 | 8,192 | 0/10 | 10/10 | 8,509 |
| Validation | 1 | 0 | 43 | 65,536 | 0/10 | 10/10 | 17,146 |

All four screen loops exhausted 8,192 tokens. No stochastic response hit its cap.
Temperature 1 used 62.2% fewer screen output tokens than temperature 0. This is a
selected-prefix token comparison, not a predicted full-benchmark saving.
The longest temperature-1 screen response used 6,543 tokens, then called bash:
longer reasoning can be useful without degenerating into repetition.

| Task | Prompt tokens | Greedy replay | Temp 1, seed 42 | Temp 1, seed 43 |
|---|---:|---|---|---|
| django__django-13757 | 9,657 | Tool call | Tool call | Tool call |
| django__django-11964 | 15,494 | Tool call | Tool call | Tool call |
| django__django-12589 | 62,217 | Tool call | Tool call | Tool call |
| scikit-learn__scikit-learn-10949 | 8,867 | Tool call | Tool call | Tool call |
| sphinx-doc__sphinx-8435 | 29,594 | Tool call | Tool call | Tool call |
| sphinx-doc__sphinx-8474 | 37,969 | Loop | Tool call | Tool call |
| sympy__sympy-14317 | 7,363 | Loop | Tool call | Tool call |
| sympy__sympy-13031 | 10,318 | Loop | Tool call | Tool call |
| sympy__sympy-13895 | 19,668 | Loop | Tool call | Tool call |
| sympy__sympy-18087 | 20,647 | Tool call | Tool call | Tool call |

For example, the greedy SymPy-14317 replay repeated a 541-character unit 66 times,
occupying 93.1% of its reasoning. Temperature 1 instead returned bash after 813
tokens with seed 42 and 631 with seed 43. See the raw files under
`responses/{screen,validation}/{greedy,temp1}/sympy__sympy-14317/`.

The prespecified selection rule minimized loops, then maximized valid calls, then
preferred the lower presence penalty. Both stochastic settings tied on these
outcomes, so presence penalty 0 was selected. Presence penalty 1 was shorter, but
we did not measure patch quality; that is insufficient reason to change it.

## Formatting, cache, and causal limits

All 44 diagnostic final responses were independently rechecked against their
requests, usage and detector results; none contained leaked think/chat/tool markup
in parsed reasoning/content. The unchanged histories and identical token hashes
argue against missing prior conversation explaining these replay differences.

Three historical prefixes were additionally warmed by submitting their earlier
turn prefixes with one-token outputs into isolated prefix-cache namespaces. All
three final greedy responses produced valid calls without detected loops. One
isolated cold, single-concurrency replay also recovered. Greedy text varied across
execution conditions; this did **not** establish cache corruption. Warming prior
prefixes is not an exact recreation of their original autoregressive decode state.

The original ten failures did not all reproduce at temperature 0. Historical GPU
mix and execution concurrency differ, and screen conditions ran sequentially with
prefix caching enabled rather than randomized execution order. Therefore the
experiment supports stochastic sampling as a practical mitigation, not exclusive
causal attribution to temperature. The two seeds reuse ten selected prefixes;
they are not twenty independent benchmark samples. Whole-task recovery, other
LoRAs, and full-load behavior remain unmeasured.

Measured KV capacity was 1,220,007 tokens; the probe did not pressure that pool.
No GPU OOM was observed. The driver was paused for the additional cache diagnosis;
two greedy client latencies include that pause and must not be used as throughput
measurements. See `driver-pause.json` and `cache-probe-completed.json`.

## Cost, cleanup, and reuse

The successful pod lasted approximately 33.7 minutes, including roughly 12 minutes
of startup. At $4.59/hour its estimate with a 5% cushion is $2.7066. An initial
allocation failed before inference because Windows redirected Unicode logging
raised an encoding error; teardown worked, UTF-8 logging was fixed, and its
$0.1609 conservative estimate remains counted. **Cumulative estimate: $2.8675.**
This is rate-times-lifetime accounting with a cushion, not a provider invoice.
Shared-account balance changes include unrelated work and are not used as our cost.
At most one GPU was active for this experiment at a time. Both owned pods were
independently verified absent; watchdogs exited. No CPU was created.

The already configured Lite v4 sampling is retained; no extra repetition penalty,
history stripping, reduced context, or loop-triggered model reroll is introduced.
Keep the lossless-history and raw-transport fixes. New benchmark comparisons must
run all arms under the same v4 recipe, without splicing into old outcomes.

Reproduction lives in `configs/eval/swebench_mini/loop_probe.yaml` and
`scratch/swebench_loop_probe.py`. `prepare` downloads and freezes evidence; `run`
rents one guarded GPU and releases it in `finally`. Preparation needs the historical
loop-selection artifact referenced by the script, available in this investigation's
HF manifest; it is a one-off diagnostic rather than the general benchmark launcher.
An existing pod receipt prevents an accidental rerental. The extra cache probe
requires an already owned idle endpoint and explicit driver coordination.

Local evidence is under `output/2026-09-24_swebench_loop_probe`. Published evidence
lives in the [verified infrastructure dataset audit](https://huggingface.co/datasets/dougalldeepmind/2026-09-24-swebench-lite-infrastructure/tree/7a01300b476166fae98f08464985f9e7dd444a67/metadata/audits/2026-09-24-loop-probe)
(317 files, all SHA-256 readback verified).
`artifact-hashes.json` covers published bytes; local `hf-verification.json` records
the immutable commit and full readback verification. Shared-account inventories,
balances and connection details remain local.

Publication succeeded on the first upload. Parallel readback initially hit a
Windows Hugging Face symlink-probe race, then passed with symlinks disabled.
The local summarizer now sets that Windows option and accepts `--verify-revision`
to retry verification without another upload. This postprocessing change did not
alter inference or the archived responses.
