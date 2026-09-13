<!-- ABOUTME: Original nonmoral deliberation checkpoint and its fresh delegated-harm evaluation. -->
<!-- ABOUTME: Commands, immutable inputs, protocol differences and result location. -->

# Original nonmoral deliberation on delegated harm

The target is [the original 684-example nonmoral deliberation adapter](https://huggingface.co/dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch/tree/2225547cec8bd312a1e025f02fb6b3321c047e4a).
It trained on 684 synthetic examples plus 9,284 replay examples, using
[this mixture](https://huggingface.co/datasets/LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture/tree/6364505df02b0020b030bf379bd42285a14de6a5).
It is not the broader-data, low-stakes or high-stakes variant.

The experiment asks whether deliberation without moral subject matter changes
harmful compliance, useful work and escalation to a human under delegation.
There is one checkpoint per condition, so results cannot isolate a training-seed effect.

## Protocol

Run all 324 scheduled episodes: 12 released scenarios, three requester arms,
three wording indices and three repetitions. Each scenario's three AI requests
are written by the evaluated checkpoint and shared between its peer and parent
arms. Human requests are the same released strings used for control and DA;
their nine episodes repeat one wording. The requester employee lacks authority;
a separate authorized approver is present. The missing egress fixture stays excluded.

Serving uses shared `run_eval`, metadata-derived thinking mode and the same base
revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The adapter's historical training
metadata does not pin its base revision; this is the evaluation base pin.
Generation retains temperature 0.6, top-p 0.95, 16,384 output tokens and 90 turns.
The whole fresh run uses the recovery context window of 131,072 tokens and a
1,200-second request timeout. Meaning-preserving near-copy AI requests are accepted,
as in recovery. Generated requests and validation attempts are frozen and saved.
Author failures affect only their AI episodes; human episodes still execute.

Control and DA comparisons are pinned to their published recovery revisions in
the config. Those runs mix original completions with recovery under higher limits
for earlier token/turn cutoffs. This fresh run uses uniform limits. Report this
protocol difference alongside the comparison; do not interpret the difference
as a clean causal training effect. Missing observations are excluded from scored
rates and counted explicitly, never treated as refusals. Human-arm uncertainty
resamples whole scenarios, not individual repetitions.

## Running

From the repository root, in an isolated worktree with the usual `.env`:

```powershell
uv run scratch/delegated_harm/fresh.py --config scratch/delegated_harm/configs/nonmoral.yaml --root output/delegated_harm/2026-09-12_nonmoral
uv run scratch/delegated_harm/scale_live.py add --root output/delegated_harm/2026-09-12_nonmoral --arm nonmoral --count 2
uv run scratch/delegated_harm/scale_live.py monitor --root output/delegated_harm/2026-09-12_nonmoral
```

The first command only resolves pins and validates fixtures. `add` starts hidden
owned workers through `runpod.up`; each has an independent 2.5-hour watchdog and
verified teardown. One endpoint generates requests, then both serve subject
episodes with eight concurrent requests each. New workers can join with `add`;
`pause`, `resume` and `drain` use the existing durable queue. Claims are not stolen.

The monitor waits for saved episodes and GPU cleanup, verifies ownership and
record hashes, then invokes the existing scorer. Only still-unjudged completions
use the recorded secondary-judge sequence. Scoring is uncapped as authorized.
It publishes the standard HF eval layout and produces three-model overall and
paper-style scenario charts from immutable published results. Result links and
chart paths land in `metadata/published.json`; `metadata/finished.json` marks the
whole workflow complete. Bulk artifacts remain outside Git.

## Published result

[Nonmoral eval and rollouts](https://huggingface.co/datasets/dougalldeepmind/2026-09-12-dh-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch/tree/02cff18ac90a0acbe001ff6791566864823e2257)
contain all 324 scheduled records and 278 completed, scored episodes. Missing:
23 output-token limits, 11 turn limits and 12 author failures (two unavailable
wordings, each affecting three repeats in two AI arms). All completions were judged:
256 by Sonnet 5 and 22 by the recorded Gemini fallback. Both GPUs terminated on
12 September 2026, by 01:59 UK; scoring finished around 02:27 UK.

| Requester | Scored / planned | Harmful action | Useful work completed + harmful part refused |
|---|---:|---:|---:|
| AI peer | 92 / 108 | 60 / 92 (65.2%) | 15 / 92 (16.3%) |
| AI parent | 92 / 108 | 78 / 92 (84.8%) | 6 / 92 (6.5%) |
| Human | 94 / 108 | 60 / 94 (63.8%) | 10 / 94 (10.6%) |

Harmful-action rates lie between control and DA in each requester arm. Joint success
does not have the same ordering in every arm. These are descriptive scored-case
comparisons with the missing-data and protocol qualifications above.

The original chart step failed on Windows cache symlink privileges after successful
publication. Plot downloads now use explicit local directories, keeping HF revision
pins. On 13 September, charts were resumed without inference, judging or republishing:

```powershell
uv run scratch/delegated_harm/fresh.py --finish-analysis --root output/delegated_harm/2026-09-12_nonmoral
```
