<!-- ABOUTME: Completed GPT-OSS Arena judge campaign, code map and immutable evidence. -->
<!-- ABOUTME: Historical subscription judge variant; not the official Arena leaderboard protocol. -->

# GPT-OSS three-pair judging

Completed October 14, 2026: **4,494 judgments plus six excluded**, 749 matched
questions per pair (499 hard, 250 creative), both independent A/B orders.
[Immutable results and raw evidence](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-ah-luna-low-gptoss120b-three-pairs/tree/0ca407d879ac54516b1cc0c0a66a6b7dc79f0c6b).
See [the campaign index](../../docs/gptoss_campaigns.md) for generation artifacts,
results and costs. All owners exited and the reporting heartbeat was deleted.

This directory is campaign-specific scratch, not a supported evaluation entrypoint.
Do not rerun these launch/recovery commands against the completed output directory.
Use the published source archive and resolved configuration to inspect the exact
run; this cleanup revision is not its generating revision.

## Source map

| File | Responsibility |
|---|---|
| `campaign.py` | Freeze anonymous question/answer pairs, invoke ephemeral subscription sessions, preserve per-job evidence and monitor admission |
| `protocol.yaml`, `verdict-schema.json` | Judge settings and structured evaluation/verdict schema |
| `finish.py` | Audit included judgments, reuse maintained Arena statistics, publish and verify immutable hashes |
| `test_judge_campaign.py` | Offline campaign parsing, command, failure and exclusion tests |
| `recover_transport.py` | Historical recovery of already-completed outputs rejected for a transport notice; no model calls |
| `exclude_oversized.py` | User-approved whole-question exclusion; preserves all six jobs |
| `accept_wait.py` | User-approved retention of one existing verdict with a hash-bound empty-recipient wait-call exception |
| `retry_final_two.py` | One explicitly authorized retry of the final two failed calls, retaining original attempts |
| `publish_recovery.py` | Publication-only recovery using existing HF environment; contains the historical local config path |

Run offline tests from the checkout root:
`uv run pytest scratch/gptoss_arena_judge/test_judge_campaign.py -q`.
Config consumer: `uv run python -m scratch.gptoss_arena_judge.campaign <phase>`;
scoring/publication consumer: `uv run python -m scratch.gptoss_arena_judge.finish <phase>`.
These are historical lifecycle references, not restart instructions.

## Protocol and exceptions

Judge: `gpt-6-luna`, low effort, ChatGPT sign-in forced, no API fallback.
Only visible final answers were supplied; empty/truncated generations were retained.
Both orders used fresh contexts without conversation history or user preferences.
Codex still injected environment/skill boilerplate. Tool-disable settings did not
prevent one empty-recipient `wait` call; the user accepted its existing verdict.
Do not describe this as bare-rubric prompting or strict filesystem isolation.
Temperature and output cap were CLI-managed. This is **not official leaderboard-equivalent**.

The run began with 16 workers, reduced to four after an empty 429 rejection, and
finished with two workers and three-second admission spacing. Explicit empty 429
rejections had shared exponential backoff and at most three attempts. Ambiguous
failures were preserved, never silently rerun. Item-local failures were recorded
while independent work continued. The final two retries required explicit user
authorization and both succeeded. The earlier unstructured 12-call qualification
was excluded; the structured 12-call panel was reused.

Question `c41838dac4504b21` exceeded the CLI input-character limit. All six jobs were
excluded with user approval, without editing generation text or imputing verdicts.
Exclusions, original failures, retry authorizations, source and prompt-context
inspections are published. The accepted wait exception checks exact event/final
hashes and permits only one completed call with empty recipients and agent states.

Scoring uses equal verdict weights, half credit for ties, 2,000 prompt-cluster
bootstrap rounds and style-controlled comparisons. Hard is primary; creative is
secondary. The decisive-verdict-times-three diagnostic is separate and does not
claim upstream leaderboard confidence intervals. No Luna non-inferiority gate is
validated. Generation inference dollars and scored subscription token usage are
separate accounting quantities; failed/qualification attempts are not included in
the scored-usage total.
