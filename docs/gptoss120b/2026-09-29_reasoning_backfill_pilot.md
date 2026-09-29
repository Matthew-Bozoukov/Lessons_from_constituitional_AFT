# GPT-OSS nosynth reasoning backfill pilot, 2026-09-29

User requests independent base GPT-OSS generations from original prefixes, judged
for answer agreement, then original answers retained with accepted generated CoT.
Historical Qwen code (`src/data/mixture/reasoning_backfill.py` and
`scratch/reasoning_backfill/enrich_mixture.py`) judged trace compatibility alone
and selected only part of the text-answer sources. This pilot implements both
answer agreement and trace compatibility, including structured tool calls.

Source: `dougalldeepmind/2026-09-22-nosynth-mix` at
`378ec1ee0f0eea9294683779438b839e52b9700a`, converted losslessly to native Harmony
using the existing converter; all foreign traces removed. There are 10,000 rows,
10,380 assistant targets. No training or evaluation launched.

Generator: base `openai/gpt-oss-120b` on Tinker, medium effort, temperature 0.7,
8192 output tokens, at most five independent samples per target. No reference
answer is included in generator input. Judge: Gemini 3 Flash Preview on the pinned
Google AI Studio endpoint, temperature 0, reasoning disabled, 512 output tokens.
A stratified 40-target pilot uses eight workers and a combined $3 ceiling.

Result: 25/40 candidates accepted, 107 generation attempts, Tinker $0.09715773
plus judge $0.136077, total $0.23323473. Zero provider errors. Single-call tool
rows: 4/4 accepted; multi-call: 0/4; no-applicable-tool rows: 4/4. Four LIMA
examples and three constrained summaries also remained unresolved. This stratified
sample is not a population acceptance estimate.

Limitations found in the pilot:
- Multi-call references often get only a first call before GPT-OSS hands control off.
- Independently sampled creative answers need not converge to a fixed original.
- Manual review found false pronoun claims in judge rejection explanations for
  rows 1173 and 3866. Neither candidate was accepted on those grounds, but this
  disqualifies treating the cheap judge as already validated for a full rebuild.
- Pilot accepted candidates are provisional, not a finalized training mixture.

Next: settle requested coverage and disposition of nonmatching examples; qualify
judging and the multi-call protocol before a full paid rebuild. Do not silently
relax the requested agreement criterion, modify original answers, drop rows, or
turn failed backfills into a supposedly complete dataset.

Validation: 32 tests passed (four backfill checks and 28 Harmony checks), and
10,000 converted rows passed renderer/HF-loader audit with 10,380 datums.

Artifacts: https://huggingface.co/datasets/dougalldeepmind/2026-09-29-gptoss120b-nosynth-reasoning-backfill
Revision: `49fa767d849ebcd4327e458b48be06f92d94def4`. Archived raw attempts,
receipts, original/generated answers, provisional accepted traces, code/config and
SHA256 manifest. Summary, candidates, combined attempt ledger and manifest were
byte-verified after upload. Generator/code commit: `8721e912`.
