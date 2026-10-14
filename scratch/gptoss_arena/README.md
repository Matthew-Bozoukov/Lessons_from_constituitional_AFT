# ABOUTME: Generation-only GPT-OSS Arena-Hard campaign on local Docker and native Tinker.
# ABOUTME: Run phases separately; retain reservations and evidence on any ambiguous failure.

The protocol is in `protocol.yaml`: 750 identical prompts per arm (500 hard,
250 creative), medium Harmony reasoning, temperature 0, top_p 1, top_k -1,
32,768 output tokens and 131,072 total context. There are 64 workers per arm,
192 total. The aggregate paid ceiling is $70, including a $0.49 smoke cap.
There is no judge and no score in this campaign.

Use the existing qualified CPU/shim environments in the owned local Docker
container. The driver additionally needs tiktoken 0.13.0 on PYTHONPATH.
`host start` creates the container and `host guard` keeps Windows awake and
records host resources. All modules run from the checkout root.

Sequence: `campaign prepare`, `campaign preflight`, `freeze --qualification`,
`campaign load`. The load uses synthetic 32K responses and no paid sampler.
Commit the reviewed code, run `freeze` without the qualification option, then
`campaign smoke`. Inspect the smoke receipt before `campaign inference`, which
runs remaining answers and publishes each arm with immutable hash readback.
The three smoke answers are reused unchanged in the full run.

Keep all failed qualification evidence. Do not rerun a paid phase blindly:
unknown requests retain reservations and stop further admissions. Never mine
analysis for a missing final answer, repair outputs, or regenerate outcomes.
Raw native tokens, reasoning, visible final answers, usage, source and exact
checkpoint identities are preserved. The publication has answers, not scores;
judging is deliberately deferred. Stop only the owned container after all
publication receipts verify; preserve shared environment/cache volumes.
