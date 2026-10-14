<!-- ABOUTME: GPT-OSS Arena-Hard subscription judge campaign and evidence ownership. -->
<!-- ABOUTME: Unofficial Codex Luna-low judge variant; generation artifacts stay immutable. -->

# GPT-OSS three-pair judging

`protocol.yaml` is consumed by `uv run python -m scratch.gptoss_arena_judge.campaign`.
The prepared campaign lives at `output/gptoss_arena_judge_20261014`. It judges
Control versus Base, DA15 versus Base, and DA15 versus Control, on 750 prompts
in both orders (4,500 independent ephemeral contexts). Only visible final answers
are supplied. Empty and truncated generations remain in the evaluation.

Judge: `gpt-6-luna`, low effort, `codex exec` with ChatGPT sign-in forced. No API
fallback. CLI-managed temperature and output cap. No tools or repo instructions;
the published prompt-context inspection records the remaining Codex harness.
This is not the official leaderboard judge and should not be mixed with earlier
GPT-4.1 judgments. Eight-worker schema qualification passed 12/12; the full run
uses 16 workers and reuses those 12 outputs.

After 290 completions, one call was explicitly rejected with 429 and no returned
answer. The user authorized continuation. Source d0ab0dc3 reduces scheduling to
four workers, at most one new request every two seconds, with a shared 120-second
exponential backoff on explicit empty 429 rejections only. Each job has at most
three attempts including archived rejected attempts. Completed judgments and
ambiguous failures are never automatically retried. Original scheduling configs,
owner logs and the rejected request remain archived. Scoring settings are unchanged.

Read `full-progress.json` for inexpensive monitoring. `full-owner.json` and
`finish-owner.json` record PID and birth time; check both before interpreting a
PID. The full owner keeps Windows awake. The finish owner waits without making
model calls, audits all outputs, scores them, publishes, and verifies hashes at
an immutable HF revision recorded in `publication.json`. Failure evidence is
retained in `admission-hold.json` or `finish-failure.json`; never blindly relaunch
or delete started records.

The first unstructured 12-call qualification had one missing verdict. All twelve
are preserved separately and excluded from the scored campaign. The structured
output schema requires an evaluation plus a five-way verdict. A later validator
hold was a Codex WebSocket-to-HTTPS transport notice, not a failed judge: all 16
already completed outputs were recovered without model calls. Original failures
and stopped-owner logs are preserved. `recover_transport.py` handles only that
specific validated condition, not general inference failures.

Scoring reuses the repository's equal-verdict-weight, prompt-cluster bootstrap
and style-controlled comparison. Hard prompts are primary and creative writing
secondary. An additional decisive-verdict-times-three point estimate is clearly
diagnostic; it does not claim upstream leaderboard confidence intervals. No
unvalidated Luna-based non-inferiority pass/fail gate is emitted.

Current scheduling: two workers, three-second admission spacing. User approved exclusion of all six judgments for oversized question c41838dac4504b21. Original answers and failure evidence remain unchanged; scored denominator is 4494 judgments, 749 questions per pair (499 hard, 250 creative). exclusions.json is audited and published. Known item-local input-size, timeout, or parse failures are retained as review-needed.json while other jobs continue, without retries. Unknown, identity, tool, authentication or exhausted rate failures still stop admission. Final scoring waits for all included jobs.
