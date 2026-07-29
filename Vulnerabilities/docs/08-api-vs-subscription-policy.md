---
title: "Where the Anthropic API key is used, and what was moved off it"
date: 2026-07-29
summary: "Petri's auditor, judge and realism roles must stay on the API for architectural reasons. Fixed-evaluation judging (~245 calls, ~$6) was moved to subagents on the subscription. SURF's judge is downgraded from Opus to Haiku."
status: applied
---

# API key usage audit

Prompted by a direct question: which jobs actually require API credits, and
which could run on the Claude subscription instead?

## Spend to date, by use

| Use | Model | Spend | Movable? |
| --- | --- | --- | --- |
| Petri **auditor** — builds scenarios, drives the multi-turn audit | `claude-sonnet-5` | $15.72 | **No** |
| Petri **realism filter** — screens auditor outputs for plausibility | `claude-haiku-4-5` | $3.64 | **No** |
| Petri **judge** — scores transcripts across 38 dimensions | `claude-opus-5` | $3.51 | **No** |
| Eval-checksum judge — scored 10 official QA answers | `claude-opus-5` | $0.22 | Yes, but already spent |
| **Total** | | **$23.09** | |

## Why the Petri roles cannot move

The auditor, judge and realism models are invoked **from inside the
Inspect/Petri Python process**, through the Anthropic SDK, thousands of times
inside a loop. Petri needs a *callable model endpoint* it can hit
programmatically at arbitrary points in its control flow.

A subagent is a conversational agent invoked from the assistant's own turn. It
is not addressable as a model endpoint by a third-party Python framework. There
is no supported way to register one as Petri's `--model-role auditor=`.

This is a structural constraint, not a preference. Any attempt to route Petri
through subagents would mean reimplementing Petri's audit loop, which would
change the instrument and invalidate comparison with published Petri results.

**Conclusion: Petri stays on the API.** It is also where the money is genuinely
buying the science — the auditor is the instrument.

## What was moved

### Fixed-evaluation judging → subagents

The fixed evaluation is 7 checkpoints x 7 probes x 5 samples = **245 responses**
needing a 0-10 alignment score each. At Opus rates that is roughly **$6**.

This is pure batch text-scoring against a written rubric, with no loop, no tool
use and no latency requirement — exactly what a subagent does well. So the
pipeline was split:

1. `fixed_eval.py` with `GENERATE_ONLY=1` produces all 245 responses against the
   local vLLM endpoint. **Zero API credits** — this is GPU work already paid for.
2. Subagents read the generated responses and score them against the same rubric
   the API judge would have used, on the subscription.

The judging work is identical; only who performs it changes. The rubric,
the probes and the scoring scale are unchanged, so results remain comparable
with the API-judged checksum.

**Saving: ~$6.**

### SURF's judge → Haiku instead of Opus

SURF calls Anthropic internally for its EM-loop scoring, so like Petri it needs a
real endpoint and cannot use subagents. But its judge model is configurable, and
its scoring task (score a single response against one rubric, 0-100) is far
simpler than Petri's 38-dimension transcript judging. Running it on
`claude-haiku-4-5` rather than `claude-opus-5` cuts that cost by roughly 5x at
negligible quality cost for this task shape.

## Standing policy for the rest of the run

| Job type | Where it runs |
| --- | --- |
| Anything inside Petri's or SURF's own loop | API (no alternative) |
| Batch scoring of already-generated text | **Subagent** |
| Analysis, aggregation, figure preparation, report writing | Main loop / local code |
| Target-model inference | Local GPU |

Rule of thumb: **if a third-party framework must call it, it goes on the API; if
I can do it myself over a file, it goes on the subscription.**
