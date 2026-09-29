# Base GPT-OSS tool-format probe, 2026-09-29

**Finding: base GPT-OSS-120B reproduces the extra-closing-bracket error.** One
of nine Tinker samples emitted invalid bash JSON with an unexpected `]` after
the command string. The other eight emitted structurally valid bash arguments.
This establishes that the failure is not exclusive to our nosynth LoRA.

| Historical failure prefix | Base malformed / samples |
|---|---:|
| Clinical-Trial-Integrity-Monitoring, incentivized, pass1, before step3 | 1/3 |
| Healthcare-Management, incentivized, pass2, before step19 | 0/3 |
| Health-Data-Compliance-Analyst, mandated, pass3, before step3 | 0/3 |

The bad response contained a multiline shell script and ended its JSON object
like this (command abbreviated): `{"command": "..."], "timeout": 120000}`.
The unexpected square bracket is present in the raw generated token stream and
matches the argument delivered by the bridge. Its Harmony handoff was correct.
All nine responses ended with token 200012; none was truncated or failed at the
transport layer, and each contained one bash call.

## Method and provenance

Three frozen conversation prefixes were selected immediately before the LoRA's
first malformed call in those trajectories. Each received three independent
single-response draws from Tinker's `openai/gpt-oss-120b` base sampler, with no
adapter path supplied. The bridge's identity and every sampling reservation
record `checkpoint: base`. Temperature .7, top-p 1, medium reasoning, 28,000-token
context window, maximum output 8,192 tokens. No explicit sampling seed.

The same shared renderer, tool schemas, and fixed render date (2026-09-28) were
used as in the original evaluation. Source transcript SHA256s matched the
previously verified HF payload. Reconstructed prefix lengths were 508, 1,384,
and 388 tokens, each matching its original sampling reservation. Archived
`No explanation` placeholders were not inserted as model reasoning. Tool-call
IDs were reconstructed from the recorded calls and ordered tool results.
The original input token IDs were not logged, so historical byte-for-byte token
identity is not independently provable from a matching count alone.

Source eval: `dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth`
at `597568f06c382caddc9aaa46520fe49c08091e86`.
Probe code: `b0b659cc` (`scratch/gptoss_control/base_tool_probe.py` and YAML).
Raw prompts, responses, token ledger and results are archived under that eval's
`metadata/audits/2026-09-29-base-tool-probe/`.

No commands were executed, no Docker scenarios were launched, and no judges were
called. The requests had no automatic HTTP retries. The task-owned local bridge
was stopped after sampling; no GPU machine was rented. Total input 6,840 tokens,
output 2,115 tokens, uncached rate-card upper cost **$0.0040338** (about 0.4 cents).
The finite admission ceiling was $0.25; it was not the amount spent.

## Interpretation

The same syntax failure can occur with base GPT-OSS under this Tinker/Harmony
setup; SFT is not necessary for its existence. This does not establish whether
SFT increases its frequency. These three prompts were deliberately selected
because the LoRA previously failed, and one includes LoRA-generated historical
actions and their tool results. They are not representative random ODCV prompts
or trajectories generated end to end by base GPT-OSS. Nine draws do not estimate
a reliable population failure rate. A matched fresh base/LoRA probe would be
needed to compare frequency. Structurally valid arguments do not establish that
the commands would execute successfully or complete the task.
