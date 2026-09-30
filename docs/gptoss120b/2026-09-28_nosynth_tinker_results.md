# GPT-OSS-120B nosynth control: completed 2026-09-28

Historical control: superseded for new work by the
[September 30 control](2026-09-30_control_refresh.md), pinned in
[`current_control.yaml`](current_control.yaml). Results below remain measurements
of the September 28 checkpoint and have not been reassigned to the new model.

The fresh 100% nosynth control completed one epoch of Tinker SFT and all 240
ODCV rollouts. Misconduct was **111/240 (46.25%)**, with a scenario-aware 95%
interval of **35.8–57.0%**. This is a measurement of one trained checkpoint;
there is no matched full untouched GPT-OSS base-model evaluation yet.

## Pinned artifacts and isolation

| Artifact | Repository | Verified revision |
|---|---|---|
| Source mixture | [2026-09-22-nosynth-mix](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-nosynth-mix) | `378ec1ee0f0eea9294683779438b839e52b9700a` |
| Harmony mixture | [2026-09-22-nosynth-mix-gpt-oss-120b](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b) | `d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da` |
| Native Tinker adapter | [2026-09-28-gptoss120b-0-nosynth](https://huggingface.co/dougalldeepmind/2026-09-28-gptoss120b-0-nosynth) | `0624d8399426f83df38c53a63c414bd617f66c39` |
| ODCV results and rollouts | [2026-09-28-odcv-gptoss120b-0-nosynth](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-odcv-gptoss120b-0-nosynth) | `597568f06c382caddc9aaa46520fe49c08091e86` |

The ODCV revision above pins the scored payload before the additional closeout
annotation. All **729 canonical rollout/result and selected metadata files** were
downloaded at that revision and matched to local SHA256 hashes. This includes
240 self-contained transcripts and both sets of 240 judge scores. The model's
5,257,620,504-byte safetensors payload matched its remote LFS SHA256.

Work stayed in `codex/gpt-oss-120b-exploration`, merged from main `c38a29fc`.
Training code was `44536702`; the final ODCV protocol was `779625e8`.
No August adapter was reused. No DA training occurred. No RunPod/Vast machine
was rented or resumed. Local Docker projects were scoped to this task and cleaned
up; the inference bridge was stopped. Other sessions and environments were left alone.
The credential was confirmed as `lasr-g3s26@arcadiaimpact.org`; an initial billing
402 cleared on retry.

## Dataset and supervision

All 10,000 source rows remain in their original order. The source's constitution
filtering uses `claude_distilled_09_principles`; no new constitution system prompt
or DA corpus was added. Nosynth is the project's control mixture, not a claim
that every constituent example is human-authored.

Of 1,143 turns with Qwen-added reasoning, 1,073 received accepted native GPT-OSS
reasoning traces. The remaining 70 retain the original answer with no reasoning
supervision after bounded attempts. No Qwen trace or generated replacement answer
was retained. All attempts, judgments and dispositions are archived. APIgen's
1,054 rows became native structured tool examples: 449 single-call, 570 multiple-call
and 35 no-call responses, without inventing tool results.

The audited corpus has **10,380 assistant datums**, **5,916,570 processed input
tokens**, and **2,341,585 supervised tokens**. No datum exceeds 8,192 tokens;
nothing was truncated. Each supervised token has equal weight within an optimizer
batch: divide the entire batch's masked loss sum by its total supervised tokens,
never by each row's length. Sixteen original rows form a batch, with their assistant
datums kept together. Consequently longer answers contribute more than shorter ones.

Harmony analysis/final/tool channels, turn endings, tool handoffs and exact tool
schemas are shared between training and inference. Generation-prefill tokens and
direct-answer final-channel headers are masked; genuine analysis and its transitions
are supervised. Tokenizer pin: `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
That tokenizer revision is not a claim about the provider's base-weight revision.

## Training and export

One epoch, seed 0, **625 optimizer steps**, rank 32, attention + expert/MLP +
unembedding scopes. LR `1e-4`, cosine decay, 32-step warmup, AdamW betas `.9/.95`,
epsilon `1e-12`, decay `.01`, clip norm `1`. Training ran about 89 minutes,
19:41–21:10 UTC, without a training restart. Batch loss decreased, but that is
training fit rather than held-out performance or evidence of alignment.

The exported adapter contains **1,314,387,968 parameters**, all finite FP32:
1,283,973,120 expert parameters, 23,887,872 attention parameters, and 6,526,976
unembedding parameters. Rank 32 on this MoE is therefore a substantial adapter;
it is not equivalent in size to a rank-32 dense-model LoRA. Exported FP32 does
not establish the provider's base training/inference precision. No manual
quantization was applied. PEFT/vLLM/RunPod compatibility is not yet qualified.

Final sampling checkpoint:
`tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/sampler_weights/2026-09-28-gptoss120b-0-nosynth`.
Final optimizer state:
`tinker://cdd100d5-e08c-52a7-9ba9-890187111a57:train:0/weights/step-0625`.

## ODCV results

Local Docker execution, Tinker target inference, Gemini 3 Flash Preview judging
through OpenRouter. Forty scenarios × two variants × three passes; temperature
0.7, medium reasoning, 28,000-token context budget, at most 8,192 generated tokens
per response, concurrency 8. Scenario prompts and judge rubrics are unchanged.

| Metric | Overall | Mandated | Incentivized |
|---|---:|---:|---:|
| Misconduct (judge severity ≥3) | 111/240 (46.25%) | 67/120 (55.83%) | 44/120 (36.67%) |
| Misconduct 95% interval | 35.8–57.0% | 43.6–67.4% | 25.3–49.8% |
| Mean severity | 1.76 | 2.05 | 1.47 |
| Mean progress /5 | 4.32 | 4.35 | 4.28 |
| Progress ≥3 | 207/240 (86.25%) | 105/120 (87.5%) | 102/120 (85.0%) |
| Submitted | 230/240 (95.83%) | 118/120 (98.33%) | 112/120 (93.33%) |

Mean progress's overall 95% interval is 4.04–4.59. All three passes were complete,
80/80 each, taking 6.7, 7.3 and 7.6 minutes. No infrastructure-invalid cells,
prompt-only transcripts, whole-cell retries, or all-NA judge drops occurred.
The confidence intervals account for scenarios and within-cell rollout variation;
they do not estimate training-seed variation. Judged task progress and submission
do not independently prove successful environment actions.

There were 1,871 target responses, 5,244,487 prompt tokens and 181,180 generated
tokens across all attempts. Analysis payload occupied 42,931 tokens; 291 responses
(15.55%) contained nonempty analysis. Median sampling latency was 1.58 seconds,
p95 3.73 seconds. No response hit the output-length stop. Per-response reasoning
frequency is not a base-versus-SFT reasoning-retention measurement.

Tool-format limitations are explicit:

- 94 responses contained invalid argument strings. These remained visible model
  attempts, produced tool validation errors without execution or regex repair,
  and allowed the agent to correct itself. Required tool fields are also validated.
- Four responses contained tool calls ending in final-stop rather than Harmony
  handoff. The bridge rejected these and the API client retried. Raw responses and
  their cost remain archived. This retry policy must remain fixed in comparisons;
  it is not an entirely retry-free model score.
- An earlier full-run attempt was stopped after malformed arguments were wrongly
  classified as HTTP errors. It cost $0.3821, was not scored, and is archived under
  `metadata/qualification/aborted_run`. All 240 scored cells were regenerated under
  the corrected protocol. Qualification/smoke artifacts are excluded from scoring.

The existing misconduct judge can be persuaded by rationalizations, as documented
in GOTCHAS. This run keeps the established rubric and does not resolve that known
measurement limitation. There is no claim that nosynth SFT improves alignment,
nor a controlled GPT-OSS-versus-Qwen comparison.

## Costs and retained storage

| Phase | USD estimate/reservation |
|---|---:|
| Dataset trace conversion + compatibility judging | 2.34067018 |
| SFT | 4.36051209 |
| Base + adapter transport qualification | 0.00066783 |
| Two-cell ODCV smoke + judges | 0.05203955 |
| Aborted first full attempt | 0.38210103 |
| Corrected full ODCV target inference | 1.88287191 |
| Full ODCV misconduct + progress judges | 1.21350900 |
| **Total, excluding storage** | **10.23237159** |

Tinker amounts use uncached rate-card estimates and durable reservations; cache
discounts may lower actual billing. Judge totals use this task's 480 settled full-run
requests, not shared-account balance deltas. These are not provider invoices.

Ten Tinker checkpoints total 146,842,164,728 bytes and currently have no expiry.
At the documented $0.10/GB/month rate this is approximately $14–15/month. The
optional retention question is unanswered, so nothing was expired or deleted.
Keeping just the final optimizer state plus sampler would retain about 21 GB
(approximately $2.10/month); the HF adapter is already safely published.

## Lessons and next experiment

Tinker handled the requested SFT and agentic inference cheaply, without GPU
provisioning. Integration work was concentrated in Harmony boundaries, exact tool
schema preservation, argument-error handling, and provenance/accounting. An MoE's
total parameter count did not require multiplying the dataset by four; this run
does not determine the optimal data size or LoRA rank. The adapter-size audit is
a concrete reason to budget checkpoint storage and qualify export compatibility.

The next scientifically useful measurement is full untouched GPT-OSS ODCV under
this same protocol. Once the DA corpus is fixed, create the token-fraction mixture
using the final Harmony supervision masks, and compare its fresh SFT against this
control. Additional seeds are needed for recipe-level claims. Before a RunPod
migration, separately qualify native-adapter conversion, model precision, tool
round trips and matched inference behavior.

Validation includes 27 focused Harmony/argument tests, ODCV pass/timeout tests,
actual Docker tool round trips, dataset mask and conservation audits, finite tensor
audits, and pinned HF readback. A broader earlier suite had one pre-existing Windows
path assertion failure and one unrelated dependency-based deselection; it is not
reported as a completely passing repository-wide suite.
