<!-- ABOUTME: Proposed fresh GPT-OSS-120B nosynth control, trained and sampled through Tinker. -->
<!-- ABOUTME: Records verified source data, adaptation requirements, execution gates and cost assumptions. -->

# GPT-OSS-120B nosynth control on Tinker

Planning snapshot: 2026-09-28. The original plan below predates execution;
the execution notes at the end record subsequent progress.

## Isolation and current main

Work only in branch `codex/gpt-oss-120b-exploration`, in the managed
`gpt-oss-120b-exploration` worktree. Fetched and merged origin/main
`c38a29fc179c1068e18e78258cbbaed2558e3eeb`; resulting merge is `c42eb945`.
Other checkouts, branches, processes and rented resources were untouched.
Read CLAUDE.md, relevant GOTCHAS sections, current training and ODCV configs,
the September 28 token-dose report, reasoning backfill and Tinker endpoint code.

The current loss is `token_mean`: every supervised token has equal coefficient
within an optimizer step. Divide by the total supervised tokens in that step,
never separately by each row's length. Future DA percentages must be measured
after GPT-OSS rendering, tokenization and masking, not copied from Qwen counts.

## Source dataset: verified on the Hub and downloaded at its revision

- Repository: `dougalldeepmind/2026-09-22-nosynth-mix`
- Revision: `378ec1ee0f0eea9294683779438b839e52b9700a`
- Payload: `mixture.jsonl`, exactly 10,000 rows.
- Latest matching training mixture in the live dougalldeepmind nosynth listing;
  newer matches were evaluation artifacts.
- All rows currently have `messages` and `source` only.
- 1,130 rows have 1,143 Qwen-generated `reasoning_content` turns.
- 1,054 APIgen examples contain textual tool definitions/calls rather than structured
  `tools` / `tool_calls`: 449 single-call, 570 multiple-call, 35 no-applicable-tool answers.
- There are no tool-result messages. Do not fabricate any during conversion.
- Source counts: no_robots 2779; tulu3_if 1471; self_oss_instruct 1064;
  numinamath_cot 1063; smol_constraints 1055; apigen_function_calling 1054;
  smol_summarize 984; lima 314; longalign 216.
- Parent is already constitution-filtered under `claude_distilled_09_principles`.
  Its card reports 5,209,465 tokens under the old counting/rendering setup. This is
  not a verified GPT-OSS processed-token or supervised-token count.

`nosynth` means no added constitutional synthetic training corpus. It does not
mean human-only data, no generated reasoning enrichment, or no prior filtering.
Do not insert the new constitution into control examples or add DA data.

## Model-specific derivative

Proposed HF name, following the user's explicit suffix preference:
`dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b`.
The inherited date identifies the source; the card must separately record the
actual conversion date and the naming exception, rather than falsely dating creation.
Refuse an unintended overwrite if that destination already exists.

Preserve row order, source membership, prompts and reference answers, except the
explicit tool-format transformation below. Preserve parent provenance and verdicts.
Keep canonical structured messages as the portable data; pin the renderer used
to obtain Harmony token IDs and masks at training time. Publish conversion audit,
source row hashes, exact revision pins, token statistics and representative decoded
examples with the derivative.

1. Replace the Qwen-added reasoning on the same 1,143 turns with fresh base
   GPT-OSS reasoning generated through Tinker at medium effort. Reuse the existing
   backfill approach: generate from the prompt, discard the generated answer,
   and judge whether the trace is consistent with the original reference answer.
   Do not claim this procedure proves that the reasoning caused the answer.
   Use bounded retries and record rejections; unresolved cases require a visible
   disposition before publishing, never silent retention of Qwen traces or invented text.
2. Convert APIgen definitions into structured function schemas and calls into
   native assistant tool calls. Normalize Python-style argument types carefully,
   including optional, union and nested list types. Preserve argument values and
   call order. Explicitly resolve unsupported schemas rather than replacing them
   with permissive schemas without a record. Remove the old requirement to emit XML.
3. Preserve the 35 no-tool answers as final assistant answers. Qualify the pinned
   renderer's handling of the 570 multiple-call examples without inventing tool
   results or sequential dependencies. Preserve a semantically equivalent batch of calls.
4. Ordinary worked math solutions remain ordinary answer content unless the source
   explicitly identifies separate private reasoning. Do not heuristically invent a
   reasoning/final split. Rows with no reasoning must not receive fake empty reasoning.
5. Run the actual Hugging Face JSON loading path on the complete derived file;
   stable nullable schema across all rows, diagnostic counts in sidecars.

## Harmony and loss qualification

HF tokenizer/config revision inspected:
`openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
This pins the local reference tokenizer, not an unverified claim about Tinker's
server-side weight revision; record the model identity Tinker actually exposes.

Verified special token IDs: start=200006, end=200007, message=200008,
channel=200005, constrain=200003, return=200002, call=200012.
The pinned tokenizer spells these `<|start|>`, `<|end|>`, `<|message|>`,
`<|channel|>`, `<|constrain|>`, `<|return|>`, `<|call|>`.

Use the pinned GPT-OSS renderer, not string substitution from Qwen templates.
Reasoning goes to analysis, user-facing answers to final, function calls to the
function recipient and commentary channel with a handoff ending. Preserve reasoning
across tool continuations; use the model's documented completed-turn history policy.
Keep complete raw reasoning in archived rollouts even when rendering later turns
omits prior completed-turn reasoning.

Qualify plain answers, analysis plus final, multiple assistant turns, single/multiple
tool calls, tool-result continuation and truncation. Check prompt token IDs match
the inference renderer and next-token targets are shifted correctly. Mask user,
system/developer, tool results and supplied generation prefills. Supervise generated
assistant payloads and the transitions/terminators the model actually must emit.
Check assistant-only examples do not accidentally train an immediate empty analysis.

Tinker's documented cross-entropy uses a weighted sum. For a logical optimizer
batch with N supervised tokens, use mask/N for all its examples, across all API
microbatches, and take exactly one optimizer step. Test the unequal-length-row case
and invariance to splitting the same logical batch. Do not average per-row means.

## Proposed training run

- Fresh base `openai/gpt-oss-120b`, standard 32K service tier; no August adapter.
- Seed 0, one epoch, 16 original rows per optimizer step: 625 steps if all 10,000
  rows are retained. If rendering requires separate training datums, retain the
  original-row grouping and avoid double-counting any supervised token.
- Rank 32 initially, Tinker's default; enable attention, MLP/expert and unembedding
  adapter scopes. Record actual adapter tensor coverage and trainable parameter count.
  This is a Tinker-native recipe, not numerically identical to Qwen rank 64.
- Initial LR 1e-4, cosine decay, 5% warmup, AdamW weight decay .01, token_mean.
  Explicitly qualify supported optimizer settings and finite losses/gradient metrics.
  Do not silently pretend Qwen's alpha=128/dropout=.05 are Tinker API controls.
- Target 8192 training tokens per example, but audit Harmony lengths first. Prefer
  a documented larger ceiling within 32K over cutting answers or dropping LongAlign
  rows merely because a different tokenizer expands them. No silent truncation.
- No manual concatenation of independent conversations without verified attention
  isolation. Let Tinker batch separate examples.
- No temperature for SFT; sampling temperature applies only to generation/backfill/eval.
- No user-selected NF4/MXFP4 requantization or local GPU rental. Tinker owns execution;
  record provider precision information if available, otherwise mark it undisclosed.
- Inspect a small initial segment for training health and continue the same run;
  no selection on ODCV. Save optimizer-bearing state every 100 steps plus final,
  sampler weights separately, and local step/shuffle/scheduler state for exact resume.
- Publish the adapter to HF with resolved config, trace provenance, training curve,
  supervision mask audit, pins and immutable Tinker checkpoint reference. Verify the
  exported adapter; no claim of RunPod/vLLM equivalence until separately tested.

## Local ODCV, Tinker model inference

Local Docker was reachable: Docker 29.6.2, 20 CPUs, approximately 17.6 GiB assigned RAM,
no running containers observed at inspection. Recheck capacity at launch because this
machine is shared. No Vast CPU or RunPod GPU is needed for this experiment.

Extend the existing integration, not a parallel eval pipeline:
- Declare reproducible Tinker SDK/cookbook dependencies in our isolated environment.
- Complete the GPT-OSS model profile, Harmony mask gate and training backend.
- Generalize existing reasoning backfill to allow Tinker generation while preserving
  its judge/provenance contract.
- Extend `src/infra/endpoints/tinker*.py` and the existing eval entrypoint/ODCV runner.
  ODCV currently refuses API targets; the bridge binds only loopback and lacks `/tokenize`.
- Provide exact `/tokenize` using the identical renderer and generation prefix, make
  the bridge reachable from Docker Desktop, and require exact counting for this run.
- Validate requested checkpoint identity, unique port ownership and local authorization;
  do not expose the Tinker credential to scenario containers. Do not accept any healthy
  unrelated service as the correct model.
- Forward or explicitly reject sampling controls, preserve reasoning and tool IDs,
  preserve commentary, distinguish stop/tool handoff/length/error, and never execute
  a truncated tool call. Bound retry costs and retain invalid attempts.
- Use namespaced Compose projects/output paths and clean up only resources owned by
  this run. No Docker restart, global pruning or changes to shared network pools.

Run a small base-model transport/tool smoke before SFT, then adapter smoke, then
the standard solo ODCV lite protocol: 40 scenarios x two variants x three passes
= 240 rollouts. Temperature .7, medium reasoning, 28,000-token window, unchanged
scenario instructions/rubric, Gemini Flash misconduct and progress judges through
the existing OpenRouter judging path. 'Tinker evaluation' means target inference;
the established independent judge remains unchanged.

Start concurrency 4, increase to 8 only after observing CPU/RAM/network capacity
and Tinker throughput. Count the complete Harmony input including tools and reasoning;
allocate output from the remaining protocol window, with up to 8192 tokens per
response including analysis. Context policy must be explicit and shared by future
GPT-OSS arms. Remove the Qwen-specific published reference from this model's metadata.

Report completed/invalid cells, misconduct with scenario-aware intervals, progress,
tool parse failures, reasoning lengths, truncation, retries, latency and billed tokens.
Prompt-only or empty runs are infrastructure failures. Preserve all rollouts and both
judge outputs under the repository's HF eval contract. The judge's documented
susceptibility to persuasive rationalizations remains a limitation of the measured score.

Recommended follow-up: matched full ODCV on untouched GPT-OSS to measure what nosynth
SFT itself changed. The immediate requested full run is the trained control; a small
base smoke cannot establish a base-versus-SFT performance difference.

## Costs and prerequisites

Live Tinker standard-32K prices per million tokens: training .737 USD; uncached
prompt .33 USD; cached prompt .066 USD; generation .84 USD. Storage is separate.
Training includes processed prompt tokens, not just supervised tokens.

Thus 5-10M processed training tokens would cost approximately $3.69-$7.37 for one
epoch. This is an illustration until the converted corpus is counted. GPT-OSS trace
generation, compatibility judging, smokes, ODCV sampling and its judges are additional.
For example, an ODCV run using 20M uncached input plus 2M output tokens would cost
$8.28 for target inference; it is not a measured forecast of this model's trajectories.
Measure a smoke and extrapolate before the full run, tracking cache hits separately.
Provisional planning allowance: $30-$50 end to end, not a quote or spending authorization.
This is above CLAUDE.md's approximately $20 spend-notification threshold.

`tinker auth status` reports no available credentials; TINKER_API_KEY is also absent
from the process and shared repository .env. HF and OpenRouter credentials are present.
Before paid execution, configure Tinker authentication locally (never paste a key
into chat), check account access/credit and model capabilities, then record a finite
run budget and cancellation policy. Local schema/rendering/bridge work can proceed first.

References checked: Tinker models/pricing, ServiceClient, cross-entropy, LoRA primer
and authentication docs; OpenAI Harmony guide; pinned HF tokenizer and source dataset.

## Execution notes

User authorized execution. The repository API key identifies
`lasr-g3s26@arcadiaimpact.org`; the initial billing-gated 402 cleared on retry.
Eight live native reasoning samples passed compatibility judging. A real Docker
container reached the authenticated bridge, obtained a native tool call, returned a
tool result, and received the correct final response. Both requests' input counts
matched `/tokenize`. This is transport qualification, not a benchmark result.

The isolated locked runtime is `src/infra/endpoints/tinker_runtime`; it avoids
changing the environments used by other sessions. The Windows-only dependency
override excludes the unavailable Inkling native renderer; GPT-OSS uses the
independent cookbook renderer. HF tokenizer spelling is verified by token IDs.
The renderer fixes the cookbook's multiple-handoff batch and historical assistant
boundary issues, and includes exact JSON Schema in tool-description comments to
preserve constraints lost by the TypeScript projection.

Backfill uses at most three attempts per originally traced turn. After those,
the explicit disposition is to retain the unchanged original answer without a
reasoning trace, following the existing backfill pipeline's rejection policy.
No Qwen trace or generated replacement answer is retained. All attempts and
rejected turn IDs are included in the dataset audit. The final audit rechecks
row/prompt/answer conservation and actual HF JSON loading before publication.

Training and target inference use Tinker; ODCV's established Gemini judging still
uses OpenRouter. Target and judge calls have separate durable cost reservations.
The ODCV bridge records raw generated token IDs and refuses malformed/truncated
tool execution. Infrastructure failures stop scoring rather than silently becoming
successful low-misconduct trials. The response cap is 8192 tokens within a 28000-token
window; the inherited harness's length-stop wording refers generically to the
window, while the actual input/output lengths and cap are in the bridge ledger.

Export will publish the **native Tinker LoRA**, with tensor shapes, finite-value
checks and the immutable sampler path. It does not require downloading the full
120B base. A later PEFT conversion / RunPod serving qualification is separate work;
the HF card must not imply that has been performed.

The optimizer records Tinker SDK defaults explicitly: beta1=.9, beta2=.95,
epsilon=1e-12, with requested weight decay=.01 and gradient clipping=1. This is
not a claim of exact optimizer equivalence with the Qwen trainer.

### Published conversion and training launch

The converted dataset is published as
`dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b`
at `d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da`, with exact payload readback verified.
All 10,000 rows remain. Of 1,143 originally traced turns, 1,073 received accepted
native GPT-OSS traces; 70 retain their original answer without reasoning supervision.
All generated attempts and the rejection disposition are archived with the dataset.
The complete HF JSON loading path was checked against the canonical payload.

The final mask audit has 10,380 assistant datums, 5,916,570 processed input tokens
and 2,341,585 supervised tokens. No datum exceeds 8,192 tokens; nothing was
truncated. The training estimate is $4.36051209 at the recorded rate. Trace
generation's uncached-price upper/reserved total is $1.56450618 and compatibility
judging's charged/reserved total is $0.776164; these are not provider invoices.

Training started on 2026-09-28 around 19:41 UTC from code commit `44536702`,
using Tinker model ID `cdd100d5-e08c-52a7-9ba9-890187111a57:train:0`.
The launch observation records the exact provider response and requested scopes.
The step-100 optimizer checkpoint has been saved; completion and ODCV results
are still pending. All 80 scenario/variant Docker images were built locally
without running agents, using this task's namespaced build projects.

### Completion update

The pending status above is superseded: SFT completed all 625 steps, the native
adapter was published and verified, and full ODCV completed and judged all 240
rollouts. Misconduct was 111/240 (46.25%; scenario-aware 95% CI 35.8–57.0%),
mean progress 4.32/5, and submission 230/240. End-to-end estimated/reserved
compute and judging cost was $10.2324, excluding retained checkpoint storage.
See [the completed results, immutable pins, format-error diagnostics and limitations](2026-09-28_nosynth_tinker_results.md).
