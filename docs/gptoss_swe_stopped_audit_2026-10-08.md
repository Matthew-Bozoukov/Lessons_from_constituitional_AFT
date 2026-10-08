<!-- ABOUTME: Offline diagnosis of the stopped GPT-OSS SWE campaign against historical ODCV fixes. -->
<!-- ABOUTME: Separates malformed calls, recovery, renderer defects, output limits and benchmark correctness. -->

# GPT-OSS SWE: stopped-run audit, 2026-10-08

The stopped campaign has a material model/tool-interface confound. The SWE shim did not incorporate the clarified ODCV tool instructions or its raw-tool-history cleanup. Offline replay confirms a history-serialization defect; saved outputs also show extensive model-generated syntax failures and repetitive generation. These findings do not establish how much any single repair would improve SWE-bench correctness. Full grading was not completed. No new inference, tool execution, infrastructure start or outcome repair was performed for this audit.

## Follow-up: teammate's October 6 runs on the current arms

After the user pointed out teammate runs on these exact arms, fetched all remotes on October 8. Newly fetched `origin/matboz/gptoss-doses` (`69162b2f`) and existing `origin/matboz/gptoss-history-rule` contain the newer arm drivers. Searching local and remote branch tips for both complete checkpoint UUIDs found no additional committed run receipt; those run outputs were gitignored. The public Hub supplies a direct control match that was missing from the initial analysis, which relied on the older September comparison.

The [October 6 control ODCV artifact](https://huggingface.co/datasets/dougalldeepmind/2026-10-06-odcv-gptoss120b-0-plain/tree/8e80b855661ea1cb114ad639021e4099ac660c47) pins the exact same `c8be8040-3551-5baf-a8aa-b9e606a91ed2:train:0/sampler_weights/2026-10-06-gptoss120b-0-plain` checkpoint in both run metadata and every sampling reservation. Recorded source is `654bad9ff1f9ca0f91b5b1285bdd320dc0718aae`. Published pass audits report three clean 80-rollout passes with zero shell/missing transcripts; both judges cover 240 rollouts. Reported misconduct is 34.2%; progress score >=3 is 92.9%, mean progress 4.49/5; formal submission is 74.6%. Progress, submission and misconduct remain different axes.

Read the actual source at the recorded commit, not just today's branch head. It already includes the custom Harmony renderer, fixed tool-format prompt by default, raw-tool-content cleanup, strict argument rejection preserving the malformed string, and specific executor feedback. Undeclared tools/wrong handoffs can be rejected by the bridge and retried through the existing HTTP path. Run metadata records T=.7, medium reasoning, max output 8192, context 28000, concurrency 10; the bridge uses top_p=1 by default and supplies no top_k override. SWE instead used T=1/top_p=.95/top_k=20, 16K output, 128K context, 500 steps, generic parser-failure feedback and the separate stock-renderer shim. These are substantial protocol differences, not evidence that the checkpoint is inherently unable to use tools.

Downloaded and decoded the pinned control sampling ledger offline (`scratch/gptoss_swe/audit_teammate_odcv.py`): 3,797 reserved/completed requests; 3,500 parsed raw tool blocks; 105 invalid-JSON blocks, of which 61 contain the confirmed extra bracket; 101 recorded invalid-argument events and 33 bridge-error events. Different counts reflect blocks versus events and bridge rejection stages. There were 23 full-8192 completions and 34 responses reaching their actual allowance including context-clipped allowances. Thus successful completion of the evaluation did not mean zero formatting errors. Full output/metadata and the audit live in `output/gptoss_swe_analysis/teammate_odcv/`.

For DA15, `scratch/gptoss_arms/da-15.yaml` and `backfill_progress.py` explicitly document evaluation and later progress judging of `2026-10-06-gptoss120b-0-da-15`; the judge script says its combined transcripts were already on the teammate's local disk. The evaluation driver reads the sampler from gitignored `trained_adapter.json`. Neither that receipt nor the DA15 result payload was found in fetched branches, accessible local worktrees, or the public Hub listing. This supports the existence of the teammate's DA15 work but does not independently bind its full result to the `5c74fa19...` checkpoint or establish its score. No result is invented. Its runtime receipt/transcripts would close that remaining evidence gap.

Updated conclusion: the exact control checkpoint demonstrably completed useful ODCV work through the established custom interface. The next compatibility baseline should be that actual implementation, adapted and tested against SWE's schemas/history/termination contract, rather than assuming our stock shim was equivalent. The new evidence strengthens the need for a serving comparison; it does not isolate temperature, prompting, rendering, retries or task differences as a single cause. No code was merged into the serving path and nothing was restarted.

## Evidence and counting

Source: the locally verified archive `output/gptoss_swe_shutdown/gptoss-three-20261008-stopped.tar.gz`, SHA256 `da646dbd2bc5480e420d4cc893cccc25aa11e7a13d449ce33945193b4e7260d1`; all 134,204 original file hashes were verified during shutdown. Active runtime was `ffeca09c1b0fa77925ba5fdbac2aa7021cff9283`. The script reads only the three parallel roots, retaining the longest saved trajectory/checkpoint per task attempt, so copied original-base outcomes are not double-counted.

The census contains 344 attempt trajectories and 23,344 distinct saved assistant responses with Tinker metadata, including partial/interrupted attempts. The final ledger has 23,389 completed request records: 45 records are outside this conversational census, and their exclusion must not be mistaken for zero usage or an established failure cause. Repeated turns within one task are not independent observations. Pending tasks are absent, and shutdown censors some recoveries. Task order prioritized historical long tasks, so this is not a representative random sample of all 300 tasks.

Reproduction: `scratch/gptoss_swe/analyze_stopped.py`; outputs `output/gptoss_swe_analysis/{summary.json,calls.jsonl,tasks.json,length_outputs.json.gz}`. The separate `replay_renderer.py` uses the cached exact cookbook 0.5.3 renderer and pinned tokenizer offline; it records `renderer_replay.json`. Its cached environment path is machine-specific. These are experimental analysis scripts, not production fixes.

## What made the earlier ODCV runs work better

Historical evidence was re-read from `codex/gpt-oss-120b-exploration`, especially `docs/gptoss120b/2026-09-29_harmony_explained_probe.md`, `2026-09-30_control_refresh.md`, `2026-09-30_base_odcv.md`, and `src/infra/endpoints/harmony.py`.

1. Explicit native-Harmony routing and JSON instructions: route to `functions.bash` on commentary; use exactly one JSON object with one string-valued `command`; distinguish this from an array, OpenAI envelope, or `tool_calls` wrapper; escape multiline content correctly; end the outer string/object without an extra `]`; wait for the tool result before claiming success. A positive Python-heredoc example contains internal arrays/dicts without introducing literal Harmony boundaries. This was prompting, not constrained decoding or silently fixing model output.
2. A custom renderer removed raw fallback text when the same response already had parsed tool calls. Otherwise the next prompt could contain the call twice, as text and as a structured call. It also handled intermediate versus final handoff boundaries in multiple-call responses, preserved tool schemas, and treated saved assistant turns as history.
3. Malformed JSON arguments remained visible as rejected calls, with explicit invalid-JSON feedback and no execution. Separately, wrong native handoff endings or other bridge-validation failures could trigger the existing HTTP retry policy. Those completion resamples were recorded; they were not whole-rollout reruns.
4. Real tool/result/continuation smoke tests supplemented prompt-token fixtures. In contrast, SWE's acceptance of an ended qualification outcome was inadequate: the DA15 qualification had 41 format errors in 42 calls. An ended outcome alone is not a healthy compatibility test.

The full September 30 ODCV comparisons used the same checkpoint within each original/fixed prompt comparison, 240 rollouts per condition:

| ODCV metric | Control original → fixed | Base original → fixed |
|---|---:|---:|
| Confirmed stray-bracket calls | 101 → 21 | 24 → 9 |
| Rollouts affected by brackets | 19 → 8 | 15 → 8 |
| Invalid-JSON calls | 102 → 23 | 24 → 16 |
| Progress score ≥3 | 196 → 218 | 221 → 231 |
| Separately bridge-rejected completions | 1 → 13 | 9 → 28 |

Thus the mitigation improved observed syntax/progress but did not eliminate native-format failures. Four fixed-prompt base rollouts exhausted the normal request retries. Control also previously issued bash and task completion together prematurely; task submission and progress are distinct from correct or ethical behavior. A separate local workspace-containment failure was recovered by preserving completed cells and running only missing cells, without disabling the guard.

The small failure-history probe found 7/30 malformed LoRA responses at T=.7 and 9/30 at T=.2 under the prior bracket-example prompt, versus 0/30 at each temperature with the explicit explanation. This supports testing the prompt/interface, not assuming colder sampling fixes formatting. It was a selected small probe, not a general failure-rate estimate. Historical checkpoints/tasks differ from the October 6 adapters and SWE tasks here.

## Sampling settings actually used

All 23,344 inspected responses record T=1.0, top_p=.95, top_k=20, and the same native return/handoff stop IDs. The run used medium reasoning, neutral penalties, 16,384 response tokens, a 131,072-token model context, 262,144 generated tokens per task, and 500 steps. This matched the chosen Qwen sampling recipe, not ODCV's recipe.

The full ODCV runs used T=.7, top_p=1, 8,192 output tokens, 28,000 context and 50 cycles. The historical ODCV implementation also removes completed assistant-turn reasoning from later history while retaining ongoing tool-cycle reasoning; SWE retained reasoning and rejected turns to match the Qwen task loop. This difference can change context pressure and cost, and should be declared rather than silently changed.

## How much was the extra square bracket?

A confirmed extra `]` means JSON decoding fails at that character and deleting that one character yields an object. This is a diagnostic test only. The strict ending subset permits only an optional final `}` and whitespace afterward.

| Saved-run metric | Base | Control | DA15 |
|---|---:|---:|---:|
| Assistant responses | 8,784 | 8,310 | 6,250 |
| Responses receiving format-error feedback | 5,548 | 1,252 | 480 |
| Invalid-JSON call blocks | 1,190 | 139 | 89 |
| Confirmed extra-`]` call blocks | 1,166 | 102 | 62 |
| Strict literal-ending `]` subset | 1,021 | 92 | 56 |
| Tasks affected by extra `]` | 96 | 62 | 38 |
| Longest consecutive format-error streak | 460 | 301 | 94 |

Across arms, 1,330/1,418 invalid-JSON call blocks (93.8%) are this bracket defect. But the 1,329 responses containing these blocks are only 18.3% of the 7,280 format-error responses: the bracket is the dominant JSON defect, not the whole failure story. One base response contains two offending blocks, explaining its call/response count difference.

## Did the next turn recover?

Here, accepted tool use means a structured call, recorded tool feedback, and no format-error feedback. It does not mean exit code zero, the intended action succeeded, a correct patch, or a resolved benchmark task.

| Next-turn result | Base | Control | DA15 |
|---|---:|---:|---:|
| Accepted tool after any format error | 343/5,548 (6.2%) | 423/1,252 (33.8%) | 204/480 (42.5%) |
| Accepted tool after an extra-bracket response | 207/1,165 (17.8%) | 81/102 (79.4%) | 55/62 (88.7%) |
| Exact same command accepted after bracket response | 16/1,165 (1.4%) | 6/102 (5.9%) | 2/62 (3.2%) |
| Any later accepted tool after bracket response | 1,052/1,165 (90.3%) | 100/102 (98.0%) | 62/62 (100%) |

The exact-command test compares the command string obtained by diagnostic bracket deletion with the next accepted command byte-for-byte. It is deliberately strict: harmless whitespace or a revised command fails the match. Broader recovery often involved a changed command and cannot be described as correcting the original action. Later recovery can take many turns. Seven base and one control bracket responses have no observed next response at shutdown. Across all error types, 37 responses have an unobserved next turn; denominators above include them.

## Other observed problems

- The SWE branch had no ODCV `harmony.py` helper; it used the stock cookbook renderer. It lacked the explicit ODCV explanation and used generic no-tool feedback after parser rejection. Correct calls did parse, but that does not prove the whole interface was healthy.
- **195 saved responses show the exact raw-fallback duplication pattern** (base 29, control 120, DA15 46). Offline replay of one case per arm confirmed that the stock next-history rendering contains the same arguments twice; removing only raw fallback text leaves one copy. This duplicates representation in the model's history, not tool execution. The pattern is established; its causal contribution to later failures is not measured.
- Unparseable Harmony header/body blocks: base 100, control 366, DA15 127. Additional malformed headers appeared inside reasoning. Examples include corrupted channel/tool routing, rather than just JSON brackets.
- Other executor-format-error responses: control 165 and DA15 36. These include undeclared `apply_patch`/`create_file`, corrupted names such as `bash,commentary`, and missing `command`. No response exposed multiple structured tool calls, so ODCV's multiple-call handoff repair is not an observed explanation for this SWE census.
- Thousands of base replies had no structured call and no recorded parser rejection. Some repeatedly apologized or claimed inability to finish, then received another no-tool correction. The 500-step allowance permitted very long error spirals.
- The shim suppresses a whole executable call batch if parsing leaves an unparsed block. This is safe against partial execution but weakens feedback to the model. ODCV preserved malformed arguments for explicit validation feedback; neither system executed bracket-repaired arguments.
- The loop detector catches only a narrow exact repeated-paragraph pattern. Empty loop diagnostics do not rule out repeated headers, numbers, near-repeated searches, or repeated errors across turns.
- The SWE shim maps a response without recognized native termination to `finish_reason=length`. One control task, `sympy__sympy-18698`, was labelled response-token-limit despite producing only 4,142 tokens against a 16,384 allowance and ending with `<|endoftext|>`. This is a termination/misclassification issue, not actual exhaustion of the configured response cap.

Simply importing the ODCV helper would require testing: its positive bash example was gated on an exact schema shape, whereas the SWE command schema includes an additional description. Verify the actual rendered SWE prompt and subsequent histories, not merely that an import succeeds.

## Would 2× or 4× output allowance help?

At shutdown there were 96 limit-ended attempts: 69 labelled response-token-limit, 23 context-limit, and four step-limit. Of the 69 response-limit labels, 68 actually generated 16,384 tokens; the unexpected-EOS case above accounts for the difference. Among all 88 saved `finish_reason=length` responses, 19 were short due to context clipping, 68 reached the full output cap, and one was the unexpected termination. Four other context failures happened before a model response.

Inspection of full-length tails found repeated empty Harmony headers/apologies in all eight base cases, and repeated zeros, numerical strings, `Search again`, and similar text in many control/DA15 cases. A descriptive screen finds an eight-token word/punctuation n-gram repeated at least 20 times in the final 14,000 characters of 60/68 capped outputs (8/8 base, 30/35 control, 22/25 DA15). This heuristic is not a validated loop classifier; the remaining cases are not proven healthy reasoning. Some output contains plausible reasoning, but saved prefixes cannot establish whether more tokens would recover.

Raising the response ceiling would allow more reasoning/action text before truncation, but would also allow longer repetitive output before feedback. It would not fix invalid JSON, duplicated history, step exhaustion, or the fixed 128K context. Longer responses can fill that context sooner and keep task slots occupied longer. The 262K generated-token task budget still applies.

At the recorded 128K rate card (output $1.94/M tokens):

| Change | Extra allowance per capped response | Maximum additional output charge for that response |
|---|---:|---:|
| 16,384 → 32,768 | 16,384 | $0.0318 |
| 16,384 → 65,536 | 49,152 | $0.0954 |

Extending each of the 68 observed capped responses to those ceilings once would add at most about $2.16 or $6.48 of output charges in that artificial calculation. **This is not a whole-run cost forecast:** changed continuations can generate further calls and cause those tokens to be replayed as input. Larger configured ceilings also increase worst-case request reservations before any actual output is generated. Across 80 simultaneous requests, the extra output reservation alone would be about $2.54 or $7.63. Actual billing is token usage, not automatically 2×/4× because the ceiling changed.

Using recorded cache hits, the conversational census accounts for about $148.04 input and $19.85 output ($167.89 total); approximately 88% was input. Calls receiving format-error feedback account for about $53.38 within that sum, but this is attribution, not a causal estimate of avoidable cost. The full ledger's $168.24 is a separate, slightly larger total and not a provider invoice. Rate reference: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/.

Recommendation: repair and offline-test known rendering/feedback problems first, explicitly qualify several tool-result-continuation and malformed-call-recovery histories, then consider a bounded matched 16K versus 32K comparison on productive long responses. Do not make 64K the blanket default on this evidence. Any future change to prompt, rendering, retries, history, sampling or limits defines a new declared protocol; preserve the stopped run rather than silently repairing or rescoring it. All services and the scheduler remain stopped.
