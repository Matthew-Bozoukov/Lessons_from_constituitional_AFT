<!-- ABOUTME: Evidence-backed inventory of GPT-OSS-120B work and recommendations for extending constitutional SFT. -->
<!-- ABOUTME: Read-only investigation on 2026-09-28; recommendations are proposed experiments, not measured outcomes. -->

# GPT-OSS-120B exploration — 2026-09-28

Worktree: `C:/Users/nikak/.codex/worktrees/gpt-oss-120b-exploration/teaching_claude_why_replication`.
Branch: `codex/gpt-oss-120b-exploration`, created from local `main@4b250268bc48bfbcd09369658ff0a86d034aea58`.
Other checkouts, branches, processes and paid infrastructure were not changed.

**Recommendation.** Start with Tinker for a small, controlled replication, using approximately the existing dataset scale. GPT-OSS's parameter count gives no defensible reason to multiply the SFT corpus by four. The immediate work is Harmony supervision, explicit LoRA scope, consistent loss normalization, and qualified evaluation. Use RunPod when control over kernels, expert/router behavior, or sustained inference throughput warrants the engineering.

**What was searched and what remains unknown.**

The checked-out source, archived configurations, research log, all-ref Git history/path inventory, original checkout's matching local output metadata, and PR 113 were inspected. Live Hub catalogues covered 147 models/603 datasets in `dougalldeepmind` and 116 models/61 datasets in `matboz`. The old `LASR-Callum` catalogue returned zero entries; the known dataset resolves under `dougalldeepmind`.

Metadata inspection covered 263 model repositories and 259 eval-tagged repositories: 247 readable adapter configurations and 1,130 run_meta files across 258 eval repositories. None identified GPT-OSS/Tinker as target/base. The remaining eval repo is the old Qwen transcript collection. Sixteen model repositories lacked adapter_config; two exposed non-GPT-OSS model configurations, fourteen lacked both standard files. All catalogue cards/descriptions/tags were also searched, including datasets without eval tags.

This establishes what is discoverable in these locations, not that nobody ran another experiment. No TINKER_API_KEY was available in the project's .env, so private Tinker run/checkpoint state, present availability and historical billing were not queried. The historical training scripts and raw GSM8K outputs named in the log were not found in this checkout or its reachable Git path inventory.

Local metadata receipts are in `output/2026-09-28_gptoss120b_inventory/`; `scratch/gptoss_inventory.py` performs the lineage scan over the saved catalogues. These are investigation artifacts, not new model evaluations.

**1. August 15: a real Tinker SFT experiment and two masking failures.**

The main record is [LOG.md, August 15](../LOG.md#2026-08-15--gpt-oss-120b-on-tinker-two-harmony-masking-bugs-found-by-ab-not-by-inspection).
The [published mixture](https://huggingface.co/datasets/dougalldeepmind/2026-08-15-table2-9284-difficult-advice-716-harmony-gpt-oss-mixture/tree/015afa654cf4bbf679532c7c21a781aeac7b788d) is pinned at `015afa654cf4bbf679532c7c21a781aeac7b788d`.

- 10,000 rows: 9,284 general Table-2 instruction examples and 716 difficult-advice examples, or 7.16% by rows.
- Rank 32, learning rate 1e-4, cosine decay, 5% warmup, global batch 16, one epoch.
- Historical log reports approximately 33 minutes and $4.52 per run.
- The card reports 6,138,470 rendered tokens; the log reports 6,128,470 training tokens. The difference is exactly one token per row, consistent with next-token shifting, but the missing trainer prevents verifying that explanation.
- After the masking fix, 44.8% of corpus tokens were supervised according to the log. Neither the total token count nor the synthetic row percentage is a measured synthetic supervised-token share.

Conversion was substantive: Qwen empty think markers were dropped, 1,597 tool calls in 952 rows became Harmony calls, and 1,004 math examples moved their derivation into analysis. The Qwen comparator did not receive that math transformation. This was consequently not a clean model-only comparison.

The first mask suppressed assistant continuation headers between analysis and final. The adapter learned an invalid final header, making correctly solved questions liable to score as failures. It also explicitly rewarded final-only openings on 8,280 traceless examples, suppressing reasoning on general prompts. The fixes supervised continuation headers, masked the final-only opener on traceless examples, and recognized the tool handoff terminator.

| Logged GSM8K check, same 60 questions | Base correct | Adapter correct | Malformed adapter finals |
| --- | ---: | ---: | ---: |
| Before mask fix | 51/60 (85.0%) | 53/60 (88.3%) | 27/60 (45.0%) |
| After mask fix | 52/60 (86.7%) | 52/60 (86.7%) | 0/60 |

The useful result is repair of output structure and recovery of reasoning on general prompts. Sixty questions do not establish a capability improvement or a precise nondeterminism noise floor. GSM8K reasoning usage was 100% in these comparisons; reasoning suppression was observed on separate general prompts.

Corrected checkpoint recorded in the log:
`tinker://d745257b-ccd9-5315-b87f-095c1b5bd351:train:0/sampler_weights/t2_da716_maskfix`.
The pre-fix checkpoint is superseded. The log left PEFT export and real misalignment evaluations as next steps. I found no subsequently published adapter or completed misalignment result tied to this checkpoint in the scanned evidence.

The [archived local training config](../../configs/train/archive/gptoss120b-table2-da-716.yaml) proposed NF4, rank 64 and a placeholder memory budget. It is a proposed local route, not evidence that this RunPod configuration completed training.

**2. September 18–19: native GPT-OSS reasoning/response datasets.**

Both datasets were generated through OpenRouter using GPT-OSS-120B on prompts from the no-synthetic instruction mixture, excluding LongAlign. Their cards describe 70% medium, 15% low and 15% high reasoning effort. I downloaded and counted the JSONL rows:

| Dataset | Revision | Rows | Rows with reasoning | Rows with calls | Rows with tool results |
| --- | --- | ---: | ---: | ---: | ---: |
| [September 18, 5k](https://huggingface.co/datasets/dougalldeepmind/2026-09-18-nosynth-gptoss120b-reasoning-5k/tree/54edb6f98070c081061a551fa1db6d2d5fb2b786) | 54edb6f98070c081061a551fa1db6d2d5fb2b786 | 5,000 | 5,000 | 597 | 0 |
| [September 19, 9771](https://huggingface.co/datasets/dougalldeepmind/2026-09-19-nosynth-gptoss120b-reasoning-9771/tree/f350cf8b6ac803c7fe91d0d5eff34e5b6615536e) | f350cf8b6ac803c7fe91d0d5eff34e5b6615536e | 9,771 | 9,771 | 943 | 0 |

The latter's actual effort counts are 6,840 medium, 1,466 low and 1,465 high. Its source counts are 2,773 no_robots; 1,471 tulu3_if; 1,064 self_oss_instruct; 1,062 numinamath_cot; 1,053 apigen; 1,050 smol_constraints; 984 smol_summarize; 314 lima.

These are usable leads for a model-native general-data control. They are newly generated answers as well as reasoning, so replacing the Qwen data with them changes the training distribution. Generation alone does not establish answer correctness. Neither dataset teaches the response-after-tool-result transition. Exact supervised-token counts must be computed after choosing the renderer and mask; row counts cannot substitute for them.

**3. September 23: a later checkpoint reference and eval integration.**

[PR 113](https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/pull/113), commit `0c40edb1c28c5aa5fdc0dda1ea99f4e887edff2e`, added the Tinker endpoint, dictator and secret_number evaluators. [The test fixture](../../tests/test_secret_number.py) contains:

`tinker://49e5b60c-8085-5a19-95d8-6490d7f41d7c:train:0/sampler_weights/gptoss120b-nosynth9771-lr1e4-3ep-sft-r32`

Its name suggests a rank-32, three-epoch, 1e-4 no-synthetic run on the 9,771 examples. That is a checkpoint reference, not independently verified training metadata.

The PR explicitly reports a successful live Tinker tool round trip followed by HTTP 402 before an episode submitted. It reports 73 passing tests, but no completed secret-number gaming measurement from the integrated harness. The config mentions earlier approximately 300-episode checkpoint comparisons; their scores and rollouts were not recovered. The dictator smoke used an OpenRouter target and encountered judge credit failures; it is not a demonstrated GPT-OSS safety result.

**4. A GPT-OSS ODCV result is bundled, but it is upstream evidence.**

The vendored benchmark's [bootstrap table](../../src/eval/misalignment/odcv/third_party/odcv-bench/existing_results/current/evaluations/judge_all/bootstrap_ci.csv) reports base GPT-OSS-120B at 36.25% misalignment overall (stored 95% interval 23.75–50.0), 27.5% incentivized, 45.0% mandated, severity 1.5688. With 40 scenarios per condition, those rates correspond to 29/80 overall, 11/40 and 18/40.

[Vendor provenance](../../src/eval/misalignment/odcv/third_party/VENDORED_FROM.txt) identifies these as retained paper results, upstream commit `7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668`; bulk upstream raw rollouts were pruned. This is not our adapter result and is not a matched baseline for today's changed ODCV protocol. It is a reason to expect some evaluable headroom, subject to a fresh baseline.

**What 120B and MoE imply for this project.**

OpenAI lists approximately 117B total parameters and 5.1B active per token. Its [pinned config](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) has 36 layers, 128 experts per layer and four selected per token; the released expert weights use MXFP4. See also the [official model description](https://developers.openai.com/api/docs/models/gpt-oss-120b).

My interpretation:

- Total parameters chiefly affect weight storage/capacity; active parameters help explain compute. Neither directly determines the amount of behavioral SFT needed.
- We are adapting an already trained model with LoRA, not learning 117B parameters from scratch. Pretraining token-to-parameter rules do not give a four-times SFT prescription.
- Nor does 5.1B active imply one-fifth the Qwen data. The selected experts vary with inputs and layers, and learned representations/behavior differ.
- Under hypothetical perfectly uniform routing, each expert sees about 4/128 = 1/32 of token routes at a layer. That is not a 32-times dataset requirement: experts start pretrained, shared attention sees all tokens, routing is nonuniform, and gradients can change behavior through shared components.
- Coverage is the useful concern: diverse domains, tool contexts and pressure mechanisms may teach more than repeatedly extending a narrow trace distribution. If on-topic improvement fails to transfer across domains, inspect expert coverage when the backend permits it.
- Model-family replication is valuable, but two families with different pretraining/post-training cannot identify a causal effect of parameter count or MoE architecture.

Start near the existing approximately 10k-example scale. Measure rows, full rendered tokens, supervised tokens, synthetic supervised-token share and truncation separately. Then use a nested 0.5x/1x/2x size ladder if needed; reserve 4x for evidence of continued gains. Keep composition and epoch policy fixed and recognize that a one-epoch size ladder also changes update count. An extra matched-exposure condition distinguishes new-example diversity from simply making more updates.

**LoRA and loss: what needs adapting.**

Start rank 32 and one epoch, with 1e-4 as a historical anchor. A small 3e-5/1e-4/3e-4 LR pilot is a reasonable proposal, not a known optimum. Tinker's [published GPT-OSS Tulu sweep](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/chat_sl/results/sft_sweep.md) favored rank 16 / 3e-4 on its held-out NLL objective; its batch/data/objective differ from ours.

Record modules, expert coverage, head training, adapter scaling and actual trainable count. Rank alone is not comparable across these models or backends. Tinker's [current implementation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/hyperparam_utils.py) shares the hidden-dimension LoRA factor across experts. Its count table gives 1,314,387,968 trainable parameters at rank 32 with all three components enabled; this is a calculation from current code, not a measurement of the historical checkpoint.

For local Transformers/PEFT, expert tensors need explicit targeting: OpenAI's [fine-tuning example](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers) uses target_parameters alongside ordinary linear modules. Our trainer currently supplies only target_modules. Avoid accidentally adapting only attention or silently adding router/head updates. I would keep router weights frozen initially and make attention-only/expert-scope changes explicit ablations.

Match the current repository's token_mean objective. Tinker's [cross-entropy primitive sums weighted token losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/); arrange weights/gradient normalization over the complete optimizer step to reproduce our intended normalization, including accumulation. Do not assume the same numeric LR makes different loss scaling, LoRA scaling, clipping and optimizer settings equivalent. The August run also predates the repository's September 21 switch from example averaging to token averaging.

Harmony requires structured treatment of analysis/final/tool turns, initial prefills versus generated continuation headers, and distinct message/assistant-turn/tool-handoff endings. Preserve the two August fixes as regression cases. Test answer-only and reasoning-only supervision explicitly; the current masking implementation assumes a Qwen-style single reasoning block and cannot become correct by substituting a few strings.

**Tinker versus RunPod.**

| Question | Tinker | Our RunPod route |
| --- | --- | --- |
| First controlled replication | Existing historical success; CPU-side driver and managed MoE execution | Requires local GPT-OSS training qualification |
| Exact adapter/kernel control | Service-defined adapter scheme and supported controls | Can choose expert scope, router policy, quantization and instrumentation |
| Memory | Managed by service | Must measure weights, adapters, optimizer state, activations and context |
| Price basis | Tokens processed | GPU-hours, startup and storage |
| Eval options now | API-capable evaluators; existing local shim | Full local eval suite after serving/profile qualification |
| Recommended role | First training/pilot backend | Exported-model evaluation or experiments requiring internals |

Current [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/) is $0.737/M training tokens, $0.33/M uncached prompt tokens and $0.84/M generated tokens for the standard 32k GPT-OSS target. Thus 6.12847M training tokens cost about $4.52 before sampling/storage. The 128k variant has a different model identifier and $2.33/M training rate. These are current listed rates, not a quote for the newer 9,771-row dataset, whose token count has not been established.

[RunPod's page](https://www.runpod.io/pricing) currently lists H200 at $4.59/hour. Illustratively, four at that rate cost $18.36/hour; to beat $4.52 they would need less than 14.8 billed minutes including startup, excluding storage. This does not claim four GPUs are required or that such throughput is achievable.

The local GPT-OSS profile is only a naming stub. No verified Harmony mask, expert target configuration, measured memory budget or GPU choice is present. The trainer's multi-GPU path is DDP: each GPU holds a complete model copy. Four torchrun processes do not pool memory for BF16 weights. A dequantized 117B checkpoint is on the order of 234GB for weights alone; local training would need an appropriate sharded approach or a validated quantized path.

Inference fitting on one H100 does not establish training memory. Unsloth [advertises](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) a 65GB GPT-OSS-120B QLoRA route, making one sufficiently large GPU plausible with that stack. It does not validate our current trainer, rank, sequence length or packaging. Do not automatically reuse the archived NF4 proposal or requantize MXFP4 without measuring its effects.

Two historical infrastructure notes have changed: current Tinker docs offer an [OpenAI-compatible beta endpoint](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/), and their [adapter export tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/) says conversion does not download the base weights. The old repo comments say otherwise. Confirm GPT-OSS-specific export/serving parity before relying on either path. Our pinned [vLLM 0.26 model table](https://docs.vllm.ai/en/v0.26.0/models/supported_models/) marks GPT-OSS LoRA supported, which is promising but does not test the exact Tinker expert adapter.

The repository shim imports tinker and tinker_cookbook, neither declared in pyproject.toml/uv.lock. A reproducible dependency environment is another prerequisite. Historical rank 32 support is documented; the old claim that it is Tinker's maximum should be rechecked against current service capabilities.

**Evaluation and proposed first experiment.**

Keep the scientific tasks and rubrics. Adapt transport, rendering and parsing, and qualify the base model before training. Pin reasoning effort (medium initially), template, model/checkpoint, generation budget, context policy, sampling and judge across GPT-OSS arms. Record parse failures, truncation, missing answers, reasoning length, tool validity, benign-task success and over-refusal alongside behavioral rates. Lower harmful-action scores are not convincing if the model simply stops acting competently.

The registry currently permits Tinker/API targets for MMLU, Arena Hard, MASK, psychosis, MoralBench, CTFish, dictator and secret_number (the latter Tinker-only). ODCV, agentic_misalignment, delegated_harm and SWE-bench do not accept this endpoint path. They need a qualified exported adapter on vLLM or a deliberate transport extension. An ODCV extension must address container reachability and token/context handling, not just toggle an allow flag.

Proposed sequence:

1. Recover/inspect the September Tinker checkpoint and its resolved training config. Qualify base and adapter rendering, masks, tool-result continuations, truncation and export parity on a small fixed set.
2. Train a matched no-synthetic control and a difficult-advice arm at the current project's chosen synthetic supervised-token dose, initially around existing corpus scale. Keep the general-data construction, mask, effort distribution and loss objective matched. Include the untouched base in evals.
3. Use MASK plus a qualified action-based evaluation, and capability/tool-use checks. Recover existing secret-number results if available; otherwise validate its full episode/scoring path before using its rates.
4. Use a pilot seed to select the numerical recipe on development data, then replicate the selected no-synthetic/DA contrast across three training seeds with a reserved test protocol. Repeated eval samples are not additional training seeds.
5. Expand data or rank only if the learning curve supports it. If transfer fails while training loss falls, inspect data diversity, supervision and baseline headroom before scaling the corpus.

This work performed no training, model sampling, GPU rental, remote publication or changes to the shared main checkout.
