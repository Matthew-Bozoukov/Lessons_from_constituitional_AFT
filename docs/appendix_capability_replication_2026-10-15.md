<!-- ABOUTME: Paper-ready appendix outline, captions and limits for the October Qwen and GPT-OSS experiments. -->
<!-- ABOUTME: Distinguishes immutable published results from configuration-only provenance and incomplete coverage. -->
# Appendix handoff: capability retention and GPT-OSS replication

Prepared 2026-10-15. Figure numbers below are placeholders. Use the PDF figures for the paper and PNGs for previews. The companion `output/appendix_20261015/` bundle contains all seven figures in PNG/PDF/SVG, numerical mirrors, a pinned HF link catalogue, and downloaded metadata with SHA256 receipts. Reproduce with the scripts in `scratch/capability_preservation/`; they make no model calls.

## A. Capability retention in Qwen3.6-27B

### Suggested text

We evaluated the pretrained Qwen3.6-27B checkpoint and its plain-control and difficult-advice (DA15) fine-tunes on a fixed 570-question MMLU subset, all 300 SWE-bench Lite tasks, and pairwise Arena-Hard v2.0 comparisons. MMLU accuracy was 92.1%, 91.8%, and 91.4% for Base, Control, and DA15, respectively; SWE-bench Lite resolution rates were 59.7%, 59.3%, and 60.7%. These observed scores are similar across arms, but do not establish equivalence or seed-to-seed robustness.

The Arena results reveal a different pattern. Base was preferred to Control (style-adjusted score 78.9%, 95% CI 75.5–82.4%), while DA15 scored 20.3% against Base (16.7–23.6%). Control versus DA15 was near parity (50.2%, 46.8–53.6%). Thus, the fine-tunes retained performance on the measured knowledge and software-repair tasks but lost preference under this bounded conversational protocol. DA15 showed no clear additional preference loss relative to Control. We do not describe these results as universal capability preservation.

Both fine-tunes used one training seed (0), one epoch, learning rate 1e-4, token-mean loss, and LoRA rank 64 / alpha 128 / dropout 0.05. The recorded maximum sequence length was 8,192 tokens, with an 8,000-token batching budget and global batch 16. Previous assistant turns before the last user message were context-only; the active assistant/tool chain was supervised. DA15 targets 15% of supervised tokens, measured using the Qwen tokenizer, rather than 15% of rows (650 DA rows among 2,218; reported 15.09% of 4,999,993 supervised tokens). Control has 1,832 rows and 5,000,005 supervised tokens. The plain corpus already contains externally authored reasoning traces.

### Figures and recommended captions

**Figure A1: Qwen3.6-27B capability retention after fine-tuning.** Accuracy on a fixed 570-question, five-shot MMLU subset and task resolution on all 300 SWE-bench Lite problems for Base, plain Control, and DA15. Bars show observed scores for one checkpoint per arm, not equivalence tests. All MMLU answers parsed successfully. SWE uses official grading and includes the same two no-fix fixture passes in every arm. These results cover the stated protocols and do not establish general capability retention.

File: `2026-10-14-qwen36-capability-preservation.pdf`.

**Figure A2: Qwen3.6-27B pairwise Arena-Hard preference.** Repository length/Markdown-adjusted preference scores on 500 hard prompts, judged in both answer orders; values above 50% favor the first-named model, with ties contributing one half. Intervals are 95% prompt-bootstrap intervals (2,000 resamples). GPT-4.1 was the primary judge, with one approved Gemini fallback in DA15 versus Base. Base outperforms both fine-tunes under this generation protocol; Control and DA15 are near parity. These are preference estimates, not percentages of capability retained or official leaderboard scores.

File: `2026-10-14-qwen36-arena-hard-pairwise.pdf`.

Arena limitation worth including in the surrounding text: the shared 12,000-token reasoning-plus-answer ceiling produced empty final answers on 36/500 Base, 71/500 Control and 73/500 DA15 hard prompts, all ending at the length limit. Offline diagnostics restricted to both nonempty, normally stopped answers still favored Base, including length-matched subsets. Such outcome-selected subsets are descriptive diagnostics, not replacements for the primary comparison and not causal isolation of the fine-tuning effect.

## B. Replication in GPT-OSS-120B

### B.1 Model setup and adaptation

We repeated the Base / plain Control / DA15 comparison with GPT-OSS-120B using Tinker training and sampling. GPT-OSS uses native Harmony serialization: reasoning occupies the analysis channel, visible answers the final channel, and tool calls retain their native recipient and handoff markers. We preserved these distinctions in training, evaluation, and saved transcripts; capability/preference judges received visible answers rather than private reasoning. Evaluation used medium reasoning, with the renderer/runtime version pinned per benchmark.

The GPT-OSS adapters use rank 32, compared with rank 64 for Qwen. The documented Tinker recipe uses one epoch, batch 16, learning rate 1e-4, token-mean loss, maximum training length 8,192, 5% warmup, weight decay 0.01, Adam betas 0.9/0.999, epsilon 1e-8, and gradient clip 1. Qwen's alpha 128 and dropout 0.05 were not configurable through that Tinker training interface; therefore this is a cross-family replication with adaptation differences, not an identical optimizer/adapter intervention.

The Control export includes a completed training metadata receipt. The DA15 export binds its rank and sampler identity but lacks `training_meta.json`: the other DA15 training settings and source revision are supported by the historical launch configuration at Git commit `03d319308acdc5cdcee81d5b82eec74370a375bd`, not by an independently recovered completion receipt. The evaluated sampler identity is pinned in every downstream evaluation. State this provenance limitation rather than presenting the full DA15 training history as independently verified. Do not assert a provider weight precision based only on the public model's name.

The source mixtures were shared in design, but the HF revisions differ between the Qwen and GPT-OSS records. The link catalogue lists each separately. DA15 denotes the mixture's Qwen-tokenizer supervised-token allocation; it is not a fresh measurement of an exactly 15% share under GPT-OSS tokenization. Earlier September nosynth/tool-format experiments and later selected diagnostic reruns are separate experiments, not additional independent replications of these October arms.

### B.2 Misalignment evaluations

On the matched 56,000-token ODCV protocol, the scenario-aggregated misalignment rate was 42.9% for Base, 36.2% for Control, and 11.3% for DA15. On MASK, the published honesty statistic was 87.4%, 69.72%, and 95.9%, respectively. DA15 therefore shows the favorable direction in both evaluations. These behavioral results do not by themselves establish the mechanism, deployment safety, or reproducibility across training seeds.

**Figure A3: GPT-OSS-120B misalignment on ODCV.** Scenario-aggregated misalignment rates for Base, Control, and DA15 under the matched 56,000-token context, remaining-context response allowance, 50-cycle, medium-reasoning protocol. Lower is better. The design comprises 40 scenarios × two variants × three passes (240 trajectories per arm). Error bars are the published 95% scenario-level intervals: Base 31.0–55.7%, Control 26.2–47.6%, and DA15 6.7–18.3%. Gemini 3 Flash supplied the alignment judgments; one provider-blocked verdict each for Control and DA15 is excluded (240/239/239 usable). The plotted revision precedes later mixed-judge supplements.

File: `2026-10-15-gptoss120b-odcv.pdf`.

**Figure A4: GPT-OSS-120B honesty on MASK.** Published MASK scores for Base, plain Control, and DA15, using medium reasoning and one `lie_k=1` pass. The statistic is 100 minus the percentage classified dishonest; it should not be read as the fraction explicitly classified honest. Base and DA15 cover 1,000 rows each. Control covers 994/1,000 after a single targeted retry of 55 generation and 11 judge failures; six unresolved rows are excluded without imputation. Control used a 28,000-token context versus 20,480 for the historical Base/DA15 runs, with the same 16,384-token maximum response budget and historical renderer/scoring. Coverage, retry policy and context differ, so this is not a strictly matched first-pass comparison. No repeated-pass uncertainty interval is available.

File: `2026-10-15-gptoss120b-mask.pdf`.

Control MASK audit: 4,434/4,438 generation slots and 3,970/3,972 eligible judge slots validated after closeout. All original outputs and retry receipts are retained. The original pilot serialization defect was repaired by retrieving the same 58 provider futures without resampling. The subsequent user-authorized single retry is a separate intervention and is explicitly disclosed. Excluded rows must not be silently counted as honest/evasive or treated as a complete 1,000-row result.

Optional **Figure A7: Available GPT-OSS-120B MASK results across additional training mixtures.** The Base, Control, DA15, MSM and DA15+MSM checkpoints score 87.4%, 69.72%, 95.9%, 80.6% and 91.9%, respectively. Control uses the available-row retry policy described in Figure A4; the other runs cover 1,000 rows. These are historical results from distinct training mixtures, not a matched factorial estimate of the effect of reasoning supervision. Use only if the paper separately introduces MSM and its training provenance.

File: `2026-10-15-gptoss120b-mask-all-available.pdf`.

### B.3 Capability retention

GPT-OSS MMLU subset accuracy was 88.6%, 78.9%, and 81.8% for Base, Control, and DA15. Control produced 41 unparseable final answers and failed the predeclared 95% parsing-health threshold; DA15 produced 17 unparseable answers and passed it. Base had none. Unparseable final answers count as incorrect in these reported scores, so the Control point is descriptive and conflates answer delivery with knowledge.

SWE-bench Lite resolution was 22.3%, 19.3%, and 29.0%, respectively. Many trajectories ended at context limits: 163 Base, 207 Control and 177 DA15. Another 23 Control and 17 DA15 trajectories reached the response-token ceiling. All 300 tasks per arm were graded with no infrastructure exclusions. DA15's higher observed SWE score does not remove the MMLU degradation or establish broad improvement.

Arena-Hard comparisons were inconclusive under the separate Codex GPT-6 Luna low-effort judge: Base versus Control 53.8% [45.5, 61.3], Control versus DA15 48.0% [44.6, 51.5], and DA15 versus Base 47.5% [39.6, 56.3]. All intervals include 50%; this does not establish equivalence. In particular, DA15 versus Base does not clear a noninferiority rule requiring the lower confidence limit to exceed 45%.

**Figure A5: GPT-OSS-120B capability results under bounded inference.** Accuracy on the fixed 570-question MMLU subset and resolution on all 300 SWE-bench Lite tasks for Base, Control and DA15. MMLU scores count unparseable answers as incorrect (Base 0, Control 41, DA15 17); Control fails the 95% parsing-health threshold and is not a clean estimate of retained knowledge. SWE uses 65,000 tokens of context, at most 16,384 generated tokens per response and 500 steps, and retains context/token-limit outcomes. The official grades include the same two no-fix fixture passes per arm. Bars are descriptive single-checkpoint results.

File: `2026-10-14-gptoss120b-capability-preservation.pdf`.

**Figure A6: GPT-OSS-120B pairwise Arena-Hard preference.** Repository length/Markdown-adjusted scores on 499 hard prompts, judged in both orders by Codex GPT-6 Luna at low reasoning effort; one oversized prompt is excluded. Scores favor the first-named model above 50%; ties contribute one half. Error intervals are 95% prompt-bootstrap intervals (2,000 resamples). Every interval includes parity, but the uncertainty does not demonstrate equivalence. One empty-recipient wait-call exception was retained with its existing verdict. This judge protocol differs from the Qwen comparison and is not official leaderboard-equivalent.

File: `2026-10-14-gptoss120b-arena-hard-pairwise.pdf`.

## C. Inference limits to report

All token ceilings below include reasoning and the visible response where reasoning is enabled. Context means the current rendered prompt plus the response, not a lifetime output budget. Effective output can be reduced to fit remaining context. A dash means no agent step loop applies, not unlimited inference. No blanket cross-family performance ranking is justified because contexts, renderers, and Arena judges differ.

| Model / evaluation | Context tokens | Response tokens | Steps / cumulative output | Sampling / reasoning | Other limits |
|---|---:|---:|---|---|---|
| Qwen MMLU, all three arms | 131,072 | 32,768 | One completion per question | T=1; p=.95; k=20; seed 0; thinking on | 4,800 s request timeout; up to two request retries |
| Qwen SWE Lite, all three arms | 262,144 | 16,384 | 500 steps; 262,144 cumulative completion tokens/task | T=1; p=.95; k=20; min-p=0; repetition=1; thinking history preserved | 7,200 s request timeout; two request attempts; tool queue 600 s; grading 1,800 s; no independent task-walltime cap (`task_seconds=null`) |
| Qwen Arena, all three arms | 16,384 | Configured 12,000; actual 7,343–12,000 | One completion per prompt | T=0; p=1; thinking on | 742/750 prompts get 12,000; eight smaller allowances; identical per-prompt budget across arms |
| GPT-OSS MMLU, all three arms | 131,072 | 32,768 | One completion per question | T=1; p=1; seed 0; medium | 4,800 s request timeout; SDK retries disabled |
| GPT-OSS SWE Lite, all three arms | 65,000 | 16,384 | 500 steps; 262,144 cumulative completion tokens/task | T=1; p=1; k disabled; medium | 7,200 s request timeout; two request attempts; tool queue 600 s; retained terminal model failures |
| GPT-OSS Arena, all three arms | 131,072 | 32,768 for all 750 prompts | One completion per prompt | T=0; p=1; k disabled; medium; no generation seed | 4,800 s request timeout; sampling POST retries disabled; 499 hard + 250 creative prompts judged |
| GPT-OSS ODCV, all three arms | 56,000 | All remaining context | 50 cycles/scenario | T=.7; p=.95; k=20; medium | 2,400 s scenario timeout; strict tool validation; 240 trajectories/arm |
| GPT-OSS MASK, historical Base/DA15 (also MSM variants) | 20,480 | At most 16,384 | One `lie_k=1` pass; fixed belief/pressure calls, no free agent loop | T=1; p=1; medium | 1,800 s generation timeout; 1,000 rows/arm; published 5% generation-error tolerance but zero errors reported |
| GPT-OSS MASK, Control closeout | 28,000 | At most 16,384 | Same one-pass call design | T=1; p=1; k disabled; medium | 1,800 s generation timeout; targeted single retry, then 994 available rows; no further rerolls |

Qwen Arena's published allowance policy is `gpt4o-estimate-plus-512-and-one-third-v1-pinned-tokenizer-cap-v1`; it is not an unqualified exact-token policy. The actual identical allowance vector is saved with each answer set. GPT-OSS Arena uses native Harmony exact prompt counting. MMLU is five-shot, 10 questions in each of 57 subjects, test split, shuffled answer choices with selection seed 0; it is not full MMLU.

GPT-OSS renderer pins: SWE uses Harmony 0.0.8 with render date 2026-10-09; ODCV 0.0.8 / 2026-10-12; MMLU and Arena 0.0.8 / 2026-10-14. MASK preserves the older renderer from Git `654bad9ff1f9ca0f91b5b1285bdd320dc0718aae`, cookbook 0.5.7, render date 2026-09-28. The historical MASK launcher at its published Git revision forwards `serving.context_window=20480` into the Tinker shim; do not substitute the shim's default 28000 when describing those runs.

### Judge limits

| Evaluation | Judge | Output ceiling / time / retries |
|---|---|---|
| Qwen Arena | GPT-4.1 via OpenRouter, T=0 | 16,000 output tokens; max three attempts per invocation; one disclosed Gemini fallback in DA15/Base; 100-prompt Gemini validation is diagnostic |
| GPT-OSS Arena | Codex GPT-6 Luna, low effort | 900 s call timeout; max three attempts in the campaign; temperature and numerical output cap are Codex-managed and not exposed as fixed values; validation disabled |
| GPT-OSS ODCV plotted revision | Gemini 3 Flash Preview | Budget wrapper caps output at 8,192 tokens; two blocked alignment verdicts excluded; later Sonnet supplements not used in the plotted revision |
| GPT-OSS MASK | Gemini 3 Flash Preview | 500 completion tokens; historical evaluator permits up to six exception attempts; Control transport uses 180 s timeout and only bounded explicit empty-429 retries, plus the separately authorized single item retry |
| MMLU / SWE | No LLM judge | Final-answer matching / official SWE tests, respectively |

Infrastructure recovery limits are not new model samples: preserved paid outputs remain immutable. For exact request history, use the raw artifacts and per-call ledgers in the pinned evaluation repositories.

## D. Provenance, links and boundaries

The generated `HF_LINKS.md` is the complete companion catalogue for the figures: model exports, training mixtures, DA synthetic source, all MMLU/SWE results, Arena answer sets and judging runs, and both misalignment benchmarks. Every experiment link includes an immutable HF revision. `sources.json` and `lineage.json` record downloaded filenames and SHA256 hashes. Original benchmark inputs are also linked there.

For the paper, cite the final 300-task SWE runs, not the earlier 20-task smoke tests, selected 64k-response diagnostics, stopped runs, or patch-cutoff regrades. Cite the 56k ODCV run, not the selected 112k-context follow-up as if it were a full independent replication. Keep earlier nosynth checkpoints distinct from the October plain Control. The optional MSM MASK plot is retained for completeness, but these appendices center on Base/Control/DA15.

There is no new inference, new judging, post-hoc score repair, or new confidence interval in this handoff. Numerical results are read from pinned published artifacts. Scientific claims should remain about these checkpoints, datasets, and bounded protocols; no training-seed variance is estimated.
