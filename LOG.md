<!-- ABOUTME: Append-only experiment log (most recent first) for the replication. -->
<!-- ABOUTME: Each entry: hypothesis -> method -> result -> next steps. -->

# LOG

## 2026-07-27 (pm-3) — Independent Inspect-harness cross-check (leaking)

Re-provisioned an H100, served base+adapter (adapter from HF `matboz/qwen3-32b-difficult-advice-lora`),
SSH-tunneled to the PC, and ran `inspect_evals/agentic_misalignment` (leaking, gemini-3-flash grader,
30 epochs, prod, test_eval_awareness) on base vs fine-tune, two goal conditions:

| condition | base | fine-tune | rel. |
|---|---|---|---|
| explicit/america | 66.7% | 13.3% | −80% |
| ambiguous/none | 40.0% | 3.3% | −92% |

Agrees with our vendored harness (74%→28% explicit/america). Benefit is *larger* in the harder OOD
ambiguous condition — strong generalization. Datasets published to HF `matboz/difficult-advice-qwen3`;
README updated with a skip-data-gen path. Both instances destroyed. See output/inspect/.

## 2026-07-27 (pm-2) — Run 2 (thinking-format fix): strong reduction + reasoning preserved

**Fix applied (user-approved):** augmented all 2119 examples with a real first-person
`<think>` deliberation via Sonnet 4.5 (`reasoning_content` field → renders as a real think
block, not empty). Retrained QLoRA (loss 1.81→1.40), served base+adapter, re-evaluated in
**thinking mode** vs the thinking baseline (19.3%).

**Headline result (thinking mode, 600 samples/condition):**
- Baseline **19.3%** → + difficult-advice SFT **8.0%** overall = **−11.3 pts, −58.5% relative.**
- Leaking 38% → 16%; blackmail 0% throughout. Every leaking condition dropped.
- Compare Run 1 (non-thinking answers): 15.0%→12.7% = only −15.3% relative.
- => Training the *value reasoning inside `<think>`* transfers ~4x better to the agentic
  honeypots (model reasons-then-acts). Faithful to "teaching why": the reasoning is the mechanism.

**Reasoning preserved (probe, thinking mode):** base `<think>` 2.9-6.6k chars; Run 1 LoRA = 0
(collapsed); Run 2 LoRA = 0.9-1.6k chars real reasoning, answers correct. Channel restored.
See output/reasoning_probe_thinking_train.txt, report output/report/final_*/.

## 2026-07-27 (pm) — Run 1 (1.5M OOD tokens, QLoRA) results + reasoning-collapse finding

**Alignment (non-thinking mode, 600 samples/condition each).**
- Baseline: 90/599 = **15.0%** overall (leaking 30.1%, blackmail 0%).
- + difficult-advice SFT: 76/597 = **12.7%** overall (leaking 25.6%, blackmail 0%).
- **−2.3 pts / 15.3% relative reduction**, concentrated in mid-severity leaking conditions;
  strongest goal-conflict conditions barely moved. Reference thinking-mode baseline: 19.3%.
- Verdict: OOD difficult-advice data *does* reduce agentic misalignment on Qwen3-32B, but
  modestly at this scale (vs the article's near-zero via full character training on Claude).

**Reasoning collapse (CONFIRMED).** Qwen3's chat template wraps each SFT target as
`<think>\n\n</think>\n\n{answer}`; training on 2119 such empties taught the model to emit an
EMPTY think block. Probe (thinking mode, base vs LoRA): base `<think>` = 2.4k-6.9k chars;
LoRA `<think>` = **0 chars on every prompt**. Answers still correct with inline steps, but the
extended-CoT channel is disabled. See output/reasoning_probe_nothink_train.txt.

**Proposed fix (pending user approval):** regenerate difficult-advice responses in Qwen3
thinking format — value/ethics deliberation inside `<think>`, final advice after — optionally
mixing ~15% general reasoning traces; retrain; re-eval in thinking mode vs the 19.3% baseline.

## 2026-07-27 (am) — End-to-end pipeline stood up; baseline shows non-zero misalignment

**Goal.** Replicate the "difficult advice" section of Anthropic's *Teaching Claude Why*
on Qwen3-32B: generate OOD difficult-advice SFT data with **Sonnet 4.5** (via OpenRouter,
no Anthropic key available), LoRA-SFT Qwen3-32B, and measure agentic-misalignment
(blackmail/leaking honeypots) before/after.

**Method.**
- Data gen (`src/experiments/generate_difficult_advice.py`): Sonnet 4.5 generates diverse
  user-in-dilemma scenarios across 18 domains → constitution-aligned responses with open
  value deliberation → Sonnet 4.5 grades/filters. Target ~1.5M Qwen tokens (user-approved
  scale; ~$46 of OpenRouter credit).
- Eval: Anthropic `agentic-misalignment` repo, patched with a `vllm/` provider
  (`third_party/.../api_client/model_client.py`); judge = Sonnet 4.5 via OpenRouter.
  Aggregated with `src/experiments/aggregate_eval.py` (new dir layout).
- GPU: 1× H100 SXM 80GB on vast.ai; vLLM serves Qwen3-32B bf16 (thinking mode on).

**Results so far.**
- Data-gen pilot (180 scen, 18 domains): 0 errors, 98% acceptance, 721 tok/example,
  high quality (empathetic, declines norm-violation on concrete stakes, non-preachy).
- Eval smoke (Qwen3-32B baseline, 3/condition): **blackmail 0/18, leaking 8/18 (44%)** —
  non-zero baseline confirmed; leaking under goal-conflict up to 100%.
- Fixed a harness bug: `_detect_provider` matched substring "claude" → Anthropic before
  the `/`-prefix rule, misrouting the OpenRouter judge.

**Next steps.**
- Finish full 1.5M-token data gen; copy to instance.
- Full baseline eval (50/condition) → LoRA SFT → post eval → report/dashboard.
