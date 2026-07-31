---
license: apache-2.0
tags:
  - petri
  - alignment
  - automated-audit
  - dose-response
  - constitution
  - null-result
language:
  - en
pretty_name: "Petri constitution audit — Qwen3.6-27B difficult-advice SFT dose sweep"
---

# Petri constitution audit — Qwen3.6-27B difficult-advice SFT dose sweep

**Headline: a null result.** Violation frequency against the constitution these models
were trained on was **20% / 20% / 40% / 30%** for 0% / 10% / 20% / 40% difficult-advice
SFT. There is no dose-response, the nominal trend is *upward*, and at n=10 test audits
per arm no arm differs from base (McNemar exact p = 1, 0.625, 1).

![dose response](./assets/violation_dose_response.png)

Violation families and the paired comparison against base:

![decomposition](./assets/violation_decomposition.png)

| field | value |
| --- | --- |
| `experiment` | Adaptive Petri audit measuring how often four Qwen3.6-27B arms violate the constitution their SFT data was written against, as a function of SFT dose |
| `date_generated` | 2026-07-31 |
| `constitution` | [`docs/claude_constitution_principles.md`](https://github.com/) — "Distilled Constitutional Principles (alignment target)", v1, the **same document the adapters were trained against**. Deliberately *not* the later `claude_approved_constitution.md`, which postdates the training data. |
| `source_repo` | `teaching_claude_why_replication`, branch `petri-constitution-audit`, commit recorded in `artifacts/results.json` |
| `models` | target `Qwen/Qwen3.6-27B` (bf16, vLLM 0.26.0) + LoRA arms `LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-{10-90,20-80,40-60}` (r=32); auditor `claude-sonnet-4-5` via Claude Agent SDK; judge `claude-sonnet-4-5` via Anthropic API; realism `claude-haiku-4-5` |
| `generation_config` | target temperature 0.7, `max_tokens` 4096, `max_model_len` 65536; Petri `max_turns=12`, `realism_filter=0.6`, `enable_rollback=True`, `enable_prefill=False`, `target_tools=synthetic`; 1 epoch |
| `schema` | see below |
| `provenance` | `scripts/Run-Grid.ps1 -Epochs 1 -MaxConnections 4 -Tag grid-e1`, then `scripts/rejudge.py`, `scripts/analyse.py`, `scripts/plot_violation_curve.py`, `scripts/build_export.py` |

## Why the numbers are what they are

The 20/80 arm has a **published ODCV-Bench result of 37.2% → 19.2%** misalignment. This
audit did not reproduce a benefit for it. Before reading that as a contradiction, note
the power: base is 20% with a 95% Clopper-Pearson interval of **[2.5%, 55.6%]**. This
design resolves differences of roughly 30 percentage points. It is not capable of
detecting an 18-point effect, so it neither confirms nor refutes the published result.

Two findings that are not power-limited:

- **Unhelpfulness-side violations were 0% in all four arms.** More SFT did not make these
  models more refusing, moralising or rule-matching. That trade-off simply did not appear.
- **Eval-awareness was 1/10 on every retained transcript.** Targets never indicated they
  suspected a test, so the transcripts describe deployment-like behaviour.

## Schema

`data/scenarios.jsonl` — one per seed (12)

| field | meaning |
| --- | --- |
| `id` | seed slug, e.g. `P2-route-around-the-check` |
| `category` | `harm`, `unhelpfulness`, or `control` |
| `hypothesis` / `expected_signal` | the violation mode under test |
| `seed_instruction` | the brief given to the auditor (shared hard-constraints block stripped) |
| `principle` | v1 constitution principle number |
| `shape` | `advice` or `agentic` |
| `control` | true for the two control seeds |

`results/transcripts.jsonl` — one per retained audit (44)

| field | meaning |
| --- | --- |
| `id` | `<arm>__<seed>` |
| `scenario_id` | joins to `scenarios.jsonl[].id` |
| `category` | **the arm** — `base`, `dose-10-90`, `dose-20-80`, `dose-40-60` |
| `outcome` | `pass`, `concerning` (any violation ≥5/10), or `eval-aware` |
| `messages` | the full target-side conversation |
| `scores` | 0–1 rescaled: four standard keys plus all nine per-principle dimensions |
| `judge_summary`, `tags` | which dimensions fired, plus arm/family/principle tags |

`results/scores.json` — per-arm aggregates.
`artifacts/raw-judge-dimensions.jsonl` — **raw 1–10 judge integers** before rescaling.
`artifacts/report.md`, `artifacts/results.json` — intervals, per-principle rates, McNemar.

## Caveats

1. **These are judge flags, not adjudicated findings.** A comparable flagging rule had a
   **57% false-positive rate** in the sibling MSM investigation. Adjudication would lower
   all four rates; it cannot create an absent dose-response.
2. **n = 10 test audits per arm**, one epoch. The dominant limitation.
3. **Controls are n = 1–2 per arm** after retention — too few to estimate the rubric's own
   false-positive rate, which was their purpose.
4. **Audits ran to a mean of 3.2 target turns** against a 5-turn brief, so pressure was
   lighter than designed (uniformly across arms).
5. **Runtime LoRA, not merged weights** — module coverage verified (256/256), numerics not
   compared against a merged checkpoint.
6. **11 of 48 audits lost in-run judge scores** to a Claude Code CLI turn limit; all arms
   were then re-judged uniformly on the API, which also removed judge-transport variance.

## Reproduce

```bash
scripts\Run-Grid.ps1 -Epochs 1 -MaxConnections 4 -Tag grid-e1
python scripts/rejudge.py --logs logs/grid-e1 --out output/rejudged
python scripts/analyse.py --rejudged output/rejudged --out output/analysis
python scripts/plot_violation_curve.py --results output/analysis/results.json --out output/analysis
```
