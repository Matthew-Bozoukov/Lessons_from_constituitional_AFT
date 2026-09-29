<!-- ABOUTME: Records the native DA advice-boundary recipe change and its blocked eighteen-row smoke. -->
<!-- ABOUTME: Distinguishes offline validation, provider failure, cost reservations and untested corpus quality. -->

# DA advice-boundary pipeline smoke, 2026-09-29

**Status: recipe implemented; live smoke blocked before any scenario was generated.**
No completed dataset, mixture, training or model evaluation was produced.

## Recipe change

The data-generating changes are in the existing
[`configs/data/synth/da.yaml`](../../configs/data/synth/da.yaml), not edits to generated
rows. Its shared generation contract reaches scenario writing, prompt drafting and
prompt revision through OmegaConf interpolation. The native engine, model split,
constitution, response format, per-row lint, scenario dedup and export are retained.

- A human with a legitimate goal faces a consequential moral gray area and asks for
  advice. The human owns the decision and action.
- The assistant's deployment system is benign, general and advisory. No malicious
  system directives, institutional agenda, jailbreak, persona attack, fictitious
  tools/access or transfer of the consequential choice to the assistant.
- AI may be the subject of advice where the actual target principle needs that
  relationship. Preserve its scope and exceptions; do not replace an AI-specific
  clause with generic human obedience or personality. Do not force unrelated
  dilemmas into AI-company settings.
- Prompt revision repairs violations of this contract even if a draft already
  directly tests the principle. The former instruction to preserve such a draft
  could preserve the wrong decision-maker.
- Response drafting and revision retain grounded moral weighing, permissions,
  timelines, uncertainty and residual costs. Proposed alternatives remain
  conditional where their availability or success is not established in the prompt.

Current main already lacked September 28's blanket human-analogue/AI-exclusion
instructions. This change therefore strengthens the advice boundary in the current
recipe; it does not claim to have removed that block from this checkout. The
[historical boundary investigation](2026-09-29_da_actor_boundary.md) explains the
distinction. The shared text is generation guidance, not an extra constitutional
principle or text exported in the assistant's system prompt.

## Native smoke and validation

The existing guarded native runner received a small extension for the normal DA
recipe, its Haiku/Sonnet model allowlist and bounded smoke sizing. Normal DA is
explicitly rejected in this runner's full mode. It still calls the standard
`pipeline.run` and shared per-request `BudgetClient`; no second generation engine
or offline data repair was introduced.

Frozen source: `144cf968fb6883e491de09887c583297efa63694`, following the merge of
`origin/main` at `b669a2b2` into the isolated `codex/da-refresh-20260929` worktree.
Other worktrees and main were not changed.

Launch: [`native_da_advice_smoke.yaml`](../../scratch/dataset_refresh/native_da_advice_smoke.yaml).
It requests **18 conversations, two per principle**, seed 0, nine workers, with
unchanged Haiku 4.5 drafting and Sonnet 5 refinement through their Anthropic provider
pins. The guard permits at most **$8 including retained failure reservations** and
180 physical calls. The original $30 investigation ceiling includes $5.9842455
already spent, so even maximum smoke exposure remains below the parent ceiling.

The final 30-scan style survey is explicitly ablated for this small diagnostic;
scenario dedup remains enabled. The intended quality review is a full read of every
exported conversation, retaining all failures and allowing no hand edits or rerolls
to obtain a pass. That review could not occur because generation never started
successfully.

Offline checks:

- All resolved stage prompts format successfully; native stage construction passes.
- The launch resolves to nine principles and 18 scenarios, rather than the standing
  production count of 2,000. Full mode and an incomplete model allowlist are rejected.
- The targeted recipe/diversity/pipeline suite reports **31 passed, 2 failed**.
  Both failing assertions also disagree with the pre-change HEAD: one expects no
  rewrite `audit` tag, and one assumes every recipe's smoke is at most 20 rows while
  `da-lowstakes-practical.yaml` already specifies 36. Neither was weakened or changed.
- `git diff --check` passes. Live provider metadata matched recorded pricing before
  dispatch: Haiku $1/$5 and Sonnet $2/$10 per million input/output tokens.

## What happened and accounting

The native run `20260929_142716` segmented all nine principles, then all nine initial
scenario requests failed with HTTP **503/529**, reporting temporary unavailability
of Anthropic's user-profiles service or service overload. The native failure alarm
halted the run after 8.1 seconds. A tiny Haiku health check and a tiny Sonnet health
check subsequently failed the same way. No provider/model fallback was used.

- Generated scenarios: **0 / 18**. Completed conversations: **0 / 18**.
- Physical requests: **11** (9 generation attempts, 2 health checks).
- New settled/billed cost confirmed by response receipts: **$0**; none returned usage.
- Retained unknown-cost reservations: **$0.457890**. They are not asserted to be free
  or to be actual charges, and were not released.
- Prior investigation actual spend plus current conservative exposure:
  **$6.4421355 / $30**.

The native manifest's zero usage is not a complete cost account for failed requests;
the durable budget ledger is authoritative. The failure manifest and nine principle
records are on [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-29-da-synth-smoke),
with a status report and sanitized accounting receipts. There is no `dataset.jsonl`.

No claims about generated quality, downstream ODCV/MASK recovery or production
readiness follow from this attempt. The next step remains the same bounded smoke
once the pinned provider is usable, followed by full reading of its outputs. Keep
the existing failure receipts and reservations when recovering. No full generation
is authorized by this smoke request.
