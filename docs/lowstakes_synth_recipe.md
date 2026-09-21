<!-- ABOUTME: Select and review the constitution-only low-stakes SynthDoc recipe. -->
<!-- ABOUTME: Records scientific differences, operational limits and offline validation. -->
# Low-stakes difficult advice

SynthDoc selects a document type through a YAML configuration. The new option is
`configs/data/synth/da-lowstakes-fresh.yaml`, using only the existing native operators.
The historical `da-lowstakes.yaml` still rewrites a saved corpus; this option does not.

## Step-by-step pipeline

1. Split `constitutions/claude_distilled_09_principles/constitution.md` into the same
   nine full principles as DA. No old examples or selected subclause map are inputs.
2. Generate diverse situations with DA's sympathetic legitimate goal, tempting
   norm-violating shortcut and cost to the legitimate path. Bound likely consequences
   to modest, recoverable costs or awkwardness. Retain DA's diversity machinery.
3. Draft a natural request from a human who decides and acts; the assistant advises
   in text. Preserve DA's prohibition on the user announcing the ethical lesson.
4. Refine the prompt for precision, realism and difficulty. Replace the unbounded
   instruction to raise refusal costs with a modest inconvenience or foregone advantage.
   The same small-consequence constraint applies at generation and refinement.
5. Judge the refined conversation's stakes. Keep 0 (trivial) and 1 (modest); drop
   2, 3 and unclear/unrecognised ratings before paying for answers.
6. Run DA's unchanged answer-drafting and answer-revision prompts, full target
   principle, style guidance and substantive lint. Values appear through concrete
   deliberation. Stakes grades and judge reasons are not passed into these prompts.
7. Judge stakes again using the final system/user/reasoning/response, catching any
   concrete escalation introduced by the answer. Filter before normal chat export.
   Preserve both grades and reasons in metadata, plus all normal stage snapshots.
8. Inspect complete examples, rejections and principle coverage before scaling.
   A magnitude rating does not certify honesty, reasoning quality or target fidelity.

The magnitude rubric uses the historical scale: likely consequences, not speculative
worst cases or moral wrongness. Ordinary missing details do not automatically make a
mundane situation unclear. The response-writing prompts remain byte-identical to DA.

## Commands

Offline estimate:

```powershell
uv run synth estimate --config configs/data/synth/da-lowstakes-fresh.yaml --overrides total_scenarios=18,scenarios_per_call=1
```

Native selection syntax, after spending approval and configuring a per-call budget guard:

```powershell
uv run synth run --config configs/data/synth/da-lowstakes-fresh.yaml --smoke
```

Smoke requests 18 candidates, two per principle; the normal recipe requests 716.
These are candidate counts, not guaranteed accepted row counts. Rejected rows are not
automatically replaced. HF naming/publication and snapshots use the native engine.

## Differences and limits

- All model roles use Sonnet. Former Haiku generation roles use Sonnet with hidden
  thinking disabled; the answer still writes the explicit supervised reasoning field.
  Existing Sonnet refinement settings are retained.
- Advice-to-a-human scope is an additional explicit restriction. Comparing against
  unrestricted DA changes actor scope as well as stakes; that is not a pure stakes effect.
- Diversity replacement rounds and content-lint re-rolls are disabled. Substantive
  lint criteria and corpus checks are retained. Native JSON/tag parse attempts remain
  bounded at three, with up to six shared transport attempts per transient request
  failure. This is not zero-retry physical dispatch. Do not enable batch mop-up/topup.
- The 25% filter-drop alarm is an operational threshold, not a validated quality bar.
  The existing filter applies it only at 20 or more inputs; inspect smoke yield directly.
- Native `budget_usd` is checked between stages. Before this campaign's live run,
  inject the existing shared `BudgetClient` through `pipeline.run(..., client=...)`.
  Keep the cumulative ledger. The YAML's $8 is neither a fresh allowance nor a strict
  per-call ceiling. Existing spend is $7.786718; the $20 total request is pending.
- Estimates use assumptions and can price corpus scans that small-sample checks skip.
  They are not actual spending or cost reservations.

## Validation status

Implemented and **offline-tested; no live generation yet**.

```powershell
uv run --no-sync python -m pytest -q scratch/dataset_refresh/test_native_lowstakes.py
```

Three tests pass: DA-stage parity; consistent prompt constraints; and full native-engine
execution with mocked external services. The latter starts 18 scenarios, blocks serious,
unclear and unrecognised prompt ratings, blocks final escalation, exhausts a malformed
judgment's bounded parse attempts, and exports only 12 eligible test rows. Stage snapshots
and failure accounting are checked. These test rows are not research data.

Combined DA regression checks: 20 pass and one pre-existing repository-wide failure:
`nonmoral-advice.yaml` sets 716 candidates without a `smoke.total_scenarios` override.
That file is unchanged. This new recipe explicitly has the 18-candidate override.
