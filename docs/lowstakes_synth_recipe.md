<!-- ABOUTME: Select and review the constitution-only low-stakes SynthDoc recipe. -->
<!-- ABOUTME: Records scientific differences, operational limits and offline validation. -->
# Low-stakes difficult advice

**Current candidate:** `configs/data/synth/da-lowstakes-practical.yaml` on
`codex/lowstakes-synthdoc-pipeline`. Nine preregistered hobby domains are crossed
with the nine full constitutional principles; each source call sees only one
trait/domain pair. Source and judge reasoning are enabled with bounded allocations.
Native DA answer prompts remain unchanged. Admission checks decision stakes and
text-advice scope; domain adherence is diagnostic, not a behavioral veto.

**Current recommendation: proceed with one bounded full-generation batch.** The
36-case smoke exported 32 rows across all nine traits and domains. All were fully
read; no clear high-stakes decisions were found in the exported prompts. The stricter
preregistered domain criteria did not pass unchanged: a tested domain veto reduced
this to 27 and erroneously rejected intended dishonesty/oversight conflicts. That
veto was removed, with the explicit policy change documented rather than called a
preregistered pass. The final policy was checked on saved real outputs; no third
paid generation followed. Shared normal-DA factual/rationalization defects remain.

Cost: $3.822942 for generation plus the same-input judge experiment; cumulative
$17.565140/$20. Offline: 77 tests pass. Full config produces 972 candidates
(9 traits x 9 domains x 12), not a guaranteed 716 accepted rows. Full-run budget and
deterministic 716-row selection must be set before launch; no full generation/SFT ran.
[Full report](dataset_audits/2026-09-21_domain_smoke.md).
[Preregistration](dataset_audits/2026-09-21_domain_preregistration.md).
[Verified complete archive](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-practical-synth-smoke/tree/febfd68c1f2e24d0a10a1f50264849329a588d15/runs/20260921_144723).

The guarded launcher selects the current candidate. Older frozen configs and reports
preserve earlier versions. The remaining sections describe the original historical
variant, including its now-removed final-answer stakes judge, not today's candidate.

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
  bounded at three. The guarded campaign launcher disables transport retries.
  Do not enable batch mop-up/topup.
- The 25% filter-drop alarm is an operational threshold, not a validated quality bar.
  The existing filter applies it only at 20 or more inputs; inspect smoke yield directly.
- Native `budget_usd` is checked between stages. This campaign uses the shared
  per-call ledger through `scratch/dataset_refresh/run_native_smoke.py` and
  `scratch/dataset_refresh/native_lowstakes_smoke.yaml`. The user approved $20 total,
  including $7.786718 already spent. The launcher's ceiling overrides the recipe's
  soft $8 setting; neither constitutes a fresh allowance on resume.
- Estimates use assumptions and can price corpus scans that small-sample checks skip.
  They are not actual spending or cost reservations.

## Validation status

Implemented, offline-tested, and **live smoke completed on 2026-09-21; content
validation failed**. Eighteen candidates yielded ten automatic exports. Both stakes
filters admitted weeks of severance pay and a $1,800 dispute; full reads found material
invented facts. Do not treat the engine's completed/PASS status as approval to scale.
[Live report](dataset_audits/2026-09-21_native_lowstakes_smoke.md) and
[same-standard old-corpus check](dataset_audits/2026-09-21_old_lowstakes_factual_check.md).
The latter confirms closely matching defects in the old corpus; no relative error
rate or causal explanation of ODCV follows from these example reads.

Campaign launcher (preserves prior spend and enforces a per-call ceiling):

```powershell
uv run --no-sync python -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_smoke.yaml
```

The completed run used $1.301356 in settled charges plus a $0.0262675 retained
reservation after a Windows settlement-file error, $1.3276235 total exposure.
Cumulative campaign exposure is $9.1143415 against the approved $20 ceiling.
Resume reused saved stages and excluded the lost response without redispatch.
The launcher supports `--resume <run_root>` with frozen configuration checks;
existing completed runs must not be regenerated to obtain a different sample.

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

Two additional recovery tests pass: a transient Windows atomic-replace error retries
only the filesystem operation, and an uncertain prior request is neither resent nor
unreserved. Combined with the three native tests: five pass.
