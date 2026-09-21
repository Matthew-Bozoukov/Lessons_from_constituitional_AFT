<!-- ABOUTME: Prospective design critique and live validation of the practical low-stakes SynthDoc variant. -->
<!-- ABOUTME: Separates mechanical integration, independent content review and untested ODCV transfer. -->
# Practical low-stakes SynthDoc variant

## Frozen plan, before paid dispatch

Select `configs/data/synth/da-lowstakes-practical.yaml`. The earlier minimal
`da-lowstakes-fresh.yaml` is retained unchanged for reproducibility. These are
configuration variants of one SynthDoc engine, not separate generation programs.
Generation uses only the new nine-principle constitution and the recipe. There are
no old examples, selected domain seeds, evaluator scenarios or curated subclause map.

1. Use native principle segmentation and diverse scenario generation.
2. Bound likely consequences while retaining a worthwhile goal, tempting shortcut
   and genuine cost to the principled route. Do not substitute toy/display stories
   or a harmlessness disclaimer for a real low-stakes dilemma.
3. Draft and refine a practical request. The human carries out the activity; the
   assistant can supply consequential text, plans or advice. Preserve plausible
   operator context where relevant. The user must not provide the ethical diagnosis.
4. One prompt-review call rates magnitude and practical-task admission separately.
   Native filters require both stakes 0/1 and an explicit keep. Unknown labels drop.
5. Native DA answer drafting and revision, with the original principle, style and
   length/identity lint. Add one shared grounding instruction: preserve source facts
   and unresolved costs; do not invent permission, actions, motives or guaranteed fixes.
6. One final-review call checks both reasoning and answer against system/user inputs.
   Reject material unsupported claims, contradictions, distorted intent, rule recital,
   absent useful deliberation or costs erased by invented facts. Qualified inferences,
   explicit hypotheticals, suggested options and placeholders remain allowed.
7. Native filter/export, source snapshots and corpus diagnostics. Review metadata
   is never inserted into training messages. Both reviews and rejected cases persist.

All paid model roles are Sonnet. One draft and one revision per accepted prompt;
no substantive lint rerolls, replacement rounds, quota top-ups or repair loops.
Native JSON/tag parsing still allows at most three attempts. Shared BudgetClient
disables transport retries, reserves before each physical dispatch and preserves
uncertain charges. The cumulative $20 authorization includes $9.1143415 prior
exposure. The new run receives no separate fresh allowance.

## Critique before observing outputs

- Practical instructions may still produce ethics quizzes. Count literal cues, but
  decide from complete requests; a natural 'should I' is not itself a failure.
- Human-facing assistance is not the full range of direct agentic action in old DA.
  Text drafting preserves responsibility without pretending this is a stakes-only
  replication. Future ODCV benefit remains a hypothesis.
- Same-family reviewers can share the generator's blind spots. Separate calls do
  not establish independence. Full manual reads must check false accepts AND rejects.
- A magnitude rater can confuse 'not catastrophic' with 'small'. Weeks of wages and
  substantial invoices are explicit boundaries, while speculative disasters are not.
- Materiality is judgment-dependent. Only reject clear decision-changing inventions,
  usable drafts containing false facts, or substantive contract failures; do not
  keep tightening a perfection standard until nothing remains.
- The inherited 700-character floor can exclude useful short answers. Retain it
  here to limit changes, but report length-only losses rather than quality failures.
- Diversity controls still cannot prove diverse underlying decisions. Read the
  mechanisms, not just labels or embeddings. With two rows per principle, a smoke
  also cannot validate full-scale diversity or acceptance rates precisely.
- 716 means requested candidates, not promised exports. Full generation would need
  an explicit finite candidate budget and acceptance policy, not an endless refill.

## Prospective smoke criteria

Eighteen candidates, two per principle; no replacements. Require at least 14 accepted
examples, at least one per principle, no clear material false accepts on complete
reads, and no single decision mechanism exceeding one third of admitted examples.
Read every refined prompt and revised answer, plus rejected judgments, and report
generator, automatic-filter and independent-review results separately. These are
operational screening criteria, not statistical certification or proof of MR benefit.

Offline: five native/recovery checks pass. The combined baseline check has twelve
passes and the previously documented unrelated `nonmoral-advice.yaml` smoke-size
failure. Full mock execution checks both magnitude and quality filters, unrecognised
labels, pre-authoring rejection, malformed judgment handling, metadata separation,
and preservation of failure receipts. No shared engine behavior was changed.

Run:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_smoke.yaml
```

Live outcome pending. No full generation, SFT or ODCV is authorized by this smoke.
