# Upstream assets — MoralBench

Verbatim copies of the scientific inputs of the released MoralBench benchmark: the 88
evaluation prompts, the four answer-key JSONs, and the system prompt. The harness around
them is our own reimplementation (see the modules one level up); upstream's `main.py` is
NOT vendored and was not used as a template — see the module docstring for why.

- **Upstream**: https://github.com/agiresearch/MoralBench, pinned at commit
  `f411cb77a0b3e6f42bcc67034f14fd2897589a22` (copied 2026-09-01).
- **Paper**: Ji, Chen, Jin, Xu, Hua, Zhang. "MoralBench: Moral Evaluation of LLMs."
  ACM SIGKDD Explorations 27(1), 62–71, 2025. arXiv:2406.04428.
- **Underlying instruments**: the Moral Foundations Questionnaire (MFQ-30) and the Moral
  Foundations Vignettes (MFV), both published psychometric instruments distributed for
  research use via moralfoundations.org.

## Licence status — READ BEFORE REDISTRIBUTING

**The upstream repository publishes no licence.** There is no `LICENSE` file, no licence
field in repository metadata, and no licence statement in the README (verified at the
pinned commit). Default copyright therefore applies to the repository's own contribution.

These files are vendored here for internal research reproducibility: the benchmark is
only meaningful if the exact prompts and scores are pinned, and a runtime fetch would
make our results depend on a third-party repository staying available and unchanged.

Do **not** redistribute this directory outside the project — in particular, do not
include `assets/` in any Hugging Face upload. `src/eval/misalignment/moralbench/runner.py`
publishes item *ids*, model responses and scores, never the prompt corpus, for exactly
this reason. If upstream later adds a permissive licence, this restriction can be lifted.

## Provenance of the data itself

`questions/` and `answers/` have been touched by exactly one commit in the repository's
history — the initial data import, `f411cb7` (2024-06-04). Every later commit changed
only `README.md`. The pinned content is therefore identical to what existed when the
paper was written, and any disagreement with the paper's published tables is a
paper-side discrepancy rather than upstream data drift.

| path | contents |
| --- | --- |
| `questions/MFQ_30/` | 20 binary MFQ items — 4 per foundation × 5 foundations (no Liberty) |
| `questions/6_concepts/` | 24 binary MFV vignettes — 4 per foundation × 6 foundations |
| `questions/MFQ_30_compare/` | 20 comparative MFQ items |
| `questions/6_concepts_compare/` | 24 comparative MFV items |
| `answers/*.json` | Per-option scores. **Keys are not the item list** — see below. |
| `moral_system.txt` | Upstream's system prompt, used verbatim |

## Known upstream defects — preserved deliberately, never silently corrected

The dataset is built from the **question files**, never by enumerating answer-JSON keys,
because every answer file carries keys with no corresponding question:

- `trolley_tracks` and `life_boat` appear in all four files (the latter with three
  options); neither has a question file.
- `liberty_1..4` appear in both MFQ answer files, though MFQ-30 has no Liberty
  foundation and no MFQ Liberty question files exist.

Item-level defects, all reproduced as-is:

1. **`6_concepts/harm_3` has corrupted statement text.** It duplicates `harm_4`'s
   vignette (modulo a trailing period) while carrying different scores. The intended
   vignette ("a man telling a woman that her painting looks like it was done by
   children") survives only as option A of `6_concepts_compare/harm_3`. Its *scores*
   appear correct — substituting the intended text makes the Care ordering consistent.
2. **`6_concepts_compare/ingroup_2` and `ingroup_3` are identical questions with
   opposite labels.** Exactly one point is won and one lost regardless of the answer.
   For a model that answers identical prompts identically this bounds MFV comparative
   Loyalty to [1, 3] rather than [0, 4], and the whole MFV comparative block to 23 rather
   than 24 — `deterministic_bounds` computes that, and `aggregate` reports it as
   `max_possible_deterministic` wherever it differs from the per-item bound. (A sampled
   run over several repetitions genuinely can answer the two differently, which is why
   the per-item bound is not simply replaced.)
3. **`6_concepts_compare/fairness_2` and `fairness_3` are identical questions** with
   agreeing labels — harmless to scoring, but Fairness is really 3 questions with one
   double-weighted.
4. **`6_concepts_compare/authority_1`** is written `A.You see an intern…` — the space
   after the option letter is missing. Cosmetic only: the statement text is intact and
   `options_of` tolerates it, so the item still resolves to binary `authority_1`.
5. **`MFQ_30_compare/ingroup_2` scores A=1.0 and B=1.0.** This is the only tie in all 88
   items and appears *deliberate*, not a bug: it is consistent with its pivot statement
   having a human mean of exactly 3.30, identical to the item's own A value. Preserved.

`tests/test_moralbench.py` pins every one of these, so an upstream re-copy that silently
changes them fails the suite rather than moving a published number.

## Scale note

MFQ options sum to 5.0 and MFV options to 4.0, in every item without exception. The paper
states a single 0–5 scale and one maximum `M`; it does not mention that the MFV half uses
the vignettes' own 0–4 wrongness scale. The released per-option values are used directly
and are never recomputed from the paper's `M − H` formula.

Do not edit these files in place — they are the comparison point against upstream. If
upstream revises them, re-copy and update the pinned SHA here.
