# Nosynth reference-answer audit — 2026-09-08

## Dataset and scope

Audited the actual 10,000-row `LASR-Callum/2026-09-05-nosynth-mix` snapshot
`a517e99ba4cfd45189ef2fafc7653ad6162a1ebe`, not upstream datasets or a new sample.
SHA-256: `3116e34ff1d832a35ddaf539ebee93e47f904e4712c2076c09116d3b4d3f9aa4`.
The local enriched copy at
`output/reasoning_backfill/enrich_20260908_114055/mixture.jsonl` has identical
sources, message roles and content, in the same order. Adding reasoning did not
repair these reference answers.

All row IDs below are **zero-based mixture indices**; JSONL line = ID + 1.
No training examples were edited. No new model/API calls were made.

## Pronoun claim: verified, with two corrections

- 984 `smol_summarize` rows: verified.
- 644 contain the exact system instruction ending “without using second or third person pronouns”: verified.
- 489 contain prohibited personal/possessive/reflexive forms: verified by case-insensitive whole-word matching against assistant answers only.
- The quoted **443 personal + 46 it-only** split is reproducible with a restricted personal-word list: `you your he him his she her they them their`.
- Expanding this to include `yours yourself yourselves himself hers herself theirs themselves` gives **444 + 45**. Row **9871** contains “himself” as well as “its”; it is not genuinely it-only. The total remains 489.
- **489 / 10,000 = 4.89%**, not ~6%. It is 49.70% of this source, or 75.93% of the 644 constrained rows. The 644 constrained rows themselves are 6.44% of the blend.

“Personal” here is the non-it bucket, not a claim that *it* is not a personal
pronoun. This is a lexical audit, not a complete grammatical parser: e.g. it
does not classify relative/indefinite pronouns. Possessive forms such as *their*
are included, as in the quoted claim.

## Findings across the other sources

| Source | Rows | Findings |
|---|---:|---|
| `smol_constraints` | 1,055 | Clear missed word limits and keyword frequencies; also contradictory prompts, which must not automatically be labelled answer failures. |
| `tulu3_if` | 1,471 | Clear missed exact word counts and keyword frequencies. Some formatting requests apply per section or to JSON fields, so naive whole-answer validators overcount failures. |
| `apigen_function_calling` | 1,054 | 64 prose refusals contradict the strict tagged-list output instruction. 80 additional rows fail the implemented explicit JSON-schema type checks. |
| `self_oss_instruct` | 1,064 | Confirmed single-loop, list-comprehension, return-value, caching and constant-space requirements violated. |
| `no_robots` | 2,779 | Confirmed forbidden-word and minimum-length failures. |
| `numinamath_cot` | 1,063 | At least three plainly incorrect worked answers in spot-checks; correctness defects, not the same phenomenon as ignoring a formatting instruction. |
| `lima` | 314 | No firm comparable instruction-following defect established by this limited check. A prior trace-consistency rejection of a creative story is not evidence that the reference story is wrong. |
| `longalign` | 216 | No firm comparable instruction-following defect established by this limited check. Context-grounded QA needs semantic verification, not just formatting checks. |

These are **not source-wide error-rate estimates**. All rows received structural
screening; only recognisable constraints received deterministic checks, with
targeted manual review and a few execution tests. Maths/code/reference grounding
were not exhaustively verified. No empty assistant answers were found.

### Smol Constraints: concrete examples

- **4015:** “less than 300 words”; reference has **442 whitespace-delimited words** (443 with the alternate lexical counter).
- **9222:** “exactly 100 words”; reference has **83** under both counters.
- **4531:** “at least 150 words”; reference has **134–135**.
- **9362:** word “cat” at least three times; reference contains **zero whole-word `cat`** occurrences. Inflected `cats` is not the requested exact word.
- **4926 — do not count as a bad answer:** prompt simultaneously requires at least 300 and fewer than 250 words. Reference correctly identifies the contradiction and asks which to prioritise.
- **4677 — bad/contradictory prompt:** requires both including and excluding the same keyword, `problem`. This needs prompt review, not blind regeneration or rejection based on one rule.
- **1756 — ambiguous, not confirmed:** exactly two placeholders; answer has three occurrences but only two distinct placeholder names. Occurrence counting alone does not settle compliance.

### Tulu instruction-following: concrete examples

- **2592:** description must be exactly 150 words; reference has **100**.
- **3040:** guide must be exactly 150 words; reference has **100**.
- **3418:** `DERMATOLOGIST` exactly four times; reference has **one** case-insensitive whole-word occurrence.
- **5139:** `passenger` exactly five times; reference has **two**.
- **6211:** no more than five all-capital words; JSON answer contains nine uppercase feedback strings, totalling **39** uppercase words.
- **5526 — requires judgment:** requested current-event bullet list, but reference refuses because it lacks current information. This is not equivalent to carelessly ignoring a constraint.
- JSON bullet representations and per-section requirements were excluded from generic Markdown-bullet counting where recognised. Remaining flags are still review candidates.

### API function calling: systematic format/type inconsistencies

All 1,054 system prompts prescribe `<tool_call>[...]</tool_call>` with no
additional text, including an empty list when no call is needed.
**64** references are instead these untagged prose sentences:

- 30: “The query cannot be answered with the provided tools.”
- 17: “The given question lacks the parameters required by the function.”
- 17: “The query cannot be answered, no tools were provided.”

The same system prompt also says to “point it out and refuse” when tools cannot
answer. This is a **prompt/template inconsistency**. Align the refusal rule and
output schema together; do not simply call all refusals bad behaviour.

The remaining **990** references parse as tagged JSON lists, and all called
function names are present in the supplied tools. The implemented schema checker
covered **426 rows with calls and explicit object schemas**; **80 rows** have
type mismatches. It supports primitive/union types, nullable values, nested
properties/items, required fields, enums and explicit additional-property bans;
it is not a full JSON Schema implementation. The other schema dialect (plain
parameter maps with Python-style type names) was not type-validated.

Example **138**: `input_dict` is declared an object, but the reference supplies
`"{'id': '12345', 'name': 'John Doe', 'age': '30'}"`, a **string containing a
Python dictionary literal**, not an object. Other references likewise stringify
arrays, booleans and integers. Some tool descriptions/schemas themselves disagree,
so these are confirmed schema mismatches, not all independently adjudicated
mistakes in the answer. The 64 format and 80 schema rows are disjoint: **144
prompt–answer format/type mismatches**, not 144 proven semantic tool-use failures.

### Self OSS Instruct: explicit requirements not implemented

- **713:** asks for set operations in a **single loop**; answer contains two sequential `for` loops.
- **781:** asks for a **list comprehension**; answer uses none. Additionally, `max(dict, key=lambda k: dict[k])[0]` returns only the first character of the winning key. Executed against `{'alpha': 10, 'beta': 20}`: returns **`'b'`**, not `'beta'`. The provided one-character-key test conceals this bug. A prior judge called this a syntax error; that label is wrong—the code runs.
- **803:** non-string input must return **`None`**; wrapping `lambda a, b: a+b` and calling with `(1, 2)` returns **3**, confirmed by execution.
- **627:** asks for caching; implementation creates a fresh `lru_cache` wrapper per call. Two identical calls invoke the underlying function **twice**, confirmed by execution.
- **883:** requires constant space; implementation builds **two input-sized sets**, while its explanation claims not to convert both arrays into sets.
- **1420:** asks for a returned function accepting a transition parameter; implementation hard-codes `t=0.5` and returns a tuple instead.

Only manually inspected, side-effect-free snippets were executed. Existing
trace/reference judge rejections were used as candidate leads, not as truth labels.

### No Robots

- **1898:** “Do not use the word ‘and.’”; reference poem uses `and` **three** times.
- **6940:** paragraph must contain at least 150 words; reference has **136** under both counters.

### NuminaMath: separate correctness defects

- **12:** retirement rule is age + years employed ≥70, hired at age30. Correct equation is `(30+w)+w=70`, hence `w=20`; first eligible in2006 means hired **1986**. Reference holds age fixed at30 and answers **1966**.
- **40:** remainder of `5283+5284+5285+5286+5287` modulo7 is **0**. Reference incorrectly starts with `5283 mod7=1` (it is5), concluding **1**.
- **43:** reference claims `sqrt(9t^4+4t^2+4t) = |t|sqrt((3t^2+2t)(3t^2+2t+2))`. At `t=1` the sides are `sqrt(17)` and `sqrt(35)`, disproving the claimed identity. The supplied multiple-choice problem also needs review.

### Additional Smol Summarize defects

A conservative sentence-boundary detector flags **267** references for exceeding
the three-sentence maximum; none of the 340 one-short-sentence rows are flagged.
Treat 267 as a **candidate count**, not a fully linguistically adjudicated count.
Nine flagged rows have no banned personal pronouns; these were manually inspected
and all clearly exceed three sentences:
`2466, 2531, 2874, 3801, 4996, 6268, 6628, 8855, 9545`.
Thus at least **498 / 984** summaries violate either the pronoun ban or the
sentence limit, even without adjudicating every sentence-count flag.

Some references also visibly end mid-sentence, e.g. **2531** ends “a significant
strategic and morale boost for”, and **2874** ends “Test flights for”. This is
evidence of incomplete references, but the upstream cause/cutoff was not verified.

## Implication for the experiment

The concern is real, but it is not confined to Smol Summarize. Several sources
supervise an explicit requirement alongside an incompatible answer; others have
conflicting instructions or incorrect reasoning. Those need different remedies.
I would audit/repair the reference blend **before** attaching more reasoning,
keep the original blend as a control, and distinguish:

1. Satisfiable constraints with bad answers: repair or exclude after validation.
2. Conflicting prompts/schema templates: resolve the specification first.
3. Code/maths correctness: use executable tests or independently checked solutions.
4. Justified refusals/clarifications: retain when appropriate, with consistent formatting.

This audit establishes data defects, **not a measured causal effect on alignment
or generalisation**. Filtering must not be conditioned on held-out evaluation
performance, and any cleaned comparison should account for changed source weights.

## Reproduction

Run `.venv/bin/python scratch/audit_nosynth_instructions.py`.
Outputs: `output/nosynth_instruction_audit/summary.json` and `flags.jsonl`.
The latter stores source, zero-based row ID, assistant turn index, check and
evidence. Generic flags are candidates, not an auto-deletion manifest. The
manual adjudications above supersede naive flags for the identified edge cases.
