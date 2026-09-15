<!-- ABOUTME: Frozen local review of twelve examples actually present in the historical nonmoral684 training mixture. -->
<!-- ABOUTME: Separates reusable prompts from complete, valid and comparison-free outputs for the strict control. -->

# Historical reuse sample — 2026-09-08

**None of these 12 historical final answers can be reused unchanged for the strict
matched experiment.** Eleven contain explicit comparisons; the remaining answer adds
unsupported game mechanics. One prompt supports a new strict pair directly, two
deserve a narrowly scoped local demonstration, and nine are presently unsuitable.
This is a diagnostic sample of three selected domains, not an estimated defect rate
for all 684 examples or evidence about their alignment effect.

## Selection and provenance

Before reading selected contents, freeze four examples from each exact metadata domain:
**UI copywriting**, **game rulebook writing**, **code documentation**. These were chosen
for potentially inspectable text, rule consistency and code contracts, not known quality.
Eligible trained counts were 18, 7 and 6. Rank each by
`SHA256("nonmoral-reuse-20260908:" + scenario_id)` and take the lowest four.
Metadata counts were inspected beforehand; one unrelated API-schema example had been
visible during file discovery. No ODCV cases, scores or labels informed selection.

The 702-row generated corpus is
`output/nonmoral_deliberation/20260902_013651/dataset.jsonl`; eligibility was restricted
to the actual 684 synthetic rows in the training mixture at HF revision
`6364505df02b0020b030bf379bd42285a14de6a5`. The proposed repository-relative
`localdata/t2_9284_nonmoral_684.jsonl` did not exist; the exact bytes were recovered
from the existing HF cache with `local_files_only=True`. Mixture SHA-256:
`0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`.
All 12 corpus prompts, system messages, final answers and CoTs match the training text.

Freeze, selected corpus/training rows, decisions and checks are in
`output/nonmoral_investigation/20260908/reuse_sample/`. Original bytes and historical
labels were not modified. Review was by an independent local agent, with identities
and previous project findings visible; it is not blinded human adjudication.

## Exact eligibility counts

| Property | Count / 12 | Meaning |
|---|---:|---|
| Prompt supports complete single-turn output without revision | 3 | Three UI tasks; two demand literal output and offer no substantive choice |
| Prompt ready for strict comparative/verification pair | **1** | Drawing conflict message |
| Prompt conditionally useful for strict pair | **2** | Six-section supplied rules; Concord example |
| Prompt currently ineligible for strict pair | **9** | Two literal-output tasks, three incomplete code inputs, missing actual settings identifiers, one contradictory docstring request, two unresolved game-rule tasks |
| Final answer reusable unchanged | **0** | Content/constraint defects as well as comparison contamination |
| Original comparative CoT approved unchanged | **0** | Rationale supports defective output, forbidden alternatives, or unsupported premises |
| Original CoT usable as strict verification control | **0** | Every original CoT evaluates alternatives |
| Final answer explicitly compares alternatives | **11** | Only the six-section rules answer lacks that comparison |

“Conditional” is not an approval. The paired-example workstream should demonstrate
that the full requested artifact can be delivered without inventing missing facts.
Zero unchanged-CoT approvals does not mean the historical corpus had no deliberation;
it means none of these traces is presently approved for the stricter new experiment.

These counts apply the user's subsequent explicit rule: **keep original prompts
unchanged and exclude incomplete cases**. Do not supply missing implementations or
config files. The initial review considered the validation docstring conditionally
documentable and interpreted “realistic identifiers” as allowing fictional UI keys.
The stricter rule excludes the former; independent paired-example review highlighted
that the latter also demands actual config-repository identifiers. That conflicting
interpretation is preserved here, and the settings example is excluded conservatively.
The frozen sample did not change, and exclusions were not replaced.

## Evidence by example

| ID | Strict-pair prompt | Historical answer/trace finding |
|---|---|---|
| `t4_b05_s004` | Ready | Refuses two-sentence problem/action template and returns 221 words. Claims preserving identifiers makes the template impossible, though the instruction forbids extra identifiers tacked on, not necessarily identifiers inside a sentence. A complete two-sentence response is possible without inventing undo functionality. |
| `t7_b02_s002` | Ineligible: literal output | Asked for only “lock”; adds “paragraph” and lengthy explanation. This is a feasible hard constraint. With the prompt unchanged, alternative wording is not a legitimate B choice. |
| `t7_b03_s004` | Ineligible: missing actual identifiers | All 45 bullets present, but headings and plain-language labels violate the requested raw-identifier presentation. A maximum-count setting is also presented as a toggle. “Realistic identifiers” could permit a fictional prototype, but the earlier actual-config requirement cannot be verified from supplied data; no silent reinterpretation. |
| `t8_b08_s005` | Ineligible: literal output | Replaces requested exact line. Claims its proposed line is roughly the same length: 73 versus 36 characters, 12 versus 5 words. Privacy implications warrant separate moral-content review; no need to resolve that to establish instruction failure. |
| `t5_b02_s007` | Conditional | Preserves six headings but adds immediate binding trades, bans notes, makes values private, and scores resource tokens without source support. Faithfully editing the supplied sections may be possible; it would not establish that the underdescribed entire game is independently playable. |
| `t3_b01_s000` | Conditional | Adds explicitly forbidden general binding rule, omits requested scores, and says Alina keeps grain she already gave Marcus. User authorizes invented names/resources/example scores; that does not authorize inventing universal withdrawal or enforcement rules. A single internally consistent example may work. |
| `t7_b05_s000` | Ineligible: unresolved | Defines the term despite explicit prohibition, invents a resale restriction, and ends the requested section with “As the game progresses, this effect compounds...” rather than completing it. Source use of “negative externality” and the missing mechanics need clarification or verification; no economic claim is certified here. |
| `t7_b04_s007` | Ineligible: missing facts | All three forbidden expressions occur in the preface; three scoring placeholders remain. Prompt uses “points” for two separate quantities, so which quantity loses 10 is unclear. Answer invents final-score penalties and group-voting consequences. In-game penalty language is not automatically moral deliberation; normative additions need review. |
| `t1_b02_s000` | Ineligible: conflicting request | Requests summary/Args/Returns, then signature/Args only. Answer invents UTC requirement, return-value types and valid feed identifiers from a bare function signature. No need for external API checks to establish absence of those facts. |
| `t8_b03_s006` | Ineligible: absent source | Actual migration code was never supplied. Answer provides two fabricated code examples, then asks for the code. It cannot be the requested complete modification across all occurrences. |
| `t8_b03_s000` | Ineligible: omitted implementation | Collapses individually requested categories; calls 11 categories 12; invents a false-if-invalid return contract. Although the prompt permits “passes through normal validation logic,” that logic is absent, so exact requested behavior cannot be certified. Excluded under the user's no-incomplete-cases rule. |
| `t8_b03_s003` | Ineligible: absent source | `[migration_script.py]` is a filename placeholder, not the 40-rule file. Answer supplies fragments and asks for the file, so single-turn complete output is unavailable without inventing source. |

These defects cannot be fixed just by deleting comparative paragraphs. Missing code,
fabricated behavior and unsupported game mechanics remain in the purported artifacts.
The historical tendency to override feasible user constraints also means regenerating
only C would confound the intended comparison with instruction compliance and quality.

## Practical next step

Use **UI text with real wording choices** and, if already present elsewhere in the
historical corpus, **small, fully supplied code/documentation contracts** as the first
few checkable families. This sample supports only the former directly; its code
examples show why “code documentation” metadata alone is insufficient.
Game-rule copyediting is conditional on supplied mechanics and a complete requested
section. Do not fill a 684-row target by duplicating the few passing cases.

Build new matched B/C examples from eligible historical prompts, retaining identical
full, comparison-free finals and a substantive nonempty verification trace for C.
Generate fresh B as well when the old rationale requires repairs. Such arms are new
controlled datasets; the historical checkpoint remains an anchor, not an exactly
matched comparator. Keep current source defects visible rather than claiming to
reproduce the original recipe after silently correcting it.

Reproduce source checks and counts with
`uv run --no-sync python scratch/nonmoral/review_reuse_sample.py` (passed locally).
It makes no network calls. Mechanical checks verify identities, selection, character
counts, list counts and forbidden literals; the eligibility judgments remain inspectable
manual decisions. No generation, GPU, training or benchmark work ran.
