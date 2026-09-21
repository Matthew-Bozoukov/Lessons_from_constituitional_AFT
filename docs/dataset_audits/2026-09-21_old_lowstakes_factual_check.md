<!-- ABOUTME: Applies the same factual-grounding concerns to the historical low-stakes corpus. -->
<!-- ABOUTME: Records verified examples and a fresh deterministic sample without claiming a population error rate. -->
# Old low-stakes factual check, 2026-09-21

**Yes: the old corpus has invented facts, arithmetic errors, unsupported guarantees,
and advice that makes the preferred option seem easier by assuming away its costs.**
These are not distinctive discoveries about the new corpus. They cannot currently
explain the old-versus-refreshed ODCV difference: we have not measured comparable
whole-corpus error rates or isolated their training effect.

The source is [the original 716 conversations](https://huggingface.co/datasets/LASR-Callum/2026-08-26-difficult-advice-low-stakes-716/blob/f268653539150af5a340164f994065f57cbef5ad/dataset.jsonl),
revision `f268653539150af5a340164f994065f57cbef5ad`, SHA256
`55c9c35f69263dfe275130677876c5adfa66ed9a9abdfe0e499fa68dc3f0936e`.
Inspect the actual training messages, not the high-stakes ancestor text in metadata.

## Directly verified old examples

| Old row | Prompt establishes | Answer adds or gets wrong | Why it matters |
|---|---|---|---|
| `t8_b02_s000`, lake-house rooms | Rooms cost $180/$150/$130, total $460; sizes are qualitative | Gives $190/$145/$130, total **$465**, while promising to keep the total; calls the small room “a third the size”; asserts most people value a private bathroom at $50–80 | Incorrect arithmetic and invented precision drive the recommended split |
| `t9_b04_s004`, secondhand book refund | Torn cover; shop normally wants the book returned | “A torn cover on a secondhand book is worth nothing to resell”; draft says “this is honestly the first issue I've had” | Neither is supplied. The zero-value assumption makes its preferred refund tactic look costless; the ready-to-send draft invents customer history |
| `t6_b02_s000`, restaurant bookings | App permits one slot per phone number; user proposes another account | Asserts systems commonly auto-cancel household duplicates using card/device/IP; says a host can override the limit and calling solves it in thirty seconds | Unsupported system capabilities and outcome confidence favor the preferred option; it later allows the double-account fallback after saying not to do it |
| `t7_b01_s003`, tool lending | Some tools have cracked handles, sticky screws, missing jigs | Says a saw with a rough handle is safe for short cuts but not extended use | An unsupported safety assurance inside an otherwise sensible warning against relabeling unsafe tools |
| `t2_b11_s006`, dog agility | Uncertain weave-pole/contact-zone performance; formal review may take weeks | Declares contact-zone changes a common early sign of joint/soft-tissue problems; recommends extra repetitions “regardless of classification” | Specific medical generalization is not established by the record; the unconditional practice advice sits uneasily with the injury concern it introduced |

The medical/tool examples are flagged for unsupported assurance, not represented as
independently medically adjudicated falsehoods. The arithmetic and invented customer
history need no external expertise to establish the defect.

## Fresh sample, beyond previously known failures

Read nine additional complete conversations (system, user, reasoning, answer), one
per principle. Selection takes the minimum SHA256 of
`2026-09-21-factual-check:<scenario_id>` within each principle, excluding the five
targeted rows above. No paid judge calls; no selection based on answer quality.

| Row | Assessment |
|---|---|
| `t1_b07_s005`, potluck tool | Real oversight reasoning. Recommends a transparent exception to prior committee approval. This is a debatable policy judgment, **not automatically invented permission**; it never claims approval was obtained |
| `t2_b05_s007`, marketplace dispute weights | Useful distinction between an infrastructure change and changed decision authority. Its 60/40-to-20/80 example is explicitly conditional; **do not count hypothetical numbers as fabricated case facts** |
| `t3_b12_s005`, PTA report | Substantive selection-bias reasoning with useful truthful wording; no clear material invention identified in this read |
| `t4_b02_s007`, workplace contact list | Treats the previous coordinator's awareness and consent as known, although the user only reports nobody mentioning the list. Some language is qualified, some categorical; an overconfident inference, not proof of her actual mental state |
| `t5_b06_s003`, craft kit allocation | Invents agreement: “nobody, including your liaison” thinks the kiln move should wait; calls it a floor “everyone agrees on.” Draft announces three chapters “are getting kilns immediately,” although stock, logistics and that decision were never established. Turns a plausible feedback-loop hypothesis into a “mechanical fact” |
| `t6_b03_s001`, craft-show voice | Useful persona distinction, but confidently diagnoses what “felt scripted” almost always means despite not seeing the work. Treat as unsupported generalization, not a measured statement about this judge |
| `t7_b05_s005`, matchmaking intake | Suggests provisional matching as if available and says “nothing for quality-review to catch”; the process permitting this status is not supplied. A proposed fast track is legitimate; guaranteeing acceptability is not |
| `t8_b15_s000`, dinner reimbursements | Gives usable weighted criteria and challenges the metric. Assumes unfamiliar cuisine explains low ratings rather than marking it as a possibility; the source says only “not their taste” |
| `t9_b11_s005`, ceramics mentoring | Says the student has “no one else” and an “exclusive” channel is proposed; source says she feels understood by this instructor, not that she lacks other relationships. Conversely, the answer appropriately admits not knowing the coordinator's kiln constraints |

This is **14 reviewed examples, five deliberately targeted and nine newly selected**,
not 14 random draws or a 716-row prevalence audit. No defensible percentage of the
old corpus containing fabricated facts follows from these observations.

## Comparison with the current native smoke

The same mechanism appears on both sides: generate a fluent moral recommendation,
then add facts that make it easier to defend. Old: the book has zero resale value,
calling definitely fixes the booking problem, everyone agrees to immediate kiln
allocation. Current smoke: only $200–400 of the invoice is disputed, HR usually has
negotiating room, the customer-retention rule supposedly exists to prevent overpromising.
These often erase the genuine cost or uncertainty the deliberation should preserve.

The current smoke also adds a concrete **selection** failure: its two stakes filters
accept a few weeks of severance pay and a $1,800 invoice as modest. Old higher-stakes
leakage was already documented: 67/716 historical serious/grave labels, with mixed
accuracy on manual inspection. Neither dataset is a clean stakes control merely
because its recipe is named low-stakes.

The old sample has good deliberation alongside factual flaws. Its good ODCV result
does not make those flaws desirable, but it shows a zero-defect corpus was not a
prerequisite for that observed score. Equally, finding these defects in a new smoke
does not predict that smoke's eventual ODCV score.

**Correction to the framing of this investigation:** we should not subject only new
data to an increasingly strict perfection test and silently treat old data as a gold
standard. Preserve the DA task and real tradeoffs; use the same explicit standard on
both. Prioritize material invented facts, contradictions, and invented guaranteed
workarounds, while allowing proposed options, explicitly conditional illustrations,
and uncertain inferences. No additional generation, repair, training, or eval follows
automatically from this audit.

Full inspected old conversations and per-row notes are archived under `review/` in
the [native smoke artifact](https://huggingface.co/datasets/dougalldeepmind/2026-09-21-da-lowstakes-fresh-synth-smoke/tree/main/review).
