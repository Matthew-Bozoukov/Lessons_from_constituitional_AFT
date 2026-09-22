<!-- ABOUTME: Prospective domain registry, sampling and readiness criteria for low-stakes DA. -->
<!-- ABOUTME: Frozen before any paid requests for this variant; no replacement rounds. -->
# Domain-constrained native DA: preregistration

The user explicitly authorized a fixed domain registry as an additional design input.
This supersedes free domain discovery from the constitution; no old corpus or examples
are used. The full target principles still come only from
`constitutions/claude_distilled_09_principles/constitution.md`.

The machine-readable registry is `write_scenarios.rotate.assigned_domain` in
`configs/data/synth/da-lowstakes-practical.yaml`. Its text is the authoritative bound
for each domain, frozen with the recipe and code before dispatch.

| Domain | Allowed scope and consequence ceiling |
|---|---|
| Tabletop games | Adult friends, no prizes/gambling; one evening or two hours preparation |
| Book/film clubs | One informal meeting or two hours; no political advocacy/broad publication |
| Hobby crafts | Private amateur activity, up to $40 affordable materials/two hours; no sales/careers |
| Adult recreation | Casual adult game/meetup, one session/two hours; no injury, prizes or official rankings |
| Community gardening | Decorative hobby plants, one weekly rota or replaceable supplies; $30/two hours per person |
| Potluck planning | One evening, voluntary preparation up to $30/two hours; no food-safety or allergy decisions |
| Personal decoration | Reversible private-room aesthetics, $40/two hours; no construction, housing dispute or product-performance claims |
| Leisure media | Adult private fiction/playlists/characters, one session/two hours; no impersonation or distress/dependency |
| Small purchases | One-off replaceable hobby goods, affordable total up to $40/person; no recurring bills/rare valuables |

These are sampling constraints, not evidence that a generated case satisfies them.
Facts in the scenario and resulting conversation must establish the scale. In
particular, a domain label cannot justify the stakes judge's grade. Excluding major
consequences should not remove minor meaningful sacrifices or turn fictional play
into wrongdoing. Full reads will inspect forced refusals and costless alternatives
as comparative diagnostics, without adding stricter factual gates than normal DA.

## Fixed allocation and minimum information per call

Native `scenarios` already assigns axes per batch. The only operator extension is
`rotate.<axis>.per_trait: true` for equal unit weights: batch b for trait t gets
registry entry `(t+b) mod 9`, with zero-based indices. Existing rotation is unchanged
when this option is absent. It stamps `assigned_domain` onto every record and export.

Full candidate generation is 9 traits x 9 domains x 12 scenarios = 972 candidates,
81 source calls. Every call sees one full principle, one domain description, a short
task/bounds contract and the output schema. It sees no other domain descriptions,
old examples or accumulated generated-example avoidance list. Existing embedding
dedup remains, with zero regeneration rounds. Drafting sees only its situation and
shortcut plus writing instructions; refinement/answers receive only the relevant
principle and case. The single pre-answer judge receives the actual system/user,
not the constitution, domain registry or answer.

Source generation uses Sonnet, temperature 0.8, reasoning enabled with 2048 tokens
within a 12288-token response cap. The existing stakes judge gets 1024 reasoning
tokens within a 4096 cap and writes its consequence ledger before its grade. These
changes are bundled with domain control; this is an engineering smoke, not a causal
test isolating which change helps. Normal DA answer prompts are unchanged.

## One smoke, declared before generation

36 scenarios: two per call, two domains per trait. Every trait and every domain is
represented, but only 18 of the 81 pairings are exercised. Four local workers,
all Sonnet, no new critics, no content rerolls or replacements, existing bounded
parse retries only. Shared cumulative ceiling stays $20, with $13.742198 already
spent and $6.257802 remaining. The per-call guard reserves worst-case cost before
dispatch. Stop on exhaustion; never reset the ledger to finish the sample.

Readiness: at least 28/36 exports, at least two exports for every trait and domain,
at least 28 initially suitable cases, no out-of-assigned-domain cases and no clear
manual stakes/scope false accepts. Disclose uncertain judgments. Retain native 25%
filter-drop alarms; these now apply because the sample exceeds 20. Read all source
cases, all judge decisions and all final reasoning/answer pairs. Record substantive
prompt repairs and shared DA defects separately. Freeze failures, not just survivors.

If suitable, full generation can use this fixed 972-candidate grid. Deterministic
selection of 716 accepted rows should preserve domain and trait balance (80 rows
for five traits, 79 for four), with predeclared seed-0 tie-breaking and explicit
quota-shortfall reporting. Selection and a full-run spending cap are still required
before launch. No automatic paid topups. A successful smoke supports proceeding;
it does not establish all 81 pairings, flawless answers or recovered ODCV performance.
