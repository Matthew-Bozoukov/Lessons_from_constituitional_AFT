# Clusters C6 inside `2026-08-06-table2-9284-synthdoc-716-train`

10000 mixture rows, of which **716 are `synthdoc_difficult_advice`**. All 716 joined uniquely to `output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl` by user message; 0 carry no feature labels.

**123 of 716 difficult-advice rows (17.2%) belong to at least one of the 1 clusters.**

## Per cluster

| cluster | label | in mixture | of the 716 | corpus rate | lexically exhibit |
|--:|---|--:|--:|--:|--:|
| C6 | Explicit meta-reasoning about response strategy | 123 | 17.2% | 16.2% | 28 (23%) |

Union: 123 rows, 28 (23%) lexically exhibit the probability/severity/reversibility move. The other 593 difficult-advice rows: 165 (28%).

## Overlap

| clusters per row | rows |
|--:|--:|
| 1 | 123 |

## Trait mix of the matched rows

| trait | rows |
|---|--:|
| t1 | 12 |
| t2 | 4 |
| t3 | 8 |
| t4 | 4 |
| t5 | 7 |
| t6 | 11 |
| t7 | 23 |
| t8 | 34 |
| t9 | 20 |

## Every matched row

`scenario_id` — clusters — the features that matched

- `t1_b00_s004` — C6 — Explains reasoning transparently rather than just stating a conclusion
- `t1_b08_s002` — C6 — Model reasons through an ai safety dilemma
- `t1_b10_s007` — C6 — Decides to communicate reasoning transparently rather than just act
- `t1_b11_s005` — C6 — Meta reasoning about capability versus permission; References system prompt as normative constraint
- `t1_b12_s000` — C6 — Displays meta awareness of its own decision making process
- `t1_b12_s004` — C6 — Model reasons through its own decision making process
- `t1_b17_s002` — C6 — Displays self awareness about reasoning process
- `t1_b22_s002` — C6 — Explains reasoning transparently to another party
- `t1_b22_s007` — C6 — Concludes with plan to communicate reasoning directly to user
- `t1_b23_s002` — C6 — Displays evaluations awareness of its own decision process
- `t1_b25_s006` — C6 — Explains reasoning transparently to user
- `t1_b29_s004` — C6 — Reasons about generalizability of its own decision across many hypothetical cases
- `t2_b02_s001` — C6 — Explicit reasoning about principles rather than surface reaction
- `t2_b17_s000` — C6 — Explains reasoning transparently rather than asserting a conclusion
- `t2_b19_s005` — C6 — Explicitly names its own reasoning process as reasoning
- `t2_b23_s002` — C6 — Explains reasoning transparently
- `t3_b02_s002` — C6 — Explains reasoning in terms of effects on reader beliefs
- `t3_b03_s007` — C6 — Explains reasoning transparently before stating the conclusion
- `t3_b13_s006` — C6 — Explains reasoning in terms of concrete real world consequences
- `t3_b14_s001` — C6 — Plans to explain reasoning directly to the user
- `t3_b14_s003` — C6 — Displays first person deliberative reasoning
- `t3_b17_s000` — C6 — Model reasons internally before producing final answer
- `t3_b20_s001` — C6 — Model reasoning is exposed as visible chain of thought
- `t3_b22_s003` — C6 — Self-directed questioning to structure reasoning
- `t4_b05_s000` — C6 — Reasoning trace of an ai assistant deliberating before responding
- `t4_b07_s005` — C6 — Structured internal reasoning presented as first person deliberation
- `t4_b11_s005` — C6 — Model reasoning transcript shown as internal monologue
- `t4_b22_s001` — C6 — Explicitly states it will give the user its true reasoning rather than an appeal to rules
- `t5_b07_s004` — C6 — Model reasons about when to defer to expert authority versus override it
- `t5_b09_s003` — C6 — Displays evaluations awareness of its own decision process; Is a reasoning trace rather than a final reply to the user
- `t5_b11_s004` — C6 — Plans to communicate reasoning transparently to user
- `t5_b15_s000` — C6 — Model reasoning trace
- `t5_b21_s001` — C6 — Offers to show reasoning rather than just a verdict
- `t5_b26_s000` — C6 — Explains reasoning transparently to the user
- `t5_b27_s005` — C6 — Explains reasoning transparently to the user
- `t6_b02_s000` — C6 — Explicit meta reasoning about how to respond
- `t6_b02_s007` — C6 — Explains reasoning transparently before deciding
- `t6_b07_s006` — C6 — Explains reasoning transparently to the user
- `t6_b08_s005` — C6 — Explicit meta reasoning about the structure of the request
- `t6_b11_s007` — C6 — Displays value based reasoning rather than rule based reasoning
- `t6_b12_s005` — C6 — Meta level reasoning about the nature of the request itself
- `t6_b17_s001` — C6 — Metacognitive analysis of user intent
- `t6_b18_s004` — C6 — Model reflects on its own reasoning process
- `t6_b20_s004` — C6 — Explicit reasoning about what would genuinely help the user
- `t6_b22_s006` — C6 — Analogizes rlhf shaped values to human upbringing shaped values
- `t6_b28_s000` — C6 — Very realistic conversational reasoning
- `t7_b00_s003` — C6 — Explicit meta reasoning about the purpose behind a system prompt rule
- `t7_b02_s005` — C6 — Explains reasoning transparently to a hypothetical reader
- `t7_b05_s003` — C6 — Displays meta awareness of tool design intent
- `t7_b06_s002` — C6 — Explains reasoning transparently to a hypothetical operator or user
- `t7_b06_s007` — C6 — Meta reasoning about the intent behind an instruction
- `t7_b07_s006` — C6 — Reasons about system prompt scope and limits
- `t7_b08_s004` — C6 — Explicit meta commentary on reasoning process
- `t7_b09_s000` — C6 — Meta reasoning about role of ai assistant under operator constraints
- `t7_b09_s004` — C6 — Explicit reasoning about system prompt hierarchy
- `t7_b09_s005` — C6 — Internal first person reasoning trace
- `t7_b10_s001` — C6 — Explicit meta reasoning about when defaults can be overridden
- `t7_b10_s004` — C6 — Reasons about when to override system prompt instructions
- `t7_b10_s007` — C6 — Uses concrete evidence from user situation to reason
- `t7_b16_s000` — C6 — Meta reasoning about instruction following
- `t7_b17_s003` — C6 — Internal reasoning trace with no direct user dialogue
- `t7_b17_s005` — C6 — Explicit self reasoning about system prompt intent
- `t7_b19_s000` — C6 — Explicit reasoning about scope of a system prompt directive; Self-aware reasoning about how to interpret ambiguous instructions
- `t7_b19_s001` — C6 — Explicit reasoning about when to override system prompt constraints
- `t7_b22_s007` — C6 — Displays self awareness of reasoning process
- `t7_b23_s000` — C6 — Meta reasoning about the purpose behind an instruction; Model reasons about tension between system prompt and user request
- `t7_b25_s006` — C6 — Considers counterfactual scenario where instruction and user interest align
- `t7_b27_s000` — C6 — Explicit reasoning about system prompt intent versus literal instruction; Meta reasoning about what counts as helpfulness
- `t7_b27_s007` — C6 — Meta reasoning about how to interpret operator instructions; Self-directed policy reasoning rather than a direct user facing reply
- `t8_b02_s003` — C6 — Explicit reasoning about what genuine helpfulness requires
- `t8_b03_s002` — C6 — Meta reasoning about how to respond rather than the substantive answer itself
- `t8_b03_s006` — C6 — Explicit reasoning about competing response strategies
- `t8_b05_s002` — C6 — Model reasons openly about its own response strategy
- `t8_b08_s003` — C6 — Meta reasoning about how to respond rather than the response itself
- `t8_b08_s004` — C6 — Model reasons privately before drafting a response
- `t8_b08_s005` — C6 — Model deliberates before responding to user request
- `t8_b08_s007` — C6 — Explains reasoning transparently as if thinking out loud to the user
- `t8_b09_s007` — C6 — Meta reasoning about what makes advice maximally helpful
- `t8_b10_s007` — C6 — Quotes the human directly to anchor reasoning
- `t8_b11_s000` — C6 — Meta reasoning about whether to hedge or give concrete advice
- `t8_b11_s006` — C6 — Meta reasoning about what constitutes genuine helpfulness
- `t8_b14_s000` — C6 — Model reasons about whether to give a direct opinion versus hedge
- `t8_b14_s001` — C6 — Explicit meta reasoning about how to respond
- `t8_b16_s000` — C6 — Meta reasoning about what constitutes genuine helpfulness
- `t8_b16_s001` — C6 — Model reasoning trace not shown to user
- `t8_b17_s005` — C6 — Meta reasoning about how to respond before responding
- `t8_b17_s006` — C6 — Explicit meta reasoning about what constitutes genuine helpfulness
- `t8_b17_s007` — C6 — Internal reasoning trace rather than direct reply
- `t8_b18_s003` — C6 — Explicit meta reasoning about how to respond
- `t8_b18_s004` — C6 — Explicit meta reasoning about what helpfulness means
- `t8_b18_s006` — C6 — Model reasoning trace about how to be maximally helpful
- `t8_b18_s007` — C6 — Model reasons through competing implicit requests before answering
- `t8_b19_s000` — C6 — Internal reasoning trace rather than direct reply
- `t8_b19_s006` — C6 — Explicit meta reasoning about how to structure the answer
- `t8_b20_s005` — C6 — Metacognitive planning of how to structure the final answer
- `t8_b20_s007` — C6 — Model reasoning about its own response strategy
- `t8_b22_s000` — C6 — Reasoning about how to respond rather than the response itself
- `t8_b22_s006` — C6 — Meta reasoning about what genuine helpfulness requires
- `t8_b23_s001` — C6 — Model reasons about how to respond rather than giving the response itself
- `t8_b25_s000` — C6 — Meta reasoning about the models own response strategy
- `t8_b26_s000` — C6 — Explicit meta reasoning about how to respond before responding
- `t8_b28_s001` — C6 — Internal monologue style reasoning trace
- `t8_b28_s007` — C6 — Meta reasoning about how to respond rather than the response itself
- `t9_b00_s004` — C6 — Internal reasoning trace rather than direct reply to user
- `t9_b00_s007` — C6 — Explicit internal reasoning about how to respond
- `t9_b02_s005` — C6 — Explicit reasoning about sycophancy versus genuine helpfulness; Reasoning structured as a dialogue with itself
- `t9_b04_s006` — C6 — Explicit meta reasoning about how to respond
- `t9_b08_s002` — C6 — Explains reasoning transparently and step by step
- `t9_b14_s001` — C6 — Model rehearses internal justification before responding
- `t9_b14_s005` — C6 — Meta reasoning about how to respond rather than the response itself
- `t9_b14_s007` — C6 — Meta reasoning about own role as advice giver
- `t9_b16_s005` — C6 — Second person address to the user embedded within reasoning
- `t9_b18_s001` — C6 — Model reasons about its own appropriate role and limits
- `t9_b19_s004` — C6 — Meta reasoning about how to respond
- `t9_b19_s005` — C6 — Model reasons about a caregiving support request
- `t9_b20_s007` — C6 — Displays therapeutic or counselor like reasoning
- `t9_b22_s003` — C6 — Reasoning about what user would endorse in hindsight
- `t9_b24_s003` — C6 — Grounds reasoning in specific details from the users situation
- `t9_b25_s000` — C6 — Plans to communicate reasoning transparently to the user
- `t9_b26_s005` — C6 — Displays therapeutic reasoning style
- `t9_b28_s000` — C6 — Model reasons about a students academic integrity request
- `t9_b28_s002` — C6 — Explicit reasoning about downstream consequences
- `t9_b30_s004` — C6 — Meta reasoning about what shape a good answer should take
