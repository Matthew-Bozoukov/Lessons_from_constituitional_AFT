# Clusters C30, C79 inside `2026-08-06-table2-9284-synthdoc-716-train`

10000 mixture rows, of which **716 are `synthdoc_difficult_advice`**. All 716 joined uniquely to `output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl` by user message; 0 carry no feature labels.

**191 of 716 difficult-advice rows (26.7%) belong to at least one of the 2 clusters.**

## Per cluster

| cluster | label | in mixture | of the 716 | corpus rate | lexically exhibit |
|--:|---|--:|--:|--:|--:|
| C30 | Explicit multi-factor harm risk assessment | 64 | 8.9% | 8.9% | 62 (97%) |
| C79 | Structured multi-factor risk assessment reasoning | 146 | 20.4% | 17.2% | 55 (38%) |

Union: 191 rows, 98 (51%) lexically exhibit the probability/severity/reversibility move. The other 525 difficult-advice rows: 95 (18%).

## Overlap

| clusters per row | rows |
|--:|--:|
| 1 | 172 |
| 2 | 19 |

## Trait mix of the matched rows

| trait | rows |
|---|--:|
| t1 | 22 |
| t2 | 9 |
| t3 | 12 |
| t4 | 69 |
| t5 | 28 |
| t6 | 3 |
| t7 | 12 |
| t8 | 26 |
| t9 | 10 |

## Every matched row

`scenario_id` — clusters — the features that matched

- `t1_b02_s004` — C79 — Weighs financial cost against safety risk explicitly
- `t1_b05_s004` — C79 — Weighs asymmetric risks of two possible actions
- `t1_b06_s006` — C79 — Explicitly weighs business pressure against safety risk
- `t1_b07_s003` — C79 — Weighs immediate cost against systemic risk
- `t1_b09_s000` — C79 — Draws parallel between its own reasoning risk and the users reasoning risk
- `t1_b09_s005` — C79 — Weighs cost and stress against safety risks
- `t1_b11_s005` — C30 — Distinguishes recoverable versus unrecoverable harms
- `t1_b13_s007` — C79 — Uses financial and legal risk scenario
- `t1_b14_s001` — C79 — Weighs asymmetry between who acts and who bears risk
- `t1_b18_s003` — C79 — Considers concrete failure modes and consequences; Warm nonpolicy tone when explaining risk
- `t1_b19_s002` — C79 — Explicit discussion of stakes and risks of the decision
- `t1_b19_s004` — C79 — Weighs good faith of requester against systemic risk
- `t1_b19_s006` — C79 — Weighs asymmetric risk of rare but catastrophic outcome against certain smaller cost
- `t1_b20_s006` — C79 — Balances helpfulness against asymmetric downside risk; Flags a chain of plausible steps leading to a risky conclusion
- `t1_b21_s007` — C79 — Warns against unilateral risk judgment under time pressure
- `t1_b25_s006` — C79 — Considers chain of small decisions compounding risk
- `t1_b25_s007` — C30 — Weighs harm of delay against harm of unverified action
- `t1_b26_s001` — C79 — Distinguishes between risk assessment and authority to decide
- `t1_b27_s003` — C79 — Bureaucratic delay versus security risk tradeoff
- `t1_b28_s001` — C79 — Model reasons about whether to bypass a safety guardrail
- `t1_b29_s001` — C79 — Weighs sympathetic urgent case against systemic risk
- `t1_b29_s004` — C79 — Draws distinction between risk assessment and procedural legitimacy
- `t2_b07_s003` — C79 — Considers personal liability risk to the user
- `t2_b11_s002` — C30 — Distinguishes recoverable versus irreversible harm
- `t2_b11_s004` — C79 — Considers multiple parties incentives and risks separately
- `t2_b12_s004` — C79 — Considers practical risk of the plan backfiring
- `t2_b12_s007` — C79 — Identifies rationalization risk in a sympathetic cause
- `t2_b14_s001` — C79 — Surfaces practical risk of detection as leverage
- `t2_b16_s003` — C79 — Concentrating power framed as core risk to avoid
- `t2_b19_s005` — C30 — Emphasizes scale of potential harm
- `t2_b25_s003` — C79 — Discusses fragility or long term risk of a strategy
- `t3_b05_s000` — C79 — Calibrated rather than alarmist risk assessment
- `t3_b06_s003` — C79 — Names asymmetry of risk explicitly
- `t3_b08_s004` — C79 — Anticipates discovery risk
- `t3_b08_s006` — C79 — Highlights retraction risk as concrete consequence
- `t3_b11_s001` — C79 — Weighs strategic risk of concealment versus disclosure
- `t3_b11_s005` — C79 — Uses dollar figures to ground abstract risk
- `t3_b20_s001` — C79 — Flags uncertainty about detection or enforcement risk
- `t3_b22_s000` — C79 — Anticipates verification or background check risk
- `t3_b23_s005` — C79 — Considers downstream consequences and tail risk scenarios
- `t3_b23_s006` — C79 — Explains relative risk versus absolute risk distinction
- `t3_b27_s002` — C79 — Highlights risk of harm from incomplete information
- `t3_b28_s001` — C79 — Calibrated discussion of real world risk rather than absolute claims
- `t4_b00_s006` — C30 — Considers reversibility and severity of harm
- `t4_b00_s007` — C79 — Explicit risk asymmetry argument
- `t4_b01_s004` — C79 — Structured risk analysis
- `t4_b02_s000` — C30 — Analyzes severity and reversibility of harm
- `t4_b03_s000` — C30 — Considers probability and severity of harm
- `t4_b03_s001` — C79 — Considers low probability high severity risk
- `t4_b04_s002` — C30 C79 — Considers probability and severity of harm; Identifies specific real world risks like fire hazard
- `t4_b04_s006` — C79 — Numbered breakdown of compounding risk factors
- `t4_b05_s000` — C30 — Weighs severity and breadth of harm explicitly
- `t4_b05_s001` — C30 — Considers probability and severity of harm separately; Weighs reversibility of potential harm
- `t4_b05_s002` — C30 C79 — Considers reversibility and severity of harm; Enumerates bullet point risk analysis; Flags broader pattern of vendor scheme as systemic risk; Uses explicit harm factor framework
- `t4_b05_s005` — C30 — Uses named analytical criteria like severity reversibility probability and breadth; Weighs probability and reversibility of potential harm
- `t4_b05_s006` — C30 — Analyzes reversibility of potential harms
- `t4_b06_s001` — C30 — Explicit harm benefit calculus using named criteria like probability severity reversibility breadth
- `t4_b07_s002` — C30 — Considers reversibility and severity of potential harms; Weighs harms on both sides of a request
- `t4_b07_s005` — C30 C79 — Considers probability and discovery risk of harmful outcome; Systematic decision framework resembling risk analysis; Weighs severity and reversibility of potential harm
- `t4_b09_s006` — C30 — Distinguishes probability from severity of risk
- `t4_b10_s003` — C30 — Uses a structured framework of severity probability and proximity
- `t4_b10_s004` — C30 C79 — Analyzes severity breadth and reversibility of harm; Identifies coercion or extortion risk in proposed action
- `t4_b11_s001` — C79 — Distinguishes between harmful assistance and legitimate risk information
- `t4_b11_s003` — C30 — Reasons about severity and reversibility of harms
- `t4_b11_s004` — C30 — Considers probability and severity of harm
- `t4_b11_s005` — C30 — Reasons about severity and reversibility of harm
- `t4_b12_s003` — C30 — Considers reversibility and severity of potential harms
- `t4_b13_s001` — C79 — Structured factor by factor risk analysis
- `t4_b13_s007` — C30 — Considers probability severity and reversibility of harm
- `t4_b14_s003` — C30 C79 — Explicit multi factor risk assessment framework; Weighs severity and reversibility of potential harms
- `t4_b15_s001` — C79 — Uses vivid concrete risk examples
- `t4_b15_s002` — C30 — Considers probability severity and reversibility of harm
- `t4_b15_s004` — C79 — Considers asymmetry of risk between two parties; Highlights hidden risk the requester may be underestimating; Uses bullet points to enumerate risk factors
- `t4_b15_s005` — C30 C79 — Analyzes rationalization tactics used to normalize risky request; Explicit weighing of probability severity reversibility and cost distribution; Names asymmetry between who bears risk and who makes the decision
- `t4_b15_s006` — C79 — Considers vulnerability and retaliation risk; Structured multi-factor risk analysis
- `t4_b16_s004` — C30 — Analyzes severity and reversibility of harm
- `t4_b17_s000` — C30 — Systematic weighing of probability severity reversibility breadth and vulnerability
- `t4_b17_s001` — C30 — Assesses probability and severity of harm
- `t4_b17_s005` — C30 C79 — Considers probability and severity of harm separately; Explicit risk assessment framework applied to a personal dilemma
- `t4_b18_s002` — C30 — Weighs severity probability and reversibility explicitly
- `t4_b18_s003` — C79 — Uses probability and risk weighing language
- `t4_b18_s004` — C30 — Considers counterfactual and severity of harm
- `t4_b19_s000` — C30 C79 — Flags a secondary unrelated risk beyond the main question; Weighs severity against reversibility of potential harms
- `t4_b19_s005` — C30 — Weighs probability and severity of harm
- `t4_b19_s006` — C30 C79 — Explicit severity probability breadth analysis; Uses a structured risk analysis framework
- `t4_b20_s002` — C79 — Weighs probability and severity of legal detection
- `t4_b21_s002` — C30 — Considers probability and severity of outcomes
- `t4_b21_s003` — C30 — References severity probability and reversibility as decision axes
- `t4_b21_s004` — C30 — Weighs severity and reversibility of harm
- `t4_b21_s005` — C30 — Weighs probability and severity of getting caught
- `t4_b22_s001` — C30 — Applies a structured harm calculus with named factors like probability severity reversibility and consent
- `t4_b22_s006` — C30 — Uses a structured framework of probability severity reversibility and consent
- `t4_b22_s007` — C79 — Weighs risk using probability severity and breadth framework
- `t4_b23_s000` — C30 C79 — Considers probability of detection or audit as a discrete factor; Weighs reversibility and severity of potential harms against costs of delay
- `t4_b23_s001` — C30 C79 — Calibrated rather than alarmist risk assessment; Distinguishes between requesters own risk and risk to third parties; Explores counterfactual legitimate alternatives to the risky plan; Weighs severity probability and reversibility of harm
- `t4_b23_s007` — C30 C79 — Assesses severity and irreversibility of harm; Considers probability and recurrence of hazard; Explicit multi factor risk analysis
- `t4_b24_s002` — C30 C79 — Analyzes risk using severity and reversibility criteria; Uses bullet points to enumerate risks
- `t4_b24_s003` — C30 — Weighs probability against severity of harm explicitly
- `t4_b24_s005` — C30 C79 — Considers asymmetry between upside and downside risk; Considers reversibility and severity of outcomes
- `t4_b24_s006` — C30 — Considers severity and reversibility of harm
- `t4_b25_s000` — C30 — Weighs probability and severity of harm explicitly
- `t4_b25_s007` — C30 — Employs harm calculus terminology like probability severity breadth; Weighs probability severity and breadth of harm
- `t4_b27_s000` — C79 — Analyzes probability of failure as near median not tail risk; Identifies who bears risk if plan fails
- `t4_b27_s007` — C30 C79 — Evaluates severity and reversibility of harm; Systematic risk assessment framework
- `t4_b28_s001` — C30 — Weighs severity probability and reversibility of harm
- `t4_b28_s003` — C30 — Analyzes reversibility of potential harms
- `t4_b28_s004` — C30 C79 — Considers probability severity reversibility and breadth of harm; Uses bullet points to organize risk factors
- `t4_b29_s000` — C30 — Considers breadth and severity of harm; Considers probability of harm
- `t4_b29_s003` — C79 — Explicit risk benefit analysis
- `t4_b29_s004` — C79 — About real estate contingency waivers and home inspection risk; Distinguishes hypothetical risk from a specific flagged professional finding; Uses a structured risk framework with named factors like severity reversibility and probability
- `t4_b29_s007` — C30 — Considers both immediate and long term harms
- `t4_b30_s001` — C30 — Considers reversibility and severity of potential harms
- `t5_b07_s002` — C79 — Identifies asymmetric risk in decision making
- `t5_b07_s007` — C79 — Considers regulatory and audit risk
- `t5_b08_s002` — C79 — Weighs whose interests bear the risk of a decision
- `t5_b08_s006` — C30 — Weighs competing harms on both sides
- `t5_b09_s001` — C30 C79 — Analyzes risk asymmetry between benefit and harm; Considers low probability severe tail outcomes
- `t5_b10_s005` — C79 — Identifies a slippery slope risk
- `t5_b12_s006` — C79 — Considers reputational and audit risk to the requester
- `t5_b14_s007` — C79 — Identifies specific combination of risk factors rather than a single cause
- `t5_b15_s004` — C79 — Analyzes credibility risk to an overall case
- `t5_b16_s002` — C79 — Explicitly names the specific risk factors combining to create concern
- `t5_b18_s003` — C79 — Weighs stakeholder risk explicitly
- `t5_b19_s001` — C79 — Prioritizes actionable plan over risk repetition
- `t5_b19_s006` — C79 — Names an asymmetry in stakes between outcomes
- `t5_b20_s006` — C79 — Considers replication risk
- `t5_b21_s002` — C79 — Considers timing and context as risk multipliers; Weighs financial risk against human stakes like employees and relationships
- `t5_b21_s004` — C79 — Weighs asymmetric risks explicitly
- `t5_b22_s002` — C79 — Considers asymmetric downside risks of each option
- `t5_b23_s003` — C79 — Analyzes who bears risk under different courses of action
- `t5_b23_s006` — C79 — Considers risk to both the individual and the institution
- `t5_b24_s002` — C79 — Reasons about who bears the most risk
- `t5_b24_s003` — C79 — Distinguishes categorical wrongdoing from a spectrum of risk; Provides risk asymmetry analysis with stakeholders
- `t5_b25_s002` — C79 — Explicit cost benefit analysis of verification effort versus risk; Rejects blanket purity rule in favor of nuanced risk assessment
- `t5_b26_s000` — C30 C79 — Considers worst case harm scenario; Weighs helpfulness against safety risk
- `t5_b26_s003` — C79 — Anticipates and addresses a counterargument about detection risk
- `t5_b27_s006` — C79 — Considers real world practical consequences like discovery risk
- `t5_b28_s000` — C79 — Names the psychological pressures pushing toward the risky choice
- `t5_b28_s003` — C79 — Identifies asymmetry between who benefits and who bears risk
- `t5_b30_s002` — C79 — Constructs a risk framework based on delay before failure, irreversibility, and inspectability; Explicitly separates two distinct engineering claims by risk profile
- `t6_b09_s003` — C79 — Uses business risk as independent justification
- `t6_b16_s002` — C79 — Identifies risk of user overreliance
- `t6_b25_s003` — C79 — Discusses risk of future inconsistency harming user
- `t7_b00_s000` — C79 — Considers who bears the risk of a decision
- `t7_b03_s003` — C79 — Identifies who bears risk in a business decision
- `t7_b05_s003` — C79 — Distinguishes generic disclaimers from case-specific risk information
- `t7_b06_s006` — C79 — Weighs risk of harm against risk of policy violation
- `t7_b06_s007` — C79 — Identifies euphemistic language masking risky action
- `t7_b07_s006` — C79 — Identifies who bears legal or financial risk
- `t7_b08_s004` — C79 — Identifies exploitation risk from predatory providers
- `t7_b10_s001` — C30 — Weighs product design intent against edge case harm
- `t7_b17_s005` — C79 — Identifies risk of withholding material information
- `t7_b22_s000` — C79 — Considers asymmetric risk and accountability
- `t7_b27_s006` — C79 — Business and legal risk framing
- `t7_b30_s000` — C79 — Identifies risk of appearing unprepared; Separates sound advice from risky application
- `t8_b02_s000` — C79 — Weighs risks of both action and inaction explicitly
- `t8_b03_s002` — C79 — Constructs a decision framework of risk tiers
- `t8_b03_s007` — C79 — Considers long term versus short term risk tradeoffs
- `t8_b06_s003` — C79 — Explains concrete causal mechanism for risk; Identifies hidden risk in proposed sequencing
- `t8_b08_s003` — C79 — Rejects false balance in addressing audit risk
- `t8_b08_s005` — C79 — Uses numbered risk analysis
- `t8_b09_s004` — C79 — Considers risks of both underexplaining and overexplaining
- `t8_b13_s005` — C30 — Weighs helpfulness against risk of harm
- `t8_b14_s002` — C79 — Explicit weighing of caution versus abandonment risk
- `t8_b15_s002` — C79 — Explicit reasoning about what constitutes actual risk
- `t8_b15_s007` — C79 — Balances speed of getting coverage against long term risk; Considers real world consequences like contestability period
- `t8_b17_s001` — C79 — Analyzes third party competence risk; Distinguishes technicality from genuine risk; Identifies real risk versus false risk
- `t8_b17_s006` — C79 — Identifies an unasked but important risk to surface
- `t8_b19_s002` — C79 — Model identifies unstated but critical risk beyond the literal question asked
- `t8_b19_s006` — C79 — Weighs legal and practical risk before giving recommendation
- `t8_b20_s005` — C79 — Explicitly weighs risks of a flawed strategy to the client
- `t8_b21_s002` — C79 — Names asymmetric risk of one option over another
- `t8_b22_s006` — C79 — Surfaces an unstated but higher-stakes risk beyond the literal request
- `t8_b23_s003` — C79 — Flags outcome switching as key risk
- `t8_b25_s004` — C79 — Discusses risk of fabrication versus helpfulness; Draws a narrow distinction between safe and risky content
- `t8_b25_s005` — C79 — About real estate contingencies and home inspection risk; Distinguishes between probability of an event and ability to absorb its cost; Weighs professional expertise against generalized risk statistics
- `t8_b26_s000` — C79 — Bullet point style enumeration of risks
- `t8_b27_s001` — C79 — Analyzes risk of a negative outcome as low but nonzero
- `t8_b27_s002` — C79 — Identifies risk of being unhelpful twice through overcorrection
- `t8_b28_s001` — C79 — Structured around two labeled risk scenarios; Weighs competing risks explicitly
- `t8_b30_s003` — C79 — Ranks options by risk and speed
- `t9_b04_s000` — C30 C79 — Analyzes request in terms of practical risk rather than moral judgment; Distinguishes between recoverable and unrecoverable harms
- `t9_b08_s005` — C79 — Identifies sycophancy risk
- `t9_b13_s006` — C79 — Identifies risk of crowding out human relationships
- `t9_b15_s000` — C79 — Uses specific percentage estimate for risk framing
- `t9_b22_s004` — C79 — Distinguishes between refusing information and flagging risk
- `t9_b23_s005` — C79 — Identifies sycophancy as a specific risk
- `t9_b25_s000` — C79 — Diagnoses root cause of burnout risk; Weighs dependency risk on the model itself
- `t9_b26_s005` — C79 — Names sycophancy as a specific risk
- `t9_b28_s002` — C79 — Identifies hidden risk in literal compliance with request
- `t9_b29_s000` — C79 — Considers risk of discovery and its unequal consequences
