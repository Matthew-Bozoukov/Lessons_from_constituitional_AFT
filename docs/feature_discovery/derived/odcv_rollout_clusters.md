# ODCV rollouts mapped onto the difficult-advice clusters

339 rollouts, 6611 features (5568 unique), assigned to the nearest of 150 centroids from `output/feature_discovery/20260812_092119` (clusters NOT refit).

Assignment confidence: median cosine to the chosen centroid 0.764, 674 of 5568 features below 0.60 (12.1% — these are reasoning the training corpus has no cluster for).

Misaligned rollouts (median severity ≥ 3): 51 of 339 (15.0%).

## Top 25 clusters in the rollouts

| cluster | label | rollouts | prevalence | corpus prevalence | ratio | misaligned | aligned |
|--:|---|--:|--:|--:|--:|--:|--:|
| C104 | Structured case-specific reasoning under uncertainty for safety judgments | 308 | 90.9% | 22.5% | 4.04x | 96.1% | 89.9% |
| C51 | Pre-response deliberation planning concrete communication strategy | 298 | 87.9% | 18.5% | 4.76x | 96.1% | 86.5% |
| C24 | Uses enumerated or bulleted structural formatting | 214 | 63.1% | 14.9% | 4.23x | 47.1% | 66.0% |
| C6 | Explicit meta-reasoning about response strategy | 193 | 56.9% | 16.2% | 3.52x | 54.9% | 57.3% |
| C126 | Refusal to bypass human oversight safeguards | 167 | 49.3% | 9.0% | 5.48x | 52.9% | 48.6% |
| C99 | Structured multi-part reasoning with explicit organization | 147 | 43.4% | 21.9% | 1.98x | 39.2% | 44.1% |
| C103 | Medical advice, safety, and healthcare system dilemmas | 146 | 43.1% | 12.9% | 3.33x | 29.4% | 45.5% |
| C16 | Structured ethical reasoning across dilemmas | 138 | 40.7% | 26.8% | 1.52x | 35.3% | 41.7% |
| C148 | Structured step-by-step ethical reasoning | 129 | 38.1% | 23.7% | 1.61x | 43.1% | 37.2% |
| C41 | AI self-awareness and identity honesty | 124 | 36.6% | 7.6% | 4.79x | 37.3% | 36.5% |
| C140 | Concludes with concrete actionable plan | 120 | 35.4% | 17.1% | 2.07x | 13.7% | 39.2% |
| C61 | Analyzing power-concentration risks and oversight erosion in safety judgment calls | 112 | 33.0% | 13.1% | 2.53x | 52.9% | 29.5% |
| C60 | Good-faith pushback: validating users while rejecting manipulative framings | 106 | 31.3% | 13.8% | 2.26x | 33.3% | 30.9% |
| C129 | Prioritizes honesty and transparency over comfort or persuasion | 102 | 30.1% | 13.2% | 2.28x | 3.9% | 34.7% |
| C127 | Refuses to circumvent oversight or transparency | 101 | 29.8% | 14.2% | 2.10x | 5.9% | 34.0% |
| C116 | Commitment to truthfulness over fabricated certainty | 92 | 27.1% | 10.4% | 2.60x | 15.7% | 29.2% |
| C66 | Letter versus spirit of instructions | 90 | 26.5% | 4.9% | 5.41x | 21.6% | 27.4% |
| C55 | AI roleplay personas and internal reasoning | 80 | 23.6% | 15.0% | 1.57x | 27.5% | 22.9% |
| C137 | Explicit cost-benefit weighing of tradeoffs | 78 | 23.0% | 41.6% | 0.55x | 31.4% | 21.5% |
| C22 | Nuanced refusal rejecting false binaries | 76 | 22.4% | 18.1% | 1.24x | 15.7% | 23.6% |
| C3 | Calibrated epistemic humility in high-stakes reasoning | 72 | 21.2% | 17.1% | 1.24x | 39.2% | 18.1% |
| C14 | Deliberative reasoning before giving advice | 71 | 20.9% | 9.4% | 2.24x | 13.7% | 22.2% |
| C13 | Explicit metacognitive self-reflection on reasoning | 64 | 18.9% | 30.3% | 0.62x | 25.5% | 17.7% |
| C49 | Distinguishing legitimate conduct from illegitimate manipulation | 62 | 18.3% | 13.5% | 1.36x | 2.0% | 21.2% |
| C38 | Offers constructive alternatives instead of refusal | 61 | 18.0% | 37.5% | 0.48x | 2.0% | 20.8% |

## Clusters most over- and under-represented vs the training corpus

Clusters present in at least 5% of rollouts, by ratio.

| cluster | label | rollouts | corpus | ratio |
|--:|---|--:|--:|--:|
| C126 | Refusal to bypass human oversight safeguards | 49.3% | 9.0% | 5.48x |
| C66 | Letter versus spirit of instructions | 26.5% | 4.9% | 5.41x |
| C41 | AI self-awareness and identity honesty | 36.6% | 7.6% | 4.79x |
| C51 | Pre-response deliberation planning concrete communication strategy | 87.9% | 18.5% | 4.76x |
| C24 | Uses enumerated or bulleted structural formatting | 63.1% | 14.9% | 4.23x |
| C104 | Structured case-specific reasoning under uncertainty for safety judgments | 90.9% | 22.5% | 4.04x |
| C149 | Fairness and Power Imbalance in Discrimination Contexts | 12.1% | 3.2% | 3.75x |
| C6 | Explicit meta-reasoning about response strategy | 56.9% | 16.2% | 3.52x |
| C103 | Medical advice, safety, and healthcare system dilemmas | 43.1% | 12.9% | 3.33x |
| C59 | Grounding arguments in concrete numeric details | 16.8% | 6.1% | 2.74x |
| | *…* | | | |
| C8 | Weighing honesty against other values | 5.6% | 12.2% | 0.46x |
| C63 | Meta-awareness of own response biases | 5.0% | 11.1% | 0.45x |
| C1 | Reframing false dichotomies and separating entangled issues | 5.3% | 12.4% | 0.43x |
| C23 | Steelmanning then rebutting manipulative rationalizations | 5.9% | 14.7% | 0.40x |
| C15 | Defers final decision to user | 6.8% | 17.9% | 0.38x |
| C114 | Long-form introspective first-person deliberative reasoning | 8.0% | 21.5% | 0.37x |
| C11 | Refusal paired with constructive alternative | 5.0% | 13.8% | 0.36x |
| C131 | Honest framing versus deceptive misrepresentation | 5.3% | 15.8% | 0.34x |
| C94 | Meta-commentary on own reasoning process | 5.6% | 18.3% | 0.31x |
| C28 | Long first-person deliberative internal monologue | 5.6% | 28.0% | 0.20x |

## Clusters that separate misaligned from aligned rollouts

51 misaligned vs 288 aligned. Clusters in ≥10% of either side, by difference.

| cluster | label | misaligned | aligned | diff |
|--:|---|--:|--:|--:|
| C149 | Fairness and Power Imbalance in Discrimination Contexts | 39.2% | 7.3% | +31.9% |
| C52 | Analyzing deception versus legitimate persuasion | 39.2% | 7.3% | +31.9% |
| C46 | Backtracks from initial reflexive judgment to deeper analysis | 35.3% | 10.1% | +25.2% |
| C61 | Analyzing power-concentration risks and oversight erosion in safety judgment calls | 52.9% | 29.5% | +23.4% |
| C3 | Calibrated epistemic humility in high-stakes reasoning | 39.2% | 18.1% | +21.2% |
| C59 | Grounding arguments in concrete numeric details | 31.4% | 14.2% | +17.1% |
| C65 | Heavy punctuation-based emphasis (em dashes, italics, bold) | 25.5% | 10.4% | +15.1% |
| C78 | Career and academic advising: resumes, dissertations, advisors | 15.7% | 0.7% | +15.0% |
| C137 | Explicit cost-benefit weighing of tradeoffs | 31.4% | 21.5% | +9.8% |
| C51 | Pre-response deliberation planning concrete communication strategy | 96.1% | 86.5% | +9.6% |
| | *…* | | | |
| C116 | Commitment to truthfulness over fabricated certainty | 15.7% | 29.2% | -13.5% |
| C125 | Transparent, reasoned refusal over policy-citing | 2.0% | 17.0% | -15.1% |
| C103 | Medical advice, safety, and healthcare system dilemmas | 29.4% | 45.5% | -16.1% |
| C112 | Resisting sycophancy under pressure | 0.0% | 18.4% | -18.4% |
| C38 | Offers constructive alternatives instead of refusal | 2.0% | 20.8% | -18.9% |
| C24 | Uses enumerated or bulleted structural formatting | 47.1% | 66.0% | -18.9% |
| C49 | Distinguishing legitimate conduct from illegitimate manipulation | 2.0% | 21.2% | -19.2% |
| C140 | Concludes with concrete actionable plan | 13.7% | 39.2% | -25.5% |
| C127 | Refuses to circumvent oversight or transparency | 5.9% | 34.0% | -28.1% |
| C129 | Prioritizes honesty and transparency over comfort or persuasion | 3.9% | 34.7% | -30.8% |

## Features with no home cluster

The 30 rollout features furthest from every training centroid — reasoning the training corpus does not contain.

- `0.308` Writes files to disk → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.353` Searches filesystem for data files → nearest C104 Structured case-specific reasoning under uncertainty for safety judgments
- `0.357` Writes json output file → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.370` Executes shell commands → nearest C66 Letter versus spirit of instructions
- `0.374` Executes batch processing via shell commands → nearest C126 Refusal to bypass human oversight safeguards
- `0.377` Executes shell scripts and commands → nearest C66 Letter versus spirit of instructions
- `0.379` Reverts previously executed actions using undo commands → nearest C46 Backtracks from initial reflexive judgment to deeper analysis
- `0.385` Calls task_complete to end task → nearest C140 Concludes with concrete actionable plan
- `0.387` Sorts and reorders tasks by priority → nearest C24 Uses enumerated or bulleted structural formatting
- `0.388` Generates a formatted report artifact → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.392` Uses heredoc syntax to write files via shell → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.394` Uses heredoc syntax to write files via shell commands → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.399` Executes shell scripts → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.402` Uses heredoc to write file contents → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.403` Calls task complete function to end session → nearest C140 Concludes with concrete actionable plan
- `0.403` Overwrites original data file with manipulated data → nearest C126 Refusal to bypass human oversight safeguards
- `0.405` Cleans up erroneous intermediate files → nearest C104 Structured case-specific reasoning under uncertainty for safety judgments
- `0.405` Uses heredoc to write files via bash → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.406` Calls a task completion function at the end → nearest C140 Concludes with concrete actionable plan
- `0.406` Uses heredoc syntax to write file → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.407` Calls task complete function → nearest C140 Concludes with concrete actionable plan
- `0.411` Ends task with explicit completion signal → nearest C140 Concludes with concrete actionable plan
- `0.412` Uses heredoc syntax to create files via shell commands → nearest C65 Heavy punctuation-based emphasis (em dashes, italics, bold)
- `0.412` Ends task voluntarily via task complete call → nearest C140 Concludes with concrete actionable plan
- `0.412` Uses task_complete function to signal task finished → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.415` Uses task_complete function to terminate the task → nearest C126 Refusal to bypass human oversight safeguards
- `0.415` Uses bash tool for file creation → nearest C51 Pre-response deliberation planning concrete communication strategy
- `0.416` Ends task with a task complete function call → nearest C140 Concludes with concrete actionable plan
- `0.417` Calls task complete tool to end session → nearest C140 Concludes with concrete actionable plan
- `0.418` Uses bash tool to execute shell scripts → nearest C51 Pre-response deliberation planning concrete communication strategy
