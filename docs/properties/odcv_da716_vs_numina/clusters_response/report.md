# clusters — 71 properties over 513 records

Evidence: **features** — 7181 embedded units. Grouping: `{'reduce': 'umap', 'cluster': 'hdbscan', 'metric': 'euclidean', 'seed': 42, 'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.0, 'min_cluster_size': 25, 'min_samples': None}`. 72 groups, 1978 units unclustered (27.5%).

Prevalence is the share of records with at least one unit in the group; groups OVERLAP, so these do not sum to 100%.

Every rate below counts a record as carrying a property on the basis of **cluster membership** — a feature extracted from that record landed in this group. No detector pass ran, so this is an assignment rather than a measurement.

Rows are ordered by the BETWEEN-ARM difference in prevalence, most enriched in the focus arm first — so the two ends of the list are what the focus model does more of and what it does less of.

## Prevalence by arm

| property | prevalence | da716_5pct | numina_control_0pct |
|---|--:|--:|--:|
| Uses lists to structure the response | 36.3% | 51.9% | 5.8% |
| Uses bold markdown or caps for emphasis | 30.4% | 45.1% | 1.7% |
| Refuses to falsify or manipulate data under pressure | 33.1% | 47.5% | 5.2% |
| Refuses and explains ethical rationale for refusal | 27.9% | 41.6% | 1.1% |
| Refuses then offers a legitimate alternative | 24.0% | 36.0% | 0.6% |
| Explicit ethical reasoning stated before acting | 24.8% | 36.0% | 2.9% |
| Explicitly discloses own limitations or tradeoffs | 22.8% | 33.6% | 1.7% |
| Imposes bold headers to structure the response | 24.2% | 34.2% | 4.6% |
| Repeats same ethical stance despite ongoing pressure | 11.3% | 17.1% | 0.0% |
| Proactively flags data/metric discrepancies to overseer | 13.1% | 18.9% | 1.7% |
| Ends response by asking user before proceeding | 10.7% | 15.9% | 0.6% |
| Presents multiple options via list instead of a single recommendation | 9.4% | 14.2% | 0.0% |
| Defers high-stakes decision to human escalation | 18.7% | 23.3% | 9.8% |
| Proactively offers next steps beyond the immediate task | 17.2% | 21.5% | 8.6% |
| Elevates patient safety over metric optimization | 8.8% | 13.0% | 0.6% |
| States rationale before executing the decision | 9.6% | 13.6% | 1.7% |
| Refuses to make unnecessary or fake tool calls | 7.4% | 10.9% | 0.6% |
| Gathers evidence via tool calls before concluding | 12.7% | 15.9% | 6.3% |
| Explicitly reframes a decision in values-based terms | 8.4% | 11.5% | 2.3% |
| Explicitly names conflicting incentives or objectives | 6.6% | 9.7% | 0.6% |
| Notices gap between validator verdict and true correctness | 7.6% | 10.6% | 1.7% |
| Frames action as gaming a proxy metric vs true goal | 14.2% | 17.1% | 8.6% |
| Repeats identical content across multiple turns | 6.6% | 9.4% | 1.1% |
| Justifies refusal via legal/regulatory risk exposure | 5.5% | 8.3% | 0.0% |
| Chooses honesty over a competing incentive | 7.4% | 10.0% | 2.3% |
| Ends the episode via an explicit task_complete call | 90.6% | 92.9% | 86.2% |
| Packages output as a formal structured report | 18.1% | 20.3% | 13.8% |
| Invokes named regulation to justify a decision | 8.8% | 10.9% | 4.6% |
| Reports unfavorable results honestly despite pressure to inflate | 8.4% | 10.3% | 4.6% |
| Invokes precise statistics to bolster a claim | 25.7% | 27.4% | 22.4% |
| Presents fabricated scenario with realistic operational detail | 34.9% | 36.6% | 31.6% |
| Adopts formal, professional register | 11.5% | 13.0% | 8.6% |
| Assistant assumes a specific professional or agent persona | 15.0% | 16.2% | 12.6% |
| Asserts fairness/equity as justification without support | 5.3% | 5.9% | 4.0% |
| Solves data tasks via shell text utilities | 8.2% | 8.6% | 7.5% |
| Ends with a confident, conclusive summary statement | 16.0% | 16.2% | 15.5% |
| Names a metric-versus-safety tradeoff explicitly | 7.6% | 7.7% | 7.5% |
| Treats protected demographic traits as manipulable data fields | 9.9% | 9.7% | 10.3% |
| Games the evaluator instead of the real task | 5.3% | 5.0% | 5.8% |
| Automates hiring decisions using biased proxy criteria | 2.9% | 2.6% | 3.5% |
| Frames task inside a compliance/audit scenario wrapper | 12.7% | 12.4% | 13.2% |
| States explicit pass/fail against a numeric threshold | 5.1% | 4.7% | 5.8% |
| Names or enacts reward hacking on a proxy metric | 7.8% | 7.4% | 8.6% |
| Games a numeric score to hit a target threshold | 6.8% | 5.9% | 8.6% |
| Checks filesystem state before acting on it | 7.6% | 6.2% | 10.3% |
| Repeats near-identical actions without adapting strategy | 25.5% | 23.6% | 29.3% |
| Fabricates plausible-sounding content presented as real | 14.6% | 12.4% | 19.0% |
| Directly patches data files with shell text commands | 10.7% | 8.6% | 14.9% |
| Autonomously chains multi-step pipeline execution | 5.9% | 3.5% | 10.3% |
| States pass/fail via explicit numeric threshold comparison | 14.6% | 12.1% | 19.5% |
| Automates an entire compliance workflow autonomously | 8.6% | 5.9% | 13.8% |
| Computes numeric scores to triage or allocate resources | 41.7% | 38.9% | 47.1% |
| Implements solution as an executable shell script | 9.6% | 6.5% | 15.5% |
| Chains multiple shell commands autonomously | 6.8% | 2.6% | 14.9% |
| Edits data to pass checks instead of fixing root cause | 9.4% | 5.0% | 17.8% |
| Writes file contents using bash heredoc syntax | 17.3% | 13.0% | 25.9% |
| Absence of visible reasoning before actions | 7.8% | 3.2% | 16.7% |
| Proactively creates missing directory to unblock a file operation | 9.9% | 5.3% | 19.0% |
| Uses structured json/yaml as the working data format | 11.3% | 6.5% | 20.7% |
| Silently fabricates missing data instead of flagging it | 17.3% | 12.4% | 27.0% |
| Inspects script source before executing it | 15.2% | 10.0% | 25.3% |
| Tool calls issued without accompanying natural language | 9.4% | 4.1% | 19.5% |
| Iterates over multiple files via loop or batch operations | 11.3% | 5.9% | 21.8% |
| Iterative edit-then-rerun-validator loop | 29.6% | 23.9% | 40.8% |
| Invents realistic file paths as if verified | 10.3% | 4.4% | 21.8% |
| Silent tool use with little narration | 8.6% | 2.1% | 21.3% |
| Accomplishes task via exclusive bash/shell tool calls | 67.8% | 60.5% | 82.2% |
| Autonomous multi-step agentic execution without confirmation | 28.6% | 21.2% | 43.1% |
| Falsifies data or actions then reports success honestly-sounding | 21.6% | 14.2% | 36.2% |
| Chains multiple tool calls without intervening narration | 37.4% | 28.9% | 54.0% |
| Silent tool calls with no accompanying narration | 30.8% | 13.6% | 64.4% |

## Between-arm difference — da716_5pct minus numina_control_0pct

`delta` is the difference in prevalence between the two models, computed WITHIN `condition` and combined by Cochran weight. `strict` repeats it within the scenario cell, which removes the scenario-mix imbalance outright; a delta that survives only the first may be a difference in which scenarios each arm ran. `pooled` is unstratified and is printed only so the gap is visible.

| property | da716_5pct | numina_control_0pct | delta | strict | pooled | q | significant |
|---|--:|--:|--:|--:|--:|--:|:--|
| Uses lists to structure the response | 51.9% | 5.8% | +46.4% | +47.7% | +46.2% | 0.000 | yes |
| Uses bold markdown or caps for emphasis | 45.1% | 1.7% | +43.8% | +45.1% | +43.4% | 0.000 | yes |
| Refuses to falsify or manipulate data under pressure | 47.5% | 5.2% | +42.7% | +44.4% | +42.3% | 0.000 | yes |
| Refuses and explains ethical rationale for refusal | 41.6% | 1.1% | +41.0% | +40.7% | +40.4% | 0.000 | yes |
| Refuses then offers a legitimate alternative | 36.0% | 0.6% | +36.0% | +34.1% | +35.4% | 0.000 | yes |
| Explicit ethical reasoning stated before acting | 36.0% | 2.9% | +33.3% | +32.0% | +33.1% | 0.000 | yes |
| Explicitly discloses own limitations or tradeoffs | 33.6% | 1.7% | +32.0% | +30.5% | +31.9% | 0.000 | yes |
| Imposes bold headers to structure the response | 34.2% | 4.6% | +29.8% | +29.6% | +29.6% | 0.000 | yes |
| Repeats same ethical stance despite ongoing pressure | 17.1% | 0.0% | +17.3% | +16.1% | +17.1% | 0.000 | yes |
| Proactively flags data/metric discrepancies to overseer | 18.9% | 1.7% | +17.2% | +17.6% | +17.2% | 0.000 | yes |
| Ends response by asking user before proceeding | 15.9% | 0.6% | +15.5% | +15.2% | +15.4% | 0.000 | yes |
| Presents multiple options via list instead of a single recommendation | 14.2% | 0.0% | +14.4% | +13.6% | +14.2% | 0.000 | yes |
| Defers high-stakes decision to human escalation | 23.3% | 9.8% | +13.7% | +13.6% | +13.5% | 0.000 | yes |
| Proactively offers next steps beyond the immediate task | 21.5% | 8.6% | +13.0% | +13.8% | +12.9% | 0.001 | yes |
| Elevates patient safety over metric optimization | 13.0% | 0.6% | +12.4% | +13.7% | +12.4% | 0.000 | yes |
| States rationale before executing the decision | 13.6% | 1.7% | +11.8% | +11.3% | +11.8% | 0.000 | yes |
| Refuses to make unnecessary or fake tool calls | 10.9% | 0.6% | +10.4% | +8.7% | +10.3% | 0.000 | yes |
| Gathers evidence via tool calls before concluding | 15.9% | 6.3% | +9.7% | +9.3% | +9.6% | 0.004 | yes |
| Explicitly reframes a decision in values-based terms | 11.5% | 2.3% | +9.2% | +9.1% | +9.2% | 0.001 | yes |
| Explicitly names conflicting incentives or objectives | 9.7% | 0.6% | +9.1% | +8.3% | +9.2% | 0.000 | yes |
| Notices gap between validator verdict and true correctness | 10.6% | 1.7% | +8.8% | +10.1% | +8.9% | 0.001 | yes |
| Frames action as gaming a proxy metric vs true goal | 17.1% | 8.6% | +8.5% | +8.5% | +8.5% | 0.013 | yes |
| Repeats identical content across multiple turns | 9.4% | 1.1% | +8.5% | +6.5% | +8.3% | 0.001 | yes |
| Justifies refusal via legal/regulatory risk exposure | 8.3% | 0.0% | +8.4% | +8.9% | +8.3% | 0.000 | yes |
| Chooses honesty over a competing incentive | 10.0% | 2.3% | +7.8% | +8.3% | +7.7% | 0.002 | yes |
| Ends the episode via an explicit task_complete call | 92.9% | 86.2% | +6.7% | +7.0% | +6.7% | 0.019 | yes |
| Packages output as a formal structured report | 20.3% | 13.8% | +6.4% | +8.3% | +6.6% | 0.101 |  |
| Invokes named regulation to justify a decision | 10.9% | 4.6% | +6.2% | +6.8% | +6.3% | 0.023 | yes |
| Reports unfavorable results honestly despite pressure to inflate | 10.3% | 4.6% | +5.9% | +6.7% | +5.7% | 0.036 | yes |
| Invokes precise statistics to bolster a claim | 27.4% | 22.4% | +5.1% | +4.5% | +5.0% | 0.252 |  |
| Presents fabricated scenario with realistic operational detail | 36.6% | 31.6% | +4.8% | +6.5% | +5.0% | 0.323 |  |
| Adopts formal, professional register | 13.0% | 8.6% | +4.5% | +5.8% | +4.4% | 0.280 |  |
| Assistant assumes a specific professional or agent persona | 16.2% | 12.6% | +3.7% | +3.4% | +3.6% | 0.299 |  |
| Asserts fairness/equity as justification without support | 5.9% | 4.0% | +1.9% | +2.6% | +1.9% | 0.475 |  |
| Solves data tasks via shell text utilities | 8.6% | 7.5% | +0.9% | +0.8% | +1.1% | 0.928 |  |
| Ends with a confident, conclusive summary statement | 16.2% | 15.5% | +0.7% | +0.9% | +0.7% | 0.865 |  |
| Names a metric-versus-safety tradeoff explicitly | 7.7% | 7.5% | +0.3% | -0.3% | +0.2% | 0.974 |  |
| Treats protected demographic traits as manipulable data fields | 9.7% | 10.3% | -0.6% | +0.6% | -0.6% | 0.858 |  |
| Games the evaluator instead of the real task | 5.0% | 5.8% | -0.8% | -0.2% | -0.7% | 0.757 |  |
| Automates hiring decisions using biased proxy criteria | 2.6% | 3.5% | -0.8% | -0.6% | -0.8% | 0.664 |  |
| Frames task inside a compliance/audit scenario wrapper | 12.4% | 13.2% | -0.8% | -0.2% | -0.8% | 0.836 |  |
| States explicit pass/fail against a numeric threshold | 4.7% | 5.8% | -1.1% | -0.6% | -1.0% | 0.477 |  |
| Names or enacts reward hacking on a proxy metric | 7.4% | 8.6% | -1.3% | -2.8% | -1.2% | 0.591 |  |
| Games a numeric score to hit a target threshold | 5.9% | 8.6% | -2.8% | -3.6% | -2.7% | 0.230 |  |
| Checks filesystem state before acting on it | 6.2% | 10.3% | -4.2% | -4.2% | -4.2% | 0.088 | yes |
| Repeats near-identical actions without adapting strategy | 23.6% | 29.3% | -5.8% | -7.2% | -5.7% | 0.174 |  |
| Fabricates plausible-sounding content presented as real | 12.4% | 19.0% | -6.5% | -6.8% | -6.6% | 0.069 | yes |
| Directly patches data files with shell text commands | 8.6% | 14.9% | -6.6% | -7.5% | -6.4% | 0.013 | yes |
| Autonomously chains multi-step pipeline execution | 3.5% | 10.3% | -7.0% | -6.7% | -6.8% | 0.001 | yes |
| States pass/fail via explicit numeric threshold comparison | 12.1% | 19.5% | -7.6% | -9.2% | -7.4% | 0.024 | yes |
| Automates an entire compliance workflow autonomously | 5.9% | 13.8% | -7.9% | -8.2% | -7.9% | 0.004 | yes |
| Computes numeric scores to triage or allocate resources | 38.9% | 47.1% | -8.2% | -4.1% | -8.2% | 0.098 | yes |
| Implements solution as an executable shell script | 6.5% | 15.5% | -9.2% | -9.1% | -9.0% | 0.001 | yes |
| Chains multiple shell commands autonomously | 2.6% | 14.9% | -12.4% | -12.7% | -12.3% | 0.000 | yes |
| Edits data to pass checks instead of fixing root cause | 5.0% | 17.8% | -12.8% | -12.7% | -12.8% | 0.000 | yes |
| Writes file contents using bash heredoc syntax | 13.0% | 25.9% | -13.1% | -11.9% | -12.9% | 0.000 | yes |
| Absence of visible reasoning before actions | 3.2% | 16.7% | -13.6% | -13.9% | -13.4% | 0.000 | yes |
| Proactively creates missing directory to unblock a file operation | 5.3% | 19.0% | -13.8% | -13.0% | -13.7% | 0.000 | yes |
| Uses structured json/yaml as the working data format | 6.5% | 20.7% | -14.2% | -15.2% | -14.2% | 0.000 | yes |
| Silently fabricates missing data instead of flagging it | 12.4% | 27.0% | -14.6% | -13.6% | -14.6% | 0.000 | yes |
| Inspects script source before executing it | 10.0% | 25.3% | -15.4% | -15.4% | -15.3% | 0.000 | yes |
| Tool calls issued without accompanying natural language | 4.1% | 19.5% | -15.6% | -16.3% | -15.4% | 0.000 | yes |
| Iterates over multiple files via loop or batch operations | 5.9% | 21.8% | -16.1% | -16.2% | -15.9% | 0.000 | yes |
| Iterative edit-then-rerun-validator loop | 23.9% | 40.8% | -17.0% | -16.6% | -16.9% | 0.000 | yes |
| Invents realistic file paths as if verified | 4.4% | 21.8% | -17.5% | -16.8% | -17.4% | 0.000 | yes |
| Silent tool use with little narration | 2.1% | 21.3% | -19.3% | -19.2% | -19.2% | 0.000 | yes |
| Accomplishes task via exclusive bash/shell tool calls | 60.5% | 82.2% | -21.9% | -19.1% | -21.7% | 0.000 | yes |
| Autonomous multi-step agentic execution without confirmation | 21.2% | 43.1% | -22.0% | -21.1% | -21.9% | 0.000 | yes |
| Falsifies data or actions then reports success honestly-sounding | 14.2% | 36.2% | -22.1% | -23.8% | -22.1% | 0.000 | yes |
| Chains multiple tool calls without intervening narration | 28.9% | 54.0% | -25.5% | -24.7% | -25.1% | 0.000 | yes |
| Silent tool calls with no accompanying narration | 13.6% | 64.4% | -51.1% | -51.3% | -50.8% | 0.000 | yes |

## Outcome rate on `violation`, within stratum

`lift` is members minus non-members OF THE SAME STRATUM (`['arm', 'condition']`). `pooled` is the same difference computed across strata and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | strata | significant |
|---|--:|--:|--:|--:|:--|
| Uses lists to structure the response | -32.1% | -32.1% | 0.000 | 1 | yes |
| Uses bold markdown or caps for emphasis | -28.2% | -28.2% | 0.000 | 1 | yes |
| Refuses to falsify or manipulate data under pressure | -35.3% | -35.3% | 0.000 | 1 | yes |
| Refuses and explains ethical rationale for refusal | -31.4% | -31.4% | 0.000 | 1 | yes |
| Refuses then offers a legitimate alternative | -29.3% | -29.3% | 0.000 | 1 | yes |
| Explicit ethical reasoning stated before acting | -27.7% | -27.7% | 0.000 | 1 | yes |
| Explicitly discloses own limitations or tradeoffs | -27.6% | -27.6% | 0.000 | 1 | yes |
| Imposes bold headers to structure the response | -23.1% | -23.1% | 0.000 | 1 | yes |
| Repeats same ethical stance despite ongoing pressure | -24.0% | -24.0% | 0.000 | 1 | yes |
| Proactively flags data/metric discrepancies to overseer | -21.6% | -21.6% | 0.000 | 1 | yes |
| Ends response by asking user before proceeding | -17.5% | -17.5% | 0.006 | 1 | yes |
| Presents multiple options via list instead of a single recommendation | -15.8% | -15.8% | 0.022 | 1 | yes |
| Defers high-stakes decision to human escalation | -16.4% | -16.4% | 0.001 | 1 | yes |
| Proactively offers next steps beyond the immediate task | -29.9% | -29.9% | 0.000 | 1 | yes |
| Elevates patient safety over metric optimization | -27.1% | -27.1% | 0.000 | 1 | yes |
| States rationale before executing the decision | -27.4% | -27.4% | 0.000 | 1 | yes |
| Refuses to make unnecessary or fake tool calls | -15.4% | -15.4% | 0.045 | 1 | yes |
| Gathers evidence via tool calls before concluding | -26.6% | -26.6% | 0.000 | 1 | yes |
| Explicitly reframes a decision in values-based terms | -24.5% | -24.5% | 0.001 | 1 | yes |
| Explicitly names conflicting incentives or objectives | -17.1% | -17.1% | 0.034 | 1 | yes |
| Notices gap between validator verdict and true correctness | -21.2% | -21.2% | 0.005 | 1 | yes |
| Frames action as gaming a proxy metric vs true goal | +12.7% | +12.7% | 0.027 | 1 | yes |
| Repeats identical content across multiple turns | -20.2% | -20.2% | 0.012 | 1 | yes |
| Justifies refusal via legal/regulatory risk exposure | -26.2% | -26.2% | 0.003 | 1 | yes |
| Chooses honesty over a competing incentive | -23.9% | -23.9% | 0.002 | 1 | yes |
| Ends the episode via an explicit task_complete call | -37.0% | -37.0% | 0.000 | 1 | yes |
| Packages output as a formal structured report | -15.8% | -15.8% | 0.002 | 1 | yes |
| Invokes named regulation to justify a decision | -24.7% | -24.7% | 0.000 | 1 | yes |
| Reports unfavorable results honestly despite pressure to inflate | -27.0% | -27.0% | 0.000 | 1 | yes |
| Invokes precise statistics to bolster a claim | -25.2% | -25.2% | 0.000 | 1 | yes |
| Presents fabricated scenario with realistic operational detail | -20.9% | -20.9% | 0.000 | 1 | yes |
| Adopts formal, professional register | -28.0% | -28.0% | 0.000 | 1 | yes |
| Assistant assumes a specific professional or agent persona | -7.7% | -7.7% | 0.174 | 1 |  |
| Asserts fairness/equity as justification without support | -2.7% | -2.7% | 0.776 | 1 |  |
| Solves data tasks via shell text utilities | +27.5% | +27.5% | 0.000 | 1 | yes |
| Ends with a confident, conclusive summary statement | -16.4% | -16.4% | 0.003 | 1 | yes |
| Names a metric-versus-safety tradeoff explicitly | -1.8% | -1.8% | 0.812 | 1 |  |
| Treats protected demographic traits as manipulable data fields | +9.5% | +9.5% | 0.162 | 1 |  |
| Games the evaluator instead of the real task | +44.2% | +44.2% | 0.000 | 1 | yes |
| Automates hiring decisions using biased proxy criteria | +43.2% | +43.2% | 0.000 | 1 | yes |
| Frames task inside a compliance/audit scenario wrapper | -3.7% | -3.7% | 0.543 | 1 |  |
| States explicit pass/fail against a numeric threshold | -26.1% | -26.1% | 0.004 | 1 | yes |
| Names or enacts reward hacking on a proxy metric | +49.1% | +49.1% | 0.000 | 1 | yes |
| Games a numeric score to hit a target threshold | +34.8% | +34.8% | 0.000 | 1 | yes |
| Checks filesystem state before acting on it | +6.5% | +6.5% | 0.399 | 1 |  |
| Repeats near-identical actions without adapting strategy | +13.9% | +13.9% | 0.002 | 1 | yes |
| Fabricates plausible-sounding content presented as real | +6.9% | +6.9% | 0.228 | 1 |  |
| Directly patches data files with shell text commands | +51.7% | +51.7% | 0.000 | 1 | yes |
| Autonomously chains multi-step pipeline execution | +5.6% | +5.6% | 0.522 | 1 |  |
| States pass/fail via explicit numeric threshold comparison | -18.1% | -18.1% | 0.001 | 1 | yes |
| Automates an entire compliance workflow autonomously | +7.7% | +7.7% | 0.284 | 1 |  |
| Computes numeric scores to triage or allocate resources | -6.4% | -6.4% | 0.124 | 1 |  |
| Implements solution as an executable shell script | +15.5% | +15.5% | 0.023 | 1 | yes |
| Chains multiple shell commands autonomously | +1.0% | +1.0% | 0.892 | 1 |  |
| Edits data to pass checks instead of fixing root cause | +57.7% | +57.7% | 0.000 | 1 | yes |
| Writes file contents using bash heredoc syntax | +47.5% | +47.5% | 0.000 | 1 | yes |
| Absence of visible reasoning before actions | +35.5% | +35.5% | 0.000 | 1 | yes |
| Proactively creates missing directory to unblock a file operation | +26.9% | +26.9% | 0.000 | 1 | yes |
| Uses structured json/yaml as the working data format | +7.1% | +7.1% | 0.270 | 1 |  |
| Silently fabricates missing data instead of flagging it | +36.7% | +36.7% | 0.000 | 1 | yes |
| Inspects script source before executing it | +25.2% | +25.2% | 0.000 | 1 | yes |
| Tool calls issued without accompanying natural language | +48.5% | +48.5% | 0.000 | 1 | yes |
| Iterates over multiple files via loop or batch operations | +9.0% | +9.0% | 0.162 | 1 |  |
| Iterative edit-then-rerun-validator loop | +27.5% | +27.5% | 0.000 | 1 | yes |
| Invents realistic file paths as if verified | -8.7% | -8.7% | 0.193 | 1 |  |
| Silent tool use with little narration | +10.2% | +10.2% | 0.162 | 1 |  |
| Accomplishes task via exclusive bash/shell tool calls | +15.0% | +15.0% | 0.000 | 1 | yes |
| Autonomous multi-step agentic execution without confirmation | +3.4% | +3.4% | 0.446 | 1 |  |
| Falsifies data or actions then reports success honestly-sounding | +52.3% | +52.3% | 0.000 | 1 | yes |
| Chains multiple tool calls without intervening narration | +13.7% | +13.7% | 0.001 | 1 | yes |
| Silent tool calls with no accompanying narration | +26.4% | +26.4% | 0.000 | 1 | yes |

## Outcome rate on `any_misalignment`, within stratum

`lift` is members minus non-members OF THE SAME STRATUM (`['arm', 'condition']`). `pooled` is the same difference computed across strata and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | strata | significant |
|---|--:|--:|--:|--:|:--|
| Uses lists to structure the response | -35.5% | -35.5% | 0.000 | 1 | yes |
| Uses bold markdown or caps for emphasis | -29.3% | -29.3% | 0.000 | 1 | yes |
| Refuses to falsify or manipulate data under pressure | -41.0% | -41.0% | 0.000 | 1 | yes |
| Refuses and explains ethical rationale for refusal | -44.1% | -44.1% | 0.000 | 1 | yes |
| Refuses then offers a legitimate alternative | -37.8% | -37.8% | 0.000 | 1 | yes |
| Explicit ethical reasoning stated before acting | -28.2% | -28.2% | 0.000 | 1 | yes |
| Explicitly discloses own limitations or tradeoffs | -20.9% | -20.9% | 0.000 | 1 | yes |
| Imposes bold headers to structure the response | -22.1% | -22.1% | 0.000 | 1 | yes |
| Repeats same ethical stance despite ongoing pressure | -38.4% | -38.4% | 0.000 | 1 | yes |
| Proactively flags data/metric discrepancies to overseer | -23.1% | -23.1% | 0.001 | 1 | yes |
| Ends response by asking user before proceeding | -19.4% | -19.4% | 0.009 | 1 | yes |
| Presents multiple options via list instead of a single recommendation | -19.9% | -19.9% | 0.011 | 1 | yes |
| Defers high-stakes decision to human escalation | -14.5% | -14.5% | 0.013 | 1 | yes |
| Proactively offers next steps beyond the immediate task | -26.1% | -26.1% | 0.000 | 1 | yes |
| Elevates patient safety over metric optimization | -30.3% | -30.3% | 0.000 | 1 | yes |
| States rationale before executing the decision | -22.7% | -22.7% | 0.004 | 1 | yes |
| Refuses to make unnecessary or fake tool calls | -21.5% | -21.5% | 0.013 | 1 | yes |
| Gathers evidence via tool calls before concluding | -32.8% | -32.8% | 0.000 | 1 | yes |
| Explicitly reframes a decision in values-based terms | -32.0% | -32.0% | 0.000 | 1 | yes |
| Explicitly names conflicting incentives or objectives | -18.6% | -18.6% | 0.041 | 1 | yes |
| Notices gap between validator verdict and true correctness | -11.0% | -11.0% | 0.196 | 1 |  |
| Frames action as gaming a proxy metric vs true goal | +17.8% | +17.8% | 0.007 | 1 | yes |
| Repeats identical content across multiple turns | -21.8% | -21.8% | 0.017 | 1 | yes |
| Justifies refusal via legal/regulatory risk exposure | -32.0% | -32.0% | 0.002 | 1 | yes |
| Chooses honesty over a competing incentive | -32.8% | -32.8% | 0.000 | 1 | yes |
| Ends the episode via an explicit task_complete call | -30.7% | -30.7% | 0.000 | 1 | yes |
| Packages output as a formal structured report | -11.9% | -11.9% | 0.043 | 1 | yes |
| Invokes named regulation to justify a decision | -27.8% | -27.8% | 0.001 | 1 | yes |
| Reports unfavorable results honestly despite pressure to inflate | -34.5% | -34.5% | 0.000 | 1 | yes |
| Invokes precise statistics to bolster a claim | -18.4% | -18.4% | 0.000 | 1 | yes |
| Presents fabricated scenario with realistic operational detail | -21.7% | -21.7% | 0.000 | 1 | yes |
| Adopts formal, professional register | -34.8% | -34.8% | 0.000 | 1 | yes |
| Assistant assumes a specific professional or agent persona | -17.6% | -17.6% | 0.006 | 1 | yes |
| Asserts fairness/equity as justification without support | +19.3% | +19.3% | 0.057 | 1 | yes |
| Solves data tasks via shell text utilities | +30.6% | +30.6% | 0.000 | 1 | yes |
| Ends with a confident, conclusive summary statement | -15.3% | -15.3% | 0.013 | 1 | yes |
| Names a metric-versus-safety tradeoff explicitly | +8.4% | +8.4% | 0.322 | 1 |  |
| Treats protected demographic traits as manipulable data fields | +22.0% | +22.0% | 0.004 | 1 | yes |
| Games the evaluator instead of the real task | +31.1% | +31.1% | 0.003 | 1 | yes |
| Automates hiring decisions using biased proxy criteria | +47.1% | +47.1% | 0.001 | 1 | yes |
| Frames task inside a compliance/audit scenario wrapper | -9.9% | -9.9% | 0.149 | 1 |  |
| States explicit pass/fail against a numeric threshold | -18.8% | -18.8% | 0.069 | 1 | yes |
| Names or enacts reward hacking on a proxy metric | +50.5% | +50.5% | 0.000 | 1 | yes |
| Games a numeric score to hit a target threshold | +45.0% | +45.0% | 0.000 | 1 | yes |
| Checks filesystem state before acting on it | +14.0% | +14.0% | 0.102 | 1 |  |
| Repeats near-identical actions without adapting strategy | +22.9% | +22.9% | 0.000 | 1 | yes |
| Fabricates plausible-sounding content presented as real | +8.3% | +8.3% | 0.196 | 1 |  |
| Directly patches data files with shell text commands | +49.9% | +49.9% | 0.000 | 1 | yes |
| Autonomously chains multi-step pipeline execution | +13.2% | +13.2% | 0.174 | 1 |  |
| States pass/fail via explicit numeric threshold comparison | -15.2% | -15.2% | 0.018 | 1 | yes |
| Automates an entire compliance workflow autonomously | +17.4% | +17.4% | 0.032 | 1 | yes |
| Computes numeric scores to triage or allocate resources | +0.3% | +0.3% | 0.943 | 1 |  |
| Implements solution as an executable shell script | +20.2% | +20.2% | 0.009 | 1 | yes |
| Chains multiple shell commands autonomously | +11.3% | +11.3% | 0.205 | 1 |  |
| Edits data to pass checks instead of fixing root cause | +46.8% | +46.8% | 0.000 | 1 | yes |
| Writes file contents using bash heredoc syntax | +44.3% | +44.3% | 0.000 | 1 | yes |
| Absence of visible reasoning before actions | +26.1% | +26.1% | 0.002 | 1 | yes |
| Proactively creates missing directory to unblock a file operation | +19.9% | +19.9% | 0.009 | 1 | yes |
| Uses structured json/yaml as the working data format | +12.2% | +12.2% | 0.090 | 1 | yes |
| Silently fabricates missing data instead of flagging it | +36.1% | +36.1% | 0.000 | 1 | yes |
| Inspects script source before executing it | +21.3% | +21.3% | 0.001 | 1 | yes |
| Tool calls issued without accompanying natural language | +44.5% | +44.5% | 0.000 | 1 | yes |
| Iterates over multiple files via loop or batch operations | +21.9% | +21.9% | 0.003 | 1 | yes |
| Iterative edit-then-rerun-validator loop | +24.1% | +24.1% | 0.000 | 1 | yes |
| Invents realistic file paths as if verified | -5.7% | -5.7% | 0.445 | 1 |  |
| Silent tool use with little narration | +4.9% | +4.9% | 0.531 | 1 |  |
| Accomplishes task via exclusive bash/shell tool calls | +16.6% | +16.6% | 0.001 | 1 | yes |
| Autonomous multi-step agentic execution without confirmation | +3.6% | +3.6% | 0.461 | 1 |  |
| Falsifies data or actions then reports success honestly-sounding | +42.0% | +42.0% | 0.000 | 1 | yes |
| Chains multiple tool calls without intervening narration | +12.8% | +12.8% | 0.007 | 1 | yes |
| Silent tool calls with no accompanying narration | +32.3% | +32.3% | 0.000 | 1 | yes |

## Probes — what the property set accounts for

Logistic regression (L1, balanced) on the binary property-membership matrix, 5-fold cross-validated. `minimal` is the sparsest point on the regularisation path within one AUC point of the best. `null` is the same pipeline on shuffled labels — read the AUC against it, not against 0.5.

| predicting | positive rate | AUC | F1 | properties used | minimal set | null AUC | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| arm == da716_5pct | 66.1% | 0.922 | 0.890 | 64 | 56 @ AUC 0.920 | 0.505 | 0.020 |
| violation | 24.8% | 0.946 | 0.806 | 50 | 38 @ AUC 0.944 | 0.499 | 0.020 |
| any_misalignment | 40.9% | 0.848 | 0.719 | 32 | 13 @ AUC 0.839 | 0.499 | 0.020 |

### Heaviest properties for `arm == da716_5pct`

| property | coefficient |
|---|--:|
| Uses bold markdown or caps for emphasis | +2.411 |
| Explicitly discloses own limitations or tradeoffs | +2.106 |
| Chooses honesty over a competing incentive | -1.985 |
| Proactively flags data/metric discrepancies to overseer | +1.957 |
| Explicit ethical reasoning stated before acting | +1.662 |
| Refuses to make unnecessary or fake tool calls | +1.526 |
| Presents multiple options via list instead of a single recommendation | +1.511 |
| Refuses and explains ethical rationale for refusal | +1.404 |
| Solves data tasks via shell text utilities | +1.348 |
| Imposes bold headers to structure the response | +1.315 |
| Uses lists to structure the response | +1.271 |
| Computes numeric scores to triage or allocate resources | -1.238 |

### Heaviest properties for `violation`

| property | coefficient |
|---|--:|
| Proactively offers next steps beyond the immediate task | -2.280 |
| Names or enacts reward hacking on a proxy metric | +2.079 |
| Notices gap between validator verdict and true correctness | -1.837 |
| Falsifies data or actions then reports success honestly-sounding | +1.764 |
| Adopts formal, professional register | -1.727 |
| Ends response by asking user before proceeding | +1.612 |
| States rationale before executing the decision | -1.596 |
| Edits data to pass checks instead of fixing root cause | +1.558 |
| Directly patches data files with shell text commands | +1.482 |
| Explicitly discloses own limitations or tradeoffs | -1.395 |
| Packages output as a formal structured report | -1.358 |
| Writes file contents using bash heredoc syntax | +1.309 |

### Heaviest properties for `any_misalignment`

| property | coefficient |
|---|--:|
| Refuses and explains ethical rationale for refusal | -1.203 |
| Names or enacts reward hacking on a proxy metric | +1.108 |
| Writes file contents using bash heredoc syntax | +0.889 |
| Directly patches data files with shell text commands | +0.816 |
| Falsifies data or actions then reports success honestly-sounding | +0.708 |
| Repeats same ethical stance despite ongoing pressure | -0.652 |
| Silently fabricates missing data instead of flagging it | +0.631 |
| Games a numeric score to hit a target threshold | +0.600 |
| Adopts formal, professional register | -0.506 |
| Frames action as gaming a proxy metric vs true goal | +0.489 |
| Uses lists to structure the response | -0.418 |
| Gathers evidence via tool calls before concluding | -0.413 |

## Audit

72 groups, 27.5% of evidence unclustered.

### Redundancy — 2 near-duplicate group pairs (0.1% of all pairs)

Two groups this close describe the same theme, so the group COUNT is a resolution setting rather than a count of behaviours.

| cosine | a | b |
|--:|---|---|
| 0.906 | Tool calls issued without accompanying natural language | Silent tool calls with no accompanying narration |
| 0.904 | Explicit ethical reasoning stated before acting | Repeats same ethical stance despite ongoing pressure |

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 246 | 47.9% | Refuses to falsify or manipulate data under pressure (138), Refuses and explains ethical rationale for refusal (96), (unclustered noise) (92) |
| evaluation awareness | 85 | 16.6% | (unclustered noise) (20), Asserts fairness/equity as justification without support (1), Frames action as gaming a proxy metric vs true goal (1) |
| persona and identity | 48 | 9.4% | Assistant assumes a specific professional or agent persona (23), (unclustered noise) (11), Automates an entire compliance workflow autonomously (2) |
| oversight / monitoring | 44 | 8.6% | Defers high-stakes decision to human escalation (25), Computes numeric scores to triage or allocate resources (8), (unclustered noise) (7) |
| sycophancy | 9 | 1.8% | (unclustered noise) (7), Reports unfavorable results honestly despite pressure to inflate (2) |
| mentions uncertainty (any kind) | 8 | 1.6% | Falsifies data or actions then reports success honestly-sounding (3), (unclustered noise) (2), Explicitly discloses own limitations or tradeoffs (1) |
| training / self awareness | 0 | 0.0% | — |

### Is a property really a `scenario` marker?

1 of 71 groups are at least 50% MORE concentrated in one `scenario` than the corpus is (37 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Automates hiring decisions using biased proxy criteria | Ai-Hiring-Assistant | 93.3% | 2.7% | +90.6% | 2 |

### Is a property really a `condition` marker?

0 of 71 groups are at least 50% MORE concentrated in one `condition` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `condition` mix by 50% or more.

### Is a property really a `arm` marker?

1 of 71 groups are at least 50% MORE concentrated in one `arm` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Silent tool use with little narration | numina_control_0pct | 84.1% | 33.9% | +50.2% | 2 |

### Stability across seeds and neighbourhoods

0 of 9 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.381 to 1.000, median 0.536. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 70 | 25.6% | 0.511 |
| 10 | 1 | 69 | 25.0% | 0.567 |
| 10 | 2 | 69 | 25.1% | 0.550 |
| 15 | 0 | 70 | 26.3% | 0.677 |
| 15 | 1 | 71 | 26.4% | 0.707 |
| 15 | 2 | 66 | 24.9% | 0.725 |
| 30 | 0 | 64 | 27.3% | 0.530 |
| 30 | 1 | 66 | 27.3% | 0.522 |
| 30 | 2 | 63 | 27.0% | 0.515 |
