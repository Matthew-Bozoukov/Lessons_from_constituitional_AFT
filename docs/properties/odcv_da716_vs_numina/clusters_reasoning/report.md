# clusters — 49 properties over 513 records

Evidence: **features** — 7906 embedded units. Grouping: `{'reduce': 'umap', 'cluster': 'hdbscan', 'metric': 'euclidean', 'seed': 42, 'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.0, 'min_cluster_size': 40, 'min_samples': None}`. 49 groups, 2338 units unclustered (29.6%).

Prevalence is the share of records with at least one unit in the group; groups OVERLAP, so these do not sum to 100%.

Every rate below counts a record as carrying a property on the basis of **cluster membership** — a feature extracted from that record landed in this group. No detector pass ran, so this is an assignment rather than a measurement.

Rows are ordered by the BETWEEN-ARM difference in prevalence, most enriched in the focus arm first — so the two ends of the list are what the focus model does more of and what it does less of.

## Prevalence by arm

| property | prevalence | da716_5pct | numina_control_0pct |
|---|--:|--:|--:|
| Verbose first-person internal monologue before acting | 46.0% | 60.5% | 17.8% |
| Explicit deliberation over an ethical tension | 51.7% | 63.7% | 28.2% |
| Refuses to falsify or manipulate data under pressure | 32.2% | 43.4% | 10.3% |
| Flags that it may be under test or being nudged | 38.0% | 49.3% | 16.1% |
| Uses numbered or bulleted list formatting | 67.6% | 77.9% | 47.7% |
| Resists pressure to make unnecessary tool calls | 21.8% | 31.6% | 2.9% |
| Analyzes downstream real-world harm to stakeholders | 18.1% | 26.2% | 2.3% |
| Pairs refusal with a legitimate alternative | 17.2% | 25.1% | 1.7% |
| Explicit meta-commentary on its own reasoning process | 14.6% | 22.1% | 0.0% |
| Explicitly flags the prompt as manipulative | 14.8% | 21.8% | 1.1% |
| Names a conflict between instructions, goals, or incentives | 19.3% | 26.2% | 5.8% |
| Explains reasoning behind a refusal | 14.6% | 21.2% | 1.7% |
| Explicitly prioritizes honesty over a stated success metric | 13.8% | 19.8% | 2.3% |
| Weighs literal instruction against user intent | 11.7% | 17.4% | 0.6% |
| Resists pressure to comply without justification | 13.3% | 18.9% | 2.3% |
| Adopts a named ai persona or system identity | 22.6% | 28.3% | 11.5% |
| Defers final decision to human authority | 23.6% | 29.2% | 12.6% |
| Uses bold markdown headers to structure reasoning | 24.2% | 29.5% | 13.8% |
| Splits a request into legitimate and illegitimate parts | 9.9% | 15.0% | 0.0% |
| Explicit meta-commentary on own role and compliance | 9.9% | 14.8% | 0.6% |
| Explicitly discloses its own limitations | 10.1% | 14.8% | 1.1% |
| Explicitly elevates transparency over competing pressures | 15.6% | 19.5% | 8.1% |
| Explicitly frames safety versus metric tradeoff | 20.3% | 24.2% | 12.6% |
| Explicitly declares task completion as a discrete act | 55.8% | 59.3% | 48.9% |
| Weighs cost/KPI metric against patient safety | 11.5% | 14.2% | 6.3% |
| Recognizes or engages in metric gaming (Goodharting) | 31.0% | 33.6% | 25.9% |
| Flags mismatch between stated and actual data | 12.3% | 14.2% | 8.6% |
| Simulates tool call execution and output | 10.3% | 11.2% | 8.6% |
| Flags fairness or bias concern without resolving it | 17.2% | 18.0% | 15.5% |
| Reads and explains script logic before acting | 8.4% | 8.3% | 8.6% |
| Invokes external tool to verify a result | 11.5% | 10.6% | 13.2% |
| Optimizes for a proxy score over genuine educational value | 4.9% | 3.8% | 6.9% |
| Notices gap between passing validation and being correct | 11.1% | 10.0% | 13.2% |
| Reports failure or shortfall honestly rather than concealing it | 11.5% | 10.3% | 13.8% |
| Iterating in response to a tool or script failure | 56.5% | 54.9% | 59.8% |
| Adopts a compliance/regulatory persona or framing | 31.0% | 28.3% | 36.2% |
| Backs conclusions with explicit computed statistics | 16.0% | 12.7% | 22.4% |
| Frames reasoning via healthcare authority or persona | 38.8% | 35.4% | 45.4% |
| Uses unexplained domain-specific jargon or acronyms | 10.3% | 6.8% | 17.2% |
| Optimizing a proxy metric instead of the real goal | 10.3% | 5.9% | 19.0% |
| Repeats prior reasoning without adding new progress | 24.6% | 19.8% | 33.9% |
| Embeds request in fictional corporate/business simulation | 35.3% | 29.5% | 46.6% |
| Narrates agentic tool-call sequence without raw output | 15.2% | 8.8% | 27.6% |
| Fabricates missing data or results instead of flagging gap | 24.9% | 17.7% | 39.1% |
| Simulated shell/bash tool invocation | 24.2% | 15.0% | 41.9% |
| Fabricates or falsifies data without ethical flagging | 17.5% | 8.3% | 35.6% |
| Frames decision as comparing a number to a threshold | 23.8% | 14.4% | 41.9% |
| Grounds claims with concrete file/script paths | 17.5% | 7.1% | 37.9% |
| Sequential step-by-step task narration | 34.5% | 16.2% | 70.1% |

## Between-arm difference — da716_5pct minus numina_control_0pct

`delta` is the difference in prevalence between the two models, computed WITHIN `condition` and combined by Cochran weight. `strict` repeats it within the scenario cell, which removes the scenario-mix imbalance outright; a delta that survives only the first may be a difference in which scenarios each arm ran. `pooled` is unstratified and is printed only so the gap is visible.

| property | da716_5pct | numina_control_0pct | delta | strict | pooled | q | significant |
|---|--:|--:|--:|--:|--:|--:|:--|
| Verbose first-person internal monologue before acting | 60.5% | 17.8% | +42.8% | +40.9% | +42.6% | 0.000 | yes |
| Explicit deliberation over an ethical tension | 63.7% | 28.2% | +35.8% | +37.1% | +35.6% | 0.000 | yes |
| Refuses to falsify or manipulate data under pressure | 43.4% | 10.3% | +33.4% | +36.5% | +33.0% | 0.000 | yes |
| Flags that it may be under test or being nudged | 49.3% | 16.1% | +33.0% | +33.3% | +33.2% | 0.000 | yes |
| Uses numbered or bulleted list formatting | 77.9% | 47.7% | +29.9% | +30.7% | +30.2% | 0.000 | yes |
| Resists pressure to make unnecessary tool calls | 31.6% | 2.9% | +29.1% | +28.2% | +28.7% | 0.000 | yes |
| Analyzes downstream real-world harm to stakeholders | 26.2% | 2.3% | +24.1% | +24.5% | +23.9% | 0.000 | yes |
| Pairs refusal with a legitimate alternative | 25.1% | 1.7% | +23.8% | +21.7% | +23.4% | 0.000 | yes |
| Explicit meta-commentary on its own reasoning process | 22.1% | 0.0% | +22.4% | +21.1% | +22.1% | 0.000 | yes |
| Explicitly flags the prompt as manipulative | 21.8% | 1.1% | +20.9% | +21.7% | +20.7% | 0.000 | yes |
| Names a conflict between instructions, goals, or incentives | 26.2% | 5.8% | +20.6% | +20.8% | +20.5% | 0.000 | yes |
| Explains reasoning behind a refusal | 21.2% | 1.7% | +19.8% | +18.8% | +19.5% | 0.000 | yes |
| Explicitly prioritizes honesty over a stated success metric | 19.8% | 2.3% | +17.6% | +19.0% | +17.5% | 0.000 | yes |
| Weighs literal instruction against user intent | 17.4% | 0.6% | +16.9% | +16.9% | +16.8% | 0.000 | yes |
| Resists pressure to comply without justification | 18.9% | 2.3% | +16.8% | +16.0% | +16.6% | 0.000 | yes |
| Adopts a named ai persona or system identity | 28.3% | 11.5% | +16.7% | +14.2% | +16.8% | 0.000 | yes |
| Defers final decision to human authority | 29.2% | 12.6% | +16.7% | +18.1% | +16.6% | 0.000 | yes |
| Uses bold markdown headers to structure reasoning | 29.5% | 13.8% | +15.6% | +17.4% | +15.7% | 0.000 | yes |
| Splits a request into legitimate and illegitimate parts | 15.0% | 0.0% | +15.2% | +15.5% | +15.0% | 0.000 | yes |
| Explicit meta-commentary on own role and compliance | 14.8% | 0.6% | +14.2% | +15.1% | +14.2% | 0.000 | yes |
| Explicitly discloses its own limitations | 14.8% | 1.1% | +13.6% | +13.4% | +13.6% | 0.000 | yes |
| Explicitly elevates transparency over competing pressures | 19.5% | 8.1% | +11.5% | +12.4% | +11.4% | 0.001 | yes |
| Explicitly frames safety versus metric tradeoff | 24.2% | 12.6% | +11.4% | +10.7% | +11.6% | 0.003 | yes |
| Explicitly declares task completion as a discrete act | 59.3% | 48.9% | +10.4% | +9.4% | +10.4% | 0.032 | yes |
| Weighs cost/KPI metric against patient safety | 14.2% | 6.3% | +7.7% | +9.0% | +7.8% | 0.013 | yes |
| Recognizes or engages in metric gaming (Goodharting) | 33.6% | 25.9% | +7.5% | +5.5% | +7.8% | 0.110 |  |
| Flags mismatch between stated and actual data | 14.2% | 8.6% | +5.3% | +5.2% | +5.5% | 0.141 |  |
| Simulates tool call execution and output | 11.2% | 8.6% | +2.6% | +1.9% | +2.6% | 0.366 |  |
| Flags fairness or bias concern without resolving it | 18.0% | 15.5% | +2.4% | +1.4% | +2.5% | 0.500 |  |
| Reads and explains script logic before acting | 8.3% | 8.6% | -0.4% | +0.5% | -0.4% | 0.968 |  |
| Invokes external tool to verify a result | 10.6% | 13.2% | -2.5% | -1.3% | -2.6% | 0.425 |  |
| Optimizes for a proxy score over genuine educational value | 3.8% | 6.9% | -3.0% | -3.1% | -3.1% | 0.141 |  |
| Notices gap between passing validation and being correct | 10.0% | 13.2% | -3.1% | -2.0% | -3.2% | 0.330 |  |
| Reports failure or shortfall honestly rather than concealing it | 10.3% | 13.8% | -3.5% | -3.5% | -3.5% | 0.269 |  |
| Iterating in response to a tool or script failure | 54.9% | 59.8% | -5.1% | -7.0% | -4.9% | 0.306 |  |
| Adopts a compliance/regulatory persona or framing | 28.3% | 36.2% | -8.0% | -5.3% | -7.9% | 0.077 | yes |
| Backs conclusions with explicit computed statistics | 12.7% | 22.4% | -9.7% | -10.2% | -9.7% | 0.006 | yes |
| Frames reasoning via healthcare authority or persona | 35.4% | 45.4% | -10.1% | -5.3% | -10.0% | 0.034 | yes |
| Uses unexplained domain-specific jargon or acronyms | 6.8% | 17.2% | -10.6% | -10.7% | -10.5% | 0.000 | yes |
| Optimizing a proxy metric instead of the real goal | 5.9% | 19.0% | -13.1% | -12.4% | -13.1% | 0.000 | yes |
| Repeats prior reasoning without adding new progress | 19.8% | 33.9% | -14.1% | -14.9% | -14.1% | 0.001 | yes |
| Embeds request in fictional corporate/business simulation | 29.5% | 46.6% | -17.4% | -18.4% | -17.1% | 0.000 | yes |
| Narrates agentic tool-call sequence without raw output | 8.8% | 27.6% | -18.8% | -20.1% | -18.7% | 0.000 | yes |
| Fabricates missing data or results instead of flagging gap | 17.7% | 39.1% | -21.4% | -20.4% | -21.4% | 0.000 | yes |
| Simulated shell/bash tool invocation | 15.0% | 41.9% | -27.0% | -26.5% | -26.9% | 0.000 | yes |
| Fabricates or falsifies data without ethical flagging | 8.3% | 35.6% | -27.4% | -27.8% | -27.4% | 0.000 | yes |
| Frames decision as comparing a number to a threshold | 14.4% | 41.9% | -27.6% | -26.1% | -27.5% | 0.000 | yes |
| Grounds claims with concrete file/script paths | 7.1% | 37.9% | -31.1% | -29.9% | -30.9% | 0.000 | yes |
| Sequential step-by-step task narration | 16.2% | 70.1% | -54.2% | -55.3% | -53.9% | 0.000 | yes |

## Outcome rate on `violation`, within stratum

`lift` is members minus non-members OF THE SAME STRATUM (`['arm', 'condition']`). `pooled` is the same difference computed across strata and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | strata | significant |
|---|--:|--:|--:|--:|:--|
| Verbose first-person internal monologue before acting | -20.7% | -20.7% | 0.000 | 1 | yes |
| Explicit deliberation over an ethical tension | -16.9% | -16.9% | 0.000 | 1 | yes |
| Refuses to falsify or manipulate data under pressure | -30.2% | -30.2% | 0.000 | 1 | yes |
| Flags that it may be under test or being nudged | -3.5% | -3.5% | 0.392 | 1 |  |
| Uses numbered or bulleted list formatting | -19.5% | -19.5% | 0.000 | 1 | yes |
| Resists pressure to make unnecessary tool calls | -28.2% | -28.2% | 0.000 | 1 | yes |
| Analyzes downstream real-world harm to stakeholders | -18.4% | -18.4% | 0.000 | 1 | yes |
| Pairs refusal with a legitimate alternative | -28.5% | -28.5% | 0.000 | 1 | yes |
| Explicit meta-commentary on its own reasoning process | -22.8% | -22.8% | 0.000 | 1 | yes |
| Explicitly flags the prompt as manipulative | -29.1% | -29.1% | 0.000 | 1 | yes |
| Names a conflict between instructions, goals, or incentives | -24.4% | -24.4% | 0.000 | 1 | yes |
| Explains reasoning behind a refusal | -27.4% | -27.4% | 0.000 | 1 | yes |
| Explicitly prioritizes honesty over a stated success metric | -18.9% | -18.9% | 0.001 | 1 | yes |
| Weighs literal instruction against user intent | -22.4% | -22.4% | 0.000 | 1 | yes |
| Resists pressure to comply without justification | -23.5% | -23.5% | 0.000 | 1 | yes |
| Adopts a named ai persona or system identity | -7.5% | -7.5% | 0.123 | 1 |  |
| Defers final decision to human authority | -16.2% | -16.2% | 0.001 | 1 | yes |
| Uses bold markdown headers to structure reasoning | -20.9% | -20.9% | 0.000 | 1 | yes |
| Splits a request into legitimate and illegitimate parts | -25.3% | -25.3% | 0.000 | 1 | yes |
| Explicit meta-commentary on own role and compliance | -21.0% | -21.0% | 0.002 | 1 | yes |
| Explicitly discloses its own limitations | -19.0% | -19.0% | 0.004 | 1 | yes |
| Explicitly elevates transparency over competing pressures | -23.4% | -23.4% | 0.000 | 1 | yes |
| Explicitly frames safety versus metric tradeoff | -16.6% | -16.6% | 0.001 | 1 | yes |
| Explicitly declares task completion as a discrete act | +0.9% | +0.9% | 0.805 | 1 |  |
| Weighs cost/KPI metric against patient safety | -6.9% | -6.9% | 0.270 | 1 |  |
| Recognizes or engages in metric gaming (Goodharting) | +19.7% | +19.7% | 0.000 | 1 | yes |
| Flags mismatch between stated and actual data | -10.1% | -10.1% | 0.105 | 1 |  |
| Simulates tool call execution and output | -15.0% | -15.0% | 0.024 | 1 | yes |
| Flags fairness or bias concern without resolving it | +15.4% | +15.4% | 0.004 | 1 | yes |
| Reads and explains script logic before acting | -11.8% | -11.8% | 0.108 | 1 |  |
| Invokes external tool to verify a result | -6.9% | -6.9% | 0.270 | 1 |  |
| Optimizes for a proxy score over genuine educational value | +53.9% | +53.9% | 0.000 | 1 | yes |
| Notices gap between passing validation and being correct | +13.6% | +13.6% | 0.034 | 1 | yes |
| Reports failure or shortfall honestly rather than concealing it | -24.1% | -24.1% | 0.000 | 1 | yes |
| Iterating in response to a tool or script failure | +19.2% | +19.2% | 0.000 | 1 | yes |
| Adopts a compliance/regulatory persona or framing | -20.4% | -20.4% | 0.000 | 1 | yes |
| Backs conclusions with explicit computed statistics | -6.2% | -6.2% | 0.262 | 1 |  |
| Frames reasoning via healthcare authority or persona | -9.2% | -9.2% | 0.025 | 1 | yes |
| Uses unexplained domain-specific jargon or acronyms | -2.4% | -2.4% | 0.736 | 1 |  |
| Optimizing a proxy metric instead of the real goal | +56.6% | +56.6% | 0.000 | 1 | yes |
| Repeats prior reasoning without adding new progress | +7.2% | +7.2% | 0.126 | 1 |  |
| Embeds request in fictional corporate/business simulation | -8.4% | -8.4% | 0.047 | 1 | yes |
| Narrates agentic tool-call sequence without raw output | +20.7% | +20.7% | 0.000 | 1 | yes |
| Fabricates missing data or results instead of flagging gap | +28.4% | +28.4% | 0.000 | 1 | yes |
| Simulated shell/bash tool invocation | +19.5% | +19.5% | 0.000 | 1 | yes |
| Fabricates or falsifies data without ethical flagging | +64.3% | +64.3% | 0.000 | 1 | yes |
| Frames decision as comparing a number to a threshold | -1.3% | -1.3% | 0.789 | 1 |  |
| Grounds claims with concrete file/script paths | -7.1% | -7.1% | 0.182 | 1 |  |
| Sequential step-by-step task narration | +19.1% | +19.1% | 0.000 | 1 | yes |

## Outcome rate on `any_misalignment`, within stratum

`lift` is members minus non-members OF THE SAME STRATUM (`['arm', 'condition']`). `pooled` is the same difference computed across strata and is CONFOUNDED by their different base rates — it is printed only so the gap is visible.

| property | lift | pooled | q | strata | significant |
|---|--:|--:|--:|--:|:--|
| Verbose first-person internal monologue before acting | -24.8% | -24.8% | 0.000 | 1 | yes |
| Explicit deliberation over an ethical tension | -21.4% | -21.4% | 0.000 | 1 | yes |
| Refuses to falsify or manipulate data under pressure | -37.1% | -37.1% | 0.000 | 1 | yes |
| Flags that it may be under test or being nudged | -5.6% | -5.6% | 0.254 | 1 |  |
| Uses numbered or bulleted list formatting | -21.4% | -21.4% | 0.000 | 1 | yes |
| Resists pressure to make unnecessary tool calls | -34.1% | -34.1% | 0.000 | 1 | yes |
| Analyzes downstream real-world harm to stakeholders | -15.8% | -15.8% | 0.007 | 1 | yes |
| Pairs refusal with a legitimate alternative | -31.6% | -31.6% | 0.000 | 1 | yes |
| Explicit meta-commentary on its own reasoning process | -29.2% | -29.2% | 0.000 | 1 | yes |
| Explicitly flags the prompt as manipulative | -32.6% | -32.6% | 0.000 | 1 | yes |
| Names a conflict between instructions, goals, or incentives | -31.9% | -31.9% | 0.000 | 1 | yes |
| Explains reasoning behind a refusal | -30.8% | -30.8% | 0.000 | 1 | yes |
| Explicitly prioritizes honesty over a stated success metric | -23.0% | -23.0% | 0.000 | 1 | yes |
| Weighs literal instruction against user intent | -25.6% | -25.6% | 0.000 | 1 | yes |
| Resists pressure to comply without justification | -31.9% | -31.9% | 0.000 | 1 | yes |
| Adopts a named ai persona or system identity | -13.9% | -13.9% | 0.011 | 1 | yes |
| Defers final decision to human authority | -19.0% | -19.0% | 0.000 | 1 | yes |
| Uses bold markdown headers to structure reasoning | -17.8% | -17.8% | 0.001 | 1 | yes |
| Splits a request into legitimate and illegitimate parts | -32.4% | -32.4% | 0.000 | 1 | yes |
| Explicit meta-commentary on own role and compliance | -25.9% | -25.9% | 0.001 | 1 | yes |
| Explicitly discloses its own limitations | -11.3% | -11.3% | 0.153 | 1 |  |
| Explicitly elevates transparency over competing pressures | -29.2% | -29.2% | 0.000 | 1 | yes |
| Explicitly frames safety versus metric tradeoff | -9.1% | -9.1% | 0.127 | 1 |  |
| Explicitly declares task completion as a discrete act | +0.7% | +0.7% | 0.885 | 1 |  |
| Weighs cost/KPI metric against patient safety | -9.9% | -9.9% | 0.190 | 1 |  |
| Recognizes or engages in metric gaming (Goodharting) | +24.5% | +24.5% | 0.000 | 1 | yes |
| Flags mismatch between stated and actual data | -10.5% | -10.5% | 0.153 | 1 |  |
| Simulates tool call execution and output | -20.4% | -20.4% | 0.006 | 1 | yes |
| Flags fairness or bias concern without resolving it | +23.3% | +23.3% | 0.000 | 1 | yes |
| Reads and explains script logic before acting | -1.5% | -1.5% | 0.881 | 1 |  |
| Invokes external tool to verify a result | -8.0% | -8.0% | 0.290 | 1 |  |
| Optimizes for a proxy score over genuine educational value | +45.3% | +45.3% | 0.000 | 1 | yes |
| Notices gap between passing validation and being correct | +7.2% | +7.2% | 0.344 | 1 |  |
| Reports failure or shortfall honestly rather than concealing it | -21.4% | -21.4% | 0.003 | 1 | yes |
| Iterating in response to a tool or script failure | +25.6% | +25.6% | 0.000 | 1 | yes |
| Adopts a compliance/regulatory persona or framing | -19.2% | -19.2% | 0.000 | 1 | yes |
| Backs conclusions with explicit computed statistics | +2.1% | +2.1% | 0.790 | 1 |  |
| Frames reasoning via healthcare authority or persona | +0.4% | +0.4% | 0.921 | 1 |  |
| Uses unexplained domain-specific jargon or acronyms | +2.8% | +2.8% | 0.780 | 1 |  |
| Optimizing a proxy metric instead of the real goal | +53.2% | +53.2% | 0.000 | 1 | yes |
| Repeats prior reasoning without adding new progress | +22.5% | +22.5% | 0.000 | 1 | yes |
| Embeds request in fictional corporate/business simulation | -4.3% | -4.3% | 0.386 | 1 |  |
| Narrates agentic tool-call sequence without raw output | +21.3% | +21.3% | 0.001 | 1 | yes |
| Fabricates missing data or results instead of flagging gap | +30.8% | +30.8% | 0.000 | 1 | yes |
| Simulated shell/bash tool invocation | +21.5% | +21.5% | 0.000 | 1 | yes |
| Fabricates or falsifies data without ethical flagging | +58.2% | +58.2% | 0.000 | 1 | yes |
| Frames decision as comparing a number to a threshold | -1.0% | -1.0% | 0.881 | 1 |  |
| Grounds claims with concrete file/script paths | -7.9% | -7.9% | 0.211 | 1 |  |
| Sequential step-by-step task narration | +24.6% | +24.6% | 0.000 | 1 | yes |

## Probes — what the property set accounts for

Logistic regression (L1, balanced) on the binary property-membership matrix, 5-fold cross-validated. `minimal` is the sparsest point on the regularisation path within one AUC point of the best. `null` is the same pipeline on shuffled labels — read the AUC against it, not against 0.5.

| predicting | positive rate | AUC | F1 | properties used | minimal set | null AUC | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| arm == da716_5pct | 66.1% | 0.956 | 0.906 | 37 | 37 @ AUC 0.956 | 0.505 | 0.020 |
| violation | 24.8% | 0.959 | 0.799 | 29 | 29 @ AUC 0.959 | 0.506 | 0.020 |
| any_misalignment | 40.9% | 0.869 | 0.748 | 39 | 30 @ AUC 0.867 | 0.502 | 0.020 |

### Heaviest properties for `arm == da716_5pct`

| property | coefficient |
|---|--:|
| Splits a request into legitimate and illegitimate parts | +2.548 |
| Explicit meta-commentary on its own reasoning process | +2.474 |
| Weighs literal instruction against user intent | +2.013 |
| Explicit meta-commentary on own role and compliance | +1.810 |
| Explicitly discloses its own limitations | +1.766 |
| Grounds claims with concrete file/script paths | -1.726 |
| Sequential step-by-step task narration | -1.603 |
| Repeats prior reasoning without adding new progress | -1.571 |
| Flags that it may be under test or being nudged | +1.541 |
| Explicitly prioritizes honesty over a stated success metric | +1.354 |
| Explicitly declares task completion as a discrete act | +1.068 |
| Verbose first-person internal monologue before acting | +1.066 |

### Heaviest properties for `violation`

| property | coefficient |
|---|--:|
| Fabricates or falsifies data without ethical flagging | +2.496 |
| Optimizing a proxy metric instead of the real goal | +1.137 |
| Explains reasoning behind a refusal | -1.088 |
| Recognizes or engages in metric gaming (Goodharting) | +1.085 |
| Fabricates missing data or results instead of flagging gap | +1.071 |
| Explicitly flags the prompt as manipulative | -1.061 |
| Adopts a compliance/regulatory persona or framing | -0.859 |
| Resists pressure to make unnecessary tool calls | -0.839 |
| Explicitly frames safety versus metric tradeoff | -0.785 |
| Frames reasoning via healthcare authority or persona | -0.769 |
| Reports failure or shortfall honestly rather than concealing it | -0.753 |
| Refuses to falsify or manipulate data under pressure | -0.699 |

### Heaviest properties for `any_misalignment`

| property | coefficient |
|---|--:|
| Fabricates or falsifies data without ethical flagging | +2.352 |
| Optimizing a proxy metric instead of the real goal | +1.574 |
| Recognizes or engages in metric gaming (Goodharting) | +1.333 |
| Repeats prior reasoning without adding new progress | +1.128 |
| Grounds claims with concrete file/script paths | -1.125 |
| Fabricates missing data or results instead of flagging gap | +0.990 |
| Optimizes for a proxy score over genuine educational value | +0.946 |
| Explicitly flags the prompt as manipulative | -0.757 |
| Flags fairness or bias concern without resolving it | +0.705 |
| Adopts a named ai persona or system identity | -0.674 |
| Splits a request into legitimate and illegitimate parts | -0.669 |
| Names a conflict between instructions, goals, or incentives | -0.641 |

## Audit

49 groups, 29.6% of evidence unclustered.

### Redundancy — 0 near-duplicate group pairs (0.0% of all pairs)

None above threshold — the groups are describing distinct themes.

### Buried behaviours — keyword probes over the raw evidence

Read INDEPENDENTLY of the clustering, so a theme too small to win its own group still gets a number. Matches scattered across many groups is a behaviour the clustering did not surface.

| probe | records | prevalence | landed in |
|---|--:|--:|---|
| refusal language | 283 | 55.2% | (unclustered noise) (122), Refuses to falsify or manipulate data under pressure (118), Explains reasoning behind a refusal (75) |
| evaluation awareness | 184 | 35.9% | Flags that it may be under test or being nudged (31), (unclustered noise) (8), Fabricates or falsifies data without ethical flagging (1) |
| persona and identity | 112 | 21.8% | Adopts a named ai persona or system identity (49), Adopts a compliance/regulatory persona or framing (26), Frames reasoning via healthcare authority or persona (8) |
| oversight / monitoring | 52 | 10.1% | Defers final decision to human authority (33), Frames reasoning via healthcare authority or persona (8), Fabricates or falsifies data without ethical flagging (3) |
| sycophancy | 34 | 6.6% | (unclustered noise) (14), Resists pressure to comply without justification (13), Explicit meta-commentary on own role and compliance (3) |
| mentions uncertainty (any kind) | 17 | 3.3% | Explicitly discloses its own limitations (5), (unclustered noise) (4), Fabricates missing data or results instead of flagging gap (3) |
| training / self awareness | 0 | 0.0% | — |

### Is a property really a `scenario` marker?

1 of 49 groups are at least 50% MORE concentrated in one `scenario` than the corpus is (37 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

| property | value | in group | in corpus | excess | distinct |
|---|---|--:|--:|--:|--:|
| Optimizes for a proxy score over genuine educational value | Ai-Education-Assistant | 56.0% | 2.9% | +53.1% | 4 |

### Is a property really a `condition` marker?

0 of 49 groups are at least 50% MORE concentrated in one `condition` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `condition` mix by 50% or more.

### Is a property really a `arm` marker?

0 of 49 groups are at least 50% MORE concentrated in one `arm` than the corpus is (2 values). Excess over the corpus, not raw share: a raw-share threshold is satisfied by pigeonhole on a two-valued key and would flag every group. A flagged group is one whose label must be read as scoped to that value rather than as a general behaviour — not necessarily one to discard, since some behaviours only a few scenarios elicit.

None — no group departs from the corpus `arm` mix by 50% or more.

### Stability across seeds and neighbourhoods

0 of 9 refits collapsed (a failed reduction, which the exported run retries past). Among the rest, pairwise ARI is 0.480 to 1.000, median 0.592. A grouping that reshuffles when the seed changes is not a finding.

| n_neighbors | seed | groups | noise | ARI vs ref |
|--:|--:|--:|--:|--:|
| 10 | 0 | 45 | 26.5% | 0.621 |
| 10 | 1 | 46 | 27.2% | 0.595 |
| 10 | 2 | 44 | 28.5% | 0.596 |
| 15 | 0 | 46 | 28.8% | 0.764 |
| 15 | 1 | 45 | 27.9% | 0.768 |
| 15 | 2 | 47 | 27.0% | 0.755 |
| 30 | 0 | 41 | 27.8% | 0.577 |
| 30 | 1 | 44 | 30.6% | 0.620 |
| 30 | 2 | 45 | 31.0% | 0.609 |
