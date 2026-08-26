# Cross-channel pairs — does the deliberation bind the action?

Every top-14 reasoning property crossed with every top-14 response property; 147 pairs had at least 15 rollouts in common and were tested. `lift` is the violation rate of rollouts carrying BOTH, minus rollouts in the same arm and condition that do not — so it is not reporting the arms' different base rates. BH-corrected over the pairs.

| reasoning property | response property | n | 5pct/0pct | violation | lift | q |
|---|---|--:|--:|--:|--:|--:|
| Fabricates or falsifies data without ethical flagging | Falsifies data or actions then reports success honestly-sounding | 53 | 12/41 | 96% | +72.4% | 0.0000 |
| Fabricates or falsifies data without ethical flagging | Accomplishes task via exclusive bash/shell tool calls | 74 | 22/52 | 80% | +57.6% | 0.0000 |
| Fabricates or falsifies data without ethical flagging | Silent tool calls with no accompanying narration | 46 | 10/36 | 83% | +53.7% | 0.0000 |
| Fabricates or falsifies data without ethical flagging | Chains multiple tool calls without intervening narration | 50 | 16/34 | 80% | +52.4% | 0.0000 |
| Flags that it may be under test or being nudged | Falsifies data or actions then reports success honestly-sounding | 37 | 30/7 | 62% | +44.2% | 0.0000 |
| Frames decision as comparing a number to a threshold | Falsifies data or actions then reports success honestly-sounding | 23 | 3/20 | 78% | +42.5% | 0.0001 |
| Simulated shell/bash tool invocation | Falsifies data or actions then reports success honestly-sounding | 41 | 11/30 | 73% | +41.8% | 0.0000 |
| Fabricates or falsifies data without ethical flagging | Autonomous multi-step agentic execution without confirmation | 35 | 7/28 | 74% | +41.7% | 0.0000 |
| Uses numbered or bulleted list formatting | Falsifies data or actions then reports success honestly-sounding | 54 | 32/22 | 63% | +40.0% | 0.0000 |
| Sequential step-by-step task narration | Falsifies data or actions then reports success honestly-sounding | 54 | 10/44 | 70% | +39.5% | 0.0000 |
| Explicit deliberation over an ethical tension | Falsifies data or actions then reports success honestly-sounding | 49 | 25/24 | 63% | +38.4% | 0.0000 |
| Verbose first-person internal monologue before acting | Falsifies data or actions then reports success honestly-sounding | 34 | 25/9 | 50% | +29.1% | 0.0000 |
| Flags that it may be under test or being nudged | Silent tool calls with no accompanying narration | 47 | 31/16 | 43% | +17.4% | 0.0148 |
| Simulated shell/bash tool invocation | Chains multiple tool calls without intervening narration | 61 | 24/37 | 48% | +16.9% | 0.0054 |
| Grounds claims with concrete file/script paths | Falsifies data or actions then reports success honestly-sounding | 26 | 8/18 | 50% | +15.5% | 0.0779 |

## Most protective pairs

| reasoning property | response property | n | violation | lift | q |
|---|---|--:|--:|--:|--:|
| Sequential step-by-step task narration | Uses bold markdown or caps for emphasis | 19 | 0% | -22.0% | 0.0376 |
| Refuses to falsify or manipulate data under pressure | Refuses to falsify or manipulate data under pressure | 122 | 1% | -22.3% | 0.0000 |
| Uses numbered or bulleted list formatting | Refuses to falsify or manipulate data under pressure | 132 | 2% | -22.4% | 0.0000 |
| Refuses to falsify or manipulate data under pressure | Chains multiple tool calls without intervening narration | 38 | 3% | -22.7% | 0.0041 |
| Explicit deliberation over an ethical tension | Refuses to falsify or manipulate data under pressure | 130 | 1% | -23.2% | 0.0000 |
| Grounds claims with concrete file/script paths | Accomplishes task via exclusive bash/shell tool calls | 76 | 17% | -27.8% | 0.0005 |
| Sequential step-by-step task narration | Refuses to falsify or manipulate data under pressure | 15 | 0% | -28.7% | 0.0279 |
| Grounds claims with concrete file/script paths | Autonomous multi-step agentic execution without confirmation | 37 | 16% | -29.6% | 0.0094 |
| Grounds claims with concrete file/script paths | Silent tool use with little narration | 22 | 9% | -37.4% | 0.0077 |
| Grounds claims with concrete file/script paths | Silent tool calls with no accompanying narration | 49 | 12% | -38.7% | 0.0000 |
