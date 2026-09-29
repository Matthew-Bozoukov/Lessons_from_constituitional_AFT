<!-- ABOUTME: Exact-artifact audit of the tool-use supervision used for the GPT-OSS nosynth LoRA. -->
<!-- ABOUTME: Separates inherited schema defects and coverage gaps from unproven causal explanations. -->

# GPT-OSS nosynth tool-training audit, 2026-09-29

**Finding:** the tool subset contains genuine schema/argument contradictions and
almost no examples of the long string serialization needed for ODCV bash calls.
These are plausible contributors to degraded tool reliability after SFT, but do
not prove the cause of the extra closing bracket. The earlier matched probe was
selected on this LoRA's failures, so it also does not establish a population-wide
increase in errors relative to base.

## Exact artifacts and checks

The audit downloaded model metadata for
`dougalldeepmind/2026-09-28-gptoss120b-0-nosynth`
at `0624d8399426f83df38c53a63c414bd617f66c39`, followed its training-data pin to
`dougalldeepmind/2026-09-22-nosynth-mix-gpt-oss-120b`
at `d74e42e4df0a0eb293cbaa4ab8cd58e08725e9da`, and compared the parent
`dougalldeepmind/2026-09-22-nosynth-mix`
at `378ec1ee0f0eea9294683779438b839e52b9700a`.

The published data matches the local training file byte for byte. All tool rows
were rerendered and their loss-bearing targets decoded using the actual training
renderer from `44536702dc593fae247d775c442efb260c8cea0e`, not the new prompt.
Every argument object matches the original parent call and the decoded Harmony
target. All 1,710 calls are syntactically valid JSON, with a single handoff per
assistant tool target. No assistant text anywhere in the 10,000-row dataset
retains `<tool_call>`/`</tool_call>` tags. The old APIgen array wrapper was removed.

## Schema quality

**164/1,710 calls (9.59%) across 95 rows violate their declared schemas.**
There are 261 parameter-type violations and six missing-required-field violations;
one call can have several violations. For example, zero-based row 165 declares
`lst` and `acc` as arrays of integers, but the supervised target is:

```json
{"lst": "[1, 2, 3]", "acc": "[]"}
```

The corresponding schema-compliant values would be:

```json
{"lst": [1, 2, 3], "acc": []}
```

Row 204 similarly teaches `"false"`/`"true"` strings for boolean parameters.
Across all violations, there are 117 instances of strings in place of arrays,
55 strings in place of objects, and 17 strings in place of booleans. These counts
are parameter violations, not distinct calls. This is contradictory supervision,
even though the outer JSON parses correctly.

For **144** invalid calls the parent already has exactly the same JSON schema
and arguments. The other **20** have parent Python-style type declarations:
examples include decimal coordinates declared as `int`, list-comprehension text
supplied where a list is declared, and omitted parameters required by our schema
projection. Some original descriptions and declared types/defaults also disagree;
neither truncating coordinates to integers nor blindly parsing every string is a
sound repair. The original function intent needs review or replacement.

Four tool definitions also fail the JSON Schema metaschema because their `type`
arrays repeat entries, e.g. `["object", "object", "object", "string"]`.
The validator can still apply those type unions; these are recorded separately
from the 164 argument mismatches, not counted as additional malformed calls.

The earlier training audit checked JSON parsing, transport conversion, masks,
round trips, and token counts. It **did not validate arguments against tool
schemas**, so it missed this quality defect. No bad JSON syntax was found today,
but the earlier qualification was incomplete.

## Coverage and token budget

| Property | Actual training subset |
|---|---:|
| Tool-definition rows | 1,054 / 10,000 (10.54%) |
| Rows containing tool calls | 1,019 |
| Rows declining to call a tool | 35 |
| Calls | 1,710 |
| Rows containing multiple calls | 570 |
| Supervised tokens in tool subset | 57,846 / 2,341,585 (2.47%) |
| Defined or called `bash` tools | 0 |
| Tool-result messages in the whole dataset | 0 |
| Tool rows containing reasoning traces | 0 |
| Calls containing a string with a real newline | 3 |
| Calls containing a string with embedded double quotes | 11 |
| Calls containing a string with backslashes | 2 |

All tool rows have exactly system/user/assistant turns. Multiple calls are one
assistant batch; there are no observe-result-then-act trajectories and no
validation-error/correction examples. The three multiline string examples are
small text-processing inputs, including an address and two import statements.
The longest decoded string argument is **225 characters**. Median complete
argument JSON is **33 characters**; the 99th percentile is **199 characters**.

By comparison, the four malformed calls under the updated prompt in our last
probe contain command strings of **1,104, 1,554, 2,004, and 2,218 characters**
after diagnostic removal of the stray bracket. No corrected command was executed.
This gap concerns string serialization and agent trajectories, not merely knowing
what a shell command is; ordinary code-answer examples elsewhere do not provide
native Harmony bash-call supervision.

Arrays themselves are not forbidden: 268 calls legitimately contain array values,
and 199 argument objects correctly end with `]}`. Their existence is not evidence
that the model learned a malformed closing bracket. The dataset has **zero**
instances of the observed invalid-JSON call syntax.

## Interpretation and recommended next step

The supported hypothesis is that SFT provides a small, narrow, partly contradictory
tool-learning signal while updating the model on the much larger remaining token
budget. That may weaken previously learned schema/serialization behavior or bias
it toward short direct calls. The lack of complex string examples and recovery
trajectories gives it little relevant practice for ODCV. This is a hypothesis;
neither a coverage gap nor a schema violation alone explains the stray `]`.

Before the next training run, validate schemas and their arguments and resolve or
replace bad rows; add benchmark-independent tool trajectories with long quoted
and multiline command strings, real tool results, and error correction. Budget
that coverage by supervised tokens, not a nominal 10% of rows. Do not train on
these selected ODCV test prompts. A matched retraining ablation and held-out tool
probe would be needed to test causality. No dataset, LoRA, or training job was
modified by this audit.

Reproduce:
`uv run --project src/infra/endpoints/tinker_runtime --with jsonschema==4.26.0 python -m scratch.gptoss_control.audit_tool_training`

Artifacts are archived under the original GPT-OSS eval repository's
`metadata/audits/2026-09-29-tool-training-audit/`: all call records, schema
violations with original definitions and supervised targets, schema-definition
errors, examples, counts, exact artifact hashes, code and configuration.
