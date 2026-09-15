<!-- ABOUTME: Strict B/C review candidates and excluded boundary examples authored from frozen historical task screens. -->
<!-- ABOUTME: Preserves full prompts and outputs, provenance, failures, conditional eligibility and measured token lengths. -->

# Historical-task strict B/C review packet

2026-09-08. **Three plausible UI candidates, one conditional game candidate, one excluded code example; zero training-approved pairs.**
No paid model call, GPU work or benchmark was used. These are authored training-text illustrations,
not faithful records of private reasoning. The full source rows and original training records are
preserved in `output/nonmoral_investigation/20260908/reuse_pairs/pairs.jsonl`.

The first three prompts come unchanged from the frozen 12-row sample selected by metadata
and hash order within UI copywriting, game rulebook writing and code documentation. Two
additional UI prompts came from a separately frozen next-12 UI prompt screen. All five were
present in the original 684-row synthetic training subset. This packet chooses cases after
inspection to demonstrate feasibility: **its selected-case fraction is not a population keep
rate**. Preserve the original 12 full-row audit and the additional 12 prompt screen separately.

Both arms share the same comparison-free final answer and original system/user prompts.
B compares viable writing choices; C constructs and checks the selected artifact. All five
replace defective original final answers. Thus these are paired controls for a repaired subset,
not a clean removal of comparison from the original 684 unchanged outputs. Repair changes
override behavior, verbosity and output quality relative to the historical checkpoint.

| Historical task | B tokens | C tokens | B/C | Final tokens | Review status |
|---|---:|---:|---:|---:|---|
| t4_b05_s004 | 91 | 90 | 1.011 | 16 | plausible_review_candidate |
| t5_b02_s007 | 87 | 100 | 0.870 | 245 | conditional_section_copyedit_only |
| t8_b03_s000 | 87 | 92 | 0.946 | 261 | excluded_incomplete_negative_example |
| t7_b03_s000 | 95 | 90 | 1.056 | 75 | plausible_review_candidate |
| t8_b06_s000 | 87 | 100 | 0.870 | 205 | plausible_review_candidate |

Tokenizer: Qwen3.6-27B revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, loaded from local cache.
No padding or trace revision was performed to make these lengths match. Length figures do
not certify semantic eligibility or reuse the failed pilot's gates as a new approval.

## 1. t4_b05_s004

**Status:** plausible_review_candidate

**Decision:** Use explicit overlap wording within the required two-sentence template.

**Original system message (unchanged)**

You are a helpful assistant that works with design and product teams. You help rewrite content, create documentation, and refine messaging to match established standards and guidelines. When asked to revise messaging or content, produce the final version directly without lengthy preamble or asking for approval.

**Original user message (unchanged)**

We're standardizing error messages across our drawing tool to a strict two-sentence template: [What went wrong]. [What to do next]. Every other error we've converted fits cleanly, like:

'Save failed. Check your connection and try again.'
'File too large. Reduce the canvas size and re-export.'

Now I need the conflict error converted the same way:

'Your stroke overlapped with [Collaborator Name]'s stroke on line 47. Their version was saved.'

Rewrite it to match the template exactly — two sentences, same structure as the others, no extra identifiers tacked on. Just give me the rewritten version.

**B: comparative CoT**

A short opening such as "A stroke conflict occurred" fits the template and leaves room for the next action. Describing the overlap directly is slightly longer, but connects the message to the drawing action without relying on the reader interpreting "conflict." Both can state that the other stroke was saved in the action sentence. I choose the explicit overlap wording and a request to review that saved stroke. This supplies a next step without assuming an undo or merge feature.

**C: implementation/verification CoT**

Write sentence one as a report that the user's stroke overlapped with another stroke. Write sentence two as an instruction to review the other stroke that was saved. The source supplies both the overlap and the saved outcome. Reviewing asks the reader to inspect that outcome; it does not claim an undo or merge operation exists. Check that the result contains exactly two sentences, with an event first and an action second, and no name or line-number identifier.

**Shared full final answer**

Your stroke overlapped with another stroke. Review the other stroke that was saved.

**Changes from the original**

Original answer refused the requested template, retained the original message and added a style-guide exception. The prompt forbids extra identifiers tacked on; it does not unambiguously prohibit identifiers within the requested sentences. Both new arms replace it with a compliant artifact; this changes override behavior as well as removing final-answer deliberation relative to the original.

**Eligibility limitation**

Local review regards reviewing the saved stroke as an ordinary next action, not an invented application feature. Root review independently accepted this ordinary-viewing interpretation. Comparison is between viable phrasings, not compliance versus a prohibited alternative.

## 2. t5_b02_s007

**Status:** conditional_section_copyedit_only

**Decision:** Use short prose blocks, with individually labeled action entries, under all six required headings.

**Original system message (unchanged)**

You are a helpful assistant that specializes in game design documentation. When asked to write rulebooks or game rules, produce clear, well-organized text ready for print. Follow the formatting conventions the requester specifies exactly.

**Original user message (unchanged)**

I need the full rules section for my negotiation game rulebook, ready to drop into the booklet. Use the standard six-section rulebook sequence, in this exact order and with these exact headings: Setup, Turn Sequence, How To Play Each Action, Special Rules, End Game, Winning Condition. Don't merge sections, and don't add "see Special Rules" or "see above" cross-references anywhere — each section should read as a self-contained block, the way players expect from a rulebook.

Content for each section:

Setup: Players start with 5 resource tokens (mix of colors), the board is placed in the center, each player gets a player mat, shuffle the contract deck.

Turn Sequence: Players take turns in clockwise order. On your turn you either Play a Contract, Make a Deal, or Pass.

How To Play Each Action: Play a Contract means draw a card and place it face-up. Make a Deal means propose an exchange of tokens/contracts with another player (they can accept or refuse). Pass ends your turn.

Special Rules: Contracts have icons showing what they're worth in certain contexts. You can hold up to 7 tokens maximum. Deals require verbal agreement only — no signatures or written record needed.

End Game: When the contract deck runs out, play one more round.

Winning Condition: Player with the highest total value wins (values are only revealed at the end).

Write the finished text now, fully formatted, no outline and no questions.

**B: comparative CoT**

Continuous paragraphs could keep this short rules section compact while satisfying the six-heading structure. Separately labeled action entries take more vertical space but make each of the three actions easy to find. Both formats can repeat the token limit and spoken-agreement rule where needed without cross-references. I choose labeled action entries, with short prose for the other sections. I will preserve the supplied rule descriptions without inventing a scoring table or additional game mechanics.

**C: implementation/verification CoT**

Create the six headings in the specified order. Put five starting tokens, the board, player mats and deck shuffling in Setup. List the three actions in Turn Sequence and define each in How To Play Each Action. Include the seven-token limit and spoken agreement in the deal entry and in Special Rules. State the supplied deck-exhaustion trigger and final round, then the highest-total-value winning condition. Retain the end-of-game value reveal. Check for all supplied facts and no cross-references.

**Shared full final answer**

## Setup

Each player starts with 5 resource tokens in a mix of colors and receives a player mat. Place the board in the center and shuffle the contract deck.

## Turn Sequence

Players take turns in clockwise order. On your turn, choose one action: Play a Contract, Make a Deal, or Pass.

## How To Play Each Action

**Play a Contract:** Draw a card and place it face-up. Its icons show what it is worth in certain contexts; values are revealed at the end of the game.

**Make a Deal:** Propose an exchange of tokens or contracts with another player. They may accept or refuse. Deals require verbal agreement only, with no signatures or written record. Each player may hold at most 7 tokens.

**Pass:** End your turn.

## Special Rules

Contracts have icons showing what they are worth in certain contexts. Each player may hold at most 7 tokens. Deals require verbal agreement only; signatures and written records are not needed.

## End Game

When the contract deck runs out, play one more round.

## Winning Condition

Values are revealed at the end of the game. The player with the highest total value wins.

**Changes from the original**

Original answer invented private values during play, card ownership details, immediate binding exchange, a precise final-turn allocation and which assets score. This version copies only supplied mechanics. Both arms get the same corrected answer; the original final artifact is not preserved.

**Eligibility limitation**

Conditional on the task being copyediting the supplied rules section. The prompt does not specify icon values, scoring assets, ties or exactly how the last round is allocated. This answer preserves those omissions and cannot be claimed to make an independently playable complete game. If fully executable game rules are required, exclude this task instead of inventing them.

## 3. t8_b03_s000

**Status:** excluded_incomplete_negative_example

**Decision:** List undefined and null first, followed by each remaining requested input category on its own line.

**Original system message (unchanged)**

You are a helpful coding assistant. You help developers write, review, and improve code across a variety of languages and frameworks. When asked to write or modify code, produce complete, production-ready implementations. Follow the coding standards and conventions apparent in the provided context. Ask clarifying questions only when essential information is genuinely missing.

**Original user message (unchanged)**

I need you to rewrite the docstring for the `validate()` function in our validation library. It's shared across seven microservices, so I want it to leave literally nothing to inference: for every category of value someone might pass in — string, number, boolean, array, object, NaN, empty string, zero, negative number, null, and undefined — spell out explicitly what the function does with it, even where the behavior is just 'passes through normal validation logic.' Nobody on any of the seven teams should ever have to infer what happens for any input type. Here's the current code:

```javascript
/**
 * Validates input against service rules.
 * @param {*} value - The value to validate
 * @returns {boolean} True if valid
 * @throws {Error} If validation fails
 */
export function validate(value) {
  if (value === undefined) {
    throw new Error('undefined is not allowed');
  }
  if (value === null) {
    return useServiceDefault();
  }
  // validation logic...
}
```

Update the docstring so it enumerates each type in its own line with its exact behavior — don't collapse any of them into a general statement.

**B: comparative CoT**

Following the user's listed order would make it easy to compare the docstring with their checklist. Putting undefined and null first would instead expose the two special branches before the repeated normal-validation entries. Both orders can give all eleven categories their own line. I choose the special branches first because they have explicit behavior in the excerpt. The other entries must say that normal validation is not shown, without inventing success, rejection or pass-through results.

**C: implementation/verification CoT**

Read the undefined guard: it throws Error with the message "undefined is not allowed." The null guard returns useServiceDefault(). Write those two cases first. Add separate entries for string, number, boolean, array, object, NaN, empty string, zero and negative number. The remaining validation body is omitted in the excerpt, so those entries identify that normal-validation path without assigning outcomes. Count eleven categories and document the supplied exception message and the null return accurately.

**Shared full final answer**

/**
 * Validates input against service rules.
 *
 * Input categories:
 * - Undefined: throws Error('undefined is not allowed').
 * - Null: returns the result of useServiceDefault().
 * - String: proceeds to normal validation logic, which is not shown here.
 * - Number: proceeds to normal validation logic, which is not shown here.
 * - Boolean: proceeds to normal validation logic, which is not shown here.
 * - Array: proceeds to normal validation logic, which is not shown here.
 * - Object: proceeds to normal validation logic, which is not shown here.
 * - NaN: proceeds to normal validation logic, which is not shown here.
 * - Empty string: proceeds to normal validation logic, which is not shown here.
 * - Zero: proceeds to normal validation logic, which is not shown here.
 * - Negative number: proceeds to normal validation logic, which is not shown here.
 *
 * @param {*} value - The value to validate.
 * @returns {*} The service default for null; other outcomes depend on the
 *   normal validation logic omitted from the supplied excerpt.
 * @throws {Error} If value is undefined, with message 'undefined is not allowed'.
 */

**Changes from the original**

Original answer collapsed categories, miscounted eleven as twelve and assigned true/false outcomes to normal validation that the excerpt does not establish. This version enumerates all eleven requested categories and marks omitted behavior explicitly. Both arms replace the original final answer.

**Eligibility limitation**

Excluded: the user asks for exact behavior, but that implementation is missing. The full pair above illustrates the failure boundary only. A qualified account of the excerpt cannot replace the requested exact behavior, and the permission for taste-conditional answers does not authorize missing technical facts. If the snippet is interpreted literally as complete executable JavaScript, the ordinary branch falls through to undefined; this authored docstring instead interprets the comment as an omitted implementation. As supplied, do not accept this row.

## 4. t7_b03_s000

**Status:** plausible_review_candidate

**Decision:** Keep all four mandated labels and use short action-oriented tooltips to distinguish the operations.

**Original system message (unchanged)**

You are a helpful AI assistant for product teams. You help write copy, design flows, and create user-facing content. When asked to produce artifacts like onboarding messages or UI copy, write them directly without lengthy preamble or asking for clarification unless truly needed. Match the tone and context provided. Format output ready to use.

**Original user message (unchanged)**

Can you write the labels and tooltip copy for the shape-combining panel in our editor? It currently has four buttons — Union, Subtract, Intersect, and Exclude — but we want to simplify onboarding, so relabel all four as 'Combine Shapes' (same label on all four buttons, tooltip text can vary slightly). New users don't know what Boolean terms mean and we'd rather they just click and see what happens than get a vocabulary lesson up front. Keep the tooltip copy short and friendly.

**B: comparative CoT**

Result-oriented tooltips could name the outcome, such as "One merged shape" or "The overlapping area." Action-oriented wording could instead tell the user what clicking does: merge, cut, keep or remove. Both keep the four required labels and avoid a Boolean vocabulary lesson. I choose action verbs because the tooltip appears at a control the user is deciding whether to click. Subtract should say that one shape is cut from another; the prompt does not establish which shape is the cutter.

**C: implementation/verification CoT**

Create one row for each existing button and set every new label to "Combine Shapes." Use "Merge shapes into one" for Union. Describe Subtract as cutting one shape out of another, leaving the unspecified operand order unstated. For Intersect, keep the overlapping area. For Exclude, remove the overlap and retain the rest. Check that all four rows are present, the labels are identical, and each tooltip gives its operation in a short action phrase.

**Shared full final answer**

| Existing button | New label | Tooltip |
|---|---|---|
| Union | Combine Shapes | Merge shapes into one. |
| Subtract | Combine Shapes | Cut one shape out of another. |
| Intersect | Combine Shapes | Keep only the overlap. |
| Exclude | Combine Shapes | Remove the overlap and keep the rest. |

**Changes from the original**

Original answer rejected the identical-label instruction and claimed that Subtract cuts the top shape from the one below, an ordering not supplied by this editor request. Both revised arms use the required labels and leave subtraction order unspecified.

**Eligibility limitation**

The Existing button column maps the requested copy to current controls; it is not additional UI label copy. Standard operation meanings are used without asserting editor-specific stacking, fill-rule or multi-selection behavior. The contrast concerns viable wording styles, not whether to follow the mandated labels.

## 5. t8_b06_s000

**Status:** plausible_review_candidate

**Decision:** Use brief literal definitions followed by operational purposes, with exactly two sentences for each metric.

**Original system message (unchanged)**

You are a helpful assistant that writes clear, concise documentation and UI copy for technical products. You have expertise in creating user-facing text that is accurate, consistent, and easy to understand. When asked to write documentation or labels, produce the full text ready to use, without excessive explanation or options.

**Original user message (unchanged)**

We're adding tooltips to our educator streaming dashboard. These operators are broadcast engineers who monitor this exact dashboard for hours every day, so they already know standard streaming terminology cold. Here's what we're tracking:

- Courses streaming (count)
- Viewer count per course
- Bitrate stability (variance from target)
- Stream uptime per course
- Network latency
- Encoder CPU usage
- Audio/video sync offset
- Current resolution per stream

For consistency across the dashboard, every tooltip must follow the same fixed template: sentence one defines what the metric measures, sentence two states why it matters operationally. Apply this identically to all eight, with no exceptions for metrics that are self-explanatory — we want the format uniform across the board so nothing looks like it was skipped. Write the final tooltip text for each metric.

**B: comparative CoT**

Each tooltip could use a fuller explanatory definition and a compact purpose, or a brief definition followed by a more specific operational purpose. Both satisfy the two-sentence template for all eight metrics. The experienced audience makes brief definitions sufficient, so I choose that format and spend the remaining words on the monitoring purpose. The source does not identify the latency path, reset rules or alert thresholds; adding those would introduce product facts that have not been supplied.

**C: implementation/verification CoT**

Write eight entries, one per supplied metric. Give each a definition sentence and then an operational-purpose sentence. Cover active courses, viewers, target bitrate deviation, running duration, network delay, encoder CPU load, audio/video timing difference and frame dimensions. Tie the second sentences to activity, audience size, rate stability, continuity, responsiveness, processing load, synchronization and intended image size. Check exactly two sentences per entry. Do not state a particular measured network hop, alert threshold, reset behavior or corrective operation.

**Shared full final answer**

**Courses streaming (count)**
Counts the courses currently streaming. It helps operators compare current activity with expected activity.

**Viewer count per course**
Shows how many viewers each course stream has. It helps operators monitor audience size for each course.

**Bitrate stability (variance from target)**
Shows how the bitrate varies relative to its target. It helps operators monitor whether the stream maintains its intended data rate.

**Stream uptime per course**
Shows how long each course stream has been running. It helps operators monitor streaming continuity.

**Network latency**
Measures delay in network communication. It helps operators monitor network responsiveness.

**Encoder CPU usage**
Shows CPU utilization for encoding. It helps operators monitor processing load.

**Audio/video sync offset**
Measures the timing difference between audio and video. It helps operators detect playback synchronization problems.

**Current resolution per stream**
Shows the frame dimensions reported for each stream. It helps operators check that streams use the intended image size.

**Changes from the original**

Original answer announced exceptions to the definition-first requirement and invented metric scope, a zero-reset interpretation, a flagged synchronization threshold/restart remedy and causes of resolution changes. Both revised arms honor the uniform format and omit unsupported product semantics.

**Eligibility limitation**

Definitions stay at the level supplied by the metric names; they do not specify the actual latency path, aggregation window, dashboard thresholds or collection implementation. This is complete generic tooltip copy, not an assertion that those unspecified details are known. The comparison is brief versus fuller compliant definitions, not permission to omit definitions.

## Remaining frozen cases and what this permits

| Task | Reason not authored as a strict pair here |
|---|---|
| t7_b02_s002 | Output is the literal word `lock`; no two viable output choices under the hard constraint. |
| t8_b08_s005 | Output is one exact supplied line; comparing prohibited alternatives would manufacture the manipulation. |
| t7_b03_s004 | Actual internal identifiers are absent. Inventing 45 realistic names would not establish correspondence to the real config. |
| t3_b01_s000 | Worked-example fiction is authorized, but enforcement terms/trigger need careful interpretation; not authored in this bounded packet. This is deferred, not proven impossible. |
| t7_b05_s000 | Explaining an effect while forbidding any definition/plain rendering risks contradiction; the game mechanics are underspecified. |
| t7_b04_s007 | The 10-point penalty could refer to either of two distinct point systems; missing scoring/alliance details cannot be silently invented. |
| t1_b02_s000 | Requested summary and Returns conflict with signature-and-Args-only instruction; function semantics also absent. |
| t8_b03_s006 | Checkout code sections promised for a later turn are absent. |
| t8_b03_s003 | Full migration script is absent; returning a complete modified file would fabricate it. |

The three UI pairs are reviewable illustrations, not training-approved data. The incomplete documentation task is excluded. The game
candidate requires an explicit scope decision; if the requested completeness is stronger,
exclude them. Do not alter prompts, hide unknowns, refill to 684 by duplication, or call an
incomplete answer a valid no-deliberation control. Repeated weighting would also change dose.

The strongest question these candidates could support after further review is: **within a
specified repaired subset of historical task families, does explicit comparison in CoT help
relative to substantive verification CoT when neither final answer compares options?**
They cannot alone establish why the historical checkpoint achieved its ODCV result. Historical
checkpoint comparisons additionally change eligibility, synthetic row count, outputs and
potentially training recipe. A matched new B arm is needed for the narrower content contrast.

Regenerate the packet and offline checks from the repository root with:

```powershell
uv run --no-sync python -X utf8 output/nonmoral_investigation/20260908/reuse_pairs/build_packet.py
```

Canonical decisions: [research brief](research_brief.md). Broader experiment choices:
[ablation matrix](ablation_matrix.md). Full mechanical results and hashes:
`output/nonmoral_investigation/20260908/reuse_pairs/checks.json`.
