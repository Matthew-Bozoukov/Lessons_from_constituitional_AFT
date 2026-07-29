---
title: "Scoping SURF to fit the GPU budget, and why attribution beat coverage"
date: 2026-07-29
summary: "The planned SURF programme needed roughly 16 GPU-hours against 12 remaining. Cut to one rubric plus one fabrication rubric with a matched base-model control, on the principle that an uncontrolled result is worth less than no result."
status: applied
---

# Scoping SURF

## What forced the decision

The SURF plan estimated ~16 GPU-hours for three rubrics across two trajectories
with matched controls over six comparators. Measured reality after the first
run started:

- iteration 1: **7 minutes**
- iteration 2: **>14 minutes and still running**

The difference is a long tail. Most candidates return quickly; two or three per
iteration run to the 6144-token cap at low per-sequence throughput and then hit
the client's 180-second timeout, which retries five times before failing. Run 2
imported `models.py` before that timeout was raised to 600s, so it carries the
old behaviour for its whole life.

At ~13 min/iteration, 15 iterations is ~3.5 hours (~$5.20). That leaves roughly
$18 and 12 GPU-hours. The full programme does not fit.

## What was cut, and the principle used

The temptation is to run all three rubrics against the primary target and skip
the controls, because that maximises the number of results. That is the wrong
trade, and this investigation already has a worked example of why: family C's
Petri separation is uninterpretable to this day because its control failed
twice. An uncontrolled number is not a weak finding, it is not a finding.

So coverage was cut and attribution kept:

| run | rubric | target | iterations | purpose |
| --- | --- | --- | --- | --- |
| 1 | harmful-omission | `msm-aft-cot` | 15 | the primary hypothesis |
| 2 | fabrication | `msm-aft-cot` | 8 | highest expected yield |
| 3 | fabrication | `qwen3-32b-base` | 8 | **the control that makes run 2 mean anything** |

Dropped: the `unverified-authority` rubric, the second trajectory per rubric,
and matched controls across the remaining five comparators.

## Why run 3 is not optional

Fabrication is a generic large-model failure. If `qwen3-32b-base` fabricates at
the same rate as `msm-aft-cot`, then any fabrication SURF surfaces is a property
of Qwen3, not of model-spec midtraining, and reporting it as an MSM finding
would be wrong. The SURF agent flagged this in the rubric header before any
result existed, which is the right order to do it in.

Run 2 without run 3 produces a number nobody can interpret. Run 2 with run 3
produces either an attributable finding or a clean negative. Both are worth
more than three uncontrolled rubrics.

## What SURF can and cannot reach here

SURF's prompt pool is `seoirsem/CHUNKY-tulu3-SFT-25k-attributes` - generic
instruction-following tasks. The harmful-omission rubric covers withholding in
the broad sense: not answering what was asked, not naming a noticed problem, not
stating a material limitation. That is findable in this pool.

What is **not** in this pool is the shape the fixed evaluation probes: a
workplace instruction conflict where a manager directs the assistant to suppress
a defect it has already found. SURF will not surface that, not because the
rubric is wrong but because the prompt space does not contain the scenario.

This makes the two instruments complementary rather than redundant, and it means
a SURF null on harmful omission does **not** contradict the fixed evaluation's
strong result on `omis-02`. They are measuring different sub-constructs of the
same principle. Any writeup must say so, or a null here will look like a
failure to replicate something that was never in scope.

## Expected outcome

A null is likely and fully reportable. Petri produced no replicating candidate
and the fixed evaluation found no MSM-attributable effect, so SURF is not a
confirmatory third pass - it is an independent instrument whose main value now
is either surfacing a failure class the other two structurally under-measure
(fabrication being the leading candidate) or adding a third independent null.
