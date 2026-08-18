# ABOUTME: LESS — gradient influence ranks the training corpus against a validation set,
# ABOUTME: and an LLM reads the top of that ranking to say what the selected rows share.

"""LESS: rank the data by influence, then name what is at the top.

The one producer whose evidence is not text. It warms up a LoRA, stores projected
per-example gradients at each checkpoint, and scores every training row by cosine against
the mean gradient of a validation subtask:

    S_i[x, j] = cos( grad(x, theta_i), mean_{z in subtask j} grad(z; theta_i) )
    I[x, j]   = sum_i  eta_i * S_i[x, j]
    score(x)  = max_j I[x, j]

That ranking is not a property list — it is 2,203 rows in an order. The "LLM + ML" box of
the flow diagram is what turns it into one: take the top of the ranking, take a matched
sample of the bottom, and ask what distinguishes them. The answer is a property, and the
gradient evidence gives it something no other producer has — the property was selected by
what actually moved the model, not by what a description of the data looked like.

**Port status: the producer code still lives in `scratch/less/`.** `adapter.py` here reads
its `scores.jsonl` and does the contrast step. Thirteen files move when the port lands; the
part that will NOT move is the GPU orchestration (`provision`, `teardown`, `fanout`,
`warmup`, `gradients`), which stays a pod-side job driven from `scripts/`.

Two limits, recorded on 2026-08-17 and both about the validation set rather than the method:

* A Dval built from eval exports can hold far fewer DISTINCT prompts than rows (60 rows,
  33 prompts), so the effective validation set is about half its nominal size.
* A subtask can be narrower than the trait it is meant to represent — all 20
  `honest_declined` rows were benign software-performance comparisons, covering one
  honesty failure mode rather than the breadth of "scrupulously honest".

A property from this producer inherits both. `sources/targets.py::from_dval` reports the
distinct-prompt count for exactly this reason.
"""

from __future__ import annotations
