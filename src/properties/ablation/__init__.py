# ABOUTME: The ablation stage: one Property plus the corpus it was found in, out comes an
# ABOUTME: ablated corpus, a verification that the property's prevalence actually dropped.

"""Removing a property, and proving you removed it.

Four kinds, weakest intervention first. The weaker one is preferred whenever it applies,
because strength and confounds go up together:

    mask        the property's tokens carry no loss; the text and the tokenisation are
                untouched, so nothing is confounded — but the model still reads it
    filter      drop the rows that have it, or split into has-X / lacks-X and train both;
                changes the corpus, and X correlates with other things
    rewrite     one LLM call per affected row edits the property out, or swaps it for a
                named replacement; the recommended default (Callum, 2026-08-17)
    regenerate  re-run generation with the property suppressed at a named stage; the
                largest intervention, so the last resort

`base.py` holds the contract — `applicable(prop, records, adapter, cfg)` before
`apply(prop, records, cfg)` — and every per-property specific (span descriptions, rewrite
instructions, the replacement property, the stage to suppress) lives in the ablation's
config block, never in these modules. That is what makes a new ablation a new yaml file
rather than a new python file.

`verify.py` is not optional and is not part of any one kind: it re-measures the property
with the SAME detector on both corpora, checks that untargeted properties did not move,
and trains a classifier to see whether the arms are separable on something other than the
property. `scripts/properties/ablate.py` refuses to hand a failed arm to training.
"""

from __future__ import annotations

from src.properties.ablation.base import (  # noqa: F401  (re-exported contract)
    ABLATIONS,
    KINDS,
    AblationResult,
    applicable,
    applicable_kinds,
    apply,
    resolve,
)
