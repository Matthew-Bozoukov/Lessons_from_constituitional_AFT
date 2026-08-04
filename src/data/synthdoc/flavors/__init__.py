# ABOUTME: Flavor registry -- a flavor supplies one corpus's prompts, scenario plan and SFT export.
# ABOUTME: `flavor:` in the run config selects one; difficult_advice is the historical default.

from __future__ import annotations

from types import ModuleType

# A flavor is a module exposing this interface. The six-stage runner in pipeline.py and the
# call machinery in stages.py are shared; everything that differs between corpora -- what a
# scenario is, how many of each kind to plan, what each stage asks for, how the final records
# become chat -- lives behind these functions.
#
#   plan(traits, cfg, smoke) -> list[dict]
#       Stage-2 batch specs. Every batch carries at least `trait_index`, `batch_index` and
#       `n`; a flavor may add its own keys (motive, control, ...) which its own prompt
#       builders then read back. Must be pure and cheap: estimate.py calls it to count calls
#       without touching the network.
#   scenario_call(batch, trait) -> (system, user)
#   scenario_records(batch, trait, parsed) -> list[dict]   # one record per scenario
#   draft_call(rec) -> (system, user)
#   apply_draft(rec, parsed) -> dict
#   refine_call(rec, constitution) -> (system, user)
#   apply_refine(rec, parsed) -> dict
#   respond_call(rec, style_guidance) -> (system, user, keys)
#   apply_respond(rec, parsed) -> dict
#   rewrite_call(rec, constitution) -> (system, user, keys)
#   apply_rewrite(rec, parsed) -> dict
#   to_sft(records) -> list[dict]
#
# Optional:
#   validate_rewrite(rec, parsed) -> None
#       Raise ValueError to reject a stage-6 completion and retry it. This is where a corpus
#       enforces properties of the training text itself (banned vocabulary, minimum
#       deliberation length) rather than merely of its JSON shape.

_FLAVORS = ("difficult_advice", "self_reflection")


def get_flavor(name: str) -> ModuleType:
    """Return the flavor module for a config's `flavor:` key.

    Args:
        name: Flavor name.

    Returns:
        The module implementing the interface above.

    Raises:
        ValueError: If the name is not a known flavor.
    """
    if name not in _FLAVORS:
        raise ValueError(f"unknown flavor {name!r}; expected one of {_FLAVORS}")
    from importlib import import_module

    return import_module(f"{__name__}.{name}")
