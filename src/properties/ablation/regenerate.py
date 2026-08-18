# ABOUTME: The strongest ablation: regenerate the corpus with the property suppressed at a
# ABOUTME: named synth stage — emits the synth config to run, and states what it confounds.

"""Regenerate the corpus instead of editing it.

The other three ablations edit an existing corpus. This one goes back to `uv run synth` and
generates a new one with the property suppressed at a named stage — either by ablating the
stage entirely (`ablate: [revise_responses]`, which the synth pipeline already supports) or
by injecting an instruction into that stage's prompt.

It is the strongest intervention and, for that reason, the one to reach for LAST. Callum's
objection to leading with it, 2026-08-17:

    "ablating the rewrite step is a very large intervention that will change many different
    aspects about the final prompt, and it might be the case that it changes three critical
    things, and only one of those things actually matters."

That is the whole argument for `rewrite.py`. Regeneration remains worth having for two
cases the editing ablations cannot reach: a property that lives in the SCENARIO (the query
channel, which a rewrite must not touch — changing the user's question changes the
experiment), and a property so pervasive that rewriting it out leaves incoherent text.

This module does not run the generation. It writes the derived synth config and returns the
command, because a generation run costs real money and takes hours, and the decision to
spend that belongs to a human reading a diff — not to an ablation driver. The rows it
returns are the ORIGINAL corpus, unchanged, so nothing downstream mistakes a plan for a
result: `verify.py` will report a zero prevalence drop, which is the truth until the
regeneration has actually run.
"""

from __future__ import annotations

from pathlib import Path

from src.properties.ablation.base import AblationResult, check_corpus
from src.properties.registry import Property
from src.properties.sources.base import Record, SourceAdapter

KIND = "regenerate"

# Appended to a stage's prompt when the config asks for suppression rather than a whole
# stage ablation. Deliberately short: a long instruction about what NOT to do is itself a
# strong steer on everything else the stage produces.
SUPPRESS_TEMPLATE = """

CONSTRAINT FOR THIS RUN: do not use the following move at all:
  {label}
  {description}
Reason through the situation in whatever other way is natural. Do not mention this \
constraint, and do not leave a gap where that move would have been."""


def applicable(prop: Property, records: list[Record], adapter: SourceAdapter,
               cfg=None) -> tuple[bool, str]:
    """Whether regeneration is set up for this property.

    Args:
        prop: The property.
        records: The corpus.
        adapter: The source adapter.
        cfg: The ablation config block; needs `synth_config` and `stage`.

    Returns:
        (ok, reason).
    """
    ok, reason = check_corpus(adapter)
    if not ok:
        return ok, reason
    if cfg is None or not hasattr(cfg, "get"):
        return True, ""
    if not cfg.get("synth_config"):
        return False, ("regenerate needs `synth_config:` — the configs/data/synth/*.yaml "
                       "that produced this corpus")
    if not cfg.get("stage"):
        return False, ("regenerate needs `stage:` — which pipeline stage introduces the "
                       "property. Ablating the wrong stage changes many things and "
                       "attributes them all to this property.")
    path = Path(str(cfg["synth_config"]))
    if not path.exists():
        return False, f"synth_config {path} does not exist"
    return True, ""


def apply(prop: Property, records: list[Record], cfg) -> AblationResult:
    """Write the derived synth config and return the command that runs it.

    Args:
        prop: The property to suppress.
        records: The corpus, returned unchanged (see the module docstring).
        cfg: The ablation config block. Keys: `synth_config` (required), `stage`
            (required), `method` (`ablate_stage` | `suppress_prompt`, default
            `suppress_prompt`), `out_config` (where to write the derived config),
            `description_override`.

    Returns:
        The AblationResult, whose `rows` are the ORIGINAL corpus and whose report carries
        the derived config path and the command to run.

    Raises:
        ValueError: If `method` is unknown, or the named stage is not in the config.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)
    synth_path = Path(str(cfg["synth_config"]))
    stage = str(cfg["stage"])
    method = str(cfg.get("method", "suppress_prompt"))
    synth = OmegaConf.load(synth_path)

    stage_names = [s.get("name") for s in (synth.get("stages") or [])]
    if stage not in stage_names:
        raise ValueError(f"stage {stage!r} is not in {synth_path} "
                         f"(stages: {stage_names})")

    if method == "ablate_stage":
        # The pipeline already supports this: a stage in `ablate:` runs its declared
        # `ablate_with` field-copy instead of the model call.
        synth.ablate = list(synth.get("ablate") or []) + [stage]
        note = (f"stage {stage!r} is ablated wholesale: it will change every property that "
                "stage introduces, not only this one. A result from this arm bounds the "
                "stage's contribution, not the property's.")
    elif method == "suppress_prompt":
        # A stage's prompt lives under `prompts:` as {system, user}. The constraint goes on
        # the SYSTEM half by default: the user half carries the constitution and the
        # scenario behind a <<<cache>>> mark, and appending after that mark would change
        # the cached prefix on every call.
        prompt_key = str(cfg.get("prompt_key", "system"))
        description = str(cfg.get("description_override") or prop.description)
        suppressed = False
        for stage_cfg in synth.stages:
            if stage_cfg.get("name") != stage:
                continue
            prompts = stage_cfg.get("prompts")
            if prompts is None or prompt_key not in prompts:
                raise ValueError(
                    f"stage {stage!r} has no prompts.{prompt_key} to append to "
                    f"(has: {sorted(prompts or {})})")
            prompts[prompt_key] = str(prompts[prompt_key]) + SUPPRESS_TEMPLATE.format(
                label=prop.label, description=description)
            suppressed = True
        if not suppressed:
            raise ValueError(f"stage {stage!r} not found in {synth_path}")
        note = (f"the stage's {prompt_key} prompt now forbids the property. This steers "
                "the generator and is not a guarantee: verify the regenerated corpus with "
                "the property's detector before treating the arm as ablated.")
    else:
        raise ValueError(f"unknown method {method!r}; "
                         "known: ablate_stage, suppress_prompt")

    slug = prop.property_id.replace(":", "_").replace("/", "_")
    out_config = Path(str(cfg.get("out_config") or
                          synth_path.with_name(f"{synth_path.stem}_ablate_{slug}.yaml")))
    out_config.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# ABOUTME: Derived from {synth_path} by src/properties/ablation/"
              f"regenerate.py to ablate one property.\n"
              f"# ABOUTME: Property {prop.property_id} — {prop.label}\n"
              f"# Run: uv run synth run --config {out_config}\n"
              f"# Method: {method} on stage {stage}. {note}\n")
    out_config.write_text(header + OmegaConf.to_yaml(synth), encoding="utf-8")

    command = f"uv run synth run --config {out_config}"
    print(f">>> wrote {out_config}\n>>> REVIEW THE DIFF, then run:\n    {command}")
    return AblationResult(
        kind=KIND, property_id=prop.property_id,
        rows=[dict(r.raw) for r in records], changed_ids=[],
        detected_ids=[],
        report={"method": method, "stage": stage, "synth_config": str(synth_path),
                "derived_config": str(out_config), "command": command, "note": note,
                "status": "PLANNED — the corpus returned here is the ORIGINAL. Run the "
                          "command above, then point the ablation config's source at the "
                          "regenerated corpus to verify the drop."})


def diff_summary(original: Path, derived: Path) -> str:
    """A human-readable diff of the two synth configs, for the review step.

    Args:
        original: The source config.
        derived: The generated one.

    Returns:
        A unified diff.
    """
    import difflib

    return "".join(difflib.unified_diff(
        original.read_text(encoding="utf-8").splitlines(keepends=True),
        derived.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile=str(original), tofile=str(derived)))
