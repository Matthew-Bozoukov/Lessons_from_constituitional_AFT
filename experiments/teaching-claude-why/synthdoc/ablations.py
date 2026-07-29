# ABOUTME: The catalogue of ablatable axes. Generated from the live registry and prompt
# ABOUTME: packs, so it cannot drift out of date the way a hand-written list would.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .control import loader
from .core import registry
from .core.recipe import RESERVED
from .core.specs import available_specs

# Axes whose value is consumed after sampling, so every arm samples an identical
# scenario set and the comparison is exactly paired.
POST_SAMPLER = "paired"
# Axes that feed the sampler: arms differ in which scenarios exist, by construction.
SAMPLER = "unpaired"


@dataclass
class Axis:
    """One ablatable dimension.

    Attributes:
        key: The dotted config key to put in `axis:` of a sweep.
        group: Section heading in the catalogue.
        varies: What changes when you move it.
        values: Legal or example values, resolved live where possible.
        pairing: POST_SAMPLER or SAMPLER.
        note: Anything a researcher would otherwise learn the hard way.
    """

    key: str
    group: str
    varies: str
    values: list[str] = field(default_factory=list)
    pairing: str = POST_SAMPLER
    note: str = ""


def catalog() -> list[Axis]:
    """Build the axis catalogue from the current registry and prompt packs.

    Returns:
        Every ablatable axis, in a sensible reading order.
    """
    axis_pack = loader.load_pack("axes")
    out: list[Axis] = [
        Axis(
            "spec.id",
            "Spec and chunking",
            "which model spec the corpus teaches",
            available_specs(),
            SAMPLER,
            "Add specs to control/specs/ or register them in control/specs/index.yaml. "
            "If your base config sets spec.path, clear it with "
            "base_overrides: {spec.path: null} so the id alone resolves.",
        ),
        Axis(
            "spec.chunker.granularity",
            "Spec and chunking",
            "how finely the spec is cut",
            registry.names("chunker"),
            SAMPLER,
            "Changes the chunk pool, so chunk_ids differ across arms and coverage "
            "reports are not directly comparable.",
        ),
        Axis(
            "recipe.n",
            "Sampling",
            "corpus size - the data-scaling curve none of the prior work ran",
            ["1000", "5000", "20000"],
            POST_SAMPLER,
            "Smaller arms are prefixes of larger ones, so a scaling sweep costs only "
            "the largest arm.",
        ),
        Axis(
            "recipe.chunks_per_example",
            "Sampling",
            "how many spec chunks share one document",
            ['{1: 1.0}', '{2: 1.0}', '{1: 0.7, 2: 0.2, 4: 0.1}'],
            SAMPLER,
        ),
        Axis(
            "recipe.grouping",
            "Sampling",
            "how co-occurring chunks are chosen",
            registry.names("grouping"),
            SAMPLER,
            "Hold group size fixed with base_overrides: "
            "{recipe.chunks_per_example: {2: 1.0}}, or you confound strategy with k.",
        ),
        Axis(
            "recipe.grouping_params",
            "Sampling",
            "similarity band for semantic grouping, section-crossing for adjacent",
            ['{semantic: {min_similarity: 0.4, max_similarity: 0.99}}'],
            SAMPLER,
        ),
        Axis(
            "recipe.doc_type",
            "Sampling",
            "the mixture of document types",
            loader.declared_doc_types(),
            SAMPLER,
            "Set one type to 1.0 for a single-type corpus.",
        ),
    ]

    for name in sorted(axis_pack):
        out.append(
            Axis(
                f"recipe.{name}",
                "Sampling (scenario axes)",
                str(axis_pack[name].get("label", name)).lower(),
                sorted((axis_pack[name].get("values") or {})),
                SAMPLER,
                "Declared in control/prompts/axes.yaml.",
            )
        )

    out += [
        Axis(
            "planning.enabled",
            "Planning",
            "whether a model chooses the situation before anything is written",
            ["true", "false"],
            note="GDM's structured scenario step. false is the control arm: generate "
            "straight from the chunk. Adds its own stage, so the effect is a stage diff.",
        ),
        Axis(
            "planning.template",
            "Planning",
            "how structured the plan is",
            sorted(loader.load_pack("planning")),
            note="what_how_why decomposes the plan; situation_only just picks a "
            "situation. Isolates whether structure matters or merely planning at all.",
        ),
        Axis(
            "planning.model",
            "Planning",
            "planner model, independent of the generator's",
            ["anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5"],
            note="A cheap planner with an expensive writer, or the reverse.",
        ),
        Axis(
            "generation.strategy",
            "Generation",
            "how many calls make one document, and in what order",
            registry.names("strategy"),
            note="draft_then_align answers with no spec then aligns (GDM); best_of_n "
            "samples and selects (Anthropic). draft_then_align requires planning.",
        ),
        Axis(
            "generation.strategy_params",
            "Generation",
            "strategy settings: best-of-n width, or what the draft phase sees",
            [
                "{n: 4, selector: judge}",
                "{draft_context: spec_in_system}",
                "{draft_context: no_spec}",
            ],
            note="draft_context spec_in_system is faithful to GDM's description; "
            "no_spec drafts blind and is our variant. See sweeps/draft_context.yaml.",
        ),
        Axis(
            "generation.model",
            "Generation",
            "the generator model - never ablated in any prior pipeline",
            ["anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5", "openai/gpt-4.1"],
        ),
        Axis(
            "generation.template",
            "Generation",
            "the generation prompt itself",
            sorted(loader.load_pack("generation")),
            note="v1 is a deliberately thin control arm.",
        ),
        Axis("generation.temperature", "Generation", "sampling diversity", ["0.7", "1.0", "1.2"]),
        Axis("generation.max_tokens", "Generation", "document length ceiling", ["2048", "4096"]),
        Axis(
            "revision",
            "Revision",
            "the revision dose - list length IS the number of passes",
            ["[]", "[{kind: critique_rewrite, model: M}]"],
            note="Shorter arms are cache prefixes of longer ones, so a dose-response "
            "sweep costs little more than its longest arm.",
        ),
        Axis(
            "revision[].kind",
            "Revision",
            "which revision strategy",
            sorted(loader.load_pack("revision")),
            note="Vary by giving each arm a one-entry revision list.",
        ),
        Axis(
            "revision[].context",
            "Revision",
            "whether the reviser sees the original generation instructions",
            ["fresh", "same"],
            note="Vary by giving each arm a full revision list with the context changed.",
        ),
        Axis(
            "revision[].model",
            "Revision",
            "reviser model, independent of the generator",
            ["anthropic/claude-sonnet-4.5", "openai/gpt-4.1"],
        ),
        Axis(
            "filters",
            "Filtering",
            "which filters run at all",
            ["[]", "[{kind: embedding_dedup, threshold: 0.87}]"],
            note="An empty list keeps everything, which is the right control arm for "
            "measuring what filtering costs you.",
        ),
        Axis(
            "filters[].threshold",
            "Filtering",
            "dedup aggressiveness",
            ["0.8", "0.87", "0.95"],
            note="Vary by giving each arm a full filters list.",
        ),
        Axis(
            "filters[].rubric",
            "Filtering",
            "autorater rubric version",
            sorted(loader.load_pack("rubrics")),
            note="Criteria names become filter_scores columns, so a rubric change "
            "changes the snapshot schema.",
        ),
        Axis("filters[].n_raters", "Filtering", "raters per document", ["1", "3", "5"]),
        Axis("filters[].min_score", "Filtering", "keep threshold", ["2", "3", "4"]),
        Axis(
            "filters[].discover",
            "Filtering",
            "pattern_scan: discover the corpus's own tics, or only check seeded ones",
            ["true", "false"],
            note="The discovered pattern list is written to the manifest and is often "
            "more informative than the filtering it drives.",
        ),
        Axis(
            "filters[].mode",
            "Filtering",
            "pattern_scan detection threshold",
            ["broad", "strict"],
            note="Running both bounds how much of the measured rate is rater latitude.",
        ),
        Axis(
            "embedder",
            "Infrastructure",
            "embedding backend for semantic grouping and dedup",
            registry.names("embedder"),
            SAMPLER,
            "Affects semantic grouping, hence the sampler.",
        ),
        Axis("llm.provider", "Infrastructure", "API backend", registry.names("llm")),
        Axis(
            "cache.scope",
            "Infrastructure",
            "which call sites are cached - not an experiment, a cost control",
            ["[plan, generate, revise, filter]", "[generate]", "[]"],
            note="Drop 'generate' to re-sample documents while replaying rating calls; "
            "drop 'filter' to re-rate an unchanged corpus. Does not change outputs.",
        ),
        Axis("seed", "Infrastructure", "resample everything - the variance baseline",
             ["0", "1", "2"], SAMPLER,
             "Run this first. An effect smaller than the seed-to-seed spread is noise."),
        Axis(
            "export.format",
            "Export",
            "output format",
            registry.names("exporter"),
        ),
        Axis(
            "export.mix",
            "Export",
            "fraction routed to a pretrain-style shard",
            ["{}", "{pretrain_shard: 0.4}"],
        ),
        Axis(
            "export.strip_system",
            "Export",
            "keep or drop in-document system turns in the training data",
            ["true", "false"],
            note="GDM removed generation system prompts before training. Ours never "
            "enter a document, so this is about tool-definition and persona turns.",
        ),
        Axis(
            "export.baseline",
            "Export",
            "how much existing SFT data to mix in",
            ["{path: null}", "{path: data/sft.jsonl, ratio: 1.0}"],
            note="GDM: mixing with baseline SFT data helped a lot against capability "
            "regressions. Ratio is baseline rows per corpus row.",
        ),
    ]
    return out


def catalog_text() -> str:
    """Render the axis catalogue for the CLI."""
    axes = catalog()
    lines = [
        "Ablatable axes — put `key` in a sweep's `axis:` field.",
        "",
        "  uv run python -m synthdoc.cli sweep --config <sweep>.yaml --dry_run",
        "",
        "pairing=paired    arms sample identical scenarios; deltas are exactly paired.",
        "pairing=unpaired  the axis feeds the sampler, so scenario sets differ by design;",
        "                  the sweep report says so and falls back to marginals.",
        "",
    ]
    group = None
    for axis in axes:
        if axis.group != group:
            group = axis.group
            lines += ["", f"== {group} ==", ""]
        values = ", ".join(str(v) for v in axis.values[:6])
        if len(axis.values) > 6:
            values += f", +{len(axis.values) - 6} more"
        lines.append(f"  {axis.key}")
        lines.append(f"      varies:  {axis.varies}")
        lines.append(f"      values:  {values or '(free-form)'}")
        lines.append(f"      pairing: {axis.pairing}")
        if axis.note:
            lines.append(f"      note:    {axis.note}")
        lines.append("")
    lines += [
        "Any other dotted config key also works — this list is the curated set, not a limit.",
        f"Reserved recipe keys (not axes): {', '.join(sorted(RESERVED))}",
    ]
    return "\n".join(lines)


def catalog_markdown() -> str:
    """Render the axis catalogue as a markdown table, for docs."""
    lines = ["| axis | varies | values | pairing |", "|---|---|---|---|"]
    for axis in catalog():
        values = ", ".join(f"`{v}`" for v in axis.values[:4])
        if len(axis.values) > 4:
            values += f", +{len(axis.values) - 4}"
        lines.append(f"| `{axis.key}` | {axis.varies} | {values} | {axis.pairing} |")
    return "\n".join(lines)


def as_dicts() -> list[dict[str, Any]]:
    """Return the catalogue as plain dicts."""
    return [
        {
            "key": a.key,
            "group": a.group,
            "varies": a.varies,
            "values": a.values,
            "pairing": a.pairing,
            "note": a.note,
        }
        for a in catalog()
    ]
