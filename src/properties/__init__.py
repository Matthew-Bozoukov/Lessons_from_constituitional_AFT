# ABOUTME: Data-property extraction: training corpora and rollouts in, one List of
# ABOUTME: Properties out, and the ablations that test whether a property is load-bearing.

"""Property extraction and ablation.

The paper's Fig-3 claim is that the *style* of a scenario format (difficult advice,
courtroom, peer critique) is not itself what moves misalignment — it is the data
*properties* each style happens to elicit. This module is the machinery for that claim:
name the properties, then remove one at a time and retrain.

The flow, and where each piece of it lives:

    sources/      a corpus or a set of rollouts  ->  Record
    producers/    Records (+ a Target)           ->  Property rows
    registry.py   every producer's rows          ->  properties.jsonl
    ablation/     one Property + the corpus      ->  an ablated corpus
                                                     -> train -> M'' -> eval

Three producers, differing only in what evidence they read and how they earn a property:

    clusters   embed evidence about each record, cluster it, label the clusters. One
               config key picks WHAT gets embedded: `evidence: features` runs an autorater
               over each record first and clusters its free-text descriptions (the
               LessWrong method); `evidence: traces` clusters the record text directly.
    turf       trace one case's behaviour back to training-data properties
    less       gradient influence ranks the corpus, then an LLM names what's on top

They share five things, and each of those is exactly one file under `shared/`, because a
producer that spells its own embedding call is a producer whose numbers cannot be compared
with anyone else's:

    embed        ONE embedding path (openrouter | runpod)
    grouping     reduce {none, umap} x cluster {kmeans, hdbscan}, plus the noise contract
    interpret    evidence -> a label AND a detector rubric (the detector is what makes a
                 property actionable: ablation and verification both run it)
    attributes   the extract-attributes prompt family, verbatim from SURF and the post
    outcomes     group membership crossed with a judged outcome, WITHIN arm, BH-corrected

The one number that has to mean the same thing across all of them is `prevalence`: the
share of records in the SAME corpus exhibiting the property. Everything else on a property row
is advisory detail a reader uses to judge the label.

Run it:

    uv run python scripts/properties/discover.py --config configs/properties/<name>.yaml
    uv run python scripts/properties/ablate.py   --config configs/properties/<name>.yaml
"""

from __future__ import annotations

from src.properties.registry import Property, PropertyRegistry  # noqa: F401


def block(cfg, key: str) -> dict:
    """One nested config block, as plain kwargs.

    Every producer and every ablation is handed a config block and passes its sub-blocks
    (`embed:`, `grouping:`, `detector:`) straight through as kwargs. That block arrives as
    an OmegaConf node when it came from a yaml and as a plain dict when it came from a
    test or a caller, and `OmegaConf.to_container` raises on the second — so the
    conversion lives here rather than being spelled, and mis-spelled, in fifteen places.

    Args:
        cfg: The config block (DictConfig, dict, or None).
        key: The sub-block to read.

    Returns:
        The sub-block as a plain dict, `{}` when absent or null.
    """
    from omegaconf import OmegaConf

    if cfg is None:
        return {}
    value = cfg.get(key) if hasattr(cfg, "get") else None
    if not value:
        return {}
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) \
        else dict(value)
