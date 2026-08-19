# ABOUTME: The four things every producer goes through — one embedding path, one grouping
# ABOUTME: stage, one interpretation step, one attribute-extraction prompt family.

"""Shared machinery, so four producers stay comparable.

A producer that writes its own embedding call, its own k-means, or its own naming prompt
produces numbers that cannot be put beside anyone else's — different embedder, different
normalisation, different notion of what an "attribute" is. Each of those four decisions
therefore lives in exactly one file here, and a producer picks its behaviour with a config
rather than by reimplementing it:

    embed       embed(texts, backend="openrouter" | "runpod") -> (vectors, EmbedMeta)
    grouping    group(vectors, GroupingParams(reduce=..., cluster=...)) -> Grouping
                assign(points, centroids) -> labels          (cross-corpus comparability)
    interpret   interpret(evidence) -> Interpretation (label + DETECTOR)
                detect(records, label, detector) -> verdicts; prevalence(verdicts)
    attributes  extract(records, AttributeSpec(style="numbered" | "freeform"))

`interpret.detect` is the load-bearing one: the same call measures a property's prevalence,
selects the rows an ablation edits, and checks afterwards that the prevalence moved.
"""

from __future__ import annotations
