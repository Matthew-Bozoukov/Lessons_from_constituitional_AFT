# ABOUTME: feature_discovery — an autorater invents its own vocabulary per trace, the
# ABOUTME: vocabulary is embedded and clustered, and each cluster becomes a property.

"""LLM-driven feature discovery (LessWrong post `WAZWA6FPQvH8okouJ`).

An autorater is shown one trace at a time, with no metadata and no schema, and asked what
its "features" are. Because it is not scoring against axes we chose in advance, it can
surface behaviours nothing anticipated — which is the whole reason to run it alongside the
other three producers rather than instead of them.

    trace -> 10-20 free-text features -> dedupe to a vocabulary -> embed -> cluster -> name

**Port status: the producer code still lives in `scratch/llm_feature_discovery/`.**
`adapter.py` here reads that module's run directory and turns its clusters into Property
rows. When the port lands, six files move in beside this one:

    prompts    the post's two verbatim prompts
    extract    trace -> features -> the unique vocabulary
    cluster    grouping + naming  (its clustering becomes shared/grouping.py)
    centroids  the noise rule, in one place
    audit      redundancy pairs, keyword probes, the clustering gate, the dashboard
    rundir     what a run directory holds

and three of its current files do NOT move, because `shared/` already holds them:
`embed.py` becomes `shared/embed.py`, `properties.py` becomes `registry.py`, and the
grouping half of `cluster.py` becomes `shared/grouping.py`.

Two caveats from the existing runs, worth re-reading before quoting a number from this
producer (the full list is in that module's README):

* The cluster count is a RESOLUTION SETTING, not a count of behaviours. 84 of 11,175
  cluster pairs sat at centroid cosine >= 0.90 in the k=150 run.
* A cluster label is not evidence a behaviour is absent. `Displays evaluations awareness`
  (89 occurrences) landed inside a generic cluster; only a keyword probe surfaced it.
"""

from __future__ import annotations
