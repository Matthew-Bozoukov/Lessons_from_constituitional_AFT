# ABOUTME: LLM-driven feature discovery: reasoning traces in, a list of named behavioural
# ABOUTME: properties out. Package marker and the map of which file does what.

"""LLM-driven feature discovery.

Replication of the LessWrong method (post `WAZWA6FPQvH8okouJ`) on this project's reasoning
traces. An autorater invents its own vocabulary for what a trace does instead of scoring it
against axes chosen in advance, so it can surface behaviours no schema anticipated.

**Input**: an SFT jsonl of reasoning traces (the training data).
**Output**: `properties.jsonl` — named behavioural properties with corpus prevalence, in
the shared interchange format, to be merged with the other producers' property lists.

Run it with `uv run python -m scratch.llm_feature_discovery <verb>`; see `__main__.py`.

One job per module:

    rundir      what a run directory holds and how each artifact is read and written
    prompts     the two verbatim prompts from the post
    extract     trace -> free-text features (autorater)
    dedupe      per-trace lists -> the unique vocabulary to embed
    podscript   the code that runs ON the rented GPU
    embed       rent the GPU, push, fetch, terminate
    cluster     embeddings -> UMAP -> HDBSCAN labels
    naming      cluster -> a ~5-word label (LLM)
    prevalence  labels + traces -> per-cluster statistics
    centroids   cluster centroids, shared with the scripts that assign against them
    audit       redundancy pairs and keyword probes
    geometry    did the reduction keep the structure
    compare     agreement with a baseline clustering, and stability
    report      markdown renderers
    dashboard   the HTML renderer
    properties  the export into the shared List of Properties
    pipeline    the order the above run in
"""

from __future__ import annotations
