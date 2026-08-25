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

Ten files. A module only stands on its own if something outside it references it, or if
it is one whole stage:

    rundir      what a run directory holds and how each artifact is read and written
                — every stage depends on it
    centroids   cluster centroids — also imported by scratch/find_harm_risk_instances.py
                and scratch/odcv_cluster_assign.py, so the noise rule lives in one place
    prompts     the two verbatim prompts from the post — read by extract and by cluster,
                and kept apart so "do not reword this" has somewhere to be written down

    extract     stages 1-2: trace -> free-text features -> the unique vocabulary
    embed       stage 3: the pod-side code, and renting the GPU that runs it
    cluster     stage 4: UMAP + HDBSCAN, LLM naming, prevalence, and the report
    audit       stages 5-6: redundancy, keyword probes, the geometry/agreement/stability
                gate, and the dashboard
    properties  stage 7: the hand-off schema other producers and the merger read

    __main__    the CLI, and the only file that knows what order stages run in
"""

from __future__ import annotations
