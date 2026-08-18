# ABOUTME: TURF — trace one case's behaviour back through the training data to the query
# ABOUTME: and reasoning properties that co-occur with responses like it.

"""TURF, adapted (arXiv 2602.05910, *Chunky Post-Training*).

The only producer that starts from an OUTCOME rather than from the corpus. Given a case —
a response we liked or disliked, from an eval — it asks which properties of the TRAINING
data co-occur with training responses similar to that one:

    1. offline: extract query + reasoning attributes (the trigger side) and response
       attributes (the behaviour side) from every training row; cluster the trigger side
    2. blind judge: 10 "The response..." attributes of the case, with no rubric in view
    3. informed judge: pick 3 VERBATIM cruxes against the rubric, guarded so a crux cannot
       merely restate one of the dataset's own generation styles
    4. retrieve the k nearest training response attributes to each crux
    5. route those hits back to their rows' trigger clusters and count the hits
    6. the case's own trigger attribute nearest the winning cluster is the candidate

So its properties come with a claim the other producers cannot make: "these training rows
are followed by responses that are good/bad in the same way as this case". That is a
correlation, not a cause — which is precisely why the property then goes to `ablation/`.

**Port status: the producer code still lives in `scratch/turf/`.** `adapter.py` here reads
its trace result and turns the ranked trigger clusters into Property rows. Seven files move
in when the port lands (`cases`, `prompts`, `extract`, `index`, `trace`, `unrender`,
`plot_turf`), while `common.py` does not: its embedding, its k-means and its row parsing
are already `shared/embed.py`, `shared/grouping.py` and `sources/base.py`.

Its rubrics move too, and the rule they carry moves with them: a rubric describes the
case's BEHAVIOUR — principle, context, what satisfying and violating look like — and never
candidate properties. `sources/targets.py::from_rubric` enforces that rather than trusting
it, because a rubric that names properties makes the discovery circular.
"""

from __future__ import annotations
