# ABOUTME: trace_clusters — embed WHOLE records, cluster them, name the clusters. The
# ABOUTME: simplest producer, and the reference implementation of the shared layer.

"""Cluster the traces themselves, not descriptions of them.

The other producers put a model between the record and the vector: an autorater writes
features or attributes, and those strings get embedded. That buys abstraction (the
autorater says what a trace DOES) at the cost of an extra model's opinion in the loop.
This producer does the direct thing — embed the record's text, cluster the embeddings,
name each cluster from its members — which is cheap, has no autorater in it, and answers a
question the others cannot: do two scenario formats occupy DIFFERENT regions of trace
space at all?

That question is the 2026-08-17 action item ("UMAP + clustering on good traces; compare DA
vs Courtroom / Peer Critique"), and it needs the direct version: an autorater's vocabulary
would smooth over exactly the surface differences the comparison is about.

Its known weakness, and why the other three producers exist: whole-text embeddings track
topic and register at least as strongly as behaviour, so a cluster here can easily be "all
the medical scenarios" rather than a move the model makes. `interpret.py` is prompted to
reject topic labels, and every property carries a detector that must be applied to a
single record — but read a trace_clusters label with more suspicion than a
feature_discovery one, and check the detector before spending a training run on it.
"""

from __future__ import annotations
