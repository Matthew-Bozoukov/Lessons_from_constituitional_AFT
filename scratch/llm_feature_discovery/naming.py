# ABOUTME: Ask the LLM for a ~5-word label per cluster, from 100 of its features sampled
# ABOUTME: at random, using the post's naming prompt verbatim.

"""Cluster naming.

The post's recipe: show the model 100 randomly sampled features from one cluster and ask
for a single concise label. Nothing about the corpus, the trait, or the other clusters is
shown, so the label describes the features and not our expectations of them.
"""

from __future__ import annotations

import random

from scratch.llm_feature_discovery.prompts import build_cluster_naming_messages
from src.endpoints.openrouter import OpenRouterClient, map_threaded

OPENROUTER_PROVIDER_ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}}
FEATURES_SHOWN_PER_CLUSTER = 100


def name_clusters(cluster_to_features: dict[int, list[str]], model: str, seed: int,
                  sample_size: int = FEATURES_SHOWN_PER_CLUSTER,
                  max_workers: int = 12) -> dict[int, str]:
    """Label every cluster.

    Args:
        cluster_to_features: Cluster id -> the feature strings in it.
        model: OpenRouter model id.
        seed: Seed for the per-cluster sample, so a rerun shows the model the same features.
        sample_size: How many features to show (the post uses 100).
        max_workers: Concurrent requests.

    Returns:
        Cluster id -> label.
    """
    client = OpenRouterClient()
    cluster_ids = sorted(cluster_to_features)
    rng = random.Random(seed)
    sampled = {c: rng.sample(cluster_to_features[c],
                             min(sample_size, len(cluster_to_features[c])))
               for c in cluster_ids}

    def name_one(index: int) -> str:
        res = client.chat(model=model,
                          messages=build_cluster_naming_messages(sampled[cluster_ids[index]]),
                          temperature=0.0, max_tokens=40,
                          extra_body=OPENROUTER_PROVIDER_ROUTING)
        return res.content.strip().strip(".").strip()

    labels = map_threaded(name_one, len(cluster_ids), max_workers=max_workers, desc="naming")
    return dict(zip(cluster_ids, labels))
