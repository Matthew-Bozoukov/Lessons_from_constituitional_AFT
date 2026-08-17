# ABOUTME: Publish the feature-discovery run to Hugging Face, including the embedding matrix
# ABOUTME: that is too large for git. Run: uv run python scratch/push_feature_discovery.py

"""Push the k=150 feature-discovery artifacts.

git carries the readable half (report, cluster definitions, the feature->cluster map) so a
fresh clone can rerun every downstream analysis; the 265MB embedding matrix lives only here,
because it is the one artifact that is both bulky and exactly reproducible from the feature
list plus a pinned embedding model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-12-difficult-advice-feature-discovery"
RUN = Path("output/feature_discovery/20260812_092119")
FILES = ("report.md", "clusters.json", "feature_cluster_map.json", "features.jsonl",
         "unique_features.txt", "feature_counts.json", "embed_meta.json",
         "report_audit.json", "dashboard.html", "embeddings.npy")


def main(private: bool = True) -> None:
    """Upload the run's artifacts with the required dataset card.

    Args:
        private: Create the repo private (default).
    """
    meta = json.loads((RUN / "clusters.json").read_text())["meta"]
    fields = {
        "title": "Difficult-advice reasoning: LLM-driven feature discovery (k=150)",
        "experiment": "Replication of the LessWrong LLM-driven feature discovery pipeline "
                      "(post WAZWA6FPQvH8okouJ) over the 2,202 difficult-advice reasoning "
                      "traces: free-text features per trace, embedded and clustered, each "
                      "cluster named from 100 sampled features.",
        "date_generated": "2026-08-12",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)"
                        " — the corpus these traces come from",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()} "
                       f"(generated at {meta['git_sha']})",
        "models": f"feature extraction + cluster naming {meta['naming_model']}; "
                  f"embeddings {meta['embedding_model']} ({meta['embedding_dim']}d)",
        "generation_config": "extraction temperature 1.0 (the post's brainstorm framing), "
                             "max_tokens 1200, 10-20 features per trace; k-means k=150 "
                             f"seed {meta['seed']}, MiniBatchKMeans over L2-normalised "
                             "fp16 vectors",
        "schema": "features.jsonl (scenario_id, trait_id, features[]); unique_features.txt "
                  "(one string per line, embedding row order); embeddings.npy "
                  "(n x 4096 fp16, L2-normalised); feature_cluster_map.json (feature -> "
                  "cluster id); clusters.json (label, prevalence, trait mix per cluster); "
                  "report.md + dashboard.html (the readable mirrors)",
        "provenance": "scratch/feature_discovery/{extract_features,prepare_features,"
                      "runpod_embed,cluster_and_name,build_report}.py, in that order",
        "scale": f"{meta['traces']} traces -> {meta['feature_instances']} feature instances "
                 f"-> {meta['unique_features']} unique -> {meta['k']} clusters",
        "embedding_sanity": f"synonym pair {meta['sanity_synonym']:.3f} vs unrelated "
                            f"{meta['sanity_unrelated']:.3f}",
        "caveat": "84 cluster pairs sit at centroid cosine >= 0.90 — k=150 is a resolution "
                  "setting, not a count of distinct behaviours. Keyword probes in report.md "
                  "measure behaviours the clustering absorbed (evaluation awareness 9.1%).",
    }
    print(push_files([RUN / f for f in FILES], REPO, fields, private=private))


if __name__ == "__main__":
    fire.Fire(main)
