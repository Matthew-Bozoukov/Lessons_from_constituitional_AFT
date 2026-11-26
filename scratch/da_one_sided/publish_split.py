# ABOUTME: Publishes the corpus check's keep/narration labels as a `splits/` file in the da-one-sided corpus repo,
# ABOUTME: so a mixture draws only from rows whose reasoning does not narrate the no-hearing rule.
# Run: uv run python scratch/da_one_sided/publish_split.py [--go]   (without --go it writes locally, pushes nothing)
"""The labels are scratch/da_one_sided/corpus_check.py's `labels.json`: one gemini-3.8-flash call per row, `narration`
when at least one sentence is the assistant talking about not weighing the options (the reviser's instruction turned
into the assistant's own thought), `keep` otherwise. A mixture source names the pool with
`split: {file: splits/<name>.json, keep: keep}` (src/data/mixture/split.py).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.mixture.split import split_counts, write_split
from src.infra.huggingface import hf_api
from src.naming import artifact_name
from src.utils import git_sha

REPO = "dougalldeepmind/2026-11-20-da-one-sided-synth"
LABELS = Path("output/2026-11-20_da_one_sided_corpus_check/labels.json")
OUT = Path("output/2026-11-20_da_one_sided_corpus_check/splits")


def main(go: bool = False) -> None:
    name = artifact_name("da-one-sided narration split", date="2026-11-20")
    api = hf_api()
    rev = api.repo_info(REPO, repo_type="dataset").sha
    corpus = [json.loads(line) for line in open(
        hf_hub_download(REPO, "dataset.jsonl", repo_type="dataset", revision=rev), encoding="utf-8")]
    doc = json.loads(LABELS.read_text(encoding="utf-8"))
    labels = dict(doc["labels"])
    ids = {r["metadata"]["scenario_id"] for r in corpus}
    if ids != set(labels):
        raise ValueError(f"{REPO}: labels cover {len(labels)} ids, corpus has {len(ids)}; "
                         f"unlabelled {sorted(ids - set(labels))[:3]}, not in corpus {sorted(set(labels) - ids)[:3]}")
    meta = {**doc["meta"], "published_by": "scratch/da_one_sided/publish_split.py", "git_sha": git_sha(),
            "corpus": f"{REPO}@{rev}", "pools": ["keep", "narration"], "labels_file": LABELS.as_posix(),
            "counts": split_counts(corpus, labels, strata=("trait_id",))}
    local = write_split(OUT / f"{name}.json", labels, meta)
    print(REPO, rev[:8], dict(Counter(labels.values())), "->", local.as_posix())
    print("keep by trait:", meta["counts"]["by_trait"]["keep"])
    if go:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=f"splits/{name}.json", repo_id=REPO,
                        repo_type="dataset",
                        commit_message=f"splits: {name} (rows whose reasoning narrates the no-hearing rule; keep: keep)")
        print(f">>> published splits/{name}.json at {REPO}@{api.repo_info(REPO, repo_type='dataset').sha}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(go="--go" in sys.argv[1:])
