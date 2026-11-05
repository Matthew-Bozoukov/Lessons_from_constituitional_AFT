# ABOUTME: Publishes the 2026-11-05 strict moral-content audit as a `splits/` labels file (neutral | moral) in each
# ABOUTME: da-amoral corpus repo, so a mixture can keep only the neutral pool (src/data/mixture/split.py).
# Run: uv run python scratch/da_amoral_neutral_split_20261105.py [--go]   (without --go it writes locally, pushes nothing)
import json
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.mixture.split import split_counts, write_split
from src.infra.huggingface import hf_api
from src.naming import artifact_name
from src.utils import git_sha

OUT = Path("output/2026-11-05_da_amoral_neutral")
RUBRIC = ("moral if the right answer depends at all on what the user owes another person or on doing right by "
          "them (honesty/disclosure, fairness, promises, another's wishes or welfare), at ANY stakes, anywhere in "
          "the conversation (system, user, reasoning, reply); neutral if only the user's own aims are at stake. "
          "Low-stakes moral dilemmas have their own arm, so this control must be neutral at any stakes.")
CORPORA = [  # (repo, revision the audit read, local labels file)
    ("dougalldeepmind/2026-11-03-da-amoral-synth", "ac8e1efe10611b1f43c8d58110b7a53b670fae52",
     Path("output/2026-11-05_da_amoral_moral_charge/labels_strict.jsonl")),
    ("dougalldeepmind/2026-11-05-da-amoral-neutral-sectors-synth", None,
     OUT / "labels_strict_new.jsonl"),
]


def main(go: bool = False) -> None:
    name = artifact_name("da-amoral moral audit")
    api = hf_api()
    for repo, rev, labels_path in CORPORA:
        rev = rev or api.repo_info(repo, repo_type="dataset").sha
        corpus = [json.loads(line) for line in open(
            hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev), encoding="utf-8")]
        verdicts = {r["scenario_id"]: r["verdict"] for r in map(json.loads, open(labels_path, encoding="utf-8"))}
        ids = {r["metadata"]["scenario_id"] for r in corpus}
        if ids != set(verdicts):
            raise ValueError(f"{repo}: audit covers {len(verdicts)} ids, corpus has {len(ids)}; "
                             f"missing {sorted(ids - set(verdicts))[:3]}")
        meta = {"made_by": "scratch/da_amoral_neutral_split_20261105.py", "git_sha": git_sha(),
                "corpus": f"{repo}@{rev}", "pools": ["neutral", "moral"], "rubric": RUBRIC,
                "rater": "Claude Opus 5.5 subagents (Claude Code), one pass, full conversation read",
                "labels_file": str(labels_path), "counts": split_counts(corpus, verdicts, strata=("trait_id", "sector"))}
        local = write_split(OUT / "splits" / repo.split("/")[1] / f"{name}.json", verdicts, meta)
        print(repo, rev[:8], dict(Counter(verdicts.values())), "->", local)
        if go:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=f"splits/{name}.json", repo_id=repo,
                            repo_type="dataset", commit_message=f"splits: {name} (strict moral-content audit; keep: neutral)")
            print(f">>> published splits/{name}.json at {repo}@{api.repo_info(repo, repo_type='dataset').sha}")


if __name__ == "__main__":
    main(go="--go" in sys.argv[1:])
