# ABOUTME: One-off (2026-09-22): publish a local MASK run that repeats an already-published same-day run
# ABOUTME: as a BRANCH of that run's repo, with a card that says what differed, and list it on main's card.

"""Run: uv run python scratch/push_mask_replicate_branch.py --run_dir output/mask/<run> \\
        --repo dougalldeepmind/2026-09-22-mask-qwen36-0-da-15 --branch run2 \\
        --differs "served on an H200 at 192 in flight (main: H100 at 32); same code otherwise"

The run dir is a `--no-push` eval run: rollouts/ results/ metadata/ and no README. The card is
main's card with the replicate's own numbers and a "Replicate" section; main's card gains a
"Replicates" table pointing at the branch. Nothing on main's own files changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fire
from huggingface_hub import HfApi, hf_hub_download


def main(run_dir: str, repo: str, branch: str, differs: str) -> None:
    run = Path(run_dir)
    assert (run / "results" / "results.json").is_file() and (run / "rollouts").is_dir(), f"{run} is not a run dir"
    assert not (run / "README.md").exists(), "this run already has a card — was it pushed?"
    api = HfApi()
    refs = api.list_repo_refs(repo, repo_type="dataset")
    assert branch not in {b.name for b in refs.branches}, f"{repo} already has a branch {branch}"
    res = json.loads((run / "results" / "results.json").read_text())
    meta = json.loads((run / "metadata" / "run_meta.json").read_text())
    score = res["overall_honesty_score"]
    per = " ".join(f"{a}={v['honesty_score']}" for a, v in res["per_archetype"].items())
    card = open(hf_hub_download(repo, "README.md", repo_type="dataset"), encoding="utf-8").read()
    main_score = json.loads(open(hf_hub_download(repo, "results/results.json", repo_type="dataset"),
                                 encoding="utf-8").read())["overall_honesty_score"]
    # the replicate's card: main's card, its own provenance, and what differed
    card_b = re.sub(r"(\| `provenance` \| ).*?( \|)", lambda m: m.group(1) + meta.get("command", "") + m.group(2), card, count=1)
    card_b = re.sub(r"(\| `generation_config` \| ).*?( \|)",
                    lambda m: m.group(1) + json.dumps({k: res.get(k) for k in ("temperature", "seed", "max_tokens", "gen_concurrency", "lie_k")}) + m.group(2),
                    card_b, count=1)
    card_b = card_b.rstrip() + (
        f"\n\n## Replicate `{branch}` of this run\n\n"
        f"This branch is a second draw of the same measurement, same target, same eval and settings "
        f"except: {differs}. It was generated later the same day (2026-09-22 UTC), so the name has no "
        f"slot for it; a same-day repeat lives as a branch of the run it repeats.\n\n"
        f"- honesty (this branch): **{score}** | {per}\n- honesty (main): {main_score}\n"
        f"- run_meta: `metadata/run_meta.json` on this branch\n")
    (run / "README.md").write_text(card_b, encoding="utf-8")
    api.create_branch(repo, branch=branch, repo_type="dataset")
    api.upload_folder(folder_path=str(run), repo_id=repo, repo_type="dataset", revision=branch,
                      commit_message=f"{branch}: same-day replicate ({differs})")
    # main's card: a Replicates table (append or extend)
    row = f"| `{branch}` | {score} | {differs} | [tree](https://huggingface.co/datasets/{repo}/tree/{branch}) |\n"
    if "## Replicates" in card:
        card_m = card.rstrip() + "\n" + row
    else:
        card_m = card.rstrip() + ("\n\n## Replicates\n\nSame-day repeats of this measurement live as branches "
                                  "of this repo; main is the first run.\n\n| branch | honesty | what differed | files |\n|---|---|---|---|\n" + row)
    api.upload_file(path_or_fileobj=card_m.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message=f"card: list replicate branch {branch}")
    print(f"pushed {run} -> https://huggingface.co/datasets/{repo}/tree/{branch} (honesty {score}; main {main_score})")


if __name__ == "__main__":
    fire.Fire(main)
