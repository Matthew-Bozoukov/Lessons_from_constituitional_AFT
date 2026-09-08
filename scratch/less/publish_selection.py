# ABOUTME: Adds the id manifest and a complete dataset card to the two published LESS
# ABOUTME: top-10% training repos, so each selection is reproducible from the pool alone.

"""Publish `selection_ids.json` + a full card to the LESS arm and control repos.

    uv run python scratch/less/publish_selection.py [--dry-run]

`uv run mix` already pushed each `mixture.jsonl`, but two things it writes are not enough
on their own:

1. The mixture rows are bare interchange messages -- the builder drops `metadata` -- so
   nothing in the published file says WHICH pool rows these are. `selection_ids.json`
   restores that: every `less_id` with the rank and per-subtask influence it was chosen
   on, which is the whole reproducibility claim for a data-selection experiment.
2. The generated card names the spec-filter judge under `models` and reports `none` when
   there is no filter, which is true but useless here -- the models that matter are the
   one the ranking's gradients were taken over and the tokenizer that counted the budget.

The card written here supersedes the builder's (push_files rewrites README.md), and adds
the YAML front matter the hub wants so `load_dataset(repo)` resolves one default config
rather than globbing both jsonl files into one schema.

Row order is NOT the id order: the builder shuffles with seed 0 before writing. The ids
are the SET that was selected; rejoin by content, or re-run select_topk.py, which
reproduces the file byte for byte from the pinned ranking.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.infra.huggingface import card_front_matter, card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

META = ROOT / "data" / "less" / "selection_meta.json"
ARM_REPO = "LASR-Callum/2026-08-19-less-top10-difficult-advice-220-train"
CTL_REPO = "LASR-Callum/2026-08-19-random-220-difficult-advice-control-train"
DATE = "2026-08-19"

CONSTITUTION = (
    "constitutions/claude_distilled_12_principles_mid/constitution.md — the constitution "
    "the scored pool was generated from. Every row traces to one of its traits via "
    "metadata.trait_id in matboz/synthdoc-v2-difficult-advice; the rows here carry only "
    "interchange messages, so use selection_ids.json to recover the trait.")

MODELS = (
    "Qwen/Qwen3.6-27B — tokenizer only in this build (token budgets and the length cap); "
    "no model was called to produce these rows. The rows themselves were generated for "
    "matboz/synthdoc-v2-difficult-advice, and the influence ranking they were selected by "
    "came from a Qwen3.6-27B r64 alpha128 warmup LoRA (bf16, 4 epochs, "
    "P=318,767,104 trainable parameters) — see the ranking repo for its full pinning.")

SCHEMA = (
    "mixture.jsonl — jsonl rows {messages: [{role, content, reasoning_content?}], source}; "
    "model-agnostic interchange, rendered with the training family's chat template at train "
    "time (src/model_profile.py ModelProfile). Every row carries a real reasoning trace on "
    "its assistant turn, which is what lets the train config declare thinking: true. "
    "mixture_stats.json — example/token counts by source. "
    "selection_ids.json — the selected less_ids with the rank and influence each was chosen "
    "on: {less_id, rank, score_max, score_mean, per_subtask, trait_id, in_warmup}. "
    "less_id is <scenario_id>#<row index into the pool file>, so the pool row is "
    "recoverable by index. NOTE: mixture.jsonl is shuffled (seed 0) and is NOT in id order.")


def ids_payload(meta: dict, side: str) -> dict:
    """The id manifest for one side, carrying why each row was selected."""
    scores = {r["less_id"]: r for r in _scores()}
    block = meta[side]
    return {
        "selected_by": ("LESS influence, top "
                        f"{meta['frac']:.0%} by {meta['order']}" if side == "selection"
                        else f"uniform random, seed {meta['control_seed']}"),
        "k": meta["k"],
        "ranking": meta["ranking"],
        "pool": meta["pool"],
        "composition": {k: block[k] for k in ("n", "traits", "argmax_subtask", "in_warmup")},
        "rows": [{
            "less_id": i,
            "rank": scores[i]["rank"],
            "score_max": scores[i]["score_max"],
            "score_mean": scores[i]["score_mean"],
            "per_subtask": scores[i]["per_subtask"],
            "trait_id": scores[i]["trait_id"],
            "in_warmup": scores[i]["in_warmup"],
        } for i in block["ids"]],
    }


def _scores() -> list[dict]:
    from src.infra.huggingface import hf_download
    from scratch.less.select_topk import SCORES_FILE, SCORES_REPO
    meta = json.loads(META.read_text(encoding="utf-8"))
    local = hf_download(SCORES_REPO, SCORES_FILE, repo_type="dataset",
                        revision=meta["ranking"]["revision"])
    return [json.loads(x) for x in Path(local).read_text(encoding="utf-8").splitlines() if x.strip()]


def card(meta: dict, side: str, repo: str, other: str) -> str:
    """Front matter + the CLAUDE.md-required fields, for one side of the pair."""
    block = meta[side]
    is_arm = side == "selection"
    what = (f"the {meta['k']} highest-influence rows (top {meta['frac']:.0%}) of the "
            f"difficult-advice pool by {meta['order']}"
            if is_arm else
            f"{meta['k']} rows drawn uniformly at random (seed {meta['control_seed']}) "
            f"from the same {meta['pool']['rows']}-row difficult-advice pool")
    role = ("THE LESS ARM" if is_arm else "THE CONTROL ARM")
    gen = {
        "seed": 0, "max_seq_len": 8192, "budget_tokenizer": "Qwen/Qwen3.6-27B",
        "k": meta["k"], "frac": meta["frac"],
        "selection": (f"top-{meta['k']} by {meta['order']}" if is_arm
                      else f"uniform random, seed {meta['control_seed']}"),
        "ranking": meta["ranking"], "pool": meta["pool"],
        "paired_with": other,
        "incidental_overlap_with_the_other_arm": meta["incidental_overlap"]["n"],
    }
    fields = {
        "title": ("LESS top-10% difficult-advice selection (220 rows, score_max)" if is_arm
                  else "Random 220-row difficult-advice control for the LESS top-10% arm"),
        "experiment": (
            f"{role} of a paired LESS (arXiv:2402.04333) data-selection experiment: "
            f"SFT training file holding {what}, trained as-is on base Qwen3.6-27B with no "
            f"other data. The two arms differ ONLY in which 220 of the same 2,203 rows "
            f"they hold — identical tokenizer, budget, seed, shuffle and training recipe — "
            f"so a difference between the trained organisms is attributable to the "
            f"selection. Paired with {other}. This is LESS as the paper runs it (rank the "
            f"pool, keep the top fraction, train on the kept rows alone), NOT the "
            f"2026-08-17 less-swap arm, which replaced 151 rows of a 10,000-row mixture."),
        "date_generated": DATE,
        "constitution": CONSTITUTION,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": MODELS,
        "generation_config": json.dumps(gen),
        "schema": SCHEMA,
        "provenance": (
            "uv run python scratch/less/select_topk.py --frac 0.10 --order "
            f"{meta['order']} --seed {meta['control_seed']} && uv run mix --config "
            f"configs/data/mixture/{'qwen36_less_top10' if is_arm else 'qwen36_less_random220'}"
            ".yaml && uv run python scratch/less/publish_selection.py"),
        "trait_composition": json.dumps(block["traits"]),
        "argmax_subtask": json.dumps(block["argmax_subtask"]),
        "warmup_rows_included": (
            f"{block['in_warmup']}/{meta['k']} rows were in the LESS warmup split "
            f"(~{round(meta['k'] * meta['k'] / meta['pool']['rows'])} expected by chance). "
            "Warmup rows were trained on before being scored, so their influence is partly "
            "self-influence; recorded, not corrected for."),
    }
    front = card_front_matter([
        {"config_name": "default", "data_files": "mixture.jsonl", "default": True}])
    return front + "\n" + card_markdown(fields)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    meta = json.loads(META.read_text(encoding="utf-8"))
    api = hf_api()

    for side, repo, other in (("selection", ARM_REPO, CTL_REPO),
                              ("control", CTL_REPO, ARM_REPO)):
        payload = ids_payload(meta, side)
        readme = card(meta, side, repo, other)
        assert len(payload["rows"]) == meta["k"], "id manifest is the wrong length"
        print(f"\n=== {repo} ===")
        print(f"    {len(payload['rows'])} ids, composition {payload['composition']['traits']}")
        if args.dry_run:
            print(readme[:1200])
            continue
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "selection_ids.json"
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            api.upload_file(path_or_fileobj=str(p), path_in_repo="selection_ids.json",
                            repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset")
        print(f"    pushed selection_ids.json + README.md -> "
              f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
