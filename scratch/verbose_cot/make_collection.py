# ABOUTME: Group the verbose-CoT artifacts under one HF Collection — datasets now, the
# ABOUTME: LoRA later. Run: uv run python scratch/verbose_cot/make_collection.py

"""HF has no folders that contain repos.

Repos are flat within an org and typed: a LoRA adapter is a `model` repo and an SFT
dataset is a `dataset` repo, so they cannot share one. A Collection is the mechanism that
does what a folder would — it holds datasets, models and spaces together on one page, and
items can be added at any time, so the adapter slots in when it exists.

Idempotent: re-running finds the existing collection by slug rather than creating a second.
"""

from __future__ import annotations

import sys

import src.endpoints.openrouter  # noqa: F401  -- its import calls load_dotenv()
from src.huggingface import hf_api

NAMESPACE = "LASR-Callum"
TITLE = "Verbose CoT — 3x deliberation"
# HF caps this at 150 characters.
DESCRIPTION = ("Does deliberation LENGTH help, holding the ideas constant? The 716 "
               "difficult-advice traces rewritten ~3x longer, and its control arm.")
ITEMS = [
    ("LASR-Callum/2026-08-25-difficult-advice-716-verbose-cot", "dataset"),
    ("LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-716-train", "dataset"),
    # The control arm, so the comparison is visible from the collection page itself.
    ("LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train", "dataset"),
]


def main() -> None:
    api = hf_api()
    existing = next((c for c in api.list_collections(owner=NAMESPACE)
                     if c.title == TITLE), None)
    if existing:
        col = api.get_collection(existing.slug)
        print(f"reusing collection {col.slug}")
    else:
        col = api.create_collection(title=TITLE, namespace=NAMESPACE,
                                    description=DESCRIPTION, private=True, exists_ok=True)
        print(f"created collection {col.slug}")

    have = {i.item_id for i in (col.items or [])}
    for repo_id, kind in ITEMS:
        if repo_id in have:
            print(f"  already in: {repo_id}")
            continue
        try:
            api.add_collection_item(col.slug, item_id=repo_id, item_type=kind,
                                    exists_ok=True)
            print(f"  added:      {repo_id}")
        except Exception as e:                        # a repo that is not pushed yet
            print(f"  SKIPPED:    {repo_id} ({type(e).__name__}: {str(e)[:90]})")
    print(f"\nhttps://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    sys.exit(main())
