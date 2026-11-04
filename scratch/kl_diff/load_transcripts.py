# ABOUTME: Build the scored steps (prompt messages + the writer's own output) from rollouts on the Hub.
# ABOUTME: Run: uv run python scratch/kl_diff/load_transcripts.py --eval mask --n 200 --out output/kl_diff/steps_mask.jsonl
"""MASK: the pressure turn of each item (system + user, or the doubling-down history), target =
that writer's reasoning + reply, pass 1, run 1. The SAME items for all three writers.
Hospital: built in hospital_steps (needs the rebuilt tool schemas, see hospital_tools.py)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from scratch.kl_diff.common import RUNS, dumps

MASK_FILES = (
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
)
GEN = "generation(System Prompt + User Prompt)_run1"
REA = "reasoning(System Prompt + User Prompt)_run1"
FIN = "finish_reason(System Prompt + User Prompt)_run1"


def _mask_csv(repo: str, archetype: str) -> pd.DataFrame:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    want = [
        f
        for f in files
        if f.endswith(f"/{archetype}.csv") and ("pass1/" in f or f.count("/") == 1)
    ]
    assert len(want) == 1, (repo, archetype, want)
    return pd.read_csv(hf_hub_download(repo, want[0], repo_type="dataset"))


def _usable(row) -> bool:
    return (
        isinstance(row.get(REA), str)
        and row[REA].strip() != ""
        and isinstance(row.get(GEN), str)
        and row[GEN].strip() != ""
        and row.get(FIN) == "stop"
    )


def mask_steps(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    per = {
        a: n // len(MASK_FILES) + (1 if i < n % len(MASK_FILES) else 0)
        for i, a in enumerate(MASK_FILES)
    }
    steps = []
    for arch in MASK_FILES:
        dfs = {
            w: _mask_csv(repo, arch).set_index("task_id")
            for w, repo in RUNS["mask"].items()
        }
        ids = sorted(
            set.intersection(
                *[{t for t, r in d.iterrows() if _usable(r)} for d in dfs.values()]
            )
        )
        pick = rng.sample(ids, min(per[arch], len(ids)))
        for tid in sorted(pick):
            for w, d in dfs.items():
                r = d.loc[tid]
                msgs = [{"role": "system", "content": r["system_prompt"]}]
                if "dd_turn_1_prompt" in d.columns:
                    msgs += [
                        {"role": "user", "content": r["dd_turn_1_prompt"]},
                        {"role": "assistant", "content": r["dd_turn_1_response"]},
                    ]
                msgs.append({"role": "user", "content": r["user_prompt"]})
                steps.append(
                    {
                        "eval": "mask",
                        "writer": w,
                        "item": f"{arch}/{tid}",
                        "step": 0,
                        "messages": msgs,
                        "tools": None,
                        "target": {
                            "reasoning_content": r[REA].strip(),
                            "content": r[GEN].strip(),
                        },
                        "meta": {"archetype": arch},
                    }
                )
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["mask", "hospital"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.eval == "mask":
        steps = mask_steps(a.n)
    else:
        from scratch.kl_diff.hospital_steps import hospital_steps

        steps = hospital_steps()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(dumps(s) + "\n" for s in steps), encoding="utf-8")
    by = {}
    for s in steps:
        by[s["writer"]] = by.get(s["writer"], 0) + 1
    print(json.dumps({"out": str(out), "steps": len(steps), "by_writer": by}))


if __name__ == "__main__":
    main()
