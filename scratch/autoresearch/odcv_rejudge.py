# ABOUTME: Re-score a subset of ODCV-lite rollouts with the vendored misalignment judge (same model, temperature 0) to
# ABOUTME: measure judge self-consistency in the gap-cluster cells; writes output/autoresearch/odcv_rejudge.jsonl.
"""uv run python scratch/autoresearch/odcv_rejudge.py [--n_cells 10]"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.eval.misalignment.odcv.odcv_judge import _load_upstream_judge, pin_provider
from src.infra.endpoints.openrouter import map_threaded

load_dotenv(".env")
OUT = Path("output/autoresearch")
ARMS = ["s25_s0", "swap_otherai", "s28_base_a", "s28_base_b", "new_otherai"]
MODEL = "google/gemini-3-flash-preview"


def main(n_cells: int = 10) -> None:
    L = pd.read_csv(OUT / "odcv_rollouts_long.csv")
    C = pd.read_csv(OUT / "odcv_cells_good_bad.csv")
    top = set(zip(C.scenario[:n_cells], C.variant[:n_cells]))
    S = L[L.arm.isin(ARMS) & pd.Series([(s, v) in top for s, v in zip(L.scenario, L.variant)], index=L.index)]
    path = OUT / "odcv_rejudge.jsonl"
    done = {(d["arm"], d["variant"], d["scenario"], d["rollout"]) for d in map(json.loads, open(path))} if path.exists() else set()
    todo = [r for r in S.itertuples() if (r.arm, r.variant, r.scenario, r.rollout) not in done]
    print(len(S), "selected,", len(todo), "todo")
    mod = _load_upstream_judge(Path("src/eval/misalignment/odcv/third_party/odcv-bench"))
    mod.OPENAI_MODEL = MODEL
    pin_provider(mod, MODEL)
    lock = threading.Lock()

    def one(i: int):
        r = todo[i]
        p = OUT / "odcv_runs" / r.run / "rollouts" / r.variant / r.scenario / r.pass_dir / "messages_record.txt"
        score, reason = mod.evaluation_routine(str(p))
        with lock, open(path, "a") as f:
            f.write(json.dumps(dict(arm=r.arm, variant=r.variant, scenario=r.scenario, rollout=int(r.rollout),
                                    orig=float(r.severity), rejudge=score, reason=reason)) + "\n")

    map_threaded(one, len(todo), max_workers=12, desc="rejudge")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
