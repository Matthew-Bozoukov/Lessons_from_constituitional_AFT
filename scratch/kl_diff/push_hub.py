# ABOUTME: Push the per-token KL scores (+ steps, run metas, aggregated results) to the Hub as one dated artifact with a full card.
# ABOUTME: Run: uv run python scratch/kl_diff/push_hub.py --scores output/kl_diff/scores_mask.jsonl output/kl_diff/scores_hospital.jsonl --results output/kl_diff/<date>_kl_results.json
"""One dataset repo `<date>-kl-diff-da-tools-mask-hospital` under the org: gzipped score files,
the step files they were scored on, score.py's run metas, and analyse.py's results json. The card
carries the card's required fields (constitution: the DA data's nine-trait constitution, by the
adapters' own cards; see `models`)."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

from scratch.kl_diff.common import ADAPTERS, BASE, RUNS
from src.infra.huggingface import push_files
from src.naming import artifact_name


def gz(p: Path) -> Path:
    out = p.with_suffix(p.suffix + ".gz")
    with open(p, "rb") as f, gzip.open(out, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument(
        "--steps",
        nargs="+",
        default=[
            "output/kl_diff/steps_mask.jsonl",
            "output/kl_diff/steps_hospital.jsonl",
        ],
    )
    ap.add_argument("--results", required=True)
    ap.add_argument("--metas", nargs="*", default=[])
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    date = time.strftime("%Y-%m-%d")
    name = artifact_name("kl-diff-da-tools-mask-hospital", date=date)
    paths = (
        [gz(Path(p)) for p in a.scores + a.steps]
        + [Path(a.results)]
        + [Path(m) for m in a.metas]
    )
    n_steps = sum(1 for p in a.scores for _ in open(p, encoding="utf-8"))
    fields = {
        "title": "Per-token KL of DA and DA + tools against nosynth on fixed MASK and Hospital transcripts",
        "experiment": (
            "Model diffing by teacher forcing: three LoRA adapters on one base read the same eval "
            "transcripts; at every reply token the trained adapters' next-token distributions are "
            "compared with the nosynth control's (KL in nats), with each token labelled reasoning / "
            f"visible / tool_call / format. {n_steps} steps scored. Card: 'Where DA changes transcripts'."
        ),
        "date_generated": date,
        "constitution": "the nine-trait distilled constitution the DA data was generated from (see the adapters' cards); the control has none",
        "source_repo": f"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git @ {sha} (scratch/kl_diff/)",
        "models": json.dumps(
            {
                "base": f"{BASE[0]}@{BASE[1]}",
                **{k: f"{r}@{v[:8]}" for k, (r, v) in ADAPTERS.items()},
            }
        ),
        "generation_config": json.dumps(
            {
                "method": "teacher forcing, no sampling",
                "kl": "KL(trained || nosynth), float32 log-softmax",
                "top5": "per model per token",
                "writers": list(ADAPTERS),
                "runs": RUNS,
                "mask": "200 items x 3 writers, pressure turn, pass1 run1, seed 0",
                "hospital": "tested seats, shifts 1-10 per writer, calls with a reasoning trace",
            }
        ),
        "schema": (
            "scores_*.jsonl.gz: one line per step {key, eval, writer, item, step, meta, n_prompt, labels[], tokens[], "
            "kl{da,da_tools}[], lp{nosynth,da,da_tools}[], top5{model:{ids[][5],p[][5]}}}. steps_*.jsonl.gz: the scored steps "
            "{eval, writer, item, step, messages, tools, target, meta}. *_kl_results.json: analyse.py output."
        ),
        "provenance": (
            "uv run python scratch/kl_diff/load_transcripts.py --eval {mask,hospital}; on a pod: uv run python "
            "scratch/kl_diff/score.py --steps <steps> --out <scores> [--seeds 1,...,10]; uv run python "
            "scratch/kl_diff/analyse.py --scores <scores...>; uv run python scratch/kl_diff/push_hub.py ..."
        ),
    }
    url = push_files(paths, name, fields, private=a.private)
    print(json.dumps({"repo": name, "url": url, "files": [p.name for p in paths]}))


if __name__ == "__main__":
    main()
