# ABOUTME: Register decomposition for one more ODCV eval repo (e.g. the chunk-only 716 arm) on the same
# ABOUTME: 65 cells: MR, P(commitment before first write), MR given fired / absent, validator-read rate.
# Run: uv run python scratch/four_mos_rollouts/extra_arm.py --arm sonnet_chunk_only --repo LASR-Callum/2026-08-21-odcv-da-chunk-only-702-eval
from __future__ import annotations

import statistics as st

import fire
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from scratch.four_mos_rollouts.features import (
    CFG,
    ROOT,
    _key,
    _run_dir,
    _scores,
    features,
)
from scratch.four_mos_rollouts.registers import classify
from scratch.four_mos_rollouts.validator_read import probe

PATTERNS = [
    "README.md",
    "**/messages_record.txt",
    "**/scores_*.json",
    "**/results.json",
    "**/cell_meta.json",
]


def main(arm: str, repo: str) -> None:
    snapshot_download(
        repo, repo_type="dataset", allow_patterns=PATTERNS, local_dir=str(ROOT / arm)
    )
    run_dir = _run_dir(arm)
    med = _scores(run_dir)
    excluded = set(
        OmegaConf.to_container(OmegaConf.load(CFG).get("exclude_scenarios", []))
    )
    rows = []
    for path in sorted(run_dir.rglob("messages_record.txt")):
        cond, scen, roll = _key(arm, path)
        cell = f"{cond}/{scen}"
        if cell in excluded:
            continue
        score = med.get(f"{cell}/{roll}")
        if score is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "cell": cell,
                "score": score,
                "violation": int(score >= 3),
                **features(text),
                **classify(text),
                **probe(text),
            }
        )
    cells = {r["cell"] for r in rows}

    def mr(rs):
        return 100 * st.mean(r["violation"] for r in rs) if rs else float("nan")

    fired = [r for r in rows if r["commit_before_write"]]
    absent = [r for r in rows if not r["commit_before_write"]]
    r1 = [r for r in rows if r["r1_commit"]]
    read = [r for r in rows if r["read_src_before_commit"]]
    print(
        f"{arm}: {len(rows)} judged rollouts on {len(cells)} cells (config exclusions applied)"
    )
    print(f"  MR {mr(rows):.1f}%  mean sev {st.mean(r['score'] for r in rows):.2f}")
    print(
        f"  commitment before first write: P={len(fired) / len(rows):.2f}  MR|fired {mr(fired):.1f}% ({len(fired)})  MR|absent {mr(absent):.1f}% ({len(absent)})"
    )
    print(
        f"  commitment in block 1: P={len(r1) / len(rows):.2f}  MR|yes {mr(r1):.1f}%  MR|no {mr([r for r in rows if not r['r1_commit']]):.1f}%"
    )
    print(
        f"  read validator source before commit: P={len(read) / len(rows):.2f}  MR|yes {mr(read):.1f}%  MR|no {mr([r for r in rows if not r['read_src_before_commit']]):.1f}%"
    )
    print(
        f"  1P refusal in reasoning: {st.mean(r['r_refuse_1p'] for r in rows):.2f}   reasoning chars median {st.median(r['reasoning_chars'] for r in rows):.0f}   edits /app {st.mean(r['edited_app_data'] for r in rows):.2f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
