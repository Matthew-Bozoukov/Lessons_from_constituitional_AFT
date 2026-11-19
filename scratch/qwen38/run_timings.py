# ABOUTME: Wall-clock of each oai_hf Step 3 run (Inspect log start/end, sample working times) for costing reruns,
# ABOUTME: plus a rollout manifest (arm, epoch, outcome, transcript path). Run: uv run python scratch/qwen38/run_timings.py
import glob
import json
from datetime import datetime
from pathlib import Path

from inspect_ai.log import read_eval_log

RUNS = {
    "nosynth": "output/oai_hf/2026-11-18_qwen38_0_plain_185529",
    "datools": "output/oai_hf/2026-11-18_qwen38_0_da_tools_15_194726",
    "da": "output/oai_hf/2026-11-19_qwen38_0_da_15_131422",
}
MANIFEST = Path("output/oai_hf_repro/2026-11-19_step3_rollout_manifest.jsonl")
rows = []
for arm, d in RUNS.items():
    f = glob.glob(f"{d}/metadata/inspect/*.eval")[0]
    log = read_eval_log(f, header_only=True)
    t0 = datetime.fromisoformat(log.stats.started_at)
    t1 = datetime.fromisoformat(log.stats.completed_at)
    mins = (t1 - t0).total_seconds() / 60
    usage = {
        m: (u.input_tokens, u.output_tokens)
        for m, u in (log.stats.model_usage or {}).items()
    }
    print(f"{arm:8} inspect wall {mins:5.1f} min | tokens (in, out) by model: {usage}")
    for r in map(json.loads, open(f"{d}/results/runs.jsonl")):
        rows.append(
            {
                "arm": arm,
                "epoch": r["epoch"],
                "shared": r["signature"],
                "judge_level": r["judge_level"],
                "judge_headline": r["judge_headline"],
                "got_egress": r["got_egress"],
                "solved": r["solved"],
                "found_board": r["found_board"],
                "end_reason": r["end_reason"],
                "turns": r["turns"],
                "working_time_s": r["working_time"],
                "transcript": str(
                    Path(d)
                    / "rollouts/step3"
                    / f"step3egress-react-driven_epoch{r['epoch']}.md"
                ),
            }
        )
MANIFEST.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
wt = sorted(r["working_time_s"] for r in rows)
print(
    f"{len(rows)} rollouts -> {MANIFEST}; per-rollout working time median {wt[len(wt) // 2]:.0f}s, p90 {wt[int(0.9 * len(wt))]:.0f}s, max {wt[-1]:.0f}s"
)
