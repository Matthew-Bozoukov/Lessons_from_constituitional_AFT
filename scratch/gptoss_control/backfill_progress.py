# ABOUTME: Runs ODCV's progress (task-completion) judge over already-combined transcripts.
# ABOUTME: For arms scored outside runner.run(), which is what emits the progress axis.
"""runner.run() judges two axes: misalignment (odcv_judge) and task progress
(progress_judge). Arms judged by hand on 2026-10-06 -- msm, da-15, da-msm-15 -- got only the
first, so the one axis that says whether the model actually did the task was missing from
exactly the arms whose competence is in question (msm reasons on 0.0% of ODCV turns and
re-issues tool calls instead of consuming results).

This runs the progress judge over each arm's existing combined directory. No sampling: the
transcripts are already on disk, so the cost is judge tokens only (~$1/arm).

Usage:
    uv run --project src/infra/endpoints/tinker_runtime python \
        scratch/gptoss_control/backfill_progress.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv(ROOT / ".env")
from src.eval.misalignment.odcv import progress_judge

ARMS = {
    "msm": ("output/gptoss_arms/msm/pass3", "2026-10-06-gptoss120b-0-msm"),
    "da-15": (None, "2026-10-06-gptoss120b-0-da-15"),          # combined lives under its odcv run
    "da-msm-15": ("output/gptoss_arms/da-msm-15/judge", "2026-10-06-gptoss120b-0-da-msm-15"),
}

def find_combined(arm: str, work: str | None, mk: str) -> Path:
    """The combined rollout directory this arm was judged from."""
    roots = [Path(work)] if work else []
    roots += sorted(Path(f"output/gptoss_arms/{arm}/odcv").iterdir()) if Path(f"output/gptoss_arms/{arm}/odcv").is_dir() else []
    for r in roots:
        hits = [p for p in r.rglob("combined*") if p.is_dir() and (p/"agent_logs").is_dir()]
        if hits:
            return sorted(hits)[-1]
    raise FileNotFoundError(f"no combined dir for {arm}")

for arm, (work, mk) in ARMS.items():
    combined = find_combined(arm, work, mk)
    n = len(list(combined.rglob("messages_record.txt")))
    print(f"\n=== {arm}: {combined} ({n} transcripts) ===", flush=True)
    cfg_path = next(p for p in (Path(work)/"odcv_config.yaml" if work else None,
                               Path(f"output/gptoss_arms/{arm}/odcv_config.yaml")) if p and p.exists())
    cfg = OmegaConf.load(cfg_path)
    cfg.model = "openai/gpt-oss-120b"; cfg.model_key = mk
    out = Path(f"output/gptoss_arms/{arm}/progress"); out.mkdir(parents=True, exist_ok=True)
    cfg.judge_budget.ledger = str((out/"judge_budget.json").resolve())
    p2 = out/"odcv_config.yaml"; OmegaConf.save(cfg, p2)
    res = progress_judge.main(rollout_dir=str(combined), config=str(p2), max_workers=8)
    o = (res or {}).get("ours", {}).get("overall", {})
    print(f"  {arm}: tp_rate {o.get('tp_rate_pct')}% "
          f"[{o.get('tp_rate_ci95_lo')}, {o.get('tp_rate_ci95_hi')}]  tp_mean {o.get('tp_mean')}",
          flush=True)
    (out/"progress_summary.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
print("\n>>> DONE", flush=True)
