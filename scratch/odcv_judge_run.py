# ABOUTME: Thin runner for ODCV judging (odcv_judge.main has no __main__ block).
# Usage: python odcv_judge_run.py <combined_rollout_dir> <config>
import sys

from src.eval.misalignment.odcv import odcv_judge

if __name__ == "__main__":
    odcv_judge.main(rollout_dir=sys.argv[1], config=sys.argv[2], max_workers=8, smoke=False)
