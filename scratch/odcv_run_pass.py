# ABOUTME: Thin runner for one ODCV rollout pass — odcv_rollout.main has no __main__ block,
# ABOUTME: so it can't be run as a script directly. Usage: python odcv_run_pass.py <config> [resume_dir]
import sys

from src.eval.misalignment.odcv import odcv_rollout

if __name__ == "__main__":
    cfg = sys.argv[1]
    resume = sys.argv[2] if len(sys.argv) > 2 else ""
    odcv_rollout.main(config=cfg, smoke=False, resume=resume)
