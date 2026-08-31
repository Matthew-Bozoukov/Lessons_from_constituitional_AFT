# ABOUTME: Add one ODCV pass's transcripts into an existing combined dir as rollout_<idx>, so the
# ABOUTME: judge (cache keyed by variant/scenario/rollout_NNN) reuses prior verdicts and only scores the new pass.
# Usage: python odcv_add_pass.py <model_key> <combined_dir> <new_pass_dir> <idx>
import shutil
import sys
from pathlib import Path

VARIANTS = ("mandated", "incentivized")


def main() -> None:
    model_key, combined_dir, new_pass_dir, idx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    n = 0
    for variant in VARIANTS:
        exp = Path(new_pass_dir) / "agent_logs" / f"{model_key}-{variant}" / "experiments"
        if not exp.is_dir():
            continue
        for scen in exp.iterdir():
            rec = scen / "messages_record.txt"
            if not (rec.is_file() and rec.stat().st_size > 0):
                continue
            dest = (Path(combined_dir) / "agent_logs" / f"{model_key}-{variant}" / "experiments"
                    / scen.name / f"rollout_{idx:03d}")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(rec, dest / "messages_record.txt")
            n += 1
    print(f"added {n} transcripts as rollout_{idx:03d} into {combined_dir}")


if __name__ == "__main__":
    main()
