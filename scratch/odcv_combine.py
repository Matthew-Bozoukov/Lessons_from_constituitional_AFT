# ABOUTME: Combine N single-rollout ODCV pass dirs into one `combinedNx_<ts>` dir with
# ABOUTME: per-scenario rollout_000.. subdirs (the "repeated" layout odcv_judge reads).
# Usage: python odcv_combine.py <model_key> <pass_dir_1> <pass_dir_2> ...
import shutil
import sys
from pathlib import Path

from src.utils import timestamp

VARIANTS = ("mandated", "incentivized")


def main() -> None:
    model_key = sys.argv[1]
    pass_dirs = [Path(p) for p in sys.argv[2:]]
    assert len(pass_dirs) >= 2, "need >=2 pass dirs"
    base = Path("output/odcv_bench") / model_key
    combined = base / f"combined{len(pass_dirs)}x_{timestamp()}"
    n_copied = 0
    for i, pdir in enumerate(pass_dirs):
        for variant in VARIANTS:
            exp = pdir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
            if not exp.is_dir():
                continue
            for scen in exp.iterdir():
                rec = scen / "messages_record.txt"
                if not (rec.is_file() and rec.stat().st_size > 0):
                    continue
                dest = (combined / "agent_logs" / f"{model_key}-{variant}" / "experiments"
                        / scen.name / f"rollout_{i:03d}")
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(rec, dest / "messages_record.txt")
                n_copied += 1
    print(f"combined {len(pass_dirs)} passes -> {combined}")
    print(f"rollout transcripts copied: {n_copied}")
    print(combined)  # last line = the combined dir path (for the judge)


if __name__ == "__main__":
    main()
