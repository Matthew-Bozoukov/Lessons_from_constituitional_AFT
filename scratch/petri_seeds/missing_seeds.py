# ABOUTME: Copy the constitution seeds that have no valid score yet in an arm's audit runs into one
# ABOUTME: directory, so `run_audit.py --seeds <dir>` reruns exactly those against a live target.
import shutil
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent))

from push_results import SEEDS, valid_scores  # noqa: E402


def main(out: str, *run_dirs: str) -> None:
    """Write the seeds lacking a valid score across `run_dirs` to `out` and print the count.

    Args:
        out: Directory to create (must not exist) holding the seed files to rerun.
        run_dirs: The arm's audit run directories, each holding `logs/*.eval`; none for an
            arm never audited, which makes every seed missing (a full audit).
    """
    logs = [log for d in run_dirs for log in Path(d, "logs").glob("*.eval")]
    assert logs or not run_dirs, f"no .eval logs under {run_dirs}"
    scores, _ = valid_scores(logs)
    missing = sorted({p.stem for p in SEEDS.glob("*.md")} - set(scores))
    assert missing, "every seed already has a valid score; nothing to rerun"
    dest = Path(out)
    dest.mkdir(parents=True)
    for seed in missing:
        shutil.copy(SEEDS / f"{seed}.md", dest / f"{seed}.md")
    print(f">>> {len(missing)} seeds to rerun -> {dest}  (valid already: {len(scores)})")


if __name__ == "__main__":
    fire.Fire(main)
