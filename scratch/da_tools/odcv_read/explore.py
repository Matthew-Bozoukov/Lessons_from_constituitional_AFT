# ABOUTME: Exploration depth per arm: rollouts that search the filesystem (find / ls -R / ls of a parent dir),
# ABOUTME: reads before the first write, and verification after writing (a read or re-run after the last write).
import re
import statistics as st
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from cmdtypes import kind  # noqa: E402
from parse import RUNS, call_list, load_run, parse  # noqa: E402

for run in RUNS:
    rows = load_run(run)
    search = verify_after_write = wrote = 0
    reads_before_write = []
    validator_runs = []
    for key, r in rows.items():
        calls = [
            (n, c if isinstance(c, str) else str(c))
            for n, c in call_list(parse(r["path"].read_text()))
            if n == "bash"
        ]
        if any(
            re.search(
                r"\bfind\b|\bls\b[^|;&]*\s/app/?(\s|$|;|&|\|)|\bls -R|\btree\b", c
            )
            for _, c in calls
        ):
            search += 1
        kinds = [kind(c) for _, c in calls]
        if "write" in kinds:
            wrote += 1
            first = kinds.index("write")
            reads_before_write.append(sum(1 for k in kinds[:first] if k == "read"))
            last = len(kinds) - 1 - kinds[::-1].index("write")
            if any(k in ("read", "run") for k in kinds[last + 1 :]):
                verify_after_write += 1
        validator_runs.append(sum(1 for k in kinds if k == "run"))
    n = len(rows)
    print(
        f"{run}: searched filesystem {100 * search / n:.0f}% | wrote a file {100 * wrote / n:.0f}% | reads before first write "
        f"{st.mean(reads_before_write):.2f} | checked after last write {100 * verify_after_write / max(wrote, 1):.0f}% of writers | script runs/rollout {st.mean(validator_runs):.2f}"
    )
