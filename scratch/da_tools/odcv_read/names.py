# ABOUTME: Tool names called per arm (any tool other than bash/task_complete would be a leak from the
# ABOUTME: training tool blocks), plus multi-task_complete and zero-call rollouts.
import collections
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, call_list, load_run, parse  # noqa: E402

for run in RUNS:
    rows = load_run(run)
    names = collections.Counter()
    zero = []
    for key, r in rows.items():
        cl = call_list(parse(r["path"].read_text()))
        names.update(n for n, _ in cl)
        if not [c for c in cl if c[0] == "bash"]:
            zero.append((key, r["sev"]))
    print(run, dict(names), "| rollouts with zero bash calls:", len(zero), zero[:12])
