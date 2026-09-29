# ABOUTME: Compact per-rollout digest for the given cells and runs: severity, nudges, each command (short),
# ABOUTME: and the task_complete reason. Run: compact.py <run[,run]> <variant/Scenario> [...]
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import call_list, load_run, parse  # noqa: E402

runs = sys.argv[1].split(",")
data = {r: load_run(r) for r in runs}
for cell in sys.argv[2:]:
    v, s = cell.split("/")
    print("=" * 100)
    print(cell)
    for run in runs:
        for k in (1, 2, 3):
            r = data[run][(v, s, k)]
            steps = parse(r["path"].read_text())
            nudges = sum(1 for x in steps[2:] if x["role"] == "user")
            calls = call_list(steps)
            print(
                f"--- {run} pass{k} sev={r['sev']} calls={len(calls)} nudges={nudges}"
            )
            for name, cmd in calls:
                if name == "task_complete":
                    reason = (
                        cmd.get("reason", "") if isinstance(cmd, dict) else str(cmd)
                    )
                    print("     TC:", reason[:500].replace("\n", " "))
                else:
                    print("     $", str(cmd)[:130].replace("\n", " \\n "))
