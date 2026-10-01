# ABOUTME: Print the judge's reasoning for every rollout in the given (variant, scenario) cells for all
# ABOUTME: three runs, with severity, progress and call counts: a quick map before reading transcripts.
# Run: uv run python judges.py incentivized/Ai-Education-Assistant [more cells ...]
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, call_list, load_run, parse  # noqa: E402

data = {run: load_run(run) for run in RUNS}
for cell in sys.argv[1:]:
    v, s = cell.split("/")
    print("=" * 100)
    print(cell)
    for run in RUNS:
        for k in (1, 2, 3):
            r = data[run][(v, s, k)]
            n = len(call_list(parse(r["path"].read_text())))
            print(f"--- {run} pass{k} sev={r['sev']} prog={r['prog']} calls={n}")
            print("   ", r["judge"][:900])
