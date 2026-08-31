# ABOUTME: Reads the incrementally-written inspect .eval log for the agentic-misalignment run
# ABOUTME: and reports completed samples, error counts and the score distribution so far.

import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

path = sorted(Path(sys.argv[1]).glob("*.eval"))[-1]
log = read_eval_log(str(path))
print(f">>> {path.name}")
print(f"    status={log.status}  model={log.eval.model}  epochs={log.eval.config.epochs}")
samples = log.samples or []
print(f"    samples completed: {len(samples)}")
errs = [s for s in samples if s.error]
print(f"    sample errors: {len(errs)}")
if errs:
    print(f"    first error: {str(errs[0].error)[:300]}")
scores: Counter = Counter()
for s in samples:
    for name, sc in (s.scores or {}).items():
        scores[f"{name}={sc.value}"] += 1
print("    score tally:", dict(scores))
if log.results:
    for m in log.results.scores:
        print(f"    {m.name}: " + ", ".join(f"{k}={v.value}" for k, v in m.metrics.items()))
