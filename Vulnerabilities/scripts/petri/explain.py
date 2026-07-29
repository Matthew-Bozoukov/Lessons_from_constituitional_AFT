#!/usr/bin/env python3
"""Print the judge's explanation and summary for named samples in an eval log.

A numerical score is not a finding. This surfaces what the judge actually cited,
so a high score on a CONTROL arm (which would undermine the design) can be
distinguished from a genuine finding.
"""
import glob
import sys

from inspect_ai.log import read_eval_log

log_dir = sys.argv[1]
wanted = sys.argv[2].split(",")
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2600

files = sorted(glob.glob(f"{log_dir}/*.eval"))
log = read_eval_log(files[-1])

for s in log.samples or []:
    if s.id not in wanted:
        continue
    print("#" * 78)
    print(f"SAMPLE: {s.id}   epoch {s.epoch}")
    for _, v in (s.scores or {}).items():
        exp = getattr(v, "explanation", None)
        if exp:
            print(exp[:limit])
        meta = getattr(v, "metadata", None) or {}
        if "summary" in meta:
            print("\n--- SUMMARY ---")
            print(str(meta["summary"])[:limit])
    print()
