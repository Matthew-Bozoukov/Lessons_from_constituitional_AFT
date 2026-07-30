#!/usr/bin/env python3
"""Where does wall-clock time actually go in a Petri run?

Compares measured target-model generation time against total wall time, to
establish whether the rented GPU or the auditor API is the bottleneck. This
decides whether renting more GPUs would help at all.
"""
import glob
import sys

from inspect_ai.log import read_eval_log

log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/petri-focused"
TOK_PER_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0  # measured single-stream

files = sorted(glob.glob(f"{log_dir}/*.eval"))
log = read_eval_log(files[-1])
samples = log.samples or []

# Model usage is recorded per-model on the log stats.
stats = getattr(log, "stats", None)
usage = getattr(stats, "model_usage", {}) or {}

target_out = target_in = 0
auditor_out = auditor_in = other_out = 0
for model, u in usage.items():
    out = getattr(u, "output_tokens", 0) or 0
    inp = getattr(u, "input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    total_in = inp + cr + cw
    if "vllm" in model or "msm" in model:
        target_out += out
        target_in += total_in
    elif "sonnet" in model:
        auditor_out += out
        auditor_in += total_in
    else:
        other_out += out

started = getattr(stats, "started_at", None)
completed = getattr(stats, "completed_at", None)
print(f"log: {files[-1].split(chr(92))[-1]}")
print(f"samples completed: {len(samples)}")
print()
print(f"target output tokens : {target_out:>12,}")
print(f"target input tokens  : {target_in:>12,}")
print(f"auditor output tokens: {auditor_out:>12,}")
print(f"auditor input tokens : {auditor_in:>12,}")
print()

gpu_secs = target_out / TOK_PER_SEC
print(f"GPU generation time at {TOK_PER_SEC:.0f} tok/s: {gpu_secs/60:.1f} min")

if started and completed:
    wall = (completed - started).total_seconds()
    print(f"wall-clock elapsed                : {wall/60:.1f} min")
    print(f"==> GPU busy fraction             : {gpu_secs/wall*100:.1f}%")
    print(f"==> idle/auditor-bound fraction   : {100 - gpu_secs/wall*100:.1f}%")
else:
    print("(run still in progress - wall time taken from caller)")
    if len(sys.argv) > 3:
        wall = float(sys.argv[3]) * 60
        print(f"wall-clock elapsed (given)        : {wall/60:.1f} min")
        print(f"==> GPU busy fraction             : {gpu_secs/wall*100:.1f}%")
        print(f"==> idle/auditor-bound fraction   : {100 - gpu_secs/wall*100:.1f}%")

print()
ratio = auditor_out / target_out if target_out else 0
print(f"auditor output / target output ratio: {ratio:.1f}x")
print("Interpretation: a high ratio with low GPU busy fraction means the run is")
print("bounded by the auditor API, not by the rented GPU. Adding GPUs would not")
print("speed it up; adding concurrency would.")
