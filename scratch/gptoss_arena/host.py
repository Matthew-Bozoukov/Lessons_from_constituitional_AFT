# ABOUTME: Reuse the qualified Windows awake/resource guard and owned local Docker launcher.
# ABOUTME: Run: uv run --project scratch/swebench_cpu_env python scratch/gptoss_arena/host.py start|guard
import sys
import time
from pathlib import Path
from scratch.gptoss_mmlu import host
host.ROOT = Path.cwd()
host.OUT = host.ROOT / "output/gptoss_arena_20261014"
host.NAME = "lasr-gptoss-arena-20261014"
def guard():
    while True:
        try:
            return host.guard()
        except PermissionError:
            # Docker/Windows may briefly hold the destination during atomic replace.
            # Stale evidence already blocks new paid admissions until this succeeds.
            time.sleep(1)

if __name__ == "__main__":
    {"start": host.start, "guard": guard}[sys.argv[1]]()
