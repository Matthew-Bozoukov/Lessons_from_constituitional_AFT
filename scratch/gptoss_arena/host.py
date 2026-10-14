# ABOUTME: Reuse the qualified Windows awake/resource guard and owned local Docker launcher.
# ABOUTME: Run: uv run --project scratch/swebench_cpu_env python scratch/gptoss_arena/host.py start|guard
import sys
from pathlib import Path
from scratch.gptoss_mmlu import host
host.ROOT = Path.cwd()
host.OUT = host.ROOT / "output/gptoss_arena_20261014"
host.NAME = "lasr-gptoss-arena-20261014"
if __name__ == "__main__":
    {"start": host.start, "guard": host.guard}[sys.argv[1]]()
