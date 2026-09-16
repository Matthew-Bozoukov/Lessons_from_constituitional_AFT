# ABOUTME: Keeps Windows awake only while this campaign has live owner processes.
# ABOUTME: Uses a temporary execution-state request and releases it on completion or the bounded deadline.
import ctypes
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from src.infra.runpod import _parent_alive

plan = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else "scratch/da_supervision/odcv_plan.yaml")
deadline = time.time() + (float(plan.max_hours) + 0.5) * 3600
try:
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    while time.time() < deadline:
        live = []
        for arm in plan.arms:
            status = Path(plan.output_root) / arm.key / "status.json"
            if status.exists():
                state = json.loads(status.read_text(encoding="utf-8"))
                live.append(_parent_alive(int(state["pid"])))
        if live and not any(live):
            break
        time.sleep(20)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
