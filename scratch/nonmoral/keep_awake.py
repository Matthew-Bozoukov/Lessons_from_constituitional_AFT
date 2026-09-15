# ABOUTME: Prevents Windows sleep while the authorized local Docker evaluations run.
# ABOUTME: Per-process request expires after nine hours or a local stop file; no permanent power settings change.
import ctypes
import json
import sys
import time
from pathlib import Path

assert sys.platform == "win32"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
stop = out / "keep_awake.stop"
assert not stop.exists(), "Prior stop marker exists; do not silently restart"
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
kernel.SetThreadExecutionState.restype = ctypes.c_uint
start = time.time()
try:
    assert kernel.SetThreadExecutionState(0x80000001), "Could not request system wakefulness"
    print("Windows sleep inhibition active for local Docker; display settings unchanged", flush=True)
    while time.time() - start < 9 * 3600 and not stop.exists():
        (out / "keep_awake.json").write_text(json.dumps({
            "active": True, "elapsed_s": time.time()-start,
            "deadline_epoch": start+9*3600,
        }), encoding="utf-8")
        time.sleep(30)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    (out / "keep_awake.json").write_text(json.dumps({"active": False}), encoding="utf-8")
    print("Windows sleep inhibition released", flush=True)
