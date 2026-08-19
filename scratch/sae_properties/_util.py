# ABOUTME: Tiny local stand-ins for src.utils (run_meta, timestamp) — the nested env
# ABOUTME: cannot import src.*, and duplicating two helpers beats coupling the envs.

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_run_meta(out_dir: Path, config: dict, extra: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path
