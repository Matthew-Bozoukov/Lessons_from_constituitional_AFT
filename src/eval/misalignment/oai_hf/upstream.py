# ABOUTME: Fetches the paper's harness (msp895/oai-hf-incident-reproduction) at the pinned commit
# ABOUTME: into a local cache and returns its docker_replication/ directory; never vendored (no LICENSE).

"""The upstream harness, at exactly the commit the config names.

The upstream repo ships no LICENSE, so this public repo does not copy it. Instead the first
run clones it into `~/.cache/oai_hf_harness/<sha>` (override with OAI_HF_HARNESS_CACHE) and
every run verifies the checkout is that sha before using it. The task files are then loaded
by path, exactly as `inspect eval <task.py>` loads them upstream.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["ensure_harness", "task_file"]

SUBDIR = Path("src/alignment_auditor/docker_replication")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure_harness(repo: str, commit: str) -> Path:
    """Return the docker_replication/ dir of `repo` at `commit`, cloning it on first use.

    Fails fast when `commit` is not a full 40-character sha or the checkout does not match.
    """
    if len(commit) != 40:
        raise SystemExit(f"oai_hf: harness.commit must be a full sha, got {commit!r}")
    root = (
        Path(
            os.environ.get("OAI_HF_HARNESS_CACHE", "~/.cache/oai_hf_harness")
        ).expanduser()
        / commit
    )
    if not (root / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        _git(root, "init", "-q")
        _git(root, "remote", "add", "origin", repo)
        _git(root, "fetch", "-q", "--depth", "1", "origin", commit)
        _git(root, "checkout", "-q", "FETCH_HEAD")
    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise SystemExit(
            f"oai_hf: harness cache {root} is at {head}, config pins {commit}; delete it and rerun"
        )
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise SystemExit(f"oai_hf: harness cache {root} has local changes:\n{dirty}")
    out = root / SUBDIR
    if not out.is_dir():
        raise SystemExit(f"oai_hf: {out} missing in the harness checkout")
    return out


def task_file(harness_dir: Path, task: str) -> Path:
    """The upstream task.py for one step directory name, e.g. `step2_reachout`."""
    f = harness_dir / task / "task.py"
    if not f.is_file():
        raise SystemExit(f"oai_hf: no task file {f}")
    return f
