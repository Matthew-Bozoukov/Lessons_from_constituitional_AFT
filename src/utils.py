# ABOUTME: Shared utilities: robust JSON extraction/JSONL io and run provenance
# ABOUTME: (git SHA, origin URL, timestamps, run_meta.json).

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file into a list of parsed records.

    Splits on "\\n" only. This is not pedantry: `str.splitlines()` also breaks on the
    Unicode line separators U+2028 and U+2029, which occur inside real prompt and
    response text (Arena-Hard's question set contains them). Since JSON encodes those
    characters literally rather than escaping them, `splitlines()` tears a single record
    in half and the parse dies with "Unterminated string" — a confusing failure that
    looks like file corruption. Iterating a file handle does not have this behaviour,
    which is why the bug hides until someone switches to `read_text()`.

    Args:
        path: Path to a `.jsonl` file.

    Returns:
        Parsed records, skipping blank lines.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def extract_json(text: str) -> Any:
    """Extract the first top-level JSON array or object from model text.

    Handles responses wrapped in prose or ```json fences.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        ParseError: If no valid JSON array/object can be parsed.
    """
    stripped = text.strip()
    # Fast path: whole string is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find the first '[' or '{' and scan to its matching close.
    start = min(
        (i for i in (stripped.find("["), stripped.find("{")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ParseError(f"No JSON found in response: {text[:200]!r}")

    open_ch = stripped[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Malformed JSON: {e}: {candidate[:200]!r}") from e
    raise ParseError(f"Unterminated JSON in response: {text[:200]!r}")


def git_sha() -> str:
    """Return the current git commit SHA, or 'nogit' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_run_meta(out_dir: Path, config: dict, extra: dict | None = None) -> Path:
    """Write a run_meta.json capturing config, git SHA, and timestamps.

    Args:
        out_dir: Directory to write into (created if missing).
        config: The resolved run config.
        extra: Additional metadata to merge in.

    Returns:
        Path to the written run_meta.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def origin_url() -> str:
    """This repo's origin URL, best-effort (provenance only)."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL).decode().strip() or "this repository"
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "this repository"
