# ABOUTME: Shared utilities: JSON/JSONL io, run provenance (git SHA, timestamps, run_meta)
# ABOUTME: and transcript rendering. THE naming law lives in src/naming.py.

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file into a list of parsed records, skipping blank lines.

    Splits on "\\n" only, not `splitlines()`, which also breaks on U+2028/U+2029 — real
    characters in prompt text (Arena-Hard's questions have them) that JSON encodes
    literally. `splitlines()` tears a record in half and the parse dies with
    "Unterminated string", which looks like file corruption.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def extract_json(text: str) -> Any:
    """The first top-level JSON array/object in model text, unwrapping prose and ```json
    fences. Raises ParseError when there is none."""
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
    """The current git commit SHA, or 'nogit' if unavailable."""
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
    """A filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_run_meta(out_dir: Path, config: dict, extra: dict | None = None) -> Path:
    """Write `out_dir/run_meta.json` — config, git SHA, timestamp, plus `extra`."""
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


def _fence(body: str, lang: str = "") -> str:
    """Code-fence verbatim content, growing the fence past any backtick run inside it."""
    run = max((len(m.group()) for m in re.finditer(r"`+", body)), default=0)
    ticks = "`" * max(3, run + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}"


def transcript_markdown(title: str, intro: str | None,
                        sections: list[tuple[int, str, str, str]]) -> str:
    """THE renderer for self-contained rollout transcripts (CLAUDE.md: "logs" means
    ROLLOUTS), so verbatim model output is delineated the same way in every eval.

    `sections` is (level, heading, kind, body) in order; an empty body renders as the bare
    heading. Kinds: `text` trusted markdown as-is; `fenced` VERBATIM model/prompt output,
    code-fenced so chain-of-thought and raw tags (<message>, <think>) cannot be swallowed
    or re-rendered by a viewer; `json` the same with a language tag.
    """
    parts = [f"# {title}"]
    if intro:
        parts.append(intro)
    for level, heading, kind, body in sections:
        parts.append(f"{'#' * level} {heading}")
        if not body:
            continue
        if kind == "text":
            parts.append(body)
        elif kind == "fenced":
            parts.append(_fence(body))
        elif kind == "json":
            parts.append(_fence(body, "json"))
        else:
            raise ValueError(f"unknown transcript section kind: {kind!r}")
    return "\n\n".join(parts) + "\n"
