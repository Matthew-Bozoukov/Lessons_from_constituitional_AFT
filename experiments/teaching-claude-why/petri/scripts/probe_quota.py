# ABOUTME: One trivial generate against a model, to test whether subscription quota is available.
# ABOUTME: A file, not a -c string: PowerShell strips quotes before python sees them.
"""Probe whether the Claude subscription window has capacity.

Prints exactly one of: OK / EMPTY / BLOCKED: <reason>

Written as a file rather than passed with `python -c` from PowerShell, because
PowerShell mangles embedded quotes before the interpreter receives them - the
same failure the repo's Invoke-Remote.ps1 was written to avoid. The first
version of the epoch loop used -c and every probe died with
`SyntaxError: unterminated string literal`, which the loop then treated as
"quota exhausted" forever.

Usage:  python scripts/probe_quota.py [model]
"""

from __future__ import annotations

import asyncio
import sys

from inspect_ai.model import get_model


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-code/claude-haiku-4-5"
    try:
        result = await get_model(model).generate("ok")
        print("OK" if (result.completion or "").strip() else "EMPTY")
    except Exception as exc:  # noqa: BLE001 - the reason is the useful part
        print(f"BLOCKED: {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
