# ABOUTME: The Inspect task the tenet audit runs: Petri's `audit`, plus the constitution prepended to
# ABOUTME: every TARGET system message when PETRI_SYSTEM_PREAMBLE_FILE is set (auditor/judge untouched).
"""`inspect eval src/eval/audits/petri/tenet_audit/task.py@tenet_audit`.

The constitution-in-the-system-prompt arms need the text in front of what the target is told,
and Petri has no hook for it: the auditor writes the target's system message through
`Controller.stage_system`. Until 2026-10-28 that was an edit to the installed package inside
one laptop's venv, so a fresh install silently ran those arms WITHOUT the constitution. The
same wrapper now lives here and is applied when this module loads -- i.e. in the process that
runs the task -- and only when the env var names a file. Unset, the task is Petri's `audit`
exactly.

`scripts/eval/audits/petri/verify_target_preamble.py` checks the written transcripts, which is
the only check that counts; the assert below only stops a run whose staging went wrong.
"""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

from inspect_ai import task
from inspect_petri import audit
from inspect_petri.target._controller import Controller

from src.eval.audits.petri.tenet_audit import PETRI_VERSION

assert version("inspect-petri") == PETRI_VERSION, (
    f"inspect-petri {version('inspect-petri')} installed; the tenet audit is pinned to "
    f"{PETRI_VERSION} (run it through run_audit.py, which installs that version)")

# A venv still carrying the old in-place edit would prepend the constitution twice.
assert Controller.stage_system.__name__ == "stage_system", (
    f"Controller.stage_system is already wrapped ({Controller.stage_system.__name__}); restore "
    f"inspect_petri/target/_controller.py from its .orig or reinstall inspect-petri")
_ORIGINAL_STAGE_SYSTEM = Controller.stage_system


def preamble() -> str:
    """The configured constitution text, or "" when PETRI_SYSTEM_PREAMBLE_FILE is unset."""
    path = os.environ.get("PETRI_SYSTEM_PREAMBLE_FILE")
    if not path:
        return ""
    text = Path(path).read_text(encoding="utf-8").rstrip()
    assert text, f"PETRI_SYSTEM_PREAMBLE_FILE is empty: {path}"
    return text


async def _stage_system_with_preamble(self, content: str) -> str:
    """Stage a target system message with the constitution first."""
    pre = preamble()
    if pre:
        content = f"{pre}\n\n{content}"
        assert content.startswith(pre[:80]), "constitution is not first in the target system message"
    return await _ORIGINAL_STAGE_SYSTEM(self, content)


Controller.stage_system = _stage_system_with_preamble


@task
def tenet_audit(**kwargs):
    """Petri's `audit` task with the preamble wrapper installed; kwargs pass straight through."""
    return audit(**kwargs)
