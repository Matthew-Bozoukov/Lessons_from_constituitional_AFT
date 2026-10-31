# ABOUTME: Puts this run's constitution preamble first in every TARGET system message Petri stages
# ABOUTME: (auditor and judge untouched); `install(None)` leaves Petri's audit byte-identical.
"""The one hook a constitution arm needs inside Petri, installed from `runner.run()`.

The auditor writes the target's system message through one tool, `set_system_message`
(`inspect_petri/tools/_messages.py`, line 116 on 3.1.0), which does nothing but
`await controller().stage_system(system_message)` -- so `Controller.stage_system` is the only
place a target system message is ever staged, and wrapping it covers every path, including a
`restart_conversation` followed by a fresh system message. The composition is
`src.eval.preamble.Preamble.prepend`, shared with MASK, agentic-misalignment and ODCV; the
runner hands the object in rather than setting an env var, so an arm ladder cannot inherit a
stale one, and `export.rollout_markdown` asserts on every published transcript that the
preamble came first. Importing this module imports `inspect_petri`; it is imported only from
the petri runner, so the eval registry stays light. (History: docs/LOG.md 2026-10-30.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_petri.target._controller import Controller

if TYPE_CHECKING:
    from src.eval.preamble import Preamble

_PREAMBLE: "Preamble | None" = None
_ORIGINAL_STAGE_SYSTEM = Controller.stage_system


async def _stage_system_with_preamble(self, content: str) -> str:
    """Stage a target system message with the run's preamble first (unset: unchanged)."""
    if _PREAMBLE is not None:
        content = _PREAMBLE.prepend(content)
    return await _ORIGINAL_STAGE_SYSTEM(self, content)


_stage_system_with_preamble._lasr_preamble_hook = True  # idempotency marker


def install(preamble: "Preamble | None") -> None:
    """Point the hook at this run's preamble (None = Petri's audit, byte-identical).

    Installs the wrapper once per process; an arm ladder only re-points the module variable.
    Call it before every `inspect_ai.eval` / `eval_retry`, since both build the `audit` task
    from the registry and the wrapper must already be in place when the auditor first calls
    `set_system_message`.
    """
    global _PREAMBLE
    _PREAMBLE = preamble
    if not getattr(Controller.stage_system, "_lasr_preamble_hook", False):
        assert Controller.stage_system is _ORIGINAL_STAGE_SYSTEM, (
            f"Controller.stage_system is already wrapped by something else "
            f"({Controller.stage_system!r}); reinstall inspect-petri")
        Controller.stage_system = _stage_system_with_preamble
