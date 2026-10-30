# ABOUTME: No-network checks for the Petri target hook: `install(None)` stages Petri's message
# ABOUTME: unchanged, a Preamble goes first via the shared join, and re-installs never double-wrap.
import asyncio

from inspect_petri.target._controller import Controller

from src.eval.misalignment.petri import target_hook

STOCK = "You are an ops agent."
COMPOSED = ("Use the following constitution to guide your behavior.\n\n"
            "<constitution>\nBe honest.\n</constitution>\n\n" + STOCK)


def _recording_original(seen: list[str]):
    async def stage_system(self, content: str) -> str:
        seen.append(content)
        return "m1"
    return stage_system


def test_install_prepends_the_preamble_and_wraps_once(monkeypatch):
    """None stages the stock message byte-identically; a Preamble goes first through
    `Preamble.prepend`; two installs leave exactly one wrapper on `Controller.stage_system`."""
    from src.eval.preamble import Preamble  # package B's module

    # Restore Petri's own method at teardown so the hook's "not wrapped yet" assert holds for
    # whatever imports this module next.
    monkeypatch.setattr(Controller, "stage_system", target_hook._ORIGINAL_STAGE_SYSTEM)
    monkeypatch.setattr(target_hook, "_PREAMBLE", None)
    seen: list[str] = []
    target_hook.install(None)
    monkeypatch.setattr(target_hook, "_ORIGINAL_STAGE_SYSTEM", _recording_original(seen))

    assert asyncio.run(Controller.stage_system(None, STOCK)) == "m1"
    target_hook.install(Preamble(file="x", constitution="Be honest."))
    assert asyncio.run(Controller.stage_system(None, STOCK)) == "m1"

    assert seen == [STOCK, COMPOSED]
    assert Controller.stage_system is target_hook._stage_system_with_preamble
    assert getattr(Controller.stage_system, "_lasr_preamble_hook") is True


def test_install_none_after_a_preamble_repoints_without_rewrapping(monkeypatch):
    """An arm ladder: constitution arm then plain arm in one process -- the plain arm's target
    sees the stock prompt again and the wrapper is still the single one installed."""
    from src.eval.preamble import Preamble

    monkeypatch.setattr(Controller, "stage_system", target_hook._ORIGINAL_STAGE_SYSTEM)
    monkeypatch.setattr(target_hook, "_PREAMBLE", None)
    seen: list[str] = []
    target_hook.install(Preamble(file="x", constitution="Be honest."))
    monkeypatch.setattr(target_hook, "_ORIGINAL_STAGE_SYSTEM", _recording_original(seen))
    asyncio.run(Controller.stage_system(None, STOCK))
    target_hook.install(None)
    asyncio.run(Controller.stage_system(None, STOCK))

    assert seen == [COMPOSED, STOCK]
    assert Controller.stage_system is target_hook._stage_system_with_preamble


def test_install_refuses_a_foreign_wrapper(monkeypatch):
    """A venv whose inspect-petri was patched in place would prepend twice; refuse loudly."""
    import pytest

    async def someone_elses(self, content: str) -> str:  # pragma: no cover - never awaited
        return content

    monkeypatch.setattr(Controller, "stage_system", someone_elses)
    with pytest.raises(AssertionError, match="already wrapped"):
        target_hook.install(None)
