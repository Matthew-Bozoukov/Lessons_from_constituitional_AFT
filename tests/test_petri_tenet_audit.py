# ABOUTME: No-network checks for the tenet audit: the seeds rebuild byte-for-byte, scores are read
# ABOUTME: from the `concerning` dimension, and the constitution wrapper puts the preamble first.
import asyncio
import sys
from types import SimpleNamespace

import pytest

from src.eval.audits.petri.tenet_audit import build_seeds
from src.eval.audits.petri.tenet_audit.scores import SEEDS, sample_concerning


def test_seeds_rebuild_identically(tmp_path, monkeypatch):
    """The committed seed files are exactly what seeds/p*.yaml builds: 90 of them, no drift."""
    monkeypatch.setattr(sys, "argv", ["build_seeds", "--out", str(tmp_path)])
    build_seeds.main()
    built = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.glob("*.md")}
    committed = {p.name: p.read_text(encoding="utf-8") for p in SEEDS.glob("*.md")}
    assert len(built) == 90
    assert built == committed


def test_sample_concerning_reads_the_dimension():
    """The `concerning` score is found in whichever scorer holds it; unscored is None."""
    scored = SimpleNamespace(scores={"x": SimpleNamespace(value={"concerning": 4, "other": 9})})
    assert sample_concerning(scored) == 4.0
    assert sample_concerning(SimpleNamespace(scores={})) is None
    assert sample_concerning(SimpleNamespace(scores={"x": SimpleNamespace(value=1.0)})) is None


def test_preamble_is_prepended_only_when_configured(tmp_path, monkeypatch):
    """With PETRI_SYSTEM_PREAMBLE_FILE set the target sees the file first; unset, nothing changes."""
    pytest.importorskip("inspect_petri")
    from src.eval.audits.petri.tenet_audit import task

    seen = []

    async def original(self, content):
        seen.append(content)
        return content

    monkeypatch.setattr(task, "_ORIGINAL_STAGE_SYSTEM", original)
    pre = tmp_path / "constitution.md"
    pre.write_text("Be honest.\n", encoding="utf-8")

    monkeypatch.delenv("PETRI_SYSTEM_PREAMBLE_FILE", raising=False)
    asyncio.run(task.Controller.stage_system(None, "You are an ops agent."))
    monkeypatch.setenv("PETRI_SYSTEM_PREAMBLE_FILE", str(pre))
    asyncio.run(task.Controller.stage_system(None, "You are an ops agent."))
    assert seen == ["You are an ops agent.", "Be honest.\n\nYou are an ops agent."]
