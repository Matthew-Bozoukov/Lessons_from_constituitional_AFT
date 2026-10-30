# ABOUTME: Pins the ONE system-preamble composition every constitution-arm eval prepends: the
# ABOUTME: framing + <constitution> text, THE join, the record, the published copy, the env path.
import hashlib

import pytest

from src.eval import preamble as mod
from src.eval.preamble import ENV_VAR, FILENAME, FRAMING, Preamble, prepend_from_env

COMPOSED = "Use the following constitution to guide your behavior.\n\n<constitution>\nBe honest.\n</constitution>"


@pytest.fixture
def honest(tmp_path):
    path = tmp_path / "constitution.md"
    path.write_text("Be honest.\n", encoding="utf-8")
    return path


def test_composition_and_the_join(honest):
    p = Preamble.from_file(honest)
    assert p.text == COMPOSED and FRAMING in p.text
    assert p.prepend("You are an ops agent.") == COMPOSED + "\n\nYou are an ops agent."
    assert p.constitution == "Be honest."


def test_from_cfg_is_none_when_unset(honest):
    assert Preamble.from_cfg({}) is None
    assert Preamble.from_cfg({"system_preamble_file": None}) is None
    assert Preamble.from_cfg({"system_preamble_file": ""}) is None
    assert Preamble.from_cfg(None) is None
    assert Preamble.from_cfg({"system_preamble_file": str(honest)}) == Preamble.from_file(honest)


def test_missing_and_empty_files_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="system_preamble_file not found"):
        Preamble.from_file(tmp_path / "nope.md")
    blank = tmp_path / "blank.md"
    blank.write_text("  \n\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="empty"):
        Preamble.from_file(blank)


def test_hashes_record_and_env(honest):
    from src.data.synth.ours.constitution import full_text

    p = Preamble.from_file(honest)
    assert p.constitution_sha256 == hashlib.sha256(full_text(honest).encode()).hexdigest()
    assert p.sha256 == hashlib.sha256(COMPOSED.encode()).hexdigest()
    rec = p.record()
    assert set(rec) == {"file", "sha256", "constitution_sha256", "framing", "text"}
    assert rec["text"] == COMPOSED and rec["framing"] == FRAMING and rec["file"] == str(honest)
    assert p.env() == {ENV_VAR: str(honest.resolve())}


def test_materialise_writes_the_published_copy_idempotently(honest, tmp_path):
    p = Preamble.from_file(honest)
    meta = tmp_path / "run" / "metadata"
    out = p.materialise(meta)
    assert out == meta / FILENAME and out.read_text(encoding="utf-8") == COMPOSED + "\n"
    assert p.materialise(meta) == out and out.read_text(encoding="utf-8") == COMPOSED + "\n"
    assert not (tmp_path / "run" / "rollouts").exists()  # creates only metadata/


def test_prepend_from_env(honest, monkeypatch):
    mod._cached.cache_clear()
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert prepend_from_env("stock") == "stock"
    monkeypatch.setenv(ENV_VAR, str(honest))
    assert prepend_from_env("stock") == COMPOSED + "\n\nstock"
    mod._cached.cache_clear()


def test_the_real_constitution_is_the_pinned_text():
    # Pins package A's edit: the file holds ONLY the priority section and the nine principles,
    # so the model-facing text carries nothing about the repo.
    p = Preamble.from_cfg({"system_preamble_file": "constitutions/claude_distilled_09_principles/constitution.md"})
    assert p.constitution.startswith("## Priority and conflict-resolution")
    assert p.constitution_sha256 == "7726f860ad97a443c8b49e12c7fc8d8bcb69d2286dc5f85934153b87a5045c11"
