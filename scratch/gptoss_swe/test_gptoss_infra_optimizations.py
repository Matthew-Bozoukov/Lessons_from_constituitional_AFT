# ABOUTME: Offline regression checks for evidence preservation and infrastructure-only optimizations.
# ABOUTME: No provider calls, model generation or rentals; Linux protocol tests use the pinned agent env.
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from scratch.gptoss_swe.native_evidence import mirror, mounts


def test_native_mirror_preserves_bytes_and_retains_unrelated_files(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "nested").mkdir()
    (src / "nested" / "raw.body").write_bytes(b"partial\x00response")
    (dst / "host-status.json").write_text("keep")
    previous = {}
    assert mirror(src, dst, previous) == 1
    assert mirror(src, dst, previous) == 0
    (src / "nested" / "raw.body").write_bytes(b"completed\x00response")
    assert mirror(src, dst, previous, verify=True) == 1
    assert (dst / "nested" / "raw.body").read_bytes() == b"completed\x00response"
    assert (dst / "host-status.json").read_text() == "keep"


def test_native_mounts_do_not_cover_config_or_credentials():
    args = mounts("campaign", "mxfp4", Path("output/run"))
    assert len(args) == 6
    assert all("/work/output/run/mxfp4/swe" in x for x in args[1::2])
    assert not any(".env" in x or "swe.yaml" in x for x in args)


def test_failed_gpu_collection_never_erases_evidence(tmp_path, monkeypatch):
    from scratch.gptoss_swe import mxfp4_pilot as pilot

    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    (tmp_path / "ssh.json").write_text(
        json.dumps(dict(key="key", port=22, host="host"))
    )
    (tmp_path / "vllm.log").write_text("valuable startup")
    monkeypatch.setattr(
        pilot.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b""),
    )
    pilot.collect_gpu()
    assert (tmp_path / "vllm.log").read_text() == "valuable startup"
    assert len(list(tmp_path.glob("gpu-collection-*.json"))) == 1


def test_final_copy_refuses_changing_source(tmp_path, monkeypatch):
    import scratch.gptoss_swe.native_evidence as native

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "data").write_bytes(b"old")
    original = native.shutil.copyfileobj

    def change(source, target):
        original(source, target)
        (src / "data").write_bytes(b"changed")

    monkeypatch.setattr(native.shutil, "copyfileobj", change)
    with pytest.raises(RuntimeError, match="still changing"):
        mirror(src, dst, verify=True)
    assert not (dst / "data").exists()
