# ABOUTME: Validate zero-request recovery admission and directory preparation without model calls.
# ABOUTME: Preserve original owner claims and reject recovery whenever paid evidence exists.
import json
import pytest
from scratch.gptoss_swe import mxfp4_odcv_recovery as recovery


def setup_root(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.pilot, "ROOT", tmp_path)
    r = tmp_path / "tinker"
    (r / "odcv").mkdir(parents=True)
    recovery.pilot.save(
        r / "odcv/status.json",
        dict(phase="held", error="FileNotFoundError odcv_config.yaml"),
    )
    (r / "odcv-owner.log").write_text(
        "OmegaConf.save(cfg, cfg_path) FileNotFoundError odcv_config.yaml"
    )
    recovery.pilot.save(r / "odcv-budget.json", dict(requests={}))
    return r


def test_admission_rejects_paid_history(tmp_path, monkeypatch):
    r = setup_root(tmp_path, monkeypatch)
    assert recovery.verify_unsent("tinker")["zero_generation_requests"]
    recovery.pilot.save(r / "odcv-budget.json", dict(requests={"paid": {}}))
    with pytest.raises(AssertionError):
        recovery.verify_unsent("tinker")


def test_admission_rejects_raw_response(tmp_path, monkeypatch):
    r = setup_root(tmp_path, monkeypatch)
    recovery.pilot.save(r / "odcv-traces/response.json", {})
    with pytest.raises(AssertionError):
        recovery.verify_unsent("tinker")


def test_child_creates_directory_preserves_original_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.pilot, "ROOT", tmp_path)
    monkeypatch.setattr(recovery.pilot, "environment", lambda: {})
    r = tmp_path / "tinker"
    r.mkdir()
    (r / "odcv-owner-started.json").write_text("original")
    original_claim = recovery.pilot.claim
    monkeypatch.setattr(recovery.pilot, "claim", original_claim)

    def actual_entry(backend):
        recovery.pilot.claim(r / "odcv-owner-started.json")
        # Reproduce the real runner's first write before it creates any pass directory.
        (r / "odcv/odcv_config.yaml").write_text("offline")

    monkeypatch.setattr(recovery.pilot, "odcv", actual_entry)
    recovery.child("tinker")
    assert (r / "odcv-owner-started.json").read_text() == "original"
    assert json.loads((r / "odcv-recovery-owner-started.json").read_text())["pid"] > 0
    assert (r / "odcv/odcv_config.yaml").exists()
