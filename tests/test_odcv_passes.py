# ABOUTME: Offline tests for ODCV multi-pass plumbing: the empty-transcript audit, the
# ABOUTME: rollout_NNN combiner, and the runner's retry-once-then-drop pass policy.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.odcv import runner
from src.eval.misalignment.odcv.passes import audit_pass, combine_passes

MK = "mk"


def make_pass(root: Path, name: str, cells, manifest: bool = True) -> Path:
    """Build a fake pass dir. cells = [(variant, scenario, transcript_text_or_None)]."""
    d = root / name
    for variant in ("mandated", "incentivized"):
        (d / "agent_logs" / f"{MK}-{variant}" / "experiments").mkdir(parents=True,
                                                                     exist_ok=True)
    for variant, scen, text in cells:
        scen_dir = d / "agent_logs" / f"{MK}-{variant}" / "experiments" / scen
        scen_dir.mkdir(parents=True, exist_ok=True)
        if text is not None:
            (scen_dir / "messages_record.txt").write_text(text)
    if manifest:
        (d / "rollout_manifest.json").write_text(json.dumps({
            "n_scenarios": len(cells),
            "rollout_cost_usd": 1.0,
            "results": [{"variant": v, "scenario": s, "status": "ok"} for v, s, _ in cells],
        }))
    return d


def test_audit_clean_pass(tmp_path):
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x"), ("incentivized", "S2", "y")])
    audit = audit_pass(d)
    assert audit["clean"] and audit["missing_cells"] == 0
    assert audit["transcripts_nonempty"] == 2 and audit["empty_transcripts"] == 0


def test_audit_flags_empty_and_missing_transcripts(tmp_path):
    # S2's transcript is zero-byte, S3's was never written — both are dirty cells even
    # though every manifest status says ok (the 2026-08-18 silent failure).
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x"), ("incentivized", "S2", ""),
                                   ("incentivized", "S3", None)])
    audit = audit_pass(d)
    assert not audit["clean"]
    assert audit["missing_cells"] == 2
    assert audit["empty_transcripts"] == 1  # S2 exists but is zero-byte


def test_audit_without_manifest_is_never_clean(tmp_path):
    d = make_pass(tmp_path, "p1", [("mandated", "S1", "x")], manifest=False)
    audit = audit_pass(d)
    assert not audit["clean"]
    assert audit["missing_cells"] is None
    assert audit["statuses"] == {"NO_MANIFEST": 1}


def test_combine_passes_builds_rollout_layout_and_skips_empty(tmp_path):
    p1 = make_pass(tmp_path, "p1", [("mandated", "S1", "a"), ("incentivized", "S2", "b")])
    p2 = make_pass(tmp_path, "p2", [("mandated", "S1", "c"), ("incentivized", "S2", "")])
    out = tmp_path / "combined"
    manifest = combine_passes([p1, p2], out, MK)

    rec = out / "agent_logs" / f"{MK}-mandated" / "experiments" / "S1"
    assert (rec / "rollout_000" / "messages_record.txt").read_text() == "a"
    assert (rec / "rollout_001" / "messages_record.txt").read_text() == "c"
    # p2's empty S2 must NOT be copied — the judge would score it as a clean rollout.
    s2 = out / "agent_logs" / f"{MK}-incentivized" / "experiments" / "S2"
    assert (s2 / "rollout_000").is_dir() and not (s2 / "rollout_001").exists()
    assert manifest["n_transcripts"] == 3
    assert manifest["skipped_empty"] == ["p2/incentivized/S2"]
    assert json.loads((out / "combine_manifest.json").read_text()) == manifest
    with pytest.raises(AssertionError, match="refusing to overwrite"):
        combine_passes([p1], out, MK)


class FakeRollout:
    """Stands in for odcv_rollout.main: fresh calls make a pass, resume calls may heal it."""

    def __init__(self, dirty_passes: set[int], heal_on_resume: bool):
        self.dirty_passes = dirty_passes  # 0-based fresh-pass indices born dirty
        self.heal_on_resume = heal_on_resume
        self.fresh = 0
        self.resumed: list[str] = []

    def __call__(self, config: str, smoke: bool = False, resume: str = ""):
        cfg = OmegaConf.load(config)
        root = Path(cfg.output_root) / cfg.model_key
        if resume:
            d = Path(resume)
            self.resumed.append(d.name)
            if self.heal_on_resume:  # write the missing transcript
                scen = d / "agent_logs" / f"{MK}-incentivized" / "experiments" / "S2"
                (scen / "messages_record.txt").write_text("healed")
            return d
        idx = self.fresh
        self.fresh += 1
        cells = [("mandated", "S1", "x"),
                 ("incentivized", "S2", None if idx in self.dirty_passes else "y")]
        return make_pass(root, f"pass{idx}", cells)


def fake_judge(rollout_dir: str, config: str, max_workers: int, smoke: bool):
    (Path(rollout_dir) / "results.json").write_text(json.dumps({"mr": 0.1}))
    evals = Path(rollout_dir) / "evaluations"
    evals.mkdir()
    (evals / "scores_grok.json").write_text("{}")


def fake_progress_judge(rollout_dir: str, config: str, max_workers: int, smoke: bool):
    """The second judging pass, stubbed like the first: these tests are about the pass
    audit, and both judges reach OpenRouter."""
    out = {"axis": "progress", "ours": {"overall": {"tp_mean": 3.0}}}
    (Path(rollout_dir) / "progress_results.json").write_text(json.dumps(out))
    return out


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_prune_networks", lambda: None)
    monkeypatch.setattr(runner.odcv_judge, "main", fake_judge)
    monkeypatch.setattr(runner.progress_judge, "main", fake_progress_judge)
    target = SimpleNamespace(model_name="m", base_url="http://localhost:8000/v1",
                             spec=SimpleNamespace(model_key=MK))
    cfg = OmegaConf.create({"passes": 2, "smoke": False, "judge_workers": 1})
    return tmp_path, monkeypatch, target, cfg


def test_runner_retries_dirty_pass_once_and_keeps_it_when_healed(runner_env):
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={1}, heal_on_resume=True)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    results = runner.run(target, cfg, tmp_path)
    assert fake.fresh == 2 and fake.resumed == ["pass1"]
    assert results["passes"] == {**results["passes"], "requested": 2, "kept": 2,
                                 "dropped": 0, "n_transcripts": 4}
    # Published layout: every transcript exactly once under rollouts/<variant>/<Scenario>/
    # pass<N>/, judge outputs under results/, provenance under metadata/, working tree gone.
    assert (tmp_path / "rollouts" / "mandated" / "S1" / "pass1"
            / "messages_record.txt").read_text() == "x"
    healed = tmp_path / "rollouts" / "incentivized" / "S2" / "pass2"
    assert (healed / "messages_record.txt").read_text() == "healed"
    meta = json.loads((healed / "cell_meta.json").read_text())
    assert meta["judged"] and meta["pass"] == 2 and meta["transcript_bytes"] > 0
    assert json.loads((tmp_path / "results" / "results.json").read_text()) == {"mr": 0.1}
    assert (tmp_path / "results" / "scores_grok.json").is_file()
    assert (tmp_path / "metadata" / "pass_summary.json").is_file()
    assert (tmp_path / "metadata" / "combine_manifest.json").is_file()
    assert (tmp_path / "metadata" / "passes" / "pass1_rollout_manifest.json").is_file()
    assert not (tmp_path / MK).exists()  # raw/combined working tree consumed


def test_runner_drops_pass_still_dirty_after_one_retry(runner_env):
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={0}, heal_on_resume=False)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    results = runner.run(target, cfg, tmp_path)
    assert fake.resumed == ["pass0"]  # exactly one retry, never two
    assert results["passes"]["kept"] == 1 and results["passes"]["dropped"] == 1
    summary = json.loads((tmp_path / "metadata" / "pass_summary.json").read_text())
    assert [a["kept"] for a in summary["audits"]] == [False, True]
    # The dropped pass (execution pass1) is preserved in rollouts/ but marked unjudged:
    # its produced transcript carries judged=False, its missing cell has meta only.
    s1 = tmp_path / "rollouts" / "mandated" / "S1" / "pass1"
    assert (s1 / "messages_record.txt").is_file()
    assert json.loads((s1 / "cell_meta.json").read_text())["judged"] is False
    s2 = tmp_path / "rollouts" / "incentivized" / "S2" / "pass1"
    assert not (s2 / "messages_record.txt").exists()
    assert json.loads((s2 / "cell_meta.json").read_text())["transcript_bytes"] == 0
    assert json.loads((tmp_path / "rollouts" / "incentivized" / "S2" / "pass2"
                       / "cell_meta.json").read_text())["judged"] is True
    assert not (tmp_path / MK).exists()


def test_runner_fails_fast_when_every_pass_drops(runner_env):
    tmp_path, monkeypatch, target, cfg = runner_env
    fake = FakeRollout(dirty_passes={0, 1}, heal_on_resume=False)
    monkeypatch.setattr(runner.odcv_rollout, "main", fake)

    with pytest.raises(RuntimeError, match="failed their audit"):
        runner.run(target, cfg, tmp_path)
    # No packaging on total failure: the forensics stay in place at the root.
    assert (tmp_path / "pass_summary.json").is_file()
    assert not (tmp_path / "rollouts").exists()
